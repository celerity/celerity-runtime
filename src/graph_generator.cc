#include "graph_generator.h"

#include <allscale/utils/string_utils.h>

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "graph_transformer.h"
#include "task.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	graph_generator::graph_generator(size_t num_nodes, task_manager& tm, command_graph& cdag) : task_mngr(tm), num_nodes(num_nodes), cdag(cdag) {
		// Build init command for each node (these are required to properly handle host-initialized buffers).
		for(auto i = 0u; i < num_nodes; ++i) {
			const auto init_cmd = cdag.create<nop_command>(i);
			node_data[i].init_cid = init_cmd->get_cid();
		}
		std::fill_n(std::back_inserter(prev_horizon_cmds), num_nodes, nullptr);
	}

	void graph_generator::add_buffer(buffer_id bid, const cl::sycl::range<3>& range) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// Initialize the whole range to all nodes, so that we always use local buffer ranges when they haven't been written to (on any node) yet.
		// TODO: Consider better handling for when buffers are not host initialized
		std::vector<node_id> all_nodes(num_nodes);
		for(auto i = 0u; i < num_nodes; ++i) {
			all_nodes[i] = i;
			node_data[i].buffer_last_writer.emplace(bid, range);
			node_data[i].buffer_last_writer.at(bid).update_region(subrange_to_grid_box({cl::sycl::id<3>(), range}), node_data[i].init_cid);
		}

		buffer_states.emplace(bid, region_map<std::vector<node_id>>{range, all_nodes});
	}

	void graph_generator::build_task(task_id tid, const std::vector<graph_transformer*>& transformers) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// TODO: Maybe assert that this task hasn't been processed before

		auto tsk = task_mngr.get_task(tid);
		if(tsk->get_type() == task_type::COLLECTIVE) {
			for(size_t nid = 0; nid < num_nodes; ++nid) {
				auto offset = cl::sycl::id<1>{nid};
				auto range = cl::sycl::range<1>{1};
				const auto sr = subrange<1>{offset, range};
				auto* cmd = cdag.create<task_command>(nid, tid, sr);

				// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
				// they are executed in the same order on every node.
				auto cgid = tsk->get_collective_group_id();
				if(auto prev = last_collective_commands.find({nid, cgid}); prev != last_collective_commands.end()) {
					cdag.add_dependency(cmd, cdag.get(prev->second), dependency_kind::ORDER_DEP);
					last_collective_commands.erase(prev);
				}
				last_collective_commands.emplace(std::pair{nid, cgid}, cmd->get_cid());
			}
		} else {
			const auto sr = subrange<3>{tsk->get_global_offset(), tsk->get_global_size()};
			cdag.create<task_command>(0, tid, sr);
		}

		for(auto& t : transformers) {
			t->transform_task(tsk, cdag);
		}

		// TODO: At some point we might want to do this also before calling transformers
		// --> So that more advanced transformations can also take data transfers into account
		process_task_data_requirements(tid);

		if(should_generate_horizon()) { generate_horizon(); }
	}

	using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<3>>>;

	buffer_requirements_map get_buffer_requirements_for_mapped_access(const task* tsk, subrange<3> sr, const cl::sycl::range<3> global_size) {
		buffer_requirements_map result;
		const auto& access_map = tsk->get_buffer_access_map();
		const auto buffers = access_map.get_accessed_buffers();
		for(const buffer_id bid : buffers) {
			const auto modes = access_map.get_access_modes(bid);
			for(auto m : modes) {
				result[bid][m] = access_map.get_requirements_for_access(bid, m, sr, global_size);
			}
		}
		return result;
	}

	void graph_generator::generate_anti_dependencies(task_id tid, buffer_id bid, const region_map<std::optional<command_id>>& last_writers_map,
	    const GridRegion<3>& write_req, abstract_command* write_cmd) {
		const auto last_writers = last_writers_map.get_region_values(write_req);
		for(auto& box_and_writers : last_writers) {
			if(box_and_writers.second == std::nullopt) continue;
			const command_id last_writer_cid = *box_and_writers.second;
			const auto last_writer_cmd = cdag.get(last_writer_cid);
			assert(!isa<task_command>(last_writer_cmd) || static_cast<task_command*>(last_writer_cmd)->get_tid() != tid);

			// Add anti-dependencies onto all dependents of the writer
			bool has_dependents = false;
			for(auto d : last_writer_cmd->get_dependents()) {
				// Only consider true dependencies
				if(d.kind != dependency_kind::TRUE_DEP) continue;

				const auto cmd = d.node;

				// We might have already generated new commands within the same task that also depend on this; in that case, skip it
				if(isa<task_command>(cmd) && static_cast<task_command*>(cmd)->get_tid() == tid) continue;

				// So far we don't know whether the dependent actually intersects with the subrange we're writing
				const auto& command_reads = command_buffer_reads[cmd->get_cid()];
				const auto buffer_reads_it = command_reads.find(bid);
				if(buffer_reads_it == command_reads.end()) continue; // The task might be a dependent because of another buffer
				if(!GridRegion<3>::intersect(write_req, buffer_reads_it->second).empty()) {
					has_dependents = true;
					cdag.add_dependency(write_cmd, cmd, dependency_kind::ANTI_DEP);
				}
			}

			// In some cases (horizons, master node host task, weird discard_* constructs...)
			// the last writer might not have any dependents. Just add the anti-dependency onto the writer itself then.
			if(!has_dependents) {
				// Don't add anti-dependencies onto the init command
				if(last_writer_cid == node_data[write_cmd->get_nid()].init_cid) continue;

				cdag.add_dependency(write_cmd, last_writer_cmd, dependency_kind::ANTI_DEP);

				// This is a good time to validate our assumption that every AWAIT_PUSH command has a dependent
				assert(!isa<await_push_command>(last_writer_cmd));
			}
		}
	}

	namespace {
		template <typename RegionMap>
		inline void add_dependencies_for_box(command_graph& cdag, abstract_command* cmd, const RegionMap& map, const GridBox<3>& box) {
			auto sources = map.get_region_values(box);
			for(const auto& source : sources) {
				auto source_cmd = cdag.get(*source.second);
				cdag.add_dependency(cmd, source_cmd);
			}
		}
	} // namespace

	void graph_generator::process_task_data_requirements(task_id tid) {
		const auto tsk = task_mngr.get_task(tid);

		// Copy the list of task commands so we can safely modify the command graph in the loop below
		// NOTE: We assume that none of these commands are deleted
		std::vector<task_command*> task_commands;
		for(auto cmd : cdag.task_commands(tid)) {
			task_commands.push_back(cmd);
		}

		// Store a list of writes (directly done by computation) while processing commands,
		// so that we can update the state with them at the end, while using only the state
		// before this task while resolving its dependencies.
		std::vector<std::tuple<node_id, buffer_id, GridRegion<3>>> buffer_state_write_list;
		std::vector<std::tuple<node_id, buffer_id, GridRegion<3>, command_id>> per_node_last_writer_update_list;

		// Remember all generated PUSHes for determining intra-task anti-dependencies.
		std::vector<push_command*> generated_pushes;

		for(auto cmd : task_commands) {
			const command_id cid = cmd->get_cid();
			const node_id nid = cmd->get_nid();

			auto requirements = get_buffer_requirements_for_mapped_access(tsk.get(), cmd->get_execution_range(), tsk->get_global_size());

			for(auto& it : requirements) {
				const buffer_id bid = it.first;
				const auto& reqs_by_mode = it.second;

				const auto& node_buffer_last_writer = node_data.at(nid).buffer_last_writer.at(bid);

				for(const auto mode : detail::access::all_modes) {
					if(reqs_by_mode.count(mode) == 0) continue;
					const auto& req = reqs_by_mode.at(mode);
					if(req.empty()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						continue;
					}

					// Add access mode and range to execution command node label for debugging
					cmd->debug_label = fmt::format("{}{} {} {}\\n", cmd->debug_label, detail::access::mode_traits::name(mode), bid, toString(req));

					if(detail::access::mode_traits::is_consumer(mode)) {
						// Store the read access for determining anti-dependencies later on
						command_buffer_reads[cid][bid] = GridRegion<3>::merge(command_buffer_reads[cid][bid], req);

						// Determine whether data transfers are required to fulfill the read requirements
						const auto buffer_source_nodes = buffer_states.at(bid).get_region_values(req);
						assert(!buffer_source_nodes.empty());

						for(auto& box_and_sources : buffer_source_nodes) {
							const auto& box = box_and_sources.first;
							const auto& box_sources = box_and_sources.second;
							assert(!box_sources.empty());

							if(std::find(box_sources.cbegin(), box_sources.cend(), nid) != box_sources.cend()) {
								// No need to push if found locally, but make sure to add dependencies
								add_dependencies_for_box(cdag, cmd, node_buffer_last_writer, box);
								continue;
							}

							// If not local, the original producer is the primary source (i.e., a task_command, as opposed to an AWAIT_PUSH)
							// it is used as the transfer source to avoid creating long dependency chains across nodes.
							// TODO: For larger numbers of nodes this might become a bottleneck.
							auto source_nid = box_sources[0];

							// Generate PUSH command
							push_command* push_cmd = nullptr;
							{
								push_cmd = cdag.create<push_command>(source_nid, bid, nid, grid_box_to_subrange(box));
								generated_pushes.push_back(push_cmd);

								// Store the read access on the pushing node
								command_buffer_reads[push_cmd->get_cid()][bid] = GridRegion<3>::merge(command_buffer_reads[push_cmd->get_cid()][bid], box);

								// Add dependencies on the source node between the PUSH and the commands that last wrote that box
								add_dependencies_for_box(cdag, push_cmd, node_data.at(source_nid).buffer_last_writer.at(bid), box);
							}

							// Generate AWAIT_PUSH command
							{
								auto await_push_cmd = cdag.create<await_push_command>(nid, push_cmd);

								cdag.add_dependency(cmd, await_push_cmd);
								generate_anti_dependencies(tid, bid, node_buffer_last_writer, box, await_push_cmd);

								// Remember the fact that we now have this valid buffer range on this node.
								auto new_box_sources = box_sources;
								new_box_sources.push_back(nid);
								buffer_states.at(bid).update_region(box, new_box_sources);

								per_node_last_writer_update_list.emplace_back(nid, bid, box, await_push_cmd->get_cid());
							}
						}
					}

					if(detail::access::mode_traits::is_producer(mode)) {
						generate_anti_dependencies(tid, bid, node_buffer_last_writer, req, cmd);

						// After this task is completed, this node and command are the last writer of this region
						buffer_state_write_list.emplace_back(nid, bid, req);
						per_node_last_writer_update_list.emplace_back(nid, bid, req, cid);
					}
				}
			}
		}

		// Update global buffer_states with writes from task_commands after handling is complete.
		for(const auto& write : buffer_state_write_list) {
			const auto& nid = std::get<0>(write);
			const auto& bid = std::get<1>(write);
			const auto& region = std::get<2>(write);
			buffer_states.at(bid).update_region(region, {nid});
		}

		// Update per-node last writer information to take into account writes from await_pushes generated for this task.
		for(const auto& write : per_node_last_writer_update_list) {
			const auto& nid = std::get<0>(write);
			const auto& bid = std::get<1>(write);
			const auto& region = std::get<2>(write);
			const auto& cid = std::get<3>(write);
			node_data.at(nid).buffer_last_writer.at(bid).update_region(region, {cid});
		}

		// As the last step, we determine potential "intra-task" race conditions.
		// These can happen in rare cases, when the node that PUSHes a buffer range also writes to that range within the same task.
		// We cannot do this while generating the PUSH command, as we may not have the writing command recorded at that point.
		for(auto cmd : generated_pushes) {
			const auto push_cmd = static_cast<push_command*>(cmd);
			const auto last_writers =
			    node_data.at(push_cmd->get_nid()).buffer_last_writer.at(push_cmd->get_bid()).get_region_values(subrange_to_grid_box(push_cmd->get_range()));
			for(auto& box_and_writer : last_writers) {
				assert(!box_and_writer.first.empty());         // If we want to push it it cannot be empty
				assert(box_and_writer.second != std::nullopt); // Exactly one command last wrote to that box
				const auto writer_cmd = cdag.get(*box_and_writer.second);
				assert(writer_cmd != nullptr);

				// We're only interested in writes that happen within the same task as the PUSH
				if(isa<task_command>(writer_cmd) && static_cast<task_command*>(writer_cmd)->get_tid() == tid) {
					// In certain situations the PUSH might have a true dependency on the last writer,
					// in that case don't add an anti-dependency (as that would cause a cycle).
					if(cmd->has_dependency(writer_cmd, dependency_kind::TRUE_DEP)) {
						// This can currently only happen for AWAIT_PUSH commands.
						assert(isa<await_push_command>(writer_cmd));
						continue;
					}
					cdag.add_dependency(writer_cmd, push_cmd, dependency_kind::ANTI_DEP);
				}
			}
		}
	}


	bool graph_generator::should_generate_horizon() const { return cdag.get_max_pseudo_critical_path_length() >= prev_horizon_cpath_max + horizon_step_size; }

	void graph_generator::generate_horizon() {
		prev_horizon_cpath_max = cdag.get_max_pseudo_critical_path_length();
		detail::command_id lowest_prev_hid = 0;
		for(node_id node = 0; node < num_nodes; ++node) {
			// TODO this could be optimized to something like cdag.apply_horizon(node_id, horizon_cmd) with much fewer internal operations
			auto previous_execution_front = cdag.get_execution_front(node);
			// Build horizon command and make current front depend on it
			auto horizon_cmd = cdag.create<horizon_command>(node);
			for(const auto& front_cmd : previous_execution_front) {
				cdag.add_dependency(horizon_cmd, front_cmd);
			}
			// Apply the previous horizon to data structures
			auto prev_horizon = prev_horizon_cmds[node];
			if(prev_horizon != nullptr) {
				// update "buffer_last_writer" structures to subsume pre-horizon commands
				auto prev_hid = prev_horizon->get_cid();
				for(auto& blw_pair : node_data[node].buffer_last_writer) {
					blw_pair.second.apply_to_values([prev_hid](std::optional<command_id> cid) -> std::optional<command_id> {
						if(!cid) return cid;
						return {std::max(prev_hid, *cid)};
					});
				}
				// update lowest previous horizon id (for later command deletion)
				if(lowest_prev_hid == 0) {
					lowest_prev_hid = prev_hid;
				} else {
					lowest_prev_hid = std::min(lowest_prev_hid, prev_hid);
				}
			}
			prev_horizon_cmds[node] = horizon_cmd;
		}
		// After updating all the data structures, delete before-cleanup-horizon commands
		// Also remove commands from command_buffer_reads (if it exists)
		if(cleanup_horizon_id > 0) {
			cdag.erase_if([&](abstract_command* cmd) {
				if(cmd->get_cid() < cleanup_horizon_id) {
					assert(cmd->is_flushed() && "Cannot delete unflushed command");
					command_buffer_reads.erase(cmd->get_cid());
					return true;
				}
				return false;
			});
		}
		cleanup_horizon_id = lowest_prev_hid;
	}

} // namespace detail
} // namespace celerity
