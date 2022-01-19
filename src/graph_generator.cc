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

	graph_generator::graph_generator(size_t num_nodes, task_manager& tm, reduction_manager& rm, command_graph& cdag)
	    : task_mngr(tm), reduction_mngr(rm), num_nodes(num_nodes), cdag(cdag) {
		// Build init command for each node (these are required to properly handle anti-dependencies on host-initialized buffers).
		// We manually generate the first set of commands; horizons are used later on (see generate_horizon).
		for(auto i = 0u; i < num_nodes; ++i) {
			const auto init_cmd = cdag.create<nop_command>(i);
			node_data[i].current_init_cid = init_cmd->get_cid();
		}
		std::fill_n(std::back_inserter(current_horizon_cmds), num_nodes, nullptr);
	}

	void graph_generator::add_buffer(buffer_id bid, const cl::sycl::range<3>& range) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// Initialize the whole range to all nodes, so that we always use local buffer ranges when they haven't been written to (on any node) yet.
		// TODO: Consider better handling for when buffers are not host initialized
		std::vector<node_id> all_nodes(num_nodes);
		for(auto i = 0u; i < num_nodes; ++i) {
			all_nodes[i] = i;
			node_data[i].buffer_last_writer.emplace(bid, range);
			node_data[i].buffer_last_writer.at(bid).update_region(subrange_to_grid_box({cl::sycl::id<3>(), range}), node_data[i].current_init_cid);
		}

		buffer_states.emplace(bid, distributed_state{{range, std::move(all_nodes)}});
	}

	void graph_generator::build_task(task_id tid, const std::vector<graph_transformer*>& transformers) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// TODO: Maybe assert that this task hasn't been processed before

		auto tsk = task_mngr.get_task(tid);

		if(tsk->get_type() == task_type::HORIZON) {
			generate_horizon(tid);
			return;
		}

		if(tsk->get_type() == task_type::COLLECTIVE) {
			for(size_t nid = 0; nid < num_nodes; ++nid) {
				auto offset = cl::sycl::id<1>{nid};
				auto range = cl::sycl::range<1>{1};
				const auto sr = subrange_cast<3>(subrange<1>{offset, range});
				auto* cmd = cdag.create<execution_command>(nid, tid, sr);

				// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
				// they are executed in the same order on every node.
				auto cgid = tsk->get_collective_group_id();
				auto& last_collective_commands = node_data.at(nid).last_collective_commands;
				if(auto prev = last_collective_commands.find(cgid); prev != last_collective_commands.end()) {
					cdag.add_dependency(cmd, cdag.get(prev->second), dependency_kind::ORDER_DEP);
					last_collective_commands.erase(prev);
				}
				last_collective_commands.emplace(cgid, cmd->get_cid());
			}
		} else {
			const auto sr = subrange<3>{tsk->get_global_offset(), tsk->get_global_size()};
			cdag.create<execution_command>(0, tid, sr);
		}

		for(auto& t : transformers) {
			t->transform_task(*tsk, cdag);
		}

#ifndef NDEBUG
		// It is currently undefined to split reduction-producer tasks into multiple chunks on the same node:
		//   - Per-node reduction intermediate results are stored with fixed access to a single SYCL buffer, so multiple chunks on the same node will race on
		//     this buffer access
		//   - Inputs to the final reduction command are ordered by origin node ids to guarantee bit-identical results. It is not possible to distinguish
		//     more than one chunk per node in the serialized commands, so different nodes can produce different final reduction results for non-associative
		//     or non-commutative operations
		if(!tsk->get_reductions().empty()) {
			std::unordered_set<node_id> producer_nids;
			for(auto& cmd : cdag.task_commands(tid)) {
				assert(producer_nids.insert(cmd->get_nid()).second);
			}
		}
#endif

		// TODO: At some point we might want to do this also before calling transformers
		// --> So that more advanced transformations can also take data transfers into account
		process_task_data_requirements(tid);
	}

	using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<3>>>;

	buffer_requirements_map get_buffer_requirements_for_mapped_access(const task* tsk, subrange<3> sr, const cl::sycl::range<3> global_size) {
		buffer_requirements_map result;
		const auto& access_map = tsk->get_buffer_access_map();
		const auto buffers = access_map.get_accessed_buffers();
		for(const buffer_id bid : buffers) {
			const auto modes = access_map.get_access_modes(bid);
			for(auto m : modes) {
				result[bid][m] = access_map.get_requirements_for_access(bid, m, tsk->get_dimensions(), sr, global_size);
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
				// Don't add anti-dependencies onto the (first! the nop_command) init command
				const auto init_cid = node_data[write_cmd->get_nid()].current_init_cid;
				if(last_writer_cid == init_cid && isa<nop_command>(cdag.get(init_cid))) continue;

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
		const auto task_commands = cdag.task_commands(tid);

		// In reductions including the current buffer value, we need to designate a single command that will read from the output buffer. To avoid unnecessary
		// data transfers to that reader, we choose a command on a node where that buffer is already present. We cannot in general guarantee an optimal choice
		// for all reductions of a task, since communicating this choice to workers must happen through the fixed-size command_data. Instead, we try to find a
		// single command_id that can initialize all reductions without requiring transfers and settle for one that requires a small-ish number of transfers if
		// that is not possible.
		command_id reduction_initializer_cid = 0;
		{
			std::unordered_set<command_id> optimal_reduction_initializer_cids;
			for(auto rid : tsk->get_reductions()) {
				auto reduction = reduction_mngr.get_reduction(rid);
				if(reduction.initialize_from_buffer) {
					if(optimal_reduction_initializer_cids.empty()) {
						// lazy-initialize (there will be no buffer-initialized reductions most of the time)
						optimal_reduction_initializer_cids.reserve(task_commands.size());
						for(auto* cmd : task_commands) {
							optimal_reduction_initializer_cids.emplace(cmd->get_cid());
						}
					}

					// If there is a command that does not need a transfer for *some* of the reductions, that's better than nothing
					reduction_initializer_cid = *optimal_reduction_initializer_cids.begin();

					auto& buffer_state = buffer_states.at(reduction.output_buffer_id);
					if(auto* distr_state = std::get_if<distributed_state>(&buffer_state)) {
						// set intersection: remove all commands on a node where the initial value is not present
						const GridBox<3> box{GridPoint<3>{1, 1, 1}};
						for(auto& [box, nids] : distr_state->region_sources.get_region_values(box)) {
							for(auto* cmd : task_commands) {
								if(std::find(nids.begin(), nids.end(), cmd->get_nid()) == nids.end()) {
									optimal_reduction_initializer_cids.erase(cmd->get_cid());
								}
							}
						}
					} // else in pending_reduction_state, we will have transfers anyway

					if(optimal_reduction_initializer_cids.empty()) break;
				}
			}

			if(!optimal_reduction_initializer_cids.empty()) {
				// There actually is a command that can initialize all reductions without a transfer
				reduction_initializer_cid = *optimal_reduction_initializer_cids.begin();
			}
		}

		// Store a list of writes (directly done by computation) while processing commands,
		// so that we can update the state with them at the end, while using only the state
		// before this task while resolving its dependencies.
		std::vector<std::tuple<node_id, buffer_id, GridRegion<3>>> buffer_state_write_list;
		std::vector<std::tuple<node_id, buffer_id, GridRegion<3>, command_id>> per_node_last_writer_update_list;
		std::unordered_map<buffer_id, std::vector<node_id>> buffer_reduction_resolve_list;

		// Remember all generated PUSHes for determining intra-task anti-dependencies.
		std::vector<push_command*> generated_pushes;

		for(auto* cmd : task_commands) {
			const command_id cid = cmd->get_cid();
			const node_id nid = cmd->get_nid();

			assert(isa<execution_command>(cmd));
			auto* ecmd = static_cast<execution_command*>(cmd);

			ecmd->set_is_reduction_initializer(cid == reduction_initializer_cid);

			auto requirements = get_buffer_requirements_for_mapped_access(tsk, ecmd->get_execution_range(), tsk->get_global_size());

			// Any reduction that includes the value previously found in the buffer (i.e. the absence of sycl::property::reduction::initialize_to_identity)
			// must read that original value in the eventual reduction_command generated by a future buffer requirement. Since whenever a buffer is used as
			// a reduction output, we replace its state with a pending_reduction_state, that original value would be lost. To avoid duplicating the buffer,
			// we simply include it in the pre-reduced state of a single execution_command.
			std::unordered_map<buffer_id, reduction_id> buffer_reduction_map;
			for(auto rid : tsk->get_reductions()) {
				auto reduction = reduction_mngr.get_reduction(rid);

				auto rmode = cl::sycl::access::mode::discard_write;
				if(ecmd->is_reduction_initializer() && reduction.initialize_from_buffer) { rmode = cl::sycl::access::mode::read_write; }

				auto bid = reduction.output_buffer_id;
#ifndef NDEBUG
				for(auto pmode : detail::access::producer_modes) {
					assert(requirements[bid].count(pmode) == 0); // We verify in the task manager that there are no reduction <-> write-access conflicts
				}
#endif

				// We need to add a proper requirement here because bid might itself be in pending_reduction_state
				requirements[bid][rmode] = GridRegion<3>{{1, 1, 1}};

				// TODO fill in debug build only
				buffer_reduction_map.emplace(bid, rid);
			}

			for(auto& it : requirements) {
				const buffer_id bid = it.first;
				const auto& reqs_by_mode = it.second;

				auto& buffer_state = buffer_states.at(bid);
				const auto& node_buffer_last_writer = node_data.at(nid).buffer_last_writer.at(bid);

				std::vector<cl::sycl::access::mode> required_modes;
				for(const auto mode : detail::access::all_modes) {
					if(auto req_it = reqs_by_mode.find(mode); req_it != reqs_by_mode.end()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						if(!req_it->second.empty()) { required_modes.push_back(mode); }
					}
				}

				// Don't add reduction commands within the loop to make sure there is at most one reduction command even in the presence of multiple
				// consumer requirements
				const bool is_pending_reduction = std::holds_alternative<pending_reduction_state>(buffer_state);
				const bool generate_reduction =
				    is_pending_reduction && std::any_of(required_modes.begin(), required_modes.end(), detail::access::mode_traits::is_consumer);

				if(is_pending_reduction && !generate_reduction) {
					// TODO the per-node reduction result is discarded - warn user about dead store
				}

				for(const auto mode : required_modes) {
					const auto& req = reqs_by_mode.at(mode);

					// Add access mode and range to execution command node label for debugging
					if(auto rid_iter = buffer_reduction_map.find(bid); rid_iter != buffer_reduction_map.end()) {
						cmd->debug_label += fmt::format("(R{}) ", rid_iter->second);
					}
					cmd->debug_label += fmt::format("{} {} {}\\n", detail::access::mode_traits::name(mode), bid, toString(req));

					if(detail::access::mode_traits::is_consumer(mode)) {
						// Store the read access for determining anti-dependencies later on
						command_buffer_reads[cid][bid] = GridRegion<3>::merge(command_buffer_reads[cid][bid], req);

						if(auto* distributed = std::get_if<distributed_state>(&buffer_state)) {
							// Determine whether data transfers are required to fulfill the read requirements
							const auto buffer_source_nodes = distributed->region_sources.get_region_values(req);
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

								// If not local, the original producer is the primary source (i.e., an execution_command, as opposed to an AWAIT_PUSH)
								// it is used as the transfer source to avoid creating long dependency chains across nodes.
								// TODO: For larger numbers of nodes this might become a bottleneck.
								auto source_nid = box_sources[0];

								// Generate PUSH command
								push_command* push_cmd;
								{
									push_cmd = cdag.create<push_command>(source_nid, bid, 0, nid, grid_box_to_subrange(box));
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
									distributed->region_sources.update_region(box, new_box_sources);

									per_node_last_writer_update_list.emplace_back(nid, bid, box, await_push_cmd->get_cid());
								}
							}
						}
					}

					if(detail::access::mode_traits::is_producer(mode)) {
						// If we are going to insert a reduction command, we will also create a true-dependency chain to the last writer. The new last writer
						// cid however is not known at this point because the the reduction command has not been generated yet. Instead, we simply skip
						// generating anti-dependencies around this requirement. This might not be valid if (multivariate) reductions ever operate on regions.
						if(!generate_reduction) { generate_anti_dependencies(tid, bid, node_buffer_last_writer, req, cmd); }

						// After this task is completed, this node and command are the last writer of this region
						buffer_state_write_list.emplace_back(nid, bid, req);
						per_node_last_writer_update_list.emplace_back(nid, bid, req, cid);
					}
				}

				if(generate_reduction) {
					auto& pending_reduction = std::get<pending_reduction_state>(buffer_state);
					auto rid = pending_reduction.reduction;

					const GridBox<3> box{GridPoint<3>{1, 1, 1}};
					const subrange<3> sr{{}, {1, 1, 1}};

					auto reduce_cmd = cdag.create<reduction_command>(nid, rid);

					// pending_reduction_state with 1 operand is equivalent to a distributed_state with 1 operand. We make sure to always generate the
					// latter since it allows us to avoid generating a no-op reduction command (and also gets rid of an edge case)
					assert(pending_reduction.operand_sources.size() > 1);

					for(auto source_nid : pending_reduction.operand_sources) {
						if(source_nid == nid) {
							add_dependencies_for_box(cdag, reduce_cmd, node_data.at(source_nid).buffer_last_writer.at(bid), box);
						} else {
							auto push_cmd = cdag.create<push_command>(source_nid, bid, rid, nid, sr);
							command_buffer_reads[push_cmd->get_cid()][bid] = GridRegion<3>::merge(command_buffer_reads[push_cmd->get_cid()][bid], box);
							add_dependencies_for_box(cdag, push_cmd, node_data.at(source_nid).buffer_last_writer.at(bid), box);

							auto await_push_cmd = cdag.create<await_push_command>(nid, push_cmd);
							cdag.add_dependency(reduce_cmd, await_push_cmd);
						}
					}

					cdag.add_dependency(cmd, reduce_cmd);

					// Unless this task also writes the reduction buffer, the reduction command becomes the last writer
					if(!std::any_of(required_modes.begin(), required_modes.end(), detail::access::mode_traits::is_producer)) {
						per_node_last_writer_update_list.emplace_back(nid, bid, box, reduce_cmd->get_cid());
					}
				}

				if(is_pending_reduction) {
					// More general than generate_reduction: A producer-only access on a pending-reduction buffer will not generate a reduction command, but
					// still change the buffer state to distributed
					buffer_reduction_resolve_list[bid].push_back(nid);
				}
			}
		}

		for(auto [bid, nids] : buffer_reduction_resolve_list) {
			GridBox<3> box{GridPoint<3>{1, 1, 1}};
			distributed_state state_after_reduction{cl::sycl::range<3>{1, 1, 1}};
			state_after_reduction.region_sources.update_region(box, nids);
			buffer_states.at(bid) = distributed_state{std::move(state_after_reduction)};
		}

		// If there is only one chunk/command, it already implicitly generates the final reduced value and the buffer does not need to be flagged as
		// a pending reduction.
		if(task_commands.size() > 1) {
			for(auto rid : tsk->get_reductions()) {
				auto bid = reduction_mngr.get_reduction(rid).output_buffer_id;
				buffer_states.at(bid) = pending_reduction_state{rid, {}};
			}
		}

		// Update global buffer_states with writes from task_commands after handling is complete.
		for(const auto& write : buffer_state_write_list) {
			const auto& nid = std::get<0>(write);
			const auto& bid = std::get<1>(write);
			const auto& region = std::get<2>(write);

			auto& state = buffer_states.at(bid);
			if(auto* distributed = std::get_if<distributed_state>(&state)) {
				distributed->region_sources.update_region(region, {nid});
			} else if(auto* pending = std::get_if<pending_reduction_state>(&state)) {
				assert(std::find(pending->operand_sources.begin(), pending->operand_sources.end(), nid) == pending->operand_sources.end());
				pending->operand_sources.push_back(nid);
			}
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

	void graph_generator::generate_horizon(task_id tid) {
		detail::command_id lowest_prev_hid = 0;
		for(node_id node = 0; node < num_nodes; ++node) {
			// TODO this could be optimized to something like cdag.apply_horizon(node_id, horizon_cmd) with much fewer internal operations
			auto previous_execution_front = cdag.get_execution_front(node);
			// Build horizon command and make current front depend on it
			auto horizon_cmd = cdag.create<horizon_command>(node, tid);
			for(const auto& front_cmd : previous_execution_front) {
				cdag.add_dependency(horizon_cmd, front_cmd);
			}
			// Apply the previous horizon to data structures
			auto* prev_horizon = current_horizon_cmds[node];
			current_horizon_cmds[node] = horizon_cmd;
			if(prev_horizon != nullptr) {
				// update "buffer_last_writer" and "last_collective_commands" structures to subsume pre-horizon commands
				const auto prev_hid = prev_horizon->get_cid();
				auto& this_node_data = node_data.at(node);
				for(auto& blw_pair : this_node_data.buffer_last_writer) {
					blw_pair.second.apply_to_values([prev_hid](std::optional<command_id> cid) -> std::optional<command_id> {
						if(!cid) return cid;
						return {std::max(prev_hid, *cid)};
					});
				}
				for(auto& [cgid, cid] : this_node_data.last_collective_commands) {
					cid = std::max(prev_hid, cid);
				}
				// update lowest previous horizon id (for later command deletion)
				if(lowest_prev_hid == 0) {
					lowest_prev_hid = prev_hid;
				} else {
					lowest_prev_hid = std::min(lowest_prev_hid, prev_hid);
				}

				// We also use the previous horizon as the new init cmd for host-initialized buffers
				node_data[node].current_init_cid = prev_hid;
			}
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
