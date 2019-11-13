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
	}

	void graph_generator::add_buffer(buffer_id bid, const cl::sycl::range<3>& range) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// Initialize the whole range to all nodes, so that we always use local buffer ranges when they haven't been written to (on any node) yet.
		// TODO: Consider better handling for when buffers are not host initialized
		std::vector<valid_buffer_source> all_nodes(num_nodes);
		for(auto i = 0u; i < num_nodes; ++i) {
			all_nodes[i].cid = node_data[i].init_cid;
			all_nodes[i].nid = i;
			node_data[i].buffer_last_writer.emplace(bid, range);
			node_data[i].buffer_last_writer.at(bid).update_region(subrange_to_grid_region({cl::sycl::id<3>(), range}), node_data[i].init_cid);
		}

		buffer_states.emplace(
		    bid, region_map<std::unordered_set<valid_buffer_source>>{range, std::unordered_set<valid_buffer_source>(all_nodes.cbegin(), all_nodes.cend())});
	}

	void graph_generator::build_task(task_id tid, const std::vector<graph_transformer*>& transformers) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// TODO: Maybe assert that this task hasn't been processed before

		auto tsk = task_mngr.get_task(tid);
		if(tsk->get_type() == task_type::COMPUTE) {
			const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
			const auto global_size = ctsk->get_global_size();
			const auto global_offset = ctsk->get_global_offset();
			cdag.create<compute_command>(0, tid, subrange<3>{global_offset, global_size});
		} else if(tsk->get_type() == task_type::MASTER_ACCESS) {
			cdag.create<master_access_command>(0, tid);
		}

		for(auto& t : transformers) {
			t->transform_task(tsk, cdag);
		}

		// TODO: At some point we might want to do this also before calling transformers
		// --> So that more advanced transformations can also take data transfers into account
		process_task_data_requirements(tid);
	}

	using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<3>>>;

	buffer_requirements_map get_buffer_requirements(const compute_task* ctsk, subrange<3> sr) {
		buffer_requirements_map result;

		const auto buffers = ctsk->get_accessed_buffers();
		for(const buffer_id bid : buffers) {
			const auto modes = ctsk->get_access_modes(bid);
			for(auto m : modes) {
				result[bid][m] = ctsk->get_requirements(bid, m, sr);
			}
		}
		return result;
	}

	buffer_requirements_map get_buffer_requirements(const master_access_task* mtsk) {
		buffer_requirements_map result;

		const auto buffers = mtsk->get_accessed_buffers();
		for(const buffer_id bid : buffers) {
			const auto modes = mtsk->get_access_modes(bid);
			for(auto m : modes) {
				result[bid][m] = mtsk->get_requirements(bid, m);
			}
		}
		return result;
	}

	void graph_generator::generate_anti_dependencies(task_id tid, buffer_id bid, const region_map<boost::optional<command_id>>& last_writers_map,
	    const GridRegion<3>& write_req, abstract_command* write_cmd) {
		const auto last_writers = last_writers_map.get_region_values(write_req);
		for(auto& box_and_writers : last_writers) {
			if(box_and_writers.second == boost::none) continue;
			const command_id last_writer_cid = *box_and_writers.second;
			const auto last_writer_cmd = cdag.get(last_writer_cid);
			assert(!isa<task_command>(last_writer_cmd) || static_cast<task_command*>(last_writer_cmd)->get_tid() != tid);

			// Add anti-dependencies onto all dependents of the writer
			bool has_dependents = false;
			for(auto d : last_writer_cmd->get_dependents()) {
				// Don't consider anti-dependents
				if(d.is_anti) continue;

				const auto cmd = d.node;

				// We might have already generated new commands within the same task that also depend on this; in that case, skip it
				if(isa<task_command>(cmd) && static_cast<task_command*>(cmd)->get_tid() == tid) continue;

				// So far we don't know whether the dependent actually intersects with the subrange we're writing
				const auto& command_reads = command_buffer_reads[cmd->get_cid()];
				const auto buffer_reads_it = command_reads.find(bid);
				if(buffer_reads_it == command_reads.end()) continue; // The task might be a dependent because of another buffer
				if(!GridRegion<3>::intersect(write_req, buffer_reads_it->second).empty()) {
					has_dependents = true;
					write_cmd->add_dependency({d.node, true});
				}
			}

			// In some cases (master access, weird discard_* constructs...)
			// the last writer might not have any dependents. Just add the anti-dependency onto the writer itself then.
			if(!has_dependents) {
				// Don't add anti-dependencies onto the init command
				if(last_writer_cid == node_data[write_cmd->get_nid()].init_cid) continue;

				write_cmd->add_dependency({last_writer_cmd, true});

				// This is a good time to validate our assumption that every AWAIT_PUSH command has a dependent
				assert(!isa<await_push_command>(last_writer_cmd));
			}
		}
	}

	void graph_generator::process_task_data_requirements(task_id tid) {
		buffer_state_map& final_buffer_states = buffer_states;

		// We keep a working copy around that is updated for data that is pulled in for the different access modes.
		// This is useful so we don't generate multiple PUSHes for the same buffer ranges.
		// Importantly, this does NOT contain the NEW buffer states produced by this task.
		// By sharing this working copy for all commands (having it outside the the loop) we also support this for multiple chunks on the same node.
		buffer_state_map working_buffer_states = buffer_states;

		const auto tsk = task_mngr.get_task(tid);

		// Copy the list of task commands so we can safely modify the command graph in the loop below
		// NOTE: We assume that none of these commands are deleted
		std::vector<task_command*> task_commands;
		for(auto cmd : cdag.task_commands<compute_command, master_access_command>(tid)) {
			task_commands.push_back(cmd);
		}

		// Remember all generated PUSHes for determining intra-task anti-dependencies.
		std::vector<push_command*> generated_pushes;

		for(auto cmd : task_commands) {
			const command_id cid = cmd->get_cid();
			const node_id nid = cmd->get_nid();
			buffer_requirements_map requirements;

			if(isa<compute_command>(cmd)) {
				const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
				requirements = get_buffer_requirements(ctsk, static_cast<compute_command*>(cmd)->get_execution_range());
			} else if(isa<master_access_command>(cmd)) {
				const auto matsk = dynamic_cast<const master_access_task*>(tsk.get());
				requirements = get_buffer_requirements(matsk);
			} else {
				assert(false);
			}

			for(auto& it : requirements) {
				const buffer_id bid = it.first;
				const auto& reqs_by_mode = it.second;

				auto& working_buffer_state = working_buffer_states.at(bid);

				// We have to make sure to update the last writer map for this node and buffer only after all new writes have been processed,
				// as we otherwise risk creating anti dependencies onto commands within the same task, that shouldn't exist.
				// (For example, an AWAIT_PUSH could be falsely identified as an anti-dependency for a "read_write" COMPUTE).
				auto working_node_buffer_last_writer = node_data.at(nid).buffer_last_writer.at(bid);

				const auto& initial_node_buffer_last_writer = node_data.at(nid).buffer_last_writer.at(bid);

				for(const auto mode : access::detail::all_modes) {
					if(reqs_by_mode.count(mode) == 0) continue;
					const auto& req = reqs_by_mode.at(mode);
					if(req.empty()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						continue;
					}

					// Add access mode and range to execution command node label for debugging
					cmd->debug_label = fmt::format("{}{} {} {}\\n", cmd->debug_label, access::detail::mode_traits::name(mode), bid, toString(req));

					if(access::detail::mode_traits::is_consumer(mode)) {
						// Store the read access for determining anti-dependencies later on
						command_buffer_reads[cid][bid] = GridRegion<3>::merge(command_buffer_reads[cid][bid], req);

						// Determine whether data transfers are required to fulfill the read requirements
						const auto buffer_sources = working_buffer_state.get_region_values(reqs_by_mode.at(mode));
						assert(!buffer_sources.empty());

						for(auto& box_and_sources : buffer_sources) {
							const auto& box = box_and_sources.first;
							const auto& box_sources = box_and_sources.second;

							bool exists_locally = false;
							for(auto& bs : box_sources) {
								if(bs.nid == nid) {
									// No need to push, but make sure to add a dependency.
									cmd->add_dependency({cdag.get(bs.cid), false});
									exists_locally = true;
									break;
								}
							}
							if(exists_locally) continue;

							// Try to find the original producer (i.e., a task_command, as opposed to an AWAIT_PUSH) to avoid
							// creating long dependency chains across nodes.
							// TODO: For larger numbers of nodes this might become a bottleneck.
							const auto source = ([&box_sources, tid, this]() {
								for(auto bs : box_sources) {
									const auto scmd = cdag.get(bs.cid);
									(void)tid;
									assert(!isa<task_command>(scmd) || static_cast<task_command*>(scmd)->get_tid() != tid);
									if(isa<task_command>(scmd)) return bs;
								}
								return *box_sources.cbegin();
							})(); // IIFE

							// Generate PUSH command
							push_command* push_cmd = nullptr;
							{
								push_cmd = cdag.create<push_command>(source.nid, bid, nid, grid_box_to_subrange(box));

								// Store the read access on the pushing node
								command_buffer_reads[push_cmd->get_cid()][bid] = GridRegion<3>::merge(command_buffer_reads[push_cmd->get_cid()][bid], box);

								// Add a dependency on the source node between the PUSH and the command that last wrote that box
								push_cmd->add_dependency({cdag.get(source.cid), false});

								generated_pushes.push_back(push_cmd);
							}

							// Generate AWAIT_PUSH command
							{
								auto await_push_cmd = cdag.create<await_push_command>(nid, push_cmd);
								cmd->add_dependency({await_push_cmd, false});

								generate_anti_dependencies(tid, bid, initial_node_buffer_last_writer, box, await_push_cmd);
								// Mark this command as the last writer of this region for this buffer and node
								working_node_buffer_last_writer.update_region(box, await_push_cmd->get_cid());

								// Finally, remember the fact that we now have this valid buffer range on this node.
								auto new_box_sources = box_sources;
								new_box_sources.insert({nid, await_push_cmd->get_cid()});
								working_buffer_state.update_region(box, new_box_sources);
								final_buffer_states.at(bid).update_region(box, new_box_sources);
							}
						}
					}

					if(access::detail::mode_traits::is_producer(mode)) {
						generate_anti_dependencies(tid, bid, initial_node_buffer_last_writer, req, cmd);

						// Mark this command as the last writer of this region for this buffer and node
						working_node_buffer_last_writer.update_region(req, cid);
						// After this task is completed, this node and command are the last writer of this region
						final_buffer_states.at(bid).update_region(req, {{nid, cid}});
					}
				}

				node_data.at(nid).buffer_last_writer.at(bid).merge(working_node_buffer_last_writer);
			}
		}

		// As the last step, we determine potential "intra-task" race conditions.
		// These can happen in rare cases, when the node that PUSHes a buffer range also writes to that range within the same task.
		// We cannot do this while generating the PUSH command, as we may not have the writing command recorded at that point.
		for(auto cmd : generated_pushes) {
			const auto push_cmd = static_cast<push_command*>(cmd);
			const auto last_writers =
			    node_data.at(push_cmd->get_nid()).buffer_last_writer.at(push_cmd->get_bid()).get_region_values(subrange_to_grid_region(push_cmd->get_range()));
			for(auto& box_and_writer : last_writers) {
				assert(!box_and_writer.first.empty());        // If we want to push it it cannot be empty
				assert(box_and_writer.second != boost::none); // Exactly one command last wrote to that box
				const auto writer_cmd = cdag.get(*box_and_writer.second);
				assert(writer_cmd != nullptr);

				// We're only interested in writes that happen within the same task as the PUSH
				if(isa<task_command>(writer_cmd) && static_cast<task_command*>(writer_cmd)->get_tid() == tid) {
					// In certain situations the PUSH might have a true dependency on the last writer,
					// in that case don't add an anti-dependency (as that would cause a cycle).
					if(cmd->has_dependency(writer_cmd, false)) {
						// This can currently only happen for AWAIT_PUSH commands.
						assert(isa<await_push_command>(writer_cmd));
						continue;
					}
					writer_cmd->add_dependency({push_cmd, true});
				}
			}
		}
	}

} // namespace detail
} // namespace celerity
