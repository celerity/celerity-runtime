#include "graph_generator.h"

#include <numeric>
#include <queue>

#include <allscale/utils/string_utils.h>

#include "access_modes.h"
#include "graph_builder.h"
#include "graph_utils.h"
#include "logger.h"
#include "task.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	std::pair<cdag_vertex, cdag_vertex> create_task_commands(const task_dag& task_graph, command_dag& command_graph, graph_builder& gb, task_id tid) {
		const auto begin_task_cmd =
		    gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, tid, command::NOP, {}, fmt::format("Begin {}", task_graph[tid].label));
		const auto end_task_cmd = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, tid, command::NOP, {}, fmt::format("End {}", task_graph[tid].label));
		gb.commit(); // Commit now so we can get the actual vertices

		const auto begin_task_cmd_v = GRAPH_PROP(command_graph, command_vertices).at(begin_task_cmd);
		const auto end_task_cmd_v = GRAPH_PROP(command_graph, command_vertices).at(end_task_cmd);
		GRAPH_PROP(command_graph, task_vertices)[tid] = std::make_pair(begin_task_cmd_v, end_task_cmd_v);

		// Add all task requirements
		graph_utils::for_predecessors(task_graph, static_cast<tdag_vertex>(tid), [&command_graph, begin_task_cmd_v](tdag_vertex requirement, tdag_edge) {
			boost::add_edge(GRAPH_PROP(command_graph, task_vertices).at(static_cast<task_id>(requirement)).second, begin_task_cmd_v, command_graph);
		});

		return std::make_pair(begin_task_cmd_v, end_task_cmd_v);
	}

	graph_generator::graph_generator(size_t num_nodes, task_manager& tm, flush_callback flush_callback)
	    : task_mngr(tm), num_nodes(num_nodes), flush_cb(flush_callback) {
		register_transformer(std::make_shared<naive_split_transformer>(num_nodes));
		build_task(tm.get_init_task_id());
	}

	void graph_generator::add_buffer(buffer_id bid, const cl::sycl::range<3>& range) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// Initialize the whole range to all nodes, so that we always use local buffer ranges when they haven't been written to (on any node) yet.
		// TODO: Consider better handling for when buffers are not host initialized
		std::vector<valid_buffer_source> all_nodes(num_nodes);
		for(auto i = 0u; i < num_nodes; ++i) {
			all_nodes[i].cid = -1; // FIXME: Not ideal
			all_nodes[i].nid = i;
			node_buffer_last_writer[i].emplace(bid, range);
		}

		buffer_states.emplace(
		    bid, region_map<std::unordered_set<valid_buffer_source>>{range, std::unordered_set<valid_buffer_source>(all_nodes.cbegin(), all_nodes.cend())});
	}

	void graph_generator::register_transformer(std::shared_ptr<graph_transformer> gt) { transformers.push_back(gt); }

	void graph_generator::build_task(task_id tid) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		// TODO: Maybe assert that this task hasn't been processed before
		graph_builder gb(command_graph);

		cdag_vertex begin_task_cmd_v, end_task_cmd_v;
		std::tie(begin_task_cmd_v, end_task_cmd_v) = create_task_commands(*task_mngr.get_task_graph(), command_graph, gb, tid);

		// TODO: Not the nicest solution.
		if(tid == task_mngr.get_init_task_id()) {
			gb.add_dependency(command_graph[end_task_cmd_v].cid, command_graph[begin_task_cmd_v].cid);
			gb.commit();
			return;
		}

		auto tsk = task_mngr.get_task(tid);
		if(tsk->get_type() == task_type::COMPUTE) {
			const node_id compute_node = 1 % num_nodes;
			const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
			const auto global_size = ctsk->get_global_size();
			const auto global_offset = ctsk->get_global_offset();
			const auto chnk = chunk<3>(global_offset, global_size, global_size);
			command_data data{};
			data.compute.subrange = command_subrange(chnk);
			gb.add_command(begin_task_cmd_v, end_task_cmd_v, compute_node, tid, command::COMPUTE, data);
		} else if(tsk->get_type() == task_type::MASTER_ACCESS) {
			command_data data{};
			data.master_access = {};
			gb.add_command(begin_task_cmd_v, end_task_cmd_v, 0, tid, command::MASTER_ACCESS, data);
		}

		gb.commit();

		scoped_graph_builder sgb(command_graph, tid);
		for(auto& t : transformers) {
			t->transform_task(tsk, sgb);
		}

		// TODO: At some point we might want to do this also before calling transformers
		// --> So that more advanced transformations can also take data transfers into account
		process_task_data_requirements(tid);
		task_mngr.mark_task_as_processed(tid);
	}

	boost::optional<task_id> graph_generator::get_unbuilt_task() const { return graph_utils::get_satisfied_task(*task_mngr.get_task_graph()); }

	void graph_generator::flush(task_id tid) const {
		const auto& tv = GRAPH_PROP(command_graph, task_vertices).at(tid);
		std::queue<cdag_vertex> cmd_queue;
		std::unordered_set<cdag_vertex> queued_cmds;
		cmd_queue.push(tv.first);
		queued_cmds.insert(tv.first);

		while(!cmd_queue.empty()) {
			const cdag_vertex v = cmd_queue.front();
			cmd_queue.pop();
			auto& cmd_v = command_graph[v];
			if(cmd_v.cmd != command::NOP) {
				const command_pkg pkg{cmd_v.tid, cmd_v.cid, cmd_v.cmd, cmd_v.data};
				const node_id target = cmd_v.nid;

				// Find all (anti-)dependencies of that command
				// TODO: We could probably do some pruning here (e.g. omit tasks we know are already finished)
				std::vector<command_id> dependencies;
				graph_utils::for_predecessors(command_graph, v, [&dependencies, this](cdag_vertex d, cdag_edge) {
					if(command_graph[d].cmd != command::NOP) { dependencies.push_back(command_graph[d].cid); }
				});
				flush_cb(target, pkg, dependencies);
			}

			std::vector<cdag_vertex> next_batch;
			graph_utils::for_successors(command_graph, v, [tid, tv, &queued_cmds, &next_batch, this](cdag_vertex s, cdag_edge) {
				if(command_graph[s].tid == tid && s != tv.second && queued_cmds.count(s) == 0) { next_batch.push_back(s); }
			});

			// Make sure to flush PUSH commands first, as we want to execute those before any COMPUTEs, in case they
			// cannot be performed in parallel (on some platforms parallel copying to host and reading from within kernel
			// is not supported).
			std::sort(next_batch.begin(), next_batch.end(),
			    [this](cdag_vertex a, cdag_vertex b) { return command_graph[a].cmd == command::PUSH && command_graph[b].cmd != command::PUSH; });

			for(auto v : next_batch) {
				cmd_queue.push(v);
				queued_cmds.insert(v);
			}
		}
	}

	void graph_generator::print_graph(logger& graph_logger) {
		if(command_graph.m_vertices.size() < 200) {
			graph_utils::print_graph(command_graph, graph_logger);
		} else {
			graph_logger.warn("Command graph is very large ({} vertices). Skipping GraphViz output", command_graph.m_vertices.size());
		}
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
	    const GridRegion<3>& write_req, command_id write_cid, graph_builder& gb) {
		const auto last_writers = last_writers_map.get_region_values(write_req);
		for(auto& box_and_writers : last_writers) {
			if(box_and_writers.second == boost::none) continue;
			const command_id last_writer_cid = *box_and_writers.second;
			const cdag_vertex cmd_v = GRAPH_PROP(command_graph, command_vertices).at(last_writer_cid);
			assert(command_graph[cmd_v].tid != tid);

			// Add anti-dependencies onto all dependants of the writer
			bool has_dependants = false;
			graph_utils::for_successors(command_graph, cmd_v, [&](cdag_vertex v, cdag_edge) {
				assert(command_graph[v].tid != tid);
				if(command_graph[v].cmd == command::NOP) return;

				// So far we don't know whether the dependant actually intersects with the subrange we're writing
				// TODO: Not the most efficient solution
				bool intersects = false;
				for(const auto& read_pair : task_buffer_reads.at(command_graph[v].tid).at(bid)) {
					if(read_pair.first == command_graph[v].cid) {
						if(!GridRegion<3>::intersect(write_req, read_pair.second).empty()) { intersects = true; }
						break;
					}
				}

				if(intersects) {
					has_dependants = true;
					gb.add_dependency(write_cid, command_graph[v].cid, true);
				}
			});

			if(!has_dependants) {
				// In some cases (master access, weird discard_* constructs...)
				// the last writer might not have any dependants. Just add the anti-dependency onto the writer itself then.
				gb.add_dependency(write_cid, last_writer_cid, true);
				// This is a good time to validate our assumption that every AWAIT_PUSH command has a dependant
				assert(command_graph[cmd_v].cmd != command::AWAIT_PUSH);
			}
		}
	}

	// TODO: We can ignore all commands that have already been flushed
	// TODO: This needs to be split up somehow
	void graph_generator::process_task_data_requirements(task_id tid) {
		buffer_state_map final_buffer_states = buffer_states;

		graph_builder gb(command_graph);
		auto tsk = task_mngr.get_task(tid);
		graph_utils::for_successors(command_graph, GRAPH_PROP(command_graph, task_vertices)[tid].first, [&](cdag_vertex v, cdag_edge) {
			const command_id cid = command_graph[v].cid;
			const node_id nid = command_graph[v].nid;
			buffer_requirements_map requirements;

			if(command_graph[v].cmd == command::COMPUTE) {
				const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
				requirements = get_buffer_requirements(ctsk, command_graph[v].data.compute.subrange);
			} else if(command_graph[v].cmd == command::MASTER_ACCESS) {
				const auto matsk = dynamic_cast<const master_access_task*>(tsk.get());
				requirements = get_buffer_requirements(matsk);
			} else {
				assert(false);
			}

			for(auto& it : requirements) {
				const buffer_id bid = it.first;
				const auto& reqs_by_mode = it.second;

				// We keep a working copy around that is updated for data that is pulled in for the different access modes.
				// This is useful so we don't generate multiple PULLs for the same buffer ranges.
				// Importantly, this does NOT contain the NEW buffer states produced by this task.
				auto working_buffer_state = buffer_states.at(bid);

				// Likewise, we have to make sure to update the last writer map for this node and buffer only after all new writes have been processed,
				// as we otherwise risk creating anti dependencies onto commands within the same task, that shouldn't exist.
				// (For example, an AWAIT_PUSH could be falsely identified as an anti-dependency for a "read_write" COMPUTE).
				auto working_node_buffer_last_writer = node_buffer_last_writer.at(nid).at(bid);

				const auto& initial_node_buffer_last_writer = node_buffer_last_writer.at(nid).at(bid);

				for(const auto mode : access::detail::all_modes) {
					if(reqs_by_mode.count(mode) == 0) continue;
					const auto& req = reqs_by_mode.at(mode);
					if(req.empty()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						continue;
					}

					// Add access mode and range to execution command node label for debugging
					command_graph[v].label = fmt::format("{}\\n{} {} {}", command_graph[v].label, access::detail::mode_traits::name(mode), bid, toString(req));

					if(access::detail::mode_traits::is_consumer(mode)) {
						// Store the read access for determining anti-dependencies later on
						task_buffer_reads[tid][bid].emplace_back(std::make_pair(cid, req));

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
									if(bs.cid != static_cast<command_id>(-1)) { gb.add_dependency(cid, bs.cid); }
									exists_locally = true;
									break;
								}
							}
							if(exists_locally) continue;

							// We just pick the first source node for now,
							// unless the sources contain the master node, in which
							// case we prefer any other node.
							const auto source = ([&box_sources]() {
								for(auto bs : box_sources) {
									if(bs.nid != 0) return bs;
								}
								return *box_sources.cbegin();
							})(); // IIFE

							// Generate PUSH command
							command_id push_cid = -1;
							{
								command_data cmd_data{};
								cmd_data.push = push_data{bid, nid, command_subrange(grid_box_to_subrange(box))};
								push_cid = gb.add_command(GRAPH_PROP(command_graph, task_vertices)[tid].first,
								    GRAPH_PROP(command_graph, task_vertices)[tid].second, source.nid, tid, command::PUSH, cmd_data);

								// Store the read access on the pushing node
								task_buffer_reads[tid][bid].emplace_back(std::make_pair(push_cid, box));

								// Add a dependency on the source node between the PUSH and the command that last wrote that box
								gb.add_dependency(push_cid, source.cid);
							}

							// Generate AWAIT_PUSH command
							{
								command_data cmd_data{};
								cmd_data.await_push = await_push_data{bid, source.nid, push_cid, command_subrange(grid_box_to_subrange(box))};
								const auto await_push_cid =
								    gb.add_command(GRAPH_PROP(command_graph, task_vertices)[tid].first, v, nid, tid, command::AWAIT_PUSH, cmd_data);

								generate_anti_dependencies(tid, bid, initial_node_buffer_last_writer, box, await_push_cid, gb);
								// Mark this command as the last writer of this region for this buffer and node
								working_node_buffer_last_writer.update_region(box, await_push_cid);

								// Finally, remember the fact that we now have this valid buffer range on this node.
								auto new_box_sources = box_sources;
								new_box_sources.insert({nid, await_push_cid});
								working_buffer_state.update_region(box, new_box_sources);
								final_buffer_states.at(bid).update_region(box, new_box_sources);
							}
						}
					}

					if(access::detail::mode_traits::is_producer(mode)) {
						generate_anti_dependencies(tid, bid, initial_node_buffer_last_writer, req, cid, gb);
						// Mark this command as the last writer of this region for this buffer and node
						working_node_buffer_last_writer.update_region(req, cid);
						// After this task is completed, this node and command are the last writer of this region
						final_buffer_states.at(bid).update_region(req, {{nid, cid}});
					}
				}

				node_buffer_last_writer.at(nid).at(bid).merge(working_node_buffer_last_writer);
			}
		});

		gb.commit();

		// As the last step, we determine potential "intra-task" race conditions.
		// These can happen in rare cases, when the node that PUSHes a buffer range also writes to that range within the same task.
		// We cannot do this while generating the PUSH command, as we may not have the writing command recorded at that point.
		graph_utils::for_successors(command_graph, GRAPH_PROP(command_graph, task_vertices)[tid].first, [&](cdag_vertex v, cdag_edge) {
			if(command_graph[v].cmd != command::PUSH) { return; }
			const command_id push_cid = command_graph[v].cid;
			const node_id push_nid = command_graph[v].nid;
			const buffer_id push_bid = command_graph[v].data.push.bid;

			const subrange<3> push_subrange = subrange<3>(command_graph[v].data.push.subrange);
			const auto last_writers = node_buffer_last_writer.at(push_nid).at(push_bid).get_region_values(subrange_to_grid_region(push_subrange));
			for(auto& box_and_writer : last_writers) {
				assert(!box_and_writer.first.empty());        // If we want to push it it cannot be empty
				assert(box_and_writer.second != boost::none); // Exactly one command last wrote to that box
				const command_id writer_cid = *box_and_writer.second;
				const cdag_vertex writer_v = GRAPH_PROP(command_graph, command_vertices)[writer_cid];

				// We're only interested in writes that happen within the same task as the PUSH
				if(command_graph[writer_v].tid == tid) { gb.add_dependency(writer_cid, push_cid, true); }
			}
		});

		gb.commit();
		buffer_states = final_buffer_states;
	}

} // namespace detail
} // namespace celerity
