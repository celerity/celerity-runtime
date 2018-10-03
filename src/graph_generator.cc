#include "graph_generator.h"

#include <queue>

#include <allscale/utils/string_utils.h>

#include "distr_queue.h"
#include "graph_builder.h"
#include "graph_utils.h"

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
		graph_utils::for_predecessors(task_graph, static_cast<tdag_vertex>(tid), [&command_graph, begin_task_cmd_v](tdag_vertex requirement) {
			boost::add_edge(GRAPH_PROP(command_graph, task_vertices).at(static_cast<task_id>(requirement)).second, begin_task_cmd_v, command_graph);
		});

		return std::make_pair(begin_task_cmd_v, end_task_cmd_v);
	}

	graph_generator::graph_generator(size_t num_nodes) : num_nodes(num_nodes) {
		register_transformer(std::make_shared<naive_split_transformer>(num_nodes > 1 ? num_nodes - 1 : 1));
	}

	void graph_generator::set_queue(distr_queue* queue) { this->queue = queue; }

	void graph_generator::add_buffer(buffer_id bid, const cl::sycl::range<3>& range) {
		// FIXME: We don't need to initialize buffer states for tasks that don't use this buffer
		for(auto& it : task_buffer_states) {
			it.second[bid] = std::make_shared<buffer_state>(range, num_nodes);
		}
		empty_buffer_states[bid] = std::make_shared<buffer_state>(range, num_nodes);
	}

	void graph_generator::register_transformer(std::shared_ptr<graph_transformer> gt) { transformers.push_back(gt); }

	void graph_generator::build_task() {
		assert(queue != nullptr);
		const auto& task_graph = queue->get_task_graph();
		auto otid = graph_utils::get_satisfied_task(task_graph);
		if(!otid) return;
		const task_id tid = *otid;
		graph_builder gb(command_graph);

		cdag_vertex begin_task_cmd_v, end_task_cmd_v;
		std::tie(begin_task_cmd_v, end_task_cmd_v) = create_task_commands(task_graph, command_graph, gb, tid);

		auto tsk = queue->get_task(tid);
		if(tsk->get_type() == task_type::COMPUTE) {
			const node_id compute_node = 1 % num_nodes;
			const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
			const auto global_size = ctsk->get_global_size();
			const auto chnk = chunk<3>(cl::sycl::id<3>(), global_size, global_size);
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
		queue->mark_task_as_processed(tid);
	}

	bool graph_generator::has_unbuilt_tasks() const {
		// TODO: It's not ideal that we're wasting this result
		return !!graph_utils::get_satisfied_task(queue->get_task_graph());
	}

	using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<3>>>;

	template <int Dims>
	buffer_requirements_map get_buffer_requirements(const compute_task* ctsk, subrange<3> sr) {
		chunk<Dims> chnk{cl::sycl::id<Dims>(sr.offset), cl::sycl::range<Dims>(sr.range), cl::sycl::range<Dims>(ctsk->get_global_size())};
		buffer_requirements_map result;

		const auto& rms = ctsk->get_range_mappers();
		for(auto& it : rms) {
			const buffer_id bid = it.first;

			for(auto& rm : it.second) {
				auto mode = rm->get_access_mode();
				assert(mode == cl::sycl::access::mode::read || mode == cl::sycl::access::mode::write);
				assert(rm->get_kernel_dimensions() == Dims);

				subrange<3> req;
				// The chunk requirements have the dimensionality of the corresponding buffer
				switch(rm->get_buffer_dimensions()) {
				default:
				case 1: {
					req = subrange<3>(rm->map_1(chnk));
				} break;
				case 2: {
					req = subrange<3>(rm->map_2(chnk));
				} break;
				case 3: {
					req = subrange<3>(rm->map_3(chnk));
				} break;
				}
				const auto& reqs = result[bid][mode];
				result[bid][mode] = GridRegion<3>::merge(reqs, detail::subrange_to_grid_region(req));
			}
		}

		return result;
	}

	// TODO: We can ignore all commands that have already been flushed
	void graph_generator::process_task_data_requirements(task_id tid) {
		buffer_state_map initial_buffer_states;

		// Build the initial buffer states for this task by merging all predecessor's final states.
		// It's important (for certain edge cases) that we do this in the same order that tasks were submitted.
		// Since walking the graph doesn't guarantee this, we have to manually sort the predecessors by task id first.
		std::vector<task_id> predecessors;
		graph_utils::for_predecessors(
		    queue->get_task_graph(), static_cast<tdag_vertex>(tid), [&](tdag_vertex v) { predecessors.push_back(static_cast<task_id>(v)); });
		std::sort(predecessors.begin(), predecessors.end());

		for(const auto t : predecessors) {
			if(initial_buffer_states.empty()) {
				initial_buffer_states = task_buffer_states[t];
			} else {
				for(auto& it : task_buffer_states[t]) {
					initial_buffer_states[it.first]->merge(*it.second);
				}
			}
		}
		if(predecessors.empty()) { initial_buffer_states = empty_buffer_states; }

		buffer_state_map final_buffer_states = initial_buffer_states;

		graph_builder gb(command_graph);
		auto tsk = queue->get_task(tid);
		graph_utils::for_successors(command_graph, GRAPH_PROP(command_graph, task_vertices)[tid].first, [&](cdag_vertex v) {
			// Work on a copy of the buffer states to ensure parallel commands don't affect each other
			buffer_state_map working_buffer_states = initial_buffer_states;
			std::unordered_set<command_id> included_await_pushes;
			buffer_requirements_map requirements;

			// NOTE: We assume here that all execution commands have NO data transfer dependencies (i.e. on a split all related transfers should be deleted)
			// TODO: We still have to take transfers into consideration that have already been flushed to a worker and are shared between multiple executions

			if(command_graph[v].cmd == command::COMPUTE) {
				const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
				switch(ctsk->get_dimensions()) {
				case 1: requirements = get_buffer_requirements<1>(ctsk, command_graph[v].data.compute.subrange); break;
				case 2: requirements = get_buffer_requirements<2>(ctsk, command_graph[v].data.compute.subrange); break;
				case 3: requirements = get_buffer_requirements<3>(ctsk, command_graph[v].data.compute.subrange); break;
				default: assert(false);
				}
			} else if(command_graph[v].cmd == command::MASTER_ACCESS) {
				const auto matsk = dynamic_cast<const master_access_task*>(tsk.get());
				const auto& buffer_accesses = matsk->get_accesses();
				for(auto& it : buffer_accesses) {
					const buffer_id bid = it.first;

					for(auto& bacc : it.second) {
						const auto req = subrange<3>{bacc.offset, bacc.range};
						const auto& reqs = requirements[bid][bacc.mode];
						requirements[bid][bacc.mode] = GridRegion<3>::merge(reqs, subrange_to_grid_region(req));
					}
				}
			}

			for(auto& it : requirements) {
				const buffer_id bid = it.first;
				const node_id nid = command_graph[v].nid;
				const cdag_vertex command_vertex = v;

				// Writes
				if(it.second.count(cl::sycl::access::mode::write) != 0) {
					const auto& write_req = it.second.at(cl::sycl::access::mode::write);
					assert(write_req.area() > 0);
					// Add to compute node label for debugging
					command_graph[command_vertex].label = fmt::format("{}\\nWrite {} {}", command_graph[command_vertex].label, bid, toString(write_req));
					final_buffer_states[bid]->update_region(write_req, {nid});
				}

				// Reads
				if(it.second.count(cl::sycl::access::mode::read) != 0) {
					const auto& read_req = it.second.at(cl::sycl::access::mode::read);
					assert(read_req.area() > 0);
					// Add to command node label for debugging
					command_graph[command_vertex].label = fmt::format("{}\\nRead {} {}", command_graph[command_vertex].label, bid, toString(read_req));
				} else {
					continue;
				}

				const auto buffer_sources = working_buffer_states[bid]->get_source_nodes(it.second.at(cl::sycl::access::mode::read));
				assert(!buffer_sources.empty());

				for(auto& box_sources : buffer_sources) {
					const auto& box = box_sources.first;
					const auto& box_src_nodes = box_sources.second;

					if(box_src_nodes.count(nid) == 1) {
						// No need to push
						continue;
					}

					// We just pick the first source node for now
					const node_id source_nid = *box_src_nodes.cbegin();

					// TODO: Update final buffer states as well (since we copied valid regions!)
					// -> Note that this might need some consideration though, as the execution order of independent commands determines the actual buffer
					// contents on a worker at any point in time (i.e. if the execution order of two independent reads of different versions of the same
					// buffer range is reversed, the buffer contents will be different afterwards).

					command_data cmd_data{};
					cmd_data.push = push_data{bid, nid, command_subrange(grid_box_to_subrange(box))};
					const auto push_cid = gb.add_command(GRAPH_PROP(command_graph, task_vertices)[tid].first,
					    GRAPH_PROP(command_graph, task_vertices)[tid].second, source_nid, tid, command::PUSH, cmd_data);

					cmd_data.await_push = await_push_data{bid, source_nid, push_cid, command_subrange(grid_box_to_subrange(box))};
					gb.add_command(GRAPH_PROP(command_graph, task_vertices)[tid].first, command_vertex, nid, tid, command::AWAIT_PUSH, cmd_data);
				}
			}
		});

		gb.commit();

		// TODO: If this task already has a final buffer state (i.e. this is called after a transformation), we have to make sure
		// the buffer state is the same by potentially inserting additional transfers. Note that this could cause some inefficiencies. We can
		// probably best avoid this by not processing the successor tasks into the dag until we're sure the current task won't change
		// (this has to be decided by the scheduler!).
		task_buffer_states[tid] = final_buffer_states;
	}

} // namespace detail
} // namespace celerity
