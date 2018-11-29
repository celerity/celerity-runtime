#include "graph_builder.h"

#include <queue>

#include "graph_utils.h"

namespace celerity {
namespace detail {

	graph_builder::graph_builder(command_dag& command_graph) : command_graph(command_graph) {}

	command_id graph_builder::add_command(cdag_vertex a, cdag_vertex b, node_id nid, task_id tid, command cmd, command_data data, std::string label) {
		add_command_op add;
		add.a = a;
		add.b = b;
		add.nid = nid;
		add.tid = tid;
		add.cid = GRAPH_PROP(command_graph, next_cmd_id)++;
		add.cmd = cmd;
		add.data = data;
		add.label = label;
		graph_ops.emplace_back<graph_op>({graph_op_type::ADD_COMMAND, add});
		return add.cid;
	}

	void graph_builder::add_dependency(command_id dependant, command_id dependency, bool anti) {
		add_dependency_op add{dependant, dependency, anti};
		graph_ops.emplace_back<graph_op>({graph_op_type::ADD_DEPENDENCY, add});
	}

	std::vector<command_id> graph_builder::get_commands(task_id tid, command cmd) const {
		std::vector<command_id> result;
		auto& tv = GRAPH_PROP(command_graph, task_vertices)[tid];

		// Iterate over all vertices within this task
		// TODO: We may want to move this functionality into graph_utils
		std::queue<cdag_vertex> cmd_queue;
		std::unordered_set<cdag_vertex> queued_cmds;
		cmd_queue.push(tv.first);
		queued_cmds.insert(tv.first);

		while(!cmd_queue.empty()) {
			const auto v = cmd_queue.front();
			cmd_queue.pop();
			auto& cmd_v = command_graph[v];
			if(cmd_v.cmd == cmd) { result.push_back(cmd_v.cid); }

			graph_utils::for_successors(command_graph, v, [tv, tid, &queued_cmds, &cmd_queue, this](cdag_vertex s, cdag_edge) {
				if(command_graph[s].tid == tid && s != tv.second && queued_cmds.count(s) == 0) {
					cmd_queue.push(s);
					queued_cmds.insert(s);
				}
				return true;
			});
		}

		return result;
	}

	const cdag_vertex_properties& graph_builder::get_command_data(command_id cid) const {
		return command_graph[GRAPH_PROP(command_graph, command_vertices).at(cid)];
	}

	// TODO: Should we take subranges rather than chunks here? We don't actually need the global size...
	void graph_builder::split_command(command_id cid, const std::vector<chunk<3>>& chunks, const std::vector<node_id>& nodes) {
		assert(chunks.size() == nodes.size());
		auto& cmdv = command_graph[GRAPH_PROP(command_graph, command_vertices).at(cid)];
		assert(cmdv.cmd == command::COMPUTE); // For now we only split COMPUTE commands

#ifndef NDEBUG
		// Make sure the sum of all chunks adds up to the original chunk
		GridRegion<3> all_chunks;
		for(auto& c : chunks) {
			all_chunks = GridRegion<3>::merge(all_chunks, subrange_to_grid_region(subrange<3>(c)));
		}
		if(GridRegion<3>::difference(all_chunks, subrange_to_grid_region(cmdv.data.compute.subrange)).area() != 0) {
			throw std::runtime_error("Invalid split");
		}
#endif

		graph_utils::for_predecessors(command_graph, GRAPH_PROP(command_graph, command_vertices).at(cid), [&](cdag_vertex v, cdag_edge) {
			assert(command_graph[v].cmd == command::NOP && "Splitting computes with existing data transfer dependencies NYI");
		});

		const auto tv = GRAPH_PROP(command_graph, task_vertices).at(cmdv.tid);
		remove_command_op rm;
		rm.cid = cid;
		graph_ops.emplace_back<graph_op>({graph_op_type::REMOVE_COMMAND, rm});

		for(auto i = 0u; i < chunks.size(); ++i) {
			command_data data{};
			data.compute = {command_subrange(chunks[i])};
			add_command(tv.first, tv.second, nodes[i], cmdv.tid, command::COMPUTE, data);
		}
	}

	void graph_builder::commit() {
		if(graph_ops.empty()) return;

		for(auto& op : graph_ops) {
			switch(op.type) {
			case graph_op_type::ADD_COMMAND: {
				auto& add_info = boost::get<add_command_op>(op.info);
				const auto v = [=] {
					if(add_info.a != cdag_vertex_none && add_info.b != cdag_vertex_none) {
						return graph_utils::insert_vertex_on_edge(add_info.a, add_info.b, command_graph);
					}
					return boost::add_vertex(command_graph);
				}(); // IIFE
				command_graph[v].cmd = add_info.cmd;
				command_graph[v].cid = add_info.cid;
				command_graph[v].nid = add_info.nid;
				command_graph[v].tid = add_info.tid;
				command_graph[v].label = add_info.label;
				command_graph[v].data = add_info.data;

				GRAPH_PROP(command_graph, command_vertices)[add_info.cid] = v;
			} break;
			case graph_op_type::REMOVE_COMMAND: {
				auto& cmd_vertices = GRAPH_PROP(command_graph, command_vertices);
				const auto cid = boost::get<remove_command_op>(op.info).cid;
				boost::clear_vertex(cmd_vertices.at(cid), command_graph);
				boost::remove_vertex(cmd_vertices.at(cid), command_graph);
				cmd_vertices.erase(cid);
			} break;
			case graph_op_type::ADD_DEPENDENCY: {
				const auto& add_info = boost::get<add_dependency_op>(op.info);
				assert(GRAPH_PROP(command_graph, command_vertices).count(add_info.dependency) == 1);
				assert(GRAPH_PROP(command_graph, command_vertices).count(add_info.dependant) == 1);
				const auto dependency_v = GRAPH_PROP(command_graph, command_vertices)[add_info.dependency];
				const auto dependant_v = GRAPH_PROP(command_graph, command_vertices)[add_info.dependant];

				// We definitely don't want that
				assert(!boost::edge(dependant_v, dependency_v, command_graph).second && "cyclic dependency");

				// Check whether edge already exists
				const auto ep = boost::edge(dependency_v, dependant_v, command_graph);
				if(ep.second) {
					if(!add_info.anti) {
						// Anti-dependencies can be overwritten by true dependencies
						command_graph[ep.first].anti_dependency = false;
					}
				} else {
					const auto new_edge = boost::add_edge(dependency_v, dependant_v, command_graph).first;
					command_graph[new_edge].anti_dependency = add_info.anti;
				}

			} break;
			default: assert(false && "Unexpected graph_op_type");
			}
		}

		graph_ops.clear();
	}

	scoped_graph_builder::scoped_graph_builder(command_dag& command_graph, task_id tid) : graph_builder(command_graph), tid(tid) {}

	std::vector<command_id> scoped_graph_builder::get_commands(command cmd) const { return graph_builder::get_commands(tid, cmd); }

} // namespace detail
} // namespace celerity
