#include "graph_utils.h"

#include <limits>
#include <queue>

#include <spdlog/fmt/fmt.h>

#include "command.h"

namespace celerity {

namespace graph_utils {
	task_vertices add_task(task_id tid, const task_dag& tdag, command_dag& cdag) {
		const vertex begin_task_v = boost::add_vertex(cdag);
		cdag[begin_task_v].label = fmt::format("Begin {}", tdag[tid].label);
		cdag[begin_task_v].tid = tid;

		// Add all task requirements
		for_predecessors(tdag, static_cast<vertex>(tid), [&cdag, begin_task_v](vertex requirement) {
			boost::add_edge(cdag[boost::graph_bundle].task_complete_vertices[static_cast<task_id>(requirement)], begin_task_v, cdag);
		});

		const vertex complete_task_v = boost::add_vertex(cdag);
		cdag[boost::graph_bundle].task_complete_vertices[tid] = complete_task_v;
		cdag[complete_task_v].label = fmt::format("Complete {}", tdag[tid].label);
		cdag[complete_task_v].tid = tid;

		return task_vertices(begin_task_v, complete_task_v);
	}

	vertex add_compute_cmd(node_id nid, const task_vertices& tv, const subrange<3>& chunk, command_dag& cdag) {
		const auto v = boost::add_vertex(cdag);
		boost::add_edge(tv.first, v, cdag);
		boost::add_edge(v, tv.second, cdag);
		cdag[v].cmd = command::COMPUTE;
		cdag[v].nid = nid;
		cdag[v].tid = cdag[tv.first].tid;
		cdag[v].label = fmt::format("Node {}:\\nCOMPUTE {}", nid, detail::subrange_to_grid_region(chunk));
		cdag[v].data.compute.chunk = command_subrange(chunk);
		return v;
	}

	vertex add_master_access_cmd(const task_vertices& tv, command_dag& cdag) {
		const node_id master_nid = 0;
		const auto v = boost::add_vertex(cdag);
		boost::add_edge(tv.first, v, cdag);
		boost::add_edge(v, tv.second, cdag);
		cdag[v].cmd = command::MASTER_ACCESS;
		cdag[v].nid = master_nid;
		cdag[v].tid = cdag[tv.first].tid;
		cdag[v].label = fmt::format("Node {}:\\nMASTER ACCESS", master_nid);
		cdag[v].data.master_access = {};
		return v;
	}

	vertex add_pull_cmd(node_id nid, node_id source_nid, buffer_id bid, const task_vertices& tv, const task_vertices& source_tv, vertex req_cmd,
	    const GridBox<3>& req, command_dag& cdag) {
		assert(cdag[req_cmd].cmd == command::COMPUTE || cdag[req_cmd].cmd == command::MASTER_ACCESS);
		const auto v = graph_utils::insert_vertex_on_edge(tv.first, req_cmd, cdag);
		cdag[v].cmd = command::PULL;
		cdag[v].nid = nid;
		cdag[v].tid = cdag[tv.first].tid;
		cdag[v].label = fmt::format("Node {}:\\nPULL {} from {}\\n {}", nid, bid, source_nid, req);
		cdag[v].data.pull.bid = bid;
		cdag[v].data.pull.source = source_nid;
		cdag[v].data.pull.subrange = command_subrange(detail::grid_box_to_subrange(req));

		// Find the compute / master access command for the source node in the writing task (or this
		// task, if no writing task has been found)
		vertex source_command_v = std::numeric_limits<size_t>::max();
		search_vertex_bf(source_tv.first, cdag, [source_nid, source_tv, &source_command_v](vertex v, const command_dag& cdag) {
			// FIXME: We have some special casing here for master access:
			// Master access only executes on the master node, which is (generally) not the source node. If the master access
			// is not in a sibling set with some writing task, we won't be able to find a compute comand for source_nid.
			// A proper solution to this will also handle the fact that in the futue we won't necessarily split every task
			// over all nodes.
			if(cdag[v].cmd == command::MASTER_ACCESS || cdag[v].cmd == command::COMPUTE && cdag[v].nid == source_nid) {
				source_command_v = v;
				return true;
			}
			return false;
		});

		// If the buffer is on the master node, chances are there isn't any master access command in the (source) task.
		// In this case, we simply add the await pull anywhere in the (source) task.
		if(source_command_v == std::numeric_limits<size_t>::max() && source_nid == 0) { source_command_v = source_tv.second; }
		assert(source_command_v != std::numeric_limits<size_t>::max());

		const auto w = graph_utils::insert_vertex_on_edge(source_tv.first, source_command_v, cdag);
		cdag[w].cmd = command::AWAIT_PULL;
		cdag[w].nid = source_nid;
		cdag[w].tid = cdag[source_tv.first].tid;
		cdag[w].label = fmt::format("Node {}:\\nAWAIT PULL {} by {}\\n {}", source_nid, bid, nid, req);
		cdag[w].data.await_pull.bid = bid;
		cdag[w].data.await_pull.target = nid;
		cdag[w].data.await_pull.target_tid = cdag[tv.first].tid;
		cdag[w].data.await_pull.subrange = command_subrange(detail::grid_box_to_subrange(req));

		// Add edges in both directions
		boost::add_edge(w, v, cdag);
		boost::add_edge(v, w, cdag);

		return v;
	}

	std::vector<task_id> get_satisfied_sibling_set(const task_dag& tdag) {
		for(auto v : tdag.vertex_set()) {
			if(tdag[v].processed || tdag[v].num_unsatisfied > 0) continue;

			std::unordered_set<task_id> checked_predecessors;
			std::unordered_set<task_id> candidates;
			std::queue<task_id> unchecked_siblings;
			candidates.insert(v);
			unchecked_siblings.push(v);

			bool abort = false;
			while(!abort && !unchecked_siblings.empty()) {
				const task_id sib = unchecked_siblings.front();
				unchecked_siblings.pop();

				abort = !for_predecessors(tdag, static_cast<vertex>(sib), [&](vertex pre) {
					if(!tdag[pre].processed) return false;
					auto pre_tid = static_cast<task_id>(pre);
					if(checked_predecessors.find(pre_tid) != checked_predecessors.end()) { return true; }
					checked_predecessors.insert(pre_tid);

					abort = !for_successors(tdag, pre, [&](vertex suc) {
						auto suc_tid = static_cast<task_id>(suc);
						if(candidates.find(suc_tid) == candidates.end()) {
							if(tdag[suc].processed || tdag[suc].num_unsatisfied > 0) { return false; }
							candidates.insert(suc_tid);
							unchecked_siblings.push(suc_tid);
						}
						return true;
					});

					// abort if v has unsatisfied sibling
					return !abort;
				});
			}

			if(!abort) {
				std::vector<task_id> result;
				result.insert(result.end(), candidates.begin(), candidates.end());
				return result;
			}
		}

		return std::vector<task_id>();
	}

	void mark_as_processed(task_id tid, task_dag& tdag) {
		tdag[tid].processed = true;
		for_successors(tdag, static_cast<vertex>(tid), [&tdag](vertex suc) {
			assert(tdag[suc].num_unsatisfied >= 1);
			tdag[suc].num_unsatisfied--;
		});
	}


	// --------------------------- Graph printing ---------------------------


	void print_graph(const celerity::task_dag& tdag, std::shared_ptr<logger> graph_logger) {
		write_graph_mux(tdag, boost::make_label_writer(boost::get(&celerity::tdag_vertex_properties::label, tdag)), boost::default_writer(), graph_logger);
	}

	void print_graph(const celerity::command_dag& cdag, std::shared_ptr<logger> graph_logger) {
		write_graph_mux(cdag,
		    [&](std::ostream& out, vertex v) {
			    const char* colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			    std::unordered_map<std::string, std::string> props;
			    props["label"] = boost::escape_dot_string(cdag[v].label);

			    props["fontcolor"] = colors[cdag[v].nid % (sizeof(colors) / sizeof(char*))];

			    switch(cdag[v].cmd) {
			    case command::NOP:
				    props["color"] = "gray50";
				    props["fontcolor"] = "gray50";
				    break;
			    case command::COMPUTE: props["shape"] = "box"; break;
			    default: break;
			    }

			    out << "[";
			    for(auto it : props) {
				    out << " " << it.first << "=" << it.second;
			    }
			    out << "]";
		    },
		    [&](std::ostream& out, auto e) {
			    vertex v0 = boost::source(e, cdag);
			    vertex v1 = boost::target(e, cdag);
			    if((cdag[v0].cmd == command::PULL || cdag[v0].cmd == command::AWAIT_PULL)
			        && (cdag[v1].cmd == command::PULL || cdag[v1].cmd == command::AWAIT_PULL)) {
				    out << "[color=gray50]";
			    }
		    },
		    graph_logger);
	}

} // namespace graph_utils

} // namespace celerity
