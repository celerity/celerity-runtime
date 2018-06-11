#include "graph_utils.h"

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

	vertex add_compute_cmd(command_id& next_cmd_id, node_id nid, const task_vertices& tv, const subrange<3>& chunk, command_dag& cdag) {
		const auto v = boost::add_vertex(cdag);
		boost::add_edge(tv.first, v, cdag);
		boost::add_edge(v, tv.second, cdag);
		cdag[v].cmd = command::COMPUTE;
		cdag[v].cid = next_cmd_id++;
		cdag[v].nid = nid;
		cdag[v].tid = cdag[tv.first].tid;
		cdag[v].label = fmt::format("[{}] Node {}:\\nCOMPUTE {}", cdag[v].cid, nid, detail::subrange_to_grid_region(chunk));
		cdag[v].data.compute.chunk = command_subrange(chunk);
		return v;
	}

	vertex add_master_access_cmd(command_id& next_cmd_id, const task_vertices& tv, command_dag& cdag) {
		const node_id master_nid = 0;
		const auto v = boost::add_vertex(cdag);
		boost::add_edge(tv.first, v, cdag);
		boost::add_edge(v, tv.second, cdag);
		cdag[v].cmd = command::MASTER_ACCESS;
		cdag[v].cid = next_cmd_id++;
		cdag[v].nid = master_nid;
		cdag[v].tid = cdag[tv.first].tid;
		cdag[v].label = fmt::format("[{}] Node {}:\\nMASTER ACCESS", cdag[v].cid, master_nid);
		cdag[v].data.master_access = {};
		return v;
	}

	vertex add_push_cmd(command_id& next_cmd_id, node_id to_nid, node_id from_nid, buffer_id bid, const task_vertices& tv, vertex req_cmd,
	    const GridBox<3>& req, command_dag& cdag) {
		assert(cdag[req_cmd].cmd == command::COMPUTE || cdag[req_cmd].cmd == command::MASTER_ACCESS);

		const auto v = insert_vertex_on_edge(tv.first, tv.second, cdag);
		cdag[v].cmd = command::PUSH;
		cdag[v].cid = next_cmd_id++;
		cdag[v].nid = from_nid;
		cdag[v].tid = cdag[tv.first].tid;
		cdag[v].label = fmt::format("[{}] Node {}:\\nPUSH {} to {}\\n {}", cdag[v].cid, from_nid, bid, to_nid, req);
		cdag[v].data.push.bid = bid;
		cdag[v].data.push.target = to_nid;
		cdag[v].data.push.subrange = command_subrange(detail::grid_box_to_subrange(req));

		const auto w = insert_vertex_on_edge(tv.first, req_cmd, cdag);
		cdag[w].cmd = command::AWAIT_PUSH;
		cdag[w].cid = next_cmd_id++;
		cdag[w].nid = to_nid;
		cdag[w].tid = cdag[tv.first].tid;
		cdag[w].label = fmt::format("[{}] Node {}:\\nAWAIT PUSH {} from {}\\n {}", cdag[w].cid, to_nid, bid, from_nid, req);
		cdag[w].data.await_push.bid = bid;
		cdag[w].data.await_push.source = from_nid;
		cdag[w].data.await_push.source_cid = cdag[v].cid;
		cdag[w].data.await_push.subrange = command_subrange(detail::grid_box_to_subrange(req));

		return v;
	}

	bool get_satisfied_task(const task_dag& tdag, task_id& tid) {
		for(auto v : tdag.vertex_set()) {
			if(tdag[v].processed || tdag[v].num_unsatisfied > 0) continue;
			tid = v;
			return true;
		}
		return false;
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
		    boost::default_writer(), graph_logger);
	}

} // namespace graph_utils

} // namespace celerity
