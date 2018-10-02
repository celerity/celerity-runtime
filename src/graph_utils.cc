#include "graph_utils.h"

#include <spdlog/fmt/fmt.h>

#include "command.h"
#include "grid.h"

namespace celerity {

namespace graph_utils {

	boost::optional<task_id> get_satisfied_task(const task_dag& tdag) {
		for(auto v : tdag.vertex_set()) {
			if(tdag[v].processed || tdag[v].num_unsatisfied > 0) continue;
			return static_cast<task_id>(v);
		}
		return boost::none;
	}

	void mark_as_processed(task_id tid, task_dag& tdag) {
		tdag[tid].processed = true;
		for_successors(tdag, static_cast<tdag_vertex>(tid), [&tdag](tdag_vertex suc) {
			assert(tdag[suc].num_unsatisfied >= 1);
			tdag[suc].num_unsatisfied--;
		});
	}


	// --------------------------- Graph printing ---------------------------


	void print_graph(const celerity::task_dag& tdag, const std::shared_ptr<logger>& graph_logger) {
		write_graph(
		    tdag, "TaskGraph", boost::make_label_writer(boost::get(&celerity::tdag_vertex_properties::label, tdag)), boost::default_writer(), graph_logger);
	}

	std::string get_command_label(const cdag_vertex_properties& props) {
		if(props.cmd == command::NOP) { return fmt::format("[{}] {}", props.cid, props.label); }
		const std::string label = fmt::format("[{}] Node {}:\\n", props.cid, props.nid);

		switch(props.cmd) {
		case command::COMPUTE: return label + fmt::format("COMPUTE {}", detail::subrange_to_grid_region(props.data.compute.subrange)) + props.label;
		case command::MASTER_ACCESS: return label + "MASTER ACCESS" + props.label;
		case command::PUSH:
			return label
			       + fmt::format("PUSH {} to {}\\n {}", props.data.push.bid, props.data.push.target, detail::subrange_to_grid_region(props.data.push.subrange));
		case command::AWAIT_PUSH:
			return label
			       + fmt::format("AWAIT PUSH {} from {}\\n {}", props.data.await_push.bid, props.data.await_push.source,
			             detail::subrange_to_grid_region(props.data.await_push.subrange));
		default: return props.label;
		}
	}

	void print_graph(celerity::command_dag& cdag, const std::shared_ptr<logger>& graph_logger) {
		// Boost's write_graphviz wants vertices with the vertex_index_t property.
		// Since we're not storing vertices in a vector (boost::vecS), we unfortunately have to populate the index ourselves.
		int idx = 0;
		BGL_FORALL_VERTICES(v, cdag, celerity::command_dag)
		boost::put(boost::vertex_index_t(), cdag, v, idx++);

		write_graph(cdag, "CommandGraph",
		    [&](std::ostream& out, auto v) {
			    const char* colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			    std::unordered_map<std::string, std::string> props;
			    props["label"] = boost::escape_dot_string(get_command_label(cdag[v]));

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
			    for(const auto& it : props) {
				    out << " " << it.first << "=" << it.second;
			    }
			    out << "]";
		    },
		    boost::default_writer(), graph_logger);
	}

} // namespace graph_utils

} // namespace celerity
