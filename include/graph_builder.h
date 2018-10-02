#pragma once

#include <boost/variant.hpp>

#include "graph.h"

namespace celerity {
namespace detail {

	enum class graph_op_type { ADD_COMMAND, REMOVE_COMMAND };

	struct add_command_op {
		cdag_vertex a = 0;
		cdag_vertex b = 0;
		node_id nid = 0;
		task_id tid = 0;
		command_id cid = 0;
		std::string label = "";
		command cmd = command::NOP;
		command_data data = {nop_data{}};
	};

	struct remove_command_op {
		command_id cid = 0;
	};

	using graph_op_info = boost::variant<add_command_op, remove_command_op>;

	struct graph_op {
		graph_op_type type = graph_op_type::ADD_COMMAND;
		graph_op_info info;
		graph_op(graph_op_type type, graph_op_info info) : type(type), info(info) {}
	};

	class graph_builder {
	  public:
		graph_builder(command_dag& command_graph);

		command_id add_command(cdag_vertex a, cdag_vertex b, node_id nid, task_id tid, command cmd, command_data data, std::string label = "");

		std::vector<command_id> get_commands(task_id tid, command cmd) const;

		const cdag_vertex_properties& get_command_data(command_id cid) const;

		// TODO: Verify command can be split (e.g. not a master access)
		// TODO: In debug mode we could check if chunks add up to original chunk
		void split_command(command_id cid, const std::vector<chunk<3>>& chunks, const std::vector<node_id>& nodes);

		// This is what does the actual graph transformation
		void commit();

	  private:
		command_dag& command_graph;
		std::vector<graph_op> graph_ops;
	};

	/**
	 * The scoped_graph_builder is scoped to a certain task and also has otherwise a more limited API.
	 *
	 * TODO: We currently don't check whether modified commands belong to the given task. Maybe we should?
	 */
	class scoped_graph_builder : private graph_builder {
	  public:
		scoped_graph_builder(command_dag& command_graph, task_id tid);

		using graph_builder::commit;
		using graph_builder::get_command_data;
		using graph_builder::split_command;

		std::vector<command_id> get_commands(command cmd) const;

	  private:
		task_id tid;
	};

} // namespace detail
} // namespace celerity
