#pragma once

#include <functional>
#include <vector>

#include "command.h"
#include "frame.h"
#include "types.h"

namespace celerity::detail {

class abstract_command;
class task_command;
class command_graph;

class graph_serializer {
	using flush_callback = std::function<void(node_id, frame_vector<command_frame>)>;

  public:
	/*
	 * @param flush_cb Callback invoked for each command that is being flushed
	 */
	graph_serializer(size_t num_nodes, command_graph& cdag, flush_callback flush_cb);

	void flush(task_id tid);

  private:
	size_t m_num_nodes;
	command_graph& m_cdag;
	flush_callback m_flush_cb;

	using command_vector = std::vector<const abstract_command*>;
	using node_command_map = std::vector<command_vector>;

	void collect_task_command(task_command* cmd, node_command_map& pending_node_cmds) const;
	void collect_dependency(abstract_command* cmd, node_command_map& pending_node_cmds) const;
	frame_vector<command_frame> serialize(const command_vector& cmds) const;
};

} // namespace celerity::detail
