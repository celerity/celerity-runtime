#pragma once

#include <functional>
#include <vector>

#include "types.h"

namespace celerity {
namespace detail {

	struct command_pkg;
	class abstract_command;
	class task_command;
	class command_graph;

	class graph_serializer {
		using flush_callback = std::function<void(node_id, command_pkg, const std::vector<command_id>&)>;

	  public:
		/*
		 * @param flush_cb Callback invoked for each command that is being flushed
		 */
		graph_serializer(command_graph& cdag, flush_callback flush_cb) : cdag(cdag), flush_cb(flush_cb) {}

		void flush(task_id tid);

		/**
		 * Serializes a list of task commands and their dependencies.
		 *
		 * @param cmds The task commands to serialize, all belonging to the same task.
		 */
		void flush(const std::vector<task_command*>& cmds);

	  private:
		command_graph& cdag;
		flush_callback flush_cb;


		void flush_dependency(abstract_command* dep) const;
		void serialize_and_flush(abstract_command* cmd, const std::vector<command_id>& dependencies) const;
	};

} // namespace detail
} // namespace celerity
