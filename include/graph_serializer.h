#pragma once

#include <functional>
#include <unordered_set>
#include <vector>

#include "command.h"
#include "frame.h"
#include "types.h"

namespace celerity {
namespace detail {

	class abstract_command;
	class task_command;
	class command_graph;

	class graph_serializer {
		using flush_callback = std::function<void(command_pkg&&)>;

	  public:
		/*
		 * @param flush_cb Callback invoked for each command that is being flushed
		 */
		graph_serializer(flush_callback flush_cb) : m_flush_cb(std::move(flush_cb)) {}

		/**
		 * Serializes a set of commands. Assumes task commands all belong to the same task.
		 */
		void flush(const command_set& cmds);

	  private:
		flush_callback m_flush_cb;

		void serialize_and_flush(abstract_command* cmd, std::vector<command_id>&& dependencies) const;
	};

} // namespace detail
} // namespace celerity
