#pragma once

#include <functional>
#include <unordered_set>

#include "command.h"
#include "frame.h"
#include "types.h"

namespace celerity {
namespace detail {

	class abstract_command;
	class task_command;
	class command_graph;

	class graph_serializer {
		using flush_callback = std::function<void(node_id, unique_frame_ptr<command_frame>)>;

	  public:
		/*
		 * @param flush_cb Callback invoked for each command that is being flushed
		 */
		graph_serializer(command_graph& cdag, flush_callback flush_cb) : m_cdag(cdag), m_flush_cb(flush_cb) {}

		/**
		 * Serializes a set of commands. Assumes task commands all belong to the same task.
		 */
		void flush(const std::unordered_set<abstract_command*>& cmds);

	  private:
		command_graph& m_cdag;
		flush_callback m_flush_cb;

		void serialize_and_flush(abstract_command* cmd, const std::vector<command_id>& dependencies) const;
	};

} // namespace detail
} // namespace celerity
