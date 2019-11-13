#pragma once

#include "task.h"

namespace celerity {
namespace detail {

	class command_graph;
	class logger;

	void print_graph(const std::unordered_map<task_id, std::shared_ptr<task>>& tdag, logger& graph_logger);
	void print_graph(const command_graph& cdag, logger& graph_logger);

} // namespace detail
} // namespace celerity
