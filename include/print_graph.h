#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "task.h"

namespace celerity {
namespace detail {

	class command_graph;
	class task_manager;

	std::string print_task_graph(const std::unordered_map<task_id, std::unique_ptr<task>>& tdag);
	std::string print_command_graph(const command_graph& cdag, const task_manager& tm);

} // namespace detail
} // namespace celerity
