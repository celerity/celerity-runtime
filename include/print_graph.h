#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "task.h"

namespace celerity {
namespace detail {

	class command_graph;

	std::string print_graph(const std::unordered_map<task_id, std::unique_ptr<task>>& tdag);
	std::string print_graph(const command_graph& cdag);

} // namespace detail
} // namespace celerity
