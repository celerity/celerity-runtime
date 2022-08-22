#pragma once

#include <memory>
#include <string>

#include "task.h"
#include "task_ring_buffer.h"

namespace celerity {
namespace detail {

	class command_graph;
	class reduction_manager;
	class task_manager;

	std::string print_task_graph(const task_ring_buffer& tdag, const reduction_manager& rm);
	std::string print_command_graph(const command_graph& cdag, const task_manager& tm, const reduction_manager& rm);

} // namespace detail
} // namespace celerity
