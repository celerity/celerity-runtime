#pragma once

#include <memory>
#include <string>

#include "task.h"
#include "task_ring_buffer.h"

namespace celerity {
namespace detail {

	class buffer_manager;
	class command_graph;
	class task_manager;

	std::string print_task_graph(const task_ring_buffer& tdag, const buffer_manager* bm);
	std::string print_command_graph(const node_id local_nid, const command_graph& cdag, const task_manager& tm, const buffer_manager* bm);
	std::string combine_command_graphs(const std::vector<std::string>& graphs);

} // namespace detail
} // namespace celerity
