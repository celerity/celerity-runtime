#pragma once

#include <memory>
#include <string>

#include "recorders.h"

namespace celerity::detail {

[[nodiscard]] std::string print_task_graph(const task_recorder& recorder);
[[nodiscard]] std::string print_command_graph(const node_id local_nid, const command_recorder& recorder);
[[nodiscard]] std::string combine_command_graphs(const std::vector<std::string>& graphs);

} // namespace celerity::detail
