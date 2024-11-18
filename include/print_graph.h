#pragma once

#include "recorders.h"
#include "types.h"

#include <string>
#include <vector>


namespace celerity::detail {

[[nodiscard]] std::string print_task_graph(const task_recorder& recorder, const std::string& title = "Task Graph");
[[nodiscard]] std::string print_command_graph(const node_id local_nid, const command_recorder& recorder, const std::string& title = "Command Graph");
[[nodiscard]] std::string combine_command_graphs(const std::vector<std::string>& graphs, const std::string& title = "Command Graph");
[[nodiscard]] std::string print_instruction_graph(const instruction_recorder& irec, const command_recorder& crec, const task_recorder& trec,
    const instruction_performance_recorder* const iprec, const std::string& title = "Instruction Graph");

} // namespace celerity::detail
