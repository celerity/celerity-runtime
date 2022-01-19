#pragma once

#include "test_utils.h"
#include "unit_test_suite.h"

namespace celerity::detail {
class command_graph;
class task_manager;
} // namespace celerity::detail

void maybe_print_graph(celerity::detail::task_manager& tm);
void maybe_print_graph(celerity::detail::command_graph& cdag);
void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx);
