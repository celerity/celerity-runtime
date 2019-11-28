#pragma once

#include "test_utils.h"
#include "unit_test_suite.h"

extern celerity::detail::logger graph_logger;

template <typename GraphOwner>
inline void maybe_print_graph(GraphOwner& go) {
	if(print_graphs) { go.print_graph(graph_logger); }
}

void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx);
