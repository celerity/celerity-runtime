#pragma once

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include "test_utils.h"

// Printing of graphs can be enabled using the "--print-graphs" command line flag
bool print_graphs = false;
celerity::detail::logger graph_logger{"test-graph", celerity::detail::log_level::trace};

template <typename GraphOwner>
void maybe_print_graph(GraphOwner& go) {
	if(print_graphs) { go.print_graph(graph_logger); }
}

void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx) {
	if(print_graphs) {
		ctx.get_task_manager().print_graph(graph_logger);
		ctx.get_command_graph().print_graph(graph_logger);
	}
}

/**
 * We provide a custom main function to add additional CLI flags.
 */
int main(int argc, char* argv[]) {
	Catch::Session session;

	using namespace Catch::clara;
	const auto cli = session.cli() | Opt(print_graphs)["--print-graphs"]("print graphs (GraphViz)");

	session.cli(cli);

	const int returnCode = session.applyCommandLine(argc, argv);
	if(returnCode != 0) { return returnCode; }

	return session.run();
}
