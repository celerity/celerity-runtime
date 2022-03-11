#include "test_utils.h"

#include <catch2/catch_session.hpp>


/**
 * We provide a custom main function to add additional CLI flags.
 */
int main(int argc, char* argv[]) {
	Catch::Session session;

	using namespace Catch::Clara;
	const auto cli = session.cli() | Opt(celerity::test_utils::print_graphs)["--print-graphs"]("print graphs (GraphViz)");

	session.cli(cli);

	int returnCode = session.applyCommandLine(argc, argv);
	if(returnCode != 0) { return returnCode; }

	celerity::detail::runtime::test_mode_enter();
	returnCode = session.run();
	celerity::detail::runtime::test_mode_exit();
	return returnCode;
}
