#include "test_utils.h"
#include "utils.h"

#include <catch2/catch_session.hpp>

/**
 * We provide a custom main function to add additional CLI flags.
 */
int main(int argc, char* argv[]) {
	Catch::Session session;

	using namespace Catch::Clara;
	const auto cli = session.cli() | Opt(celerity::test_utils::print_graphs)["--print-graphs"]("print graphs (GraphViz)");

	session.cli(cli);

	int return_code = session.applyCommandLine(argc, argv);
	if(return_code != 0) { return return_code; }

	// allow unit tests to catch and recover from panics
	celerity::detail::utils::set_panic_solution(celerity::detail::utils::panic_solution::throw_logic_error);

	celerity::detail::runtime::test_mode_enter();
	return_code = session.run();
	celerity::detail::runtime::test_mode_exit();
	return return_code;
}
