#include "unit_test_suite.h"

#include <catch2/catch_session.hpp>

bool print_graphs = false;

namespace celerity::test_utils {

struct test_mode_guard {
	test_mode_guard() { test_run_started_callback(); }
	test_mode_guard(const test_mode_guard&) = delete;
	test_mode_guard& operator=(const test_mode_guard&) = delete;
	~test_mode_guard() { test_run_ended_callback(); }
};

} // namespace celerity::test_utils


/**
 * We provide a custom main function to add additional CLI flags.
 */
int main(int argc, char* argv[]) {
	Catch::Session session;

	using namespace Catch::Clara;
	const auto cli = session.cli() | Opt(print_graphs)["--print-graphs"]("print graphs (GraphViz)");

	session.cli(cli);

	int returnCode = session.applyCommandLine(argc, argv);
	if(returnCode != 0) { return returnCode; }

	celerity::test_utils::test_mode_guard test_mode;
	return session.run();
}
