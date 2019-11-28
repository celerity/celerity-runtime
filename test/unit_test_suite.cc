#include "unit_test_suite.h"

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

bool print_graphs = false;

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

struct GlobalSetupAndTeardown : Catch::TestEventListenerBase {
	using TestEventListenerBase::TestEventListenerBase;
	void testRunStarting(const Catch::TestRunInfo&) override { detail::test_started_callback(); }
	void testCaseEnded(const Catch::TestCaseStats&) override { detail::test_ended_callback(); }
};

struct GlobalSetupAndTeardown;
CATCH_REGISTER_LISTENER(GlobalSetupAndTeardown)
