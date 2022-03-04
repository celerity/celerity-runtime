#include "unit_test_suite.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

bool print_graphs = false;

/**
 * We provide a custom main function to add additional CLI flags.
 */
int main(int argc, char* argv[]) {
	Catch::Session session;

	using namespace Catch::Clara;
	const auto cli = session.cli() | Opt(print_graphs)["--print-graphs"]("print graphs (GraphViz)");

	session.cli(cli);

	const int returnCode = session.applyCommandLine(argc, argv);
	if(returnCode != 0) { return returnCode; }

	return session.run();
}

struct GlobalSetupAndTeardown : Catch::EventListenerBase {
	using EventListenerBase::EventListenerBase;
	void testRunStarting(const Catch::TestRunInfo&) override { detail::test_run_started_callback(); }
	void testCaseEnded(const Catch::TestCaseStats&) override { detail::test_case_ended_callback(); }
	void testRunEnded(const Catch::TestRunStats&) override { detail::test_run_ended_callback(); }
};

struct GlobalSetupAndTeardown;
CATCH_REGISTER_LISTENER(GlobalSetupAndTeardown)
