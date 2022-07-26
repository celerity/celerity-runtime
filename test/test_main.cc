#include "test_utils.h"

#include <catch2/catch_session.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

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

struct global_setup_and_teardown : Catch::EventListenerBase {
	using EventListenerBase::EventListenerBase;
	void testCasePartialEnded(const Catch::TestCaseStats&, uint64_t) override {
		// Reset REQUIRE_LOOP after each test case, section or generator value.
		celerity::test_utils::require_loop_assertion_registry::get_instance().reset();
	}
};

CATCH_REGISTER_LISTENER(global_setup_and_teardown);
