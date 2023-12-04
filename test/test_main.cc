#include "test_utils.h"
#include "utils.h"

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

	int return_code = session.applyCommandLine(argc, argv);
	if(return_code != 0) { return return_code; }

	// allow unit tests to catch and recover from panics
	celerity::detail::utils::set_panic_solution(celerity::detail::utils::panic_solution::throw_logic_error);

	celerity::detail::runtime::test_mode_enter();
	return_code = session.run();
	celerity::detail::runtime::test_mode_exit();
	return return_code;
}

struct global_setup_and_teardown : Catch::EventListenerBase {
	using EventListenerBase::EventListenerBase;

	void testRunStarting(const Catch::TestRunInfo& /* info */) override { celerity::detail::closure_hydrator::make_available(); }

	void testCasePartialEnded(const Catch::TestCaseStats&, uint64_t) override {
		// Reset REQUIRE_LOOP after each test case, section or generator value.
		celerity::test_utils::require_loop_assertion_registry::get_instance().reset();
	}
};

CATCH_REGISTER_LISTENER(global_setup_and_teardown);
