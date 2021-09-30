#include "unit_test_suite_celerity.h"

celerity::detail::logger graph_logger{"test-graph", celerity::detail::log_level::trace};

namespace detail {
void test_run_started_callback() {
	celerity::detail::runtime::enable_test_mode();
}
void test_case_ended_callback() {
	if(celerity::detail::runtime::is_initialized()) { celerity::detail::runtime::teardown(); }
}
void test_run_ended_callback() {
	celerity::detail::runtime::finish_test_mode();
}
} // namespace detail

void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx) {
	if(print_graphs) {
		ctx.get_task_manager().print_graph(graph_logger);
		ctx.get_command_graph().print_graph(graph_logger);
	}
}
