#include "unit_test_suite_celerity.h"

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

void maybe_print_graph(celerity::detail::task_manager& tm) {
	if(print_graphs) {
		const auto graph_str = tm.print_graph();
		assert(graph_str.has_value());
		CELERITY_INFO("Task graph:\n\n{}\n", *graph_str);
	}
}

void maybe_print_graph(celerity::detail::command_graph& cdag) {
	if(print_graphs) {
		const auto graph_str = cdag.print_graph();
		assert(graph_str.has_value());
		CELERITY_INFO("Command graph:\n\n{}\n", *graph_str);
	}
}

void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx) {
	if(print_graphs) {
		maybe_print_graph(ctx.get_task_manager());
		maybe_print_graph(ctx.get_command_graph());
	}
}
