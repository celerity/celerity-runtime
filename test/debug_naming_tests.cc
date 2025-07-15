#include "cgf.h"
#include "types.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "task_graph_test_utils.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("debug names can be set and retrieved from tasks", "[debug]") {
	const std::string task_name = "sample task";

	test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

	SECTION("Host Task") {
		const auto tid_a = tctx.master_node_host_task().name(task_name).submit();
		const auto tid_b = tctx.master_node_host_task().submit();

		CHECK(test_utils::get_task(tctx.get_task_graph(), tid_a)->get_debug_name() == task_name);
		CHECK(test_utils::get_task(tctx.get_task_graph(), tid_b)->get_debug_name().empty());
	}

	SECTION("Compute Task") {
		const auto tid_a = tctx.device_compute(range<1>(ones)).name(task_name).submit();
		const auto tid_b = tctx.device_compute<class compute_task_unnamed>(range<1>(ones)).submit();

		CHECK(test_utils::get_task(tctx.get_task_graph(), tid_a)->get_debug_name() == task_name);
		CHECK_THAT(test_utils::get_task(tctx.get_task_graph(), tid_b)->get_debug_name(), Catch::Matchers::ContainsSubstring("compute_task_unnamed"));
	}

	SECTION("ND Range Task") {
		const auto tid_a = tctx.device_compute(nd_range<1>{range<1>{1}, range<1>{1}}).name(task_name).submit();
		const auto tid_b = tctx.device_compute<class nd_range_task_unnamed>(nd_range<1>{range<1>{1}, range<1>{1}}).submit();

		CHECK(test_utils::get_task(tctx.get_task_graph(), tid_a)->get_debug_name() == task_name);
		CHECK_THAT(test_utils::get_task(tctx.get_task_graph(), tid_b)->get_debug_name(), Catch::Matchers::ContainsSubstring("nd_range_task_unnamed"));
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "buffers allow setting and retrieving debug names", "[debug]") {
	celerity::buffer<int, 1> buff_a(16);
	std::string buff_name{"my_buffer"};
	debug::set_buffer_name(buff_a, buff_name);
	CHECK(debug::get_buffer_name(buff_a) == buff_name);
}


namespace foo {
class MySecondKernel;
}

template <typename T>
class MyThirdKernel;

TEST_CASE("device_compute tasks derive debug name from kernel name", "[task]") {
	const auto cg1 = invoke_command_group_function([](handler& cgh) { cgh.parallel_for<class MyFirstKernel>(range<1>{1}, [](id<1>) {}); });
	const auto cg2 = invoke_command_group_function([](handler& cgh) { cgh.parallel_for<foo::MySecondKernel>(range<1>{1}, [](id<1>) {}); });
	const auto cg3 = invoke_command_group_function([](handler& cgh) { cgh.parallel_for<MyThirdKernel<int>>(range<1>{1}, [](id<1>) {}); });
	CHECK(cg1.task_name == "MyFirstKernel");
	CHECK(cg2.task_name == "MySecondKernel");
	CHECK(cg3.task_name == "MyThirdKernel<...>");
}
