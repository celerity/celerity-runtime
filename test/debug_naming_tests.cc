#include "task.h"
#include "task_manager.h"
#include "types.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("debug names can be set and retrieved from tasks", "[debug]") {
	const std::string task_name = "sample task";

	auto tt = test_utils::task_test_context{};

	SECTION("Host Task") {
		const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { celerity::debug::set_task_name(cgh, task_name); });

		const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {});

		CHECK(tt.tm.get_task(tid_a)->get_debug_name() == task_name);
		CHECK(tt.tm.get_task(tid_b)->get_debug_name().empty());
	}

	SECTION("Compute Task") {
		const auto tid_a = test_utils::add_compute_task<class compute_task>(tt.tm, [&](handler& cgh) { celerity::debug::set_task_name(cgh, task_name); });

		const auto tid_b = test_utils::add_compute_task<class compute_task_unnamed>(tt.tm, [&](handler& cgh) {});

		CHECK(tt.tm.get_task(tid_a)->get_debug_name() == task_name);
		CHECK_THAT(tt.tm.get_task(tid_b)->get_debug_name(), Catch::Matchers::ContainsSubstring("compute_task_unnamed"));
	}

	SECTION("ND Range Task") {
		const auto tid_a =
		    test_utils::add_nd_range_compute_task<class nd_range_task>(tt.tm, [&](handler& cgh) { celerity::debug::set_task_name(cgh, task_name); });

		const auto tid_b = test_utils::add_compute_task<class nd_range_task_unnamed>(tt.tm, [&](handler& cgh) {});

		CHECK(tt.tm.get_task(tid_a)->get_debug_name() == task_name);
		CHECK_THAT(tt.tm.get_task(tid_b)->get_debug_name(), Catch::Matchers::ContainsSubstring("nd_range_task_unnamed"));
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
	auto tm = celerity::detail::task_manager(1, nullptr);
	const auto t1 = tm.get_task(tm.submit_command_group([](handler& cgh) { cgh.parallel_for<class MyFirstKernel>(range<1>{1}, [](id<1>) {}); }));
	const auto t2 = tm.get_task(tm.submit_command_group([](handler& cgh) { cgh.parallel_for<foo::MySecondKernel>(range<1>{1}, [](id<1>) {}); }));
	const auto t3 = tm.get_task(tm.submit_command_group([](handler& cgh) { cgh.parallel_for<MyThirdKernel<int>>(range<1>{1}, [](id<1>) {}); }));
	CHECK(t1->get_debug_name() == "MyFirstKernel");
	CHECK(t2->get_debug_name() == "MySecondKernel");
	CHECK(t3->get_debug_name() == "MyThirdKernel<...>");
}
