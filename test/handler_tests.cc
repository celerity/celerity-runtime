#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <celerity.h>

using namespace celerity;
using namespace celerity::detail;


TEST_CASE("handler::parallel_for accepts nd_range", "[handler]") {
	SECTION("1D") {
		CHECK_NOTHROW(invoke_command_group_function([](handler& cgh) {
			cgh.parallel_for(celerity::nd_range<1>{{256}, {64}}, [](nd_item<1> item) {
				group_barrier(item.get_group());
				group_broadcast(item.get_group(), 42);
			});
		}));
	}

	SECTION("2D") {
		CHECK_NOTHROW(invoke_command_group_function([](handler& cgh) {
			cgh.parallel_for(celerity::nd_range<2>{{64, 64}, {8, 8}}, [](nd_item<2> item) {
				group_barrier(item.get_group());
				group_broadcast(item.get_group(), 42, 25);
			});
		}));
	}

	SECTION("3D") {
		CHECK_NOTHROW(invoke_command_group_function([](handler& cgh) {
			cgh.parallel_for(celerity::nd_range<3>{{16, 16, 16}, {4, 4, 4}}, [](nd_item<3> item) {
				group_barrier(item.get_group());
				group_broadcast(item.get_group(), 42, {1, 2, 3});
			});
		}));
	}
}

TEST_CASE("handler throws if effective split constraint does not evenly divide global size", "[handler]") {
	const auto submit = [](auto range, auto constraint) {
		invoke_command_group_function([&](handler& cgh) {
			experimental::constrain_split(cgh, constraint);
			cgh.parallel_for(range, [=](auto) {});
		});
	};

	CHECK_THROWS_WITH(submit(range<1>{10}, range<1>{0}), "Split constraint cannot be 0");
	CHECK_THROWS_WITH(submit(range<2>{10, 10}, range<2>{2, 0}), "Split constraint cannot be 0");
	CHECK_THROWS_WITH(submit(range<3>{10, 10, 10}, range<3>{2, 2, 0}), "Split constraint cannot be 0");

	CHECK_NOTHROW(submit(range<1>{10}, range<1>{2}));
	CHECK_NOTHROW(submit(range<2>{10, 8}, range<2>{2, 4}));
	CHECK_NOTHROW(submit(range<3>{10, 8, 16}, range<3>{2, 4, 8}));

	CHECK_THROWS_WITH(submit(range<1>{10}, range<1>{3}), "The split constraint [3] does not evenly divide the kernel global size [10]");
	CHECK_THROWS_WITH(submit(range<2>{10, 8}, range<2>{2, 5}), "The split constraint [2,5] does not evenly divide the kernel global size [10,8]");
	CHECK_THROWS_WITH(submit(range<3>{10, 8, 16}, range<3>{2, 4, 9}), "The split constraint [2,4,9] does not evenly divide the kernel global size [10,8,16]");

	CHECK_THROWS_WITH(submit(range<1>{10}, range<1>{20}), "The split constraint [20] does not evenly divide the kernel global size [10]");

	CHECK_NOTHROW(submit(nd_range<1>{100, 10}, range<1>{2}));
	CHECK_NOTHROW(submit(nd_range<2>{{100, 80}, {10, 20}}, range<2>{2, 4}));
	CHECK_NOTHROW(submit(nd_range<3>{{100, 80, 60}, {1, 2, 30}}, range<3>{2, 4, 20}));

	CHECK_THROWS_WITH(submit(nd_range<1>{100, 10}, range<1>{3}), "The effective split constraint [30] does not evenly divide the kernel global size [100]");
	CHECK_THROWS_WITH(submit(nd_range<2>{{100, 80}, {10, 20}}, range<2>{2, 3}),
	    "The effective split constraint [10,60] does not evenly divide the kernel global size [100,80]");
	CHECK_THROWS_WITH(submit(nd_range<3>{{100, 80, 60}, {1, 2, 30}}, range<3>{1, 2, 40}),
	    "The effective split constraint [1,2,120] does not evenly divide the kernel global size [100,80,60]");
}

TEST_CASE("host_task(once) is equivalent to a host task with unit range", "[handler]") {
	raw_command_group cg;

	SECTION("with an argument-less functor") {
		cg = invoke_command_group_function([](handler& cgh) { cgh.host_task(once, [] {}); });
	}
	SECTION("with a unary functor") {
		cg = invoke_command_group_function([](handler& cgh) { cgh.host_task(once, [](partition<0> part) {}); });
	}

	CHECK(cg.geometry.value().dimensions == 0);
	CHECK(cg.geometry.value().global_size.size() == 1);
	CHECK(cg.geometry.value().global_offset == zeros);
	CHECK(cg.geometry.value().granularity == ones);
	CHECK(cg.task_type == task_type::host_compute); // NOT the magic "master node task" type
}

TEST_CASE("parallel_for(size_t, ...) acts as a shorthand for parallel_for(range<1>, ...)", "[handler]") {
	const auto cg = invoke_command_group_function([](handler& cgh) { cgh.parallel_for(10, [](item<1>) {}); });
	CHECK(cg.geometry.value().global_size == range_cast<3>(range(10)));
	CHECK(cg.geometry.value().global_offset == zeros);
	CHECK(cg.geometry.value().granularity == ones);
}

TEST_CASE("parallel_for(size_t, size_t,, ...) acts as a shorthand for parallel_for(range<1>, id<1>,, ...)", "[handler]") {
	const auto cg = invoke_command_group_function([](handler& cgh) { cgh.parallel_for(10, 11, [](item<1>) {}); });
	CHECK(cg.geometry.value().global_size == range_cast<3>(range(10)));
	CHECK(cg.geometry.value().global_offset == id_cast<3>(id(11)));
	CHECK(cg.geometry.value().granularity == ones);
}
