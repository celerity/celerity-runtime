#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <celerity.h>

using namespace celerity;
using namespace celerity::detail;


TEST_CASE("handler::parallel_for accepts nd_range", "[handler]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);

	SECTION("1D") {
		CHECK_NOTHROW(cgh.parallel_for(celerity::nd_range<1>{{256}, {64}}, [](nd_item<1> item) {
			group_barrier(item.get_group());
			group_broadcast(item.get_group(), 42);
		}));
	}

	SECTION("2D") {
		CHECK_NOTHROW(cgh.parallel_for(celerity::nd_range<2>{{64, 64}, {8, 8}}, [](nd_item<2> item) {
			group_barrier(item.get_group());
			group_broadcast(item.get_group(), 42, 25);
		}));
	}

	SECTION("3D") {
		CHECK_NOTHROW(cgh.parallel_for(celerity::nd_range<3>{{16, 16, 16}, {4, 4, 4}}, [](nd_item<3> item) {
			group_barrier(item.get_group());
			group_broadcast(item.get_group(), 42, {1, 2, 3});
		}));
	}
}

TEST_CASE("handler throws if effective split constraint does not evenly divide global size", "[handler]") {
	const auto submit = [](auto range, auto constraint) {
		auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
		experimental::constrain_split(cgh, constraint);
		cgh.parallel_for(range, [=](auto) {});
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


class my_hint : public detail::hint_base {
  public:
	my_hint(int value) : m_value(value) {}
	int get_value() const { return m_value; }

  private:
	int m_value;
};

class my_other_hint : public detail::hint_base {
  private:
	void validate(const hint_base& other) const override {
		if(auto ptr = dynamic_cast<const my_hint*>(&other); ptr != nullptr) {
			if(ptr->get_value() != 1337) throw std::runtime_error("not leet enough");
		}
	}
};

TEST_CASE("hints can be attached to and retrieved from tasks", "[handler][task-hints]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
	experimental::hint(cgh, my_hint{1337});
	cgh.parallel_for(range<1>{1}, [](item<1>) {});

	const auto tsk = into_task(std::move(cgh));
	const auto hint = tsk->get_hint<my_hint>();
	REQUIRE(hint != nullptr);
	CHECK(hint->get_value() == 1337);
}

TEST_CASE("providing a hint of a particular type more than once throws", "[handler][task-hints]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
	CHECK_NOTHROW(experimental::hint(cgh, my_hint{1337}));
	CHECK_NOTHROW(experimental::hint(cgh, my_other_hint{}));
	CHECK_THROWS_WITH(experimental::hint(cgh, my_hint{1337}), "Providing more than one hint of the same type is not allowed");
}

TEST_CASE("hints can ensure combinations with other hints are valid", "[handler][task-hints]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
	CHECK_NOTHROW(experimental::hint(cgh, my_other_hint{}));
	CHECK_THROWS_WITH(experimental::hint(cgh, my_hint{1336}), "not leet enough");
}

TEST_CASE("split_1d and split_2d hints cannot be combined", "[handler][task-hints]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
	SECTION("1d then 2d") {
		CHECK_NOTHROW(experimental::hint(cgh, experimental::hints::split_1d{}));
		CHECK_THROWS_WITH(experimental::hint(cgh, experimental::hints::split_2d{}), "Cannot combine split_1d and split_2d hints");
	}
	SECTION("2d then 1d") {
		CHECK_NOTHROW(experimental::hint(cgh, experimental::hints::split_2d{}));
		CHECK_THROWS_WITH(experimental::hint(cgh, experimental::hints::split_1d{}), "Cannot combine split_1d and split_2d hints");
	}
}

TEST_CASE("host_task(once) is equivalent to a host task with unit range", "[handler]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);

	SECTION("with an argument-less functor") {
		cgh.host_task(once, [] {});
	}
	SECTION("with a unary functor") {
		cgh.host_task(once, [](partition<0> part) {});
	}

	auto tsk = detail::into_task(std::move(cgh));
	CHECK(tsk->get_geometry().dimensions == 0);
	CHECK(tsk->get_geometry().global_size.size() == 1);
	CHECK(tsk->get_geometry().global_offset == zeros);
	CHECK(tsk->get_geometry().granularity == ones);
	CHECK(tsk->get_type() == task_type::host_compute); // NOT the magic "master node task" type
}

TEST_CASE("parallel_for(size_t, ...) acts as a shorthand for parallel_for(range<1>, ...)", "[handler]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
	cgh.parallel_for(10, [](item<1>) {});

	const auto tsk = into_task(std::move(cgh));
	CHECK(tsk->get_geometry().global_size == range_cast<3>(range(10)));
	CHECK(tsk->get_geometry().global_offset == zeros);
	CHECK(tsk->get_geometry().granularity == ones);
}

TEST_CASE("parallel_for(size_t, size_t,, ...) acts as a shorthand for parallel_for(range<1>, id<1>,, ...)", "[handler]") {
	auto cgh = detail::make_command_group_handler(task_id(1), 1 /* num_collective_nodes */);
	cgh.parallel_for(10, 11, [](item<1>) {});

	const auto tsk = into_task(std::move(cgh));
	CHECK(tsk->get_geometry().global_size == range_cast<3>(range(10)));
	CHECK(tsk->get_geometry().global_offset == id_cast<3>(id(11)));
	CHECK(tsk->get_geometry().granularity == ones);
}
