// This diagnostic must be disabled here, because ComputeCpp appears to override it when specified on the command line.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "ranges.h"
#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

namespace celerity {
namespace detail {

	TEST_CASE_METHOD(test_utils::runtime_fixture, "any number of distr_queues can be created", "[deprecated][distr_queue][lifetime]") {
		distr_queue q1;
		auto q2{q1};    // Copying is allowed
		distr_queue q3; // so is creating new ones as of Celerity 0.7.0
		distr_queue q4;
	}

	TEST_CASE_METHOD(
	    test_utils::runtime_fixture, "new distr_queues can be created after the last one has been destroyed", "[deprecated][distr_queue][lifetime]") {
		distr_queue{};
		CHECK(runtime::has_instance()); // ~distr_queue does not shut down the runtime as of Celerity 0.7.0
		distr_queue{};
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "distr_queue implicitly initializes the runtime", "[deprecated][distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::has_instance());
		distr_queue queue;
		REQUIRE(runtime::has_instance());
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "an explicit device can be provided to distr_queue", "[deprecated][distr_queue]") {
		sycl::device device;

		SECTION("before the runtime is initialized") {
			REQUIRE_FALSE(runtime::has_instance());
			REQUIRE_NOTHROW(distr_queue{std::vector{device}});
		}

		SECTION("but not once the runtime has been initialized") {
			REQUIRE_FALSE(runtime::has_instance());
			runtime::init(nullptr, nullptr);
			REQUIRE_THROWS_WITH(distr_queue{std::vector{device}}, "Passing explicit device list not possible, runtime has already been initialized.");
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture,
	    "distr_queue::submit(allow_by_ref_t, ...) and creation of accessors/side-effects/reductions from const buffers/host-objects continues to work",
	    "[handler][deprecated][reduction]") {
		distr_queue q;
		buffer<size_t, 1> buf{32};
		buffer<size_t, 1> reduction_buf{1};
		experimental::host_object<size_t> ho;
		int my_int = 33;
		q.submit(allow_by_ref, [= /* capture buffer/host-object by value */, &my_int](handler& cgh) {
			accessor acc{buf, cgh, celerity::access::all{}, celerity::write_only_host_task, celerity::no_init};
			experimental::side_effect se{ho, cgh};
			cgh.host_task(on_master_node, [=, &my_int] {
				(void)acc;
				(void)se;
				my_int = 42;
			});
		});
		q.submit([= /* capture by value */](handler& cgh) {
			accessor acc{buf, cgh, celerity::access::one_to_one{}, celerity::read_only};
			cgh.parallel_for(range<1>(32), reduction(reduction_buf, cgh, sycl::plus<size_t>{}, property::reduction::initialize_to_identity{}),
			    [=](item<1>, auto&) { (void)acc; });
		});
		q.slow_full_sync(); // `my_int` must not go out of scope before host task finishes executing
		CHECK(my_int == 42);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "get_access can still be called on a const buffer", "[buffer]") {
		const range<2> range{32, 64};
		std::vector<float> init(range.size());
		buffer<float, 2> buf_a{init.data(), range};
		const auto cg = invoke_command_group_function([&](handler& cgh) {
			auto acc = std::as_const(buf_a).get_access<access_mode::read>(cgh, celerity::access::one_to_one{});
			cgh.parallel_for(range, [=](item<2>) { (void)acc; });
		});
		CHECK(cg.buffer_accesses.size() == 1);
		CHECK(cg.buffer_accesses[0].bid == get_buffer_id(buf_a));
		CHECK(cg.buffer_accesses[0].mode == access_mode::read);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "experimental::fence continues to work", "[deprecated][fence]") {
		distr_queue q;

		std::vector<int> init(16);
		std::iota(init.begin(), init.end(), 0);
		buffer<int, 1> buf(init.data(), init.size());

		experimental::host_object<int> ho(42);

		experimental::buffer_snapshot<int, 1> full_snapshot = experimental::fence(q, buf).get();
		experimental::buffer_snapshot<int, 1> partial_snapshot = experimental::fence(q, buf, subrange<1>(8, 8)).get();
		const int ho_value = experimental::fence(q, ho).get();

		CHECK(full_snapshot.get_range() == range<1>(16));
		CHECK(std::equal(init.begin(), init.end(), full_snapshot.get_data()));
		CHECK(partial_snapshot.get_range() == range<1>(8));
		CHECK(partial_snapshot.get_offset() == id<1>(8));
		CHECK(std::equal(init.begin() + 8, init.end(), partial_snapshot.get_data()));
		CHECK(ho_value == 42);
	}

	TEST_CASE("neighborhood range mapper can still be constructed with coordinate-lists", "[deprecated][range-mapper]") {
		const celerity::access::neighborhood deprecated_n1(1);
		const celerity::access::neighborhood deprecated_n2(1, 2);
		const celerity::access::neighborhood deprecated_n3(1, 2, 3);

		const celerity::access::neighborhood new_n1({1});
		const celerity::access::neighborhood new_n2({1, 2});
		const celerity::access::neighborhood new_n3({1, 2, 3});

		CHECK(range_mapper_testspy::neighborhood_equals(deprecated_n1, new_n1));
		CHECK(range_mapper_testspy::neighborhood_equals(deprecated_n2, new_n2));
		CHECK(range_mapper_testspy::neighborhood_equals(deprecated_n3, new_n3));
	}

} // namespace detail
} // namespace celerity
