#include "../unit_test_suite_celerity.h"

#include <algorithm>

#include <catch2/catch.hpp>

#include <celerity.h>

#include "ranges.h"

namespace celerity {
namespace detail {

	template <typename T>
	struct unknown_identity_maximum {
		T operator()(T a, T b) const { return a < b ? b : a; }
	};

#if !WORKAROUND_COMPUTECPP

	TEST_CASE("simple reductions produce the expected results", "[reductions]") {
		size_t N = 1000;
		buffer<size_t, 1> sum_buf{{1}};
		buffer<size_t, 1> max_buf{{1}};

		distr_queue q;
		q.submit([=](handler& cgh) {
			auto sum_r = reduction(sum_buf, cgh, cl::sycl::plus<size_t>{});
			auto max_r = reduction(max_buf, cgh, size_t{0}, unknown_identity_maximum<size_t>{});
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, cl::sycl::id{1}, sum_r, max_r, [=](cl::sycl::item<1> item, auto& sum, auto& max) {
				sum += item.get_id(0);
				max.combine(item.get_id(0));
			});
		});
		q.submit([=](handler& cgh) {
			auto sum_acc = sum_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<1>{});
			auto max_acc = max_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<1>{});
			cgh.host_task(on_master_node, [=] {
				CHECK(sum_acc[0] == (N + 1) * (N / 2));
				CHECK(max_acc[0] == N);
			});
		});
	}

	// Regression test: The host -> device transfer previously caused an illegal nested sycl::queue::submit call which deadlocks
	// Distributed test, since the single-node case optimizes the reduction command away
	TEST_CASE("reduction commands perform host -> device transfers if necessary", "[reductions]") {
		distr_queue q;

		REQUIRE(runtime::get_instance().get_num_nodes() > 1);

		const int N = 1000;
		const int init = 42;
		buffer<int, 1> sum(&init, cl::sycl::range{1});
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, reduction(sum, cgh, cl::sycl::plus<int>{} /* don't initialize to identity */),
			    [=](cl::sycl::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([=](handler& cgh) {
			auto acc = sum.get_access<cl::sycl::access_mode::read, cl::sycl::target::host_buffer>(cgh, celerity::access::all<1>{});
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == N + init); });
		});
	}

	TEST_CASE("multiple chained reductions produce correct results", "[reductions]") {
		distr_queue q;

		const int N = 1000;

		buffer<int, 1> sum(cl::sycl::range{1});
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N},
			    reduction(sum, cgh, cl::sycl::plus<int>{}, cl::sycl::property::reduction::initialize_to_identity{}),
			    [=](cl::sycl::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, reduction(sum, cgh, cl::sycl::plus<int>{} /* include previous reduction result */),
			    [=](cl::sycl::item<1> item, auto& sum) { sum += 2; });
		});

		q.submit([=](handler& cgh) {
			auto acc = sum.get_access<cl::sycl::access_mode::read_write, cl::sycl::target::host_buffer>(cgh, celerity::access::all<1>{});
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == 3 * N); });
		});
	}

#endif // !WORKAROUND_COMPUTECPP

} // namespace detail
} // namespace celerity