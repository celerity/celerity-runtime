#include "../unit_test_suite_celerity.h"

#include <algorithm>

#include <catch2/catch.hpp>

#include <celerity.h>

#include "ranges.h"

namespace celerity {
namespace detail {

#if !WORKAROUND_COMPUTECPP && (!WORKAROUND_HIPSYCL || CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)

	template <typename T>
	struct unknown_identity_maximum {
		T operator()(T a, T b) const { return a < b ? b : a; }
	};

	TEST_CASE("simple reductions produce the expected results", "[reductions]") {
		size_t N = 1000;
		buffer<size_t, 1> sum_buf{{1}};
		buffer<size_t, 1> max_buf{{1}};

		distr_queue q;
		const auto initialize_to_identity = cl::sycl::property::reduction::initialize_to_identity{};

#if WORKAROUND_DPCPP // DPC++ can handle at most 1 reduction variable per kernel
		q.submit([=](handler& cgh) {
			auto sum_r = reduction(sum_buf, cgh, cl::sycl::plus<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, cl::sycl::id{1}, sum_r, [=](celerity::item<1> item, auto& sum) { sum += item.get_id(0); });
		});

		q.submit([=](handler& cgh) {
			auto max_r = reduction(max_buf, cgh, size_t{0}, unknown_identity_maximum<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(
			    cl::sycl::range{N}, cl::sycl::id{1}, max_r, [=](celerity::item<1> item, auto& max) { max.combine(item.get_id(0)); });
		});
#else
		q.submit([=](handler& cgh) {
			auto sum_r = reduction(sum_buf, cgh, cl::sycl::plus<size_t>{}, initialize_to_identity);
			auto max_r = reduction(max_buf, cgh, size_t{0}, unknown_identity_maximum<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, cl::sycl::id{1}, sum_r, max_r, [=](celerity::item<1> item, auto& sum, auto& max) {
				sum += item.get_id(0);
				max.combine(item.get_id(0));
			});
		});
#endif

		q.submit([=](handler& cgh) {
			accessor sum_acc{sum_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
			accessor max_acc{max_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
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
			    [=](celerity::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([=](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
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
			    [=](celerity::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, reduction(sum, cgh, cl::sycl::plus<int>{} /* include previous reduction result */),
			    [=](celerity::item<1> item, auto& sum) { sum += 2; });
		});

		q.submit([=](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_write_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == 3 * N); });
		});
	}

#endif // !WORKAROUND_COMPUTECPP && (!WORKAROUND_HIPSYCL || CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)

} // namespace detail
} // namespace celerity