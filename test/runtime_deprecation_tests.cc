// This diagnostic must be disabled here, because ComputeCpp appears to override it when specified on the command line.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "sycl_wrappers.h"

#include <algorithm>
#include <memory>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "ranges.h"

#include "buffer_manager_test_utils.h"

namespace celerity {
namespace detail {

	TEST_CASE_METHOD(test_utils::runtime_fixture,
	    "distr_queue::submit(allow_by_ref_t, ...) and creation of accessors/side-effects/reductions from const buffers/host-objects continues to work",
	    "[handler][deprecated]") {
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
#if CELERITY_FEATURE_SCALAR_REDUCTIONS
			auto red = reduction(reduction_buf, cgh, std::plus<size_t>{});
#endif
			cgh.parallel_for(range<1>{32}, [=](item<1>) { (void)acc; });
		});
		SUCCEED();
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "experimental::fence continues to work", "[deprecated][fence]") {
		distr_queue q;

		std::vector<int> init(16);
		std::iota(init.begin(), init.end(), 0);
		buffer<int, 1> buf(init.data(), init.size());

		experimental::host_object<int> ho(42);

		experimental::buffer_snapshot<int, 1> full_snapshot = experimental::fence(q, buf).get();
		experimental::buffer_snapshot<int, 1> partial_snapshot = experimental::fence(q, buf, subrange<1>(8, 8)).get();
		int ho_value = experimental::fence(q, ho).get();

		CHECK(full_snapshot.get_range() == range<1>(16));
		CHECK(std::equal(init.begin(), init.end(), full_snapshot.get_data()));
		CHECK(partial_snapshot.get_range() == range<1>(8));
		CHECK(partial_snapshot.get_offset() == id<1>(8));
		CHECK(std::equal(init.begin() + 8, init.end(), partial_snapshot.get_data()));
		CHECK(ho_value == 42);
	}

} // namespace detail
} // namespace celerity
