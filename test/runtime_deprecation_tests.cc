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

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	TEST_CASE("deprecated range mapper templates continue to work", "[range-mapper][deprecated]") {
		const auto chunk1d = chunk<1>{1, 2, 3};
		const auto chunk2d = chunk<2>{{1, 1}, {2, 2}, {3, 3}};
		const auto chunk3d = chunk<3>{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
		const auto range1d = range<1>{4};
		const auto range2d = range<2>{4, 4};
		const auto range3d = range<3>{4, 4, 4};
		const auto subrange1d = subrange<1>{{}, range1d};
		const auto subrange2d = subrange<2>{{}, range2d};
		const auto subrange3d = subrange<3>{{}, range3d};

		CHECK(one_to_one<1>{}(chunk1d) == subrange{chunk1d});
		CHECK(one_to_one<2>{}(chunk2d) == subrange{chunk2d});
		CHECK(one_to_one<3>{}(chunk3d) == subrange{chunk3d});

		CHECK(fixed<1, 3>{subrange3d}(chunk1d) == subrange3d);
		CHECK(fixed<2, 2>{subrange2d}(chunk2d) == subrange2d);
		CHECK(fixed<3, 1>{subrange1d}(chunk3d) == subrange1d);

		CHECK(all<1>{}(chunk1d, range1d) == subrange1d);
		CHECK(all<2>{}(chunk2d, range2d) == subrange2d);
		CHECK(all<3>{}(chunk3d, range3d) == subrange3d);
		CHECK(all<1, 3>{}(chunk1d, range3d) == subrange3d);
		CHECK(all<2, 2>{}(chunk2d, range2d) == subrange2d);
		CHECK(all<3, 1>{}(chunk3d, range1d) == subrange1d);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture,
	    "distr_queue::submit(allow_by_ref_t, ...) and creation of accessors/side-effects/reductions from const buffers/host-objects continues to work",
	    "[handler][deprecated]") {
		distr_queue q;
		buffer<size_t, 1> buf{32};
		buffer<size_t, 1> reduction_buf{1};
		experimental::host_object<size_t> ho;
		int my_int = 33;
		q.submit(allow_by_ref, [= /* capture buffer/host-object by value */, &my_int](handler& cgh) {
			accessor acc{buf, cgh, celerity::access::all{}, celerity::write_only_host_task};
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

} // namespace detail
} // namespace celerity
