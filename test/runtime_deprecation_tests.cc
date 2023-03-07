// This diagnostic must be disabled here, because ComputeCpp appears to override it when specified on the command line.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "sycl_wrappers.h"

#include <algorithm>
#include <memory>
#include <random>

#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "ranges.h"
#include "region_map.h"

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
		const auto range1d = cl::sycl::range<1>{4};
		const auto range2d = cl::sycl::range<2>{4, 4};
		const auto range3d = cl::sycl::range<3>{4, 4, 4};
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

	TEST_CASE_METHOD(test_utils::runtime_fixture, "deprecated host_memory_layout continues to work", "[task][deprecated]") {
		distr_queue q;

		std::vector<char> memory1d(10);
		buffer<char, 1> buf1d(memory1d.data(), cl::sycl::range<1>(10));

		q.submit([=](handler& cgh) {
			auto b = buf1d.get_access<cl::sycl::access::mode::discard_write, target::host_task>(cgh, all{});
			cgh.host_task(on_master_node, [=](partition<0> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 1);
				CHECK(dims[0].get_global_offset() == 0);
				CHECK(dims[0].get_local_offset() == 0);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_size() >= 10);
				CHECK(dims[0].get_extent() == 10);
			});
		});

		q.submit([=](handler& cgh) {
			auto b = buf1d.get_access<cl::sycl::access::mode::discard_write, target::host_task>(cgh, one_to_one{});
			cgh.host_task(cl::sycl::range<1>(6), cl::sycl::id<1>(2), [=](partition<1> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 1);
				CHECK(dims[0].get_global_offset() == 2);
				CHECK(dims[0].get_local_offset() <= 2);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_size() >= 6);
				CHECK(dims[0].get_local_size() <= 10);
				CHECK(dims[0].get_extent() == 6);
			});
		});

		std::vector<char> memory2d(10 * 10);
		buffer<char, 2> buf2d(memory2d.data(), cl::sycl::range<2>(10, 10));

		q.submit([=](handler& cgh) {
			auto b = buf2d.get_access<cl::sycl::access::mode::discard_write, target::host_task>(cgh, one_to_one{});
			cgh.host_task(cl::sycl::range<2>(5, 6), cl::sycl::id<2>(1, 2), [=](partition<2> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 2);
				CHECK(dims[0].get_global_offset() == 1);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_offset() <= 1);
				CHECK(dims[0].get_local_size() >= 6);
				CHECK(dims[0].get_local_size() <= 10);
				CHECK(dims[0].get_extent() == 5);
				CHECK(dims[1].get_global_offset() == 2);
				CHECK(dims[1].get_global_size() == 10);
				CHECK(dims[1].get_extent() == 6);
			});
		});

		std::vector<char> memory3d(10 * 10 * 10);
		buffer<char, 3> buf3d(memory3d.data(), cl::sycl::range<3>(10, 10, 10));

		q.submit([=](handler& cgh) {
			auto b = buf3d.get_access<cl::sycl::access::mode::discard_write, target::host_task>(cgh, one_to_one{});
			cgh.host_task(cl::sycl::range<3>(5, 6, 7), cl::sycl::id<3>(1, 2, 3), [=](partition<3> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 3);
				CHECK(dims[0].get_global_offset() == 1);
				CHECK(dims[0].get_local_offset() <= 1);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_size() >= 5);
				CHECK(dims[0].get_local_size() <= 10);
				CHECK(dims[0].get_extent() == 5);
				CHECK(dims[1].get_global_offset() == 2);
				CHECK(dims[1].get_local_offset() <= 2);
				CHECK(dims[1].get_global_size() == 10);
				CHECK(dims[1].get_local_size() >= 6);
				CHECK(dims[1].get_local_size() <= 10);
				CHECK(dims[1].get_extent() == 6);
				CHECK(dims[2].get_global_offset() == 3);
				CHECK(dims[2].get_global_size() == 10);
				CHECK(dims[2].get_extent() == 7);
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "Kernels receiving cl::sycl::item<Dims> (deprecated) continue to work", "[handler][deprecated]") {
		distr_queue q;

		buffer<int, 1> buf1d{{1}};
		q.submit([=](handler& cgh) {
			accessor acc{buf1d, cgh, celerity::access::one_to_one{}, celerity::read_write, celerity::no_init};
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<1>{1}, [=](cl::sycl::item<1> id) { acc[id] = 0; });
		});

		buffer<int, 2> buf2d{{1, 1}};
		q.submit([=](handler& cgh) {
			accessor acc{buf2d, cgh, celerity::access::one_to_one{}, celerity::read_write, celerity::no_init};
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<2>{1, 1}, [=](cl::sycl::item<2> id) { acc[id] = 0; });
		});

		buffer<int, 3> buf3d{{1, 1, 1}};
		q.submit([=](handler& cgh) {
			accessor acc{buf3d, cgh, celerity::access::one_to_one{}, celerity::read_write, celerity::no_init};
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<3>{1, 1, 1}, [=](cl::sycl::item<3> id) { acc[id] = 0; });
		});
	}

} // namespace detail
} // namespace celerity
