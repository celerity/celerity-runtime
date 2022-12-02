#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <sycl/sycl.hpp>

#include "buffer_storage.h"
#include "ndvbuffer.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

namespace {

CUdevice get_cuda_drv_device(const sycl::device& d) {
	// TODO: It's not entirely clear whether this returns a CUDA runtime device or driver API device
	const auto rt_dev = sycl::get_native<sycl::backend::cuda>(d);
	CUdevice drv_dev;
	cuDeviceGet(&drv_dev, rt_dev);
	return drv_dev;
}

template <typename T, int Dims>
void write_global_linear_ids(sycl::queue& q, ndv::accessor<T, Dims> acc) {
	q.submit([=](sycl::handler& cgh) {
		const auto b = acc.get_box();
		const auto buf_extent = acc.get_buffer_extent();
		const sycl::range<Dims> e = b.get_extent();
		cgh.parallel_for(e, [=](sycl::item<Dims> itm) {
			const auto offset_id = ndv::point<Dims>::make_from(b.min() + itm.get_id());
			acc[offset_id] = ndv::get_linear_id(buf_extent, offset_id);
		});
	});
	q.wait();
}

// FIXME: Buffer is only needed for getting context (see copy issue below)
template <typename T, int Dims>
void verify_global_linear_ids(
    sycl::queue& q, ndv::buffer<T, Dims>& buf, ndv::accessor<T, Dims> acc, const std::optional<ndv::box<Dims>>& verify_region = std::nullopt) {
	const auto buf_extent = acc.get_buffer_extent();
	const auto box = acc.get_box();
	const auto acc_extent = box.get_extent();
	std::vector<T> host_buf(acc_extent.size());

	buf.copy_to(host_buf.data(), acc_extent, box, {{}, acc_extent});

	const auto acc_r3 = range_cast<3>(acc_extent);
	for(size_t k = 0; k < acc_r3[0]; ++k) {
		for(size_t j = 0; j < acc_r3[1]; ++j) {
			for(size_t i = 0; i < acc_r3[2]; ++i) {
				const auto offset_id = ndv::point<Dims>::make_from(box.min() + id_cast<Dims>(id<3>{k, j, i}));
				const size_t expected = ndv::get_linear_id(buf_extent, offset_id);
				if(!verify_region.has_value() || verify_region->contains(ndv::point<Dims>{k, j, i})) {
					REQUIRE_LOOP(static_cast<size_t>(host_buf[(k * acc_r3[1] * acc_r3[2]) + (j * acc_r3[2]) + i]) == expected);
				}
			}
		}
	}
}

} // namespace

// TODO:
// - [x] Turn existing checks into test cases
// - [x] Move to exclusive upper bound
// - [x] Add support for 1D/3D buffers
// - [ ] Device selection?
// - [x] D2D copies? Do they work? Is the address space virtualized across all devices?!

namespace {

template <size_t Size>
struct type_of_size {
	type_of_size() {
		static_assert(sizeof(type_of_size<Size>) == Size);
		*this = size_t(0);
	}
	type_of_size(size_t v) { *this = v; }
	type_of_size& operator=(size_t v) {
		*reinterpret_cast<size_t*>(data) = v;
		return *this;
	}
	operator size_t&() { return *reinterpret_cast<size_t*>(data); }
	unsigned char data[Size];
};

// TODO: This assumes a fixed page size ("allocation granularity") of 2 MiB. Need to make more generic.
constexpr size_t page_size = 2 * 1024 * 1024;
using one_page_t = type_of_size<page_size>;

// Since pages are quite large, using basic types such as size_t does not exercise all branches
// in our allocation logic (unless we use very large extents, which makes it harder to debug things).
using quarter_page_t = type_of_size<page_size / 4>;

} // namespace

TEMPLATE_TEST_CASE_SIG("basic full access", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};
	const ndv::extent<Dims> ext{6, 7, 8};
	ndv::buffer<quarter_page_t, Dims> buf{get_cuda_drv_device(q.get_device()), ext};
	auto acc = buf.access({{}, buf.get_extent()});
	write_global_linear_ids(q, acc);
	verify_global_linear_ids(q, buf, acc);
}

TEMPLATE_TEST_CASE_SIG("access with offset", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};
	const ndv::extent<Dims> ext{6, 7, 8};
	const ndv::point<Dims> offset{1, 0, 3};
	ndv::buffer<quarter_page_t, Dims> buf{get_cuda_drv_device(q.get_device()), ext};
	auto acc = buf.access({offset, ext - ndv::extent<Dims>{1, 1, 1}});
	write_global_linear_ids(q, acc);
	verify_global_linear_ids(q, buf, acc);
}

TEST_CASE("buffer can be moved", "[ndvbuffer]") {
	sycl::queue q{sycl::gpu_selector_v};
	const ndv::extent<1> ext{32};
	ndv::buffer<quarter_page_t, 1> buf1{get_cuda_drv_device(q.get_device()), ext};
	write_global_linear_ids(q, buf1.access({{}, 16}));

	const auto extent = buf1.get_extent();
	const auto alloc_size = buf1.get_allocated_size();
	const auto granularity = buf1.get_allocation_granularity();
	const auto ctx = buf1.get_ctx();
	const auto ptr = buf1.get_pointer();

	ndv::buffer<quarter_page_t, 1> buf2{std::move(buf1)};
	CHECK(buf2.get_extent() == extent);
	CHECK(buf1.get_extent() == ndv::extent<1>{});
	CHECK(buf2.get_allocated_size() == alloc_size);
	CHECK(buf1.get_allocated_size() == 0);
	CHECK(buf2.get_allocation_granularity() == granularity);
	CHECK(buf1.get_allocation_granularity() == 0);
	CHECK(buf2.get_ctx() == ctx);
	CHECK(buf1.get_ctx() == 0);
	CHECK(buf2.get_pointer() == ptr);
	CHECK(buf1.get_pointer() == 0);

	write_global_linear_ids(q, buf2.access({16, 32}));
	verify_global_linear_ids(q, buf2, buf2.access({{}, ext}));
}

TEST_CASE("virtual buffer extent can exceed physical device memory") {
	// NOCOMMIT TODO: Here and others: Need NVIDIA GPU! (Btw, how do we select devices in other test suites..?)
	sycl::queue q{sycl::gpu_selector_v};
	const auto mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
	ndv::buffer<char, 1> buf{get_cuda_drv_device(q.get_device()), {mem_size * 2}};
	auto acc = buf.access({{10}, {20}});
	write_global_linear_ids(q, acc);
	verify_global_linear_ids(q, buf, acc);
}

TEMPLATE_TEST_CASE_SIG("buffers may contain types whose size does not evenly divide page size", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};

	// FIXME: Hardcoded for 2 MiB page size.
	using my_type = type_of_size<48>;

	const size_t buf_size = 2 * (page_size / sizeof(my_type)); // Ensure we'll need two pages to cover the allocation
	ndv::buffer<my_type, 1> buf1(get_cuda_drv_device(q.get_device()), {buf_size});
	REQUIRE(buf1.get_allocation_granularity() % sizeof(my_type) != 0);
	const ndv::box<1> copy_box{{buf_size / 2 - 100}, {buf_size / 2 + 100}};
	write_global_linear_ids(q, buf1.access(copy_box));

	ndv::buffer<my_type, 1> buf2(get_cuda_drv_device(q.get_device()), {buf_size});
	buf2.copy_from(buf1, copy_box, copy_box);
	verify_global_linear_ids(q, buf2, buf1.access(copy_box));
}

TEST_CASE("physical regions are allocated lazily upon access (1D)") {
	sycl::queue q{sycl::gpu_selector_v};
	ndv::buffer<one_page_t, 1> buf(get_cuda_drv_device(q.get_device()), {256});
	REQUIRE(buf.get_allocation_granularity() == sizeof(one_page_t));

	auto acc1 = buf.access({{10}, {15}});
	write_global_linear_ids(q, acc1);
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 5);

	buf.access({{8}, {15}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 7);

	buf.access({{99}, {100}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 8);

	buf.access({{0}, {2}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 10);
}

TEST_CASE("physical regions are allocated lazily upon access (2D)") {
	sycl::queue q{sycl::gpu_selector_v};
	ndv::buffer<one_page_t, 2> buf(get_cuda_drv_device(q.get_device()), {8, 8});
	REQUIRE(buf.get_allocation_granularity() == sizeof(one_page_t));

	auto acc1 = buf.access({{1, 1}, {3, 3}});
	write_global_linear_ids(q, acc1);
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 4);
	verify_global_linear_ids(q, buf, acc1);

	auto acc2 = buf.access({{0, 0}, {3, 3}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 9);
	verify_global_linear_ids(q, buf, acc2, std::optional{ndv::box<2>{{1, 1}, {3, 3}}});

	auto acc3 = buf.access({{0, 0}, {4, 4}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 16);
	verify_global_linear_ids(q, buf, acc3, std::optional{ndv::box<2>{{1, 1}, {3, 3}}});

	// Create disjoint allocation at various locations relative to existing one ("to the side", "below", ...).
	// TODO: Probably not exhaustive
	SECTION("pattern 1") {
		buf.access({{4, 4}, {6, 6}});
		CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 20);
	}

	SECTION("pattern 2") {
		buf.access({{0, 4}, {2, 6}});
		CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 20);
	}

	SECTION("pattern 3") {
		buf.access({{1, 4}, {3, 6}});
		CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 20);
	}
}

TEST_CASE("physical regions are allocated lazily upon access (3D)") {
	sycl::queue q{sycl::gpu_selector_v};
	ndv::buffer<one_page_t, 3> buf(get_cuda_drv_device(q.get_device()), {8, 8, 8});
	REQUIRE(buf.get_allocation_granularity() == sizeof(one_page_t));

	auto acc1 = buf.access({{4, 4, 4}, {6, 6, 6}});
	write_global_linear_ids(q, acc1);
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 8);

	buf.access({{0, 0, 0}, {1, 1, 1}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 9);

	buf.access({{7, 7, 7}, {8, 8, 8}});
	CHECK(buf.get_allocated_size() / sizeof(one_page_t) == 10);

	// TODO: Test more access patterns
}

// TOOD: For some reason the pointers returned by cumMemAddressReserve do not seem to be "universal", i.e., CUDA cannot infer the corresponding context / device
// from them. This is annoying because it means we cannot use cudaMemcpy with cudaMemcpyDefault. Instead we need to do an explicit cuMemcpyPeer...
// It also means that we need to call cuCtxSetCurrent before doing a D2H copy.
TEST_CASE("WIP: why are returned pointers not 'UVA pointers' ?!") {
	const auto print_ptr_attrs = [](CUdeviceptr ptr) {
		CUcontext ctx;
		CHECK_DRV(cuPointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, ptr));
		CUmemorytype memtype;
		CHECK_DRV(cuPointerGetAttribute(&memtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr));
		CUdeviceptr devptr;
		CHECK_DRV(cuPointerGetAttribute(&devptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr));
		// void* hostptr;
		// CHECK_DRV(cuPointerGetAttribute(&hostptr, CU_POINTER_ATTRIBUTE_HOST_POINTER, ptr));
		bool is_managed;
		CHECK_DRV(cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, ptr));
		int device_ordinal;
		CHECK_DRV(cuPointerGetAttribute(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr));
		bool is_mapped;
		CHECK_DRV(cuPointerGetAttribute(&is_mapped, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr));

		fmt::print("Attributes of pointer {}:\n", ptr);
		fmt::print("\tcontext = {}\n", (void*)ctx);
		fmt::print("\tmemtype = {}\n", memtype);
		fmt::print("\tdevptr = {}\n", devptr);
		// fmt::print("\thostptr = {}\n", hostptr);
		fmt::print("\tis_managed = {}\n", is_managed);
		fmt::print("\tdevice = {}\n", device_ordinal);
		fmt::print("\tis_mapped = {}\n", is_mapped);
	};

	auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	REQUIRE(devices.size() >= 2);
	sycl::queue q1{devices[0]};
	sycl::queue q2{devices[1]};

	// NOCOMMIT Don't hardcode ranges
	ndv::buffer<size_t, 2> buf1{get_cuda_drv_device(devices[0]), {8, 8}};
	auto acc1 = buf1.access({{0, 0}, {8, 8}});
	write_global_linear_ids(q1, acc1);

	ndv::buffer<size_t, 2> buf2{get_cuda_drv_device(devices[1]), {8, 8}};
	auto acc2 = buf2.access({{0, 0}, {8, 8}});

	cuCtxSetCurrent(buf1.get_ctx());
	print_ptr_attrs((CUdeviceptr)acc1.get_pointer());
	cuCtxSetCurrent(buf2.get_ctx());
	print_ptr_attrs((CUdeviceptr)acc2.get_pointer());

	// cudaSetDevice(0);
	// void* ptr123;
	// cudaMalloc(&ptr123, 1024 * 1024);
	// print_ptr_attrs((CUdeviceptr)ptr123);

	// cudaMemcpy(acc1.get_pointer(), acc2.get_pointer(), 100 * 1024, cudaMemcpyDefault);
	CHECK_DRV(cuMemcpyPeer((CUdeviceptr)acc2.get_pointer(), buf2.get_ctx(), (CUdeviceptr)acc1.get_pointer(), buf1.get_ctx(), 8 * 8 * sizeof(size_t)));

	verify_global_linear_ids(q2, buf2, acc2);
}

// Smoke test: CUDA for some (of course undocumented!) reason seems to check whether all pages in the copied region are mapped.
// We work around this by mapping all unallocated address ranges to a "zero page" using virtual aliasing.
// See also https://forums.developer.nvidia.com/t/strange-behavior-of-2d-3d-copies-in-partially-mapped-virtual-address-space/235533/2
TEMPLATE_TEST_CASE_SIG("copy works when stride skips over non-allocated pages", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};

	const ndv::extent<Dims> ext{9, 4, 7};
	const ndv::box<Dims> copy_box{{3, 1, 2}, {5, 2, 3}};

	SECTION("copy from other buffer") {
		ndv::buffer<one_page_t, Dims> buf1{get_cuda_drv_device(q.get_device()), ext};
		buf1.access(copy_box);
		const ndv::extent<Dims> offset{1, 1, 1};
		ndv::buffer<one_page_t, Dims> buf2{get_cuda_drv_device(q.get_device()), ext + offset};
		buf2.copy_from(buf1, copy_box, ndv::box<Dims>{copy_box.min() + offset, copy_box.max() + offset});
	}

	SECTION("copy from host buffer") {
		const std::vector<one_page_t> host_buf(ext.size());
		ndv::buffer<one_page_t, Dims> buf{get_cuda_drv_device(q.get_device()), ext};
		buf.copy_from(host_buf.data(), ext, copy_box, copy_box);
	}

	SECTION("copy to host buffer") {
		ndv::buffer<one_page_t, Dims> buf{get_cuda_drv_device(q.get_device()), ext};
		buf.access(copy_box);
		std::vector<one_page_t> host_buf(ext.size());
		buf.copy_to(host_buf.data(), ext, copy_box, copy_box);
	}
}

TEMPLATE_TEST_CASE_SIG("copy parts between buffers on different devices", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	REQUIRE(devices.size() >= 2);
	sycl::queue q1{devices[0]};
	sycl::queue q2{devices[1]};

	// TODO: Also copy between differently sized buffers (needs custom verification though)
	const ndv::extent<Dims> ext{8, 7, 6};
	ndv::buffer<quarter_page_t, Dims> buf1{get_cuda_drv_device(devices[0]), ext};
	auto acc1 = buf1.access({{}, ext});
	write_global_linear_ids(q1, acc1);

	ndv::buffer<quarter_page_t, Dims> buf2{get_cuda_drv_device(devices[1]), ext};

	const ndv::box<Dims> copy_box{{2, 1, 3}, {7, 6, 4}};
	// TODO: Also copy between different locations (needs custom verification though)
	buf2.copy_from(buf1, copy_box, copy_box);

	auto acc2 = buf2.access({{}, ext});
	// TODO: Negative testing - check that we don't copy anything other than copy_box
	verify_global_linear_ids(q2, buf2, acc2, std::optional{copy_box});
}

// TODO: Also copy between different locations (needs custom verification though)
TEMPLATE_TEST_CASE_SIG("copy from/to host memory", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};

	// TODO: Also copy between differently sized buffers (needs custom verification though)
	const ndv::extent<Dims> ext{8, 7, 6};
	std::vector<quarter_page_t> host_buf(ext.size());

	const auto host_for_each = [&ext](const auto& cb) {
		for(size_t k = 0; k < ext[0]; ++k) {
			for(size_t j = 0; j < (Dims > 1 ? ext[1] : 1); ++j) {
				for(size_t i = 0; i < (Dims == 3 ? ext[2] : 1); ++i) {
					cb(ndv::point<Dims>{k, j, i});
				}
			}
		}
	};

	const ndv::box<Dims> copy_box{{2, 1, 3}, {7, 6, 4}};
	ndv::buffer<quarter_page_t, Dims> buf{get_cuda_drv_device(q.get_device()), ext};

	SECTION("copy from host") {
		host_for_each([&](const ndv::point<Dims>& pt) {
			const auto linear_id = ndv::get_linear_id(ext, pt);
			host_buf[linear_id] = linear_id;
		});
		buf.copy_from(host_buf.data(), ext, copy_box, copy_box);
		auto acc = buf.access({{}, ext});
		// TODO: Negative testing - check that we don't copy anything other than copy_box
		verify_global_linear_ids(q, buf, acc, std::optional{copy_box});
	}

	SECTION("copy to host") {
		write_global_linear_ids(q, buf.access({{}, ext}));
		buf.copy_to(host_buf.data(), ext, copy_box, copy_box);
		host_for_each([&](const ndv::point<Dims>& pt) {
			const auto linear_id = ndv::get_linear_id(ext, pt);
			if(copy_box.contains(pt)) { REQUIRE_LOOP(static_cast<size_t>(host_buf[linear_id]) == linear_id); }
		});
	}
}

// Had to roll my own benchmark utility b/c Catch2 doesn't allow control over the number of benchmark runs,
// which leads us to run out of GPU memory. See also https://github.com/catchorg/Catch2/issues/2150
template <typename SetupCb, typename RunCb, typename TeardownCb>
void run_benchmark(const std::string& name, const size_t iterations, SetupCb setup, RunCb run, TeardownCb teardown) {
	const auto format_duration = [](const double& us) {
		std::vector<std::string> units = {"us", "ms", "s"};
		size_t unit_i = 0;
		double v = us;
		while(v > 1000 && unit_i < units.size()) {
			v /= 1000;
			unit_i++;
		}
		return fmt::format("{:.2f}{}", v, units[unit_i]);
	};

	const size_t warmups = std::max(size_t(1), iterations / 10);
	setup(warmups);
	for(size_t i = 0; i < warmups; ++i) {
		run(i);
	}
	teardown(warmups);

	setup(iterations);
	std::vector<std::chrono::microseconds> times(iterations);
	for(size_t i = 0; i < iterations; ++i) {
		const auto before = std::chrono::steady_clock::now();
		run(i);
		const auto after = std::chrono::steady_clock::now();
		times[i] = std::chrono::duration_cast<std::chrono::microseconds>(after - before);
	}
	teardown(iterations);

	std::sort(times.begin(), times.end());
	const auto sum = std::accumulate(times.begin(), times.end(), std::chrono::microseconds{0});
	const double avg = sum.count() / double(iterations);
	const auto min = times.front().count();
	const auto max = times.back().count();
	const auto median = iterations % 2 == 1 ? double(times[iterations / 2].count()) : (times[iterations / 2].count() + times[iterations / 2 + 1].count()) / 2.0;
	fmt::print("{}\n\tavg={}, median={}, min={}, max={}\n\n", name, format_duration(avg), format_duration(median), format_duration(min), format_duration(max));
}

// NOTE: We are currently measuring performance for working with single pages (e.g. 2 MiB).
//       Allocating larger blocks would be a lot more efficient in practice.
TEST_CASE("CUDA baseline performance") {
	sycl::queue q{sycl::gpu_selector_v};
	const auto device_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
	CUcontext ctx;
	CHECK_DRV(cuInit(0));
	CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, get_cuda_drv_device(q.get_device())));
	ndv::activate_cuda_context act{ctx};

	CUmemAllocationProp props = {};
	props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	props.location.id = get_cuda_drv_device(q.get_device());
	size_t granularity = 0;
	CHECK_DRV(cuMemGetAllocationGranularity(&granularity, &props, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

	{
		const size_t iterations = 100;
		REQUIRE(device_mem_size >= granularity * iterations);
		std::vector<CUmemGenericAllocationHandle> alloc_handles;
		run_benchmark(
		    "allocating one page", iterations,
		    [&](const size_t n) {
			    alloc_handles.clear();
			    alloc_handles.resize(n);
		    },
		    [&](const size_t i) { CHECK_DRV(cuMemCreate(&alloc_handles[i], granularity, &props, 0)); },
		    [&](const size_t n) {
			    for(size_t i = 0; i < n; ++i) {
				    CHECK_DRV(cuMemRelease(alloc_handles[i]));
			    }
		    });
	}

	// FIXME: Results fluctuate wildly (100-800us), but hand-rolled benchmark is seemingly always faster (e.g. 193us vs 135us) - what is going on?
	// (NOTE: This is actually broken in that the inner callback runs multiple times, thus leaking memory!)
	// BENCHMARK_ADVANCED("allocating one page")(Catch::Benchmark::Chronometer meter) {
	// 	CUmemGenericAllocationHandle alloc_handle;
	// 	meter.measure([&] { CHECK_DRV(cuMemCreate(&alloc_handle, granularity, &props, 0)); });
	// 	cuMemRelease(alloc_handle);
	// };

	{
		const size_t iterations = 50;
		const size_t alloc_size = 100 * 1024 * 1024;
		const size_t num_pages = alloc_size / granularity;
		REQUIRE(device_mem_size >= alloc_size * iterations);
		std::vector<CUmemGenericAllocationHandle> alloc_handles;
		run_benchmark(
		    "allocating 100 MiB worth of pages", iterations,
		    [&](const size_t n) {
			    alloc_handles.clear();
			    alloc_handles.resize(n * num_pages);
		    },
		    [&](const size_t i) {
			    for(size_t j = 0; j < num_pages; ++j) {
				    CHECK_DRV(cuMemCreate(&alloc_handles[i * num_pages + j], granularity, &props, 0));
			    }
		    },
		    [&](const size_t n) {
			    for(size_t i = 0; i < n * num_pages; ++i) {
				    CHECK_DRV(cuMemRelease(alloc_handles[i]));
			    }
		    });
	}

	const size_t virtual_size = 100ull * 1024 * 1024 * 1024;
	CUdeviceptr base_ptr = 0;
	CHECK_DRV(cuMemAddressReserve(&base_ptr, virtual_size, 0, 0, 0));

	{
		const size_t iterations = 100;
		REQUIRE(device_mem_size >= iterations * granularity);
		std::vector<CUmemGenericAllocationHandle> alloc_handles(iterations, {});
		for(size_t i = 0; i < iterations; ++i) {
			CHECK_DRV(cuMemCreate(&alloc_handles[i], granularity, &props, 0));
		}
		run_benchmark(
		    "mapping one page", iterations, [&](const size_t n) {},
		    [&](const size_t i) { CHECK_DRV(cuMemMap(base_ptr + i * granularity, granularity, 0, alloc_handles[i], 0)); },
		    [&](const size_t n) {
			    for(size_t i = 0; i < n; ++i) {
				    CHECK_DRV(cuMemUnmap(base_ptr + i * granularity, granularity));
			    }
		    });
		for(size_t i = 0; i < iterations; ++i) {
			CHECK_DRV(cuMemRelease(alloc_handles[i]));
		}
	}

	{
		const size_t iterations = 50;
		const size_t alloc_size = 100 * 1024 * 1024;
		const size_t num_pages = alloc_size / granularity;
		REQUIRE(device_mem_size >= alloc_size * iterations);
		std::vector<CUmemGenericAllocationHandle> alloc_handles(num_pages * iterations, {});
		for(size_t i = 0; i < num_pages * iterations; ++i) {
			CHECK_DRV(cuMemCreate(&alloc_handles[i], granularity, &props, 0));
		}
		run_benchmark(
		    "mapping 100 MiB worth of pages", iterations, [&](const size_t n) {},
		    [&](const size_t i) {
			    for(size_t j = 0; j < num_pages; ++j) {
				    CHECK_DRV(cuMemMap(base_ptr + (i * num_pages + j) * granularity, granularity, 0, alloc_handles[i], 0));
			    }
		    },
		    [&](const size_t n) {
			    for(size_t i = 0; i < n * num_pages; ++i) {
				    CHECK_DRV(cuMemUnmap(base_ptr + i * granularity, granularity));
			    }
		    });
		for(size_t i = 0; i < num_pages * iterations; ++i) {
			CHECK_DRV(cuMemRelease(alloc_handles[i]));
		}
	}

	for(bool in_bulk : {false, true}) {
		const size_t iterations = 50;
		const size_t alloc_size = 100 * 1024 * 1024;
		const size_t num_pages = alloc_size / granularity;
		REQUIRE(device_mem_size >= alloc_size * iterations);
		std::vector<CUmemGenericAllocationHandle> alloc_handles(num_pages * iterations, {});
		for(size_t i = 0; i < num_pages * iterations; ++i) {
			CHECK_DRV(cuMemCreate(&alloc_handles[i], granularity, &props, 0));
			CHECK_DRV(cuMemMap(base_ptr + i * granularity, granularity, 0, alloc_handles[i], 0));
		}
		CUmemAccessDesc desc;
		desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		desc.location.id = get_cuda_drv_device(q.get_device());
		desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		const auto title = fmt::format("setting access flags on 100 MiB worth of pages ({})", in_bulk ? "in bulk" : "one by one");
		run_benchmark(
		    title, iterations, [&](const size_t n) {},
		    [&](const size_t i) {
			    if(in_bulk) {
				    CHECK_DRV(cuMemSetAccess(base_ptr + i * num_pages * granularity, num_pages * granularity, &desc, 1));
			    } else {
				    for(size_t j = 0; j < num_pages; ++j) {
					    CHECK_DRV(cuMemSetAccess(base_ptr + (i * num_pages + j) * granularity, granularity, &desc, 1));
				    }
			    }
		    },
		    [&](const size_t n) {});
		for(size_t i = 0; i < num_pages * iterations; ++i) {
			CHECK_DRV(cuMemUnmap(base_ptr + i * granularity, granularity));
			CHECK_DRV(cuMemRelease(alloc_handles[i]));
		}
	}

	{
		const size_t iterations = 50;
		const size_t alloc_size = 100 * 1024 * 1024;
		const size_t num_pages = alloc_size / granularity;
		REQUIRE(device_mem_size >= alloc_size * iterations);
		std::vector<CUmemGenericAllocationHandle> alloc_handles(num_pages * iterations, {});
		for(size_t i = 0; i < num_pages * iterations; ++i) {
			CHECK_DRV(cuMemCreate(&alloc_handles[i], granularity, &props, 0));
		}
		CUmemAccessDesc desc;
		desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		desc.location.id = get_cuda_drv_device(q.get_device());
		desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		run_benchmark(
		    "unmapping 100 MiB worth of pages (with set access flags)", iterations,
		    [&](const size_t n) {
			    for(size_t i = 0; i < n * num_pages; ++i) {
				    CHECK_DRV(cuMemMap(base_ptr + i * granularity, granularity, 0, alloc_handles[i], 0));
				    CHECK_DRV(cuMemSetAccess(base_ptr + i * granularity, granularity, &desc, 1));
			    }
		    },
		    [&](const size_t i) {
			    for(size_t j = 0; j < num_pages; ++j) {
				    CHECK_DRV(cuMemUnmap(base_ptr + (i * num_pages + j) * granularity, granularity));
			    }
		    },
		    [&](const size_t n) {});
		for(size_t i = 0; i < num_pages * iterations; ++i) {
			CHECK_DRV(cuMemRelease(alloc_handles[i]));
		}
	}

	CHECK_DRV(cuMemAddressFree(base_ptr, virtual_size));
}

TEST_CASE("buffer performance") {
	sycl::queue q{sycl::gpu_selector_v};
	const auto device_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();

	{
		const size_t iterations = 50;
		const size_t alloc_size = 100 * 1024 * 1024;
		const size_t num_pages = alloc_size / page_size;
		REQUIRE(device_mem_size >= iterations * alloc_size);

		std::vector<ndv::buffer<one_page_t, 1>> buffers;
		run_benchmark(
		    "allocating a 100 MiB buffer", iterations,
		    [&](const size_t n) {
			    buffers.clear();
			    for(size_t i = 0; i < n; ++i) {
				    buffers.emplace_back(get_cuda_drv_device(q.get_device()), ndv::extent<1>{num_pages});
			    }
		    },
		    [&](const size_t i) {
			    buffers[i].access({{}, {num_pages}});
		    },
		    [&](const size_t n) {});
	}

	{
		const size_t iterations = 50;
		const size_t alloc_size = 100 * 1024 * 1024;
		const size_t num_pages = alloc_size / page_size;
		REQUIRE(device_mem_size >= iterations * alloc_size);
		ndv::buffer<one_page_t, 1> buf{get_cuda_drv_device(q.get_device()), ndv::extent<1>{num_pages}};
		buf.access({{}, {num_pages}});
		run_benchmark(
		    "accessing 100 MiB of already allocated memory", iterations, [&](const size_t n) {},
		    [&](const size_t i) {
			    buf.access({{}, {num_pages}});
		    },
		    [&](const size_t n) {});
	}
}
