#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

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

template <template <int> class OutType, int DimsOut, template <int> class InType, int DimsIn>
OutType<DimsOut> coordinate_cast(const InType<DimsIn>& other) {
	OutType<DimsOut> result;
	for(int o = 0; o < DimsOut; ++o) {
		result[o] = o < DimsIn ? other[o] : 0;
	}
	return result;
}

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
		const auto buffer_range = acc.get_buffer_range();
		const sycl::range<Dims> r = b.get_range();
		cgh.parallel_for(r, [=](sycl::item<Dims> itm) {
			const auto offset_id = coordinate_cast<ndv::point, Dims>(b.min() + itm.get_id());
			acc[offset_id] = ndv::get_linear_id(buffer_range, offset_id);
		});
	});
	q.wait();
}

// FIXME: Buffer is only needed for getting context (see copy issue below)
template <typename T, int Dims>
void verify_global_linear_ids(
    sycl::queue& q, ndv::buffer<T, Dims>& buf, ndv::accessor<T, Dims> acc, const std::optional<ndv::box<Dims>>& nonzero_region = std::nullopt) {
	const auto buf_range = acc.get_buffer_range();
	const auto box = acc.get_box();
	range<Dims> acc_range = box.get_range();
	std::vector<T> host_buf(acc_range.size());
	cuCtxSetCurrent(buf.get_ctx());

	memcpy_strided_device(q, acc.get_pointer(), host_buf.data(), sizeof(T), buf_range, box.min(), acc_range, sycl::id<Dims>{}, acc_range);

	const auto acc_r3 = range_cast<3>(acc_range);
	for(size_t k = 0; k < acc_r3[0]; ++k) {
		for(size_t j = 0; j < acc_r3[1]; ++j) {
			for(size_t i = 0; i < acc_r3[2]; ++i) {
				const auto offset_id = coordinate_cast<ndv::point, Dims>(box.min() + id_cast<Dims>(id<3>{k, j, i}));
				size_t expected = ndv::get_linear_id(buf_range, offset_id);
				if(nonzero_region.has_value() && !nonzero_region->contains(ndv::point<Dims>{k, j, i})) { expected = 0; }
				REQUIRE_LOOP(static_cast<size_t>(host_buf[(k * acc_r3[1] * acc_r3[2]) + (j * acc_r3[2]) + i]) == expected);
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
// - [ ] D2D copies? Do they work? Is the address space virtualized across all devices?!

TEMPLATE_TEST_CASE_SIG("basic full access", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};
	const ndv::extent<Dims> r{6, 7, 8};
	ndv::buffer<size_t, Dims> buf{get_cuda_drv_device(q.get_device()), r};
	auto acc = buf.access({{}, buf.get_range()});
	write_global_linear_ids(q, acc);
	verify_global_linear_ids(q, buf, acc);
}

TEMPLATE_TEST_CASE_SIG("access with offset", "[ndvbuffer]", ((int Dims), Dims), 1, 2, 3) {
	sycl::queue q{sycl::gpu_selector_v};
	const ndv::extent<Dims> r{6, 7, 8};
	const ndv::point<Dims> o{1, 2, 3};
	ndv::buffer<size_t, Dims> buf{get_cuda_drv_device(q.get_device()), r};
	auto acc = buf.access({o, r - ndv::extent<Dims>{1, 1, 1}});
	write_global_linear_ids(q, acc);
	verify_global_linear_ids(q, buf, acc);
}

TEST_CASE("virtual buffer range can exceed physical device memory") {
	// NOCOMMIT TODO: Here and others: Need NVIDIA GPU! (Btw, how do we select devices in other tests suites..?)
	sycl::queue q{sycl::gpu_selector_v};
	const auto mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
	ndv::buffer<char, 1> buf{get_cuda_drv_device(q.get_device()), {mem_size * 2}};
	auto acc = buf.access({{10}, {20}});
	write_global_linear_ids(q, acc);
	verify_global_linear_ids(q, buf, acc);
}

namespace {
// TODO: This assumes a fixed granularity (2 MiB). Need to make more generic.
struct very_large_type {
	very_large_type() { *this = size_t(0); }
	very_large_type(size_t v) { *this = v; }
	very_large_type& operator=(size_t v) {
		*reinterpret_cast<size_t*>(data) = v;
		return *this;
	}
	operator size_t&() { return *reinterpret_cast<size_t*>(data); }

	unsigned char data[2 * 1024 * 1024];
};
} // namespace

TEST_CASE("physical regions are allocated lazily upon access (1D)") {
	sycl::queue q{sycl::gpu_selector_v};
	ndv::buffer<very_large_type, 1> buf(get_cuda_drv_device(q.get_device()), {256});
	REQUIRE(buf.get_allocation_granularity() == sizeof(very_large_type));

	auto acc1 = buf.access({{10}, {15}});
	write_global_linear_ids(q, acc1);
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 5);

	buf.access({{8}, {15}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 7);

	buf.access({{99}, {100}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 8);

	buf.access({{0}, {2}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 10);
}

// FIXME: For some reason cudaMemcpy2D fails for very_large_type, so we cannot verify the global ids. It works when changing the size to 1 MiB...
TEST_CASE("physical regions are allocated lazily upon access (2D)") {
	sycl::queue q{sycl::gpu_selector_v};
	ndv::buffer<very_large_type, 2> buf(get_cuda_drv_device(q.get_device()), {8, 8});
	REQUIRE(buf.get_allocation_granularity() == sizeof(very_large_type));

	auto acc1 = buf.access({{1, 1}, {3, 3}});
	write_global_linear_ids(q, acc1);
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 4);
	// verify_global_linear_ids(q, buf, acc1);

	buf.access({{0, 0}, {3, 3}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 9);
	// verify_global_linear_ids(q, buf, acc2, ndv::extent{{1, 1}, {2, 2}});

	buf.access({{0, 0}, {4, 4}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 16);
	// verify_global_linear_ids(q, buf, acc3, ndv::extent{{1, 1}, {2, 2}});

	// Create disjoint allocation at various locations relative to existing one ("to the side", "below", ...).
	// TODO: Probably not exhaustive
	SECTION("pattern 1") {
		buf.access({{4, 4}, {6, 6}});
		CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 20);
	}

	SECTION("pattern 2") {
		buf.access({{0, 4}, {2, 6}});
		CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 20);
	}

	SECTION("pattern 3") {
		buf.access({{1, 4}, {3, 6}});
		CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 20);
	}
}

TEST_CASE("physical regions are allocated lazily upon access (3D)") {
	sycl::queue q{sycl::gpu_selector_v};
	ndv::buffer<very_large_type, 3> buf(get_cuda_drv_device(q.get_device()), {8, 8, 8});
	REQUIRE(buf.get_allocation_granularity() == sizeof(very_large_type));

	auto acc1 = buf.access({{4, 4, 4}, {6, 6, 6}});
	write_global_linear_ids(q, acc1);
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 8);

	buf.access({{0, 0, 0}, {1, 1, 1}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 9);

	buf.access({{7, 7, 7}, {8, 8, 8}});
	CHECK(buf.get_allocated_size() / sizeof(very_large_type) == 10);

	// TODO: Test more access patterns
}

// TOOD: For some reason the pointers returned by cumMemAddressReserve do not seem to be "universal", i.e., CUDA cannot infer the corresponding context / device
// from them. This is annoying because it means we cannot use cudaMemcpy with cudaMemcpyDefault. Instead we need to do an explicit cuMemcpyPeer...
// It also means that we need to call cuCtxSetCurrent before doing a D2H copy.
TEST_CASE("device to device copies") {
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
