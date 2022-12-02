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
