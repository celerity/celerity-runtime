#include "test_utils.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

namespace celerity::detail {

template <int Dims>
class dim_device_queue_fixture : public test_utils::device_queue_fixture {};

template <int Dims>
class accessor_range_test_kernel;

TEMPLATE_TEST_CASE_METHOD_SIG(dim_device_queue_fixture, "ranged_sycl_access works around differences in offset computation for device accessors",
    "[sycl][accessor]", ((int Dims), Dims), 1, 2, 3) {
	constexpr static size_t tile_width = 2;
	constexpr static size_t num_tiles[] = {2, 3, 4};

	const auto tile_range = range_cast<Dims>(sycl::range<3>{tile_width, tile_width, tile_width});
	const auto buffer_range = range_cast<Dims>(sycl::range<3>{2 * tile_width, 3 * tile_width, 4 * tile_width});

	// GENERATE macros cannot be executed in a loop
	sycl::id<Dims> offset;
	if constexpr(Dims >= 1) { offset[0] = GENERATE(Catch::Generators::range(size_t{0}, num_tiles[0])) * tile_width; }
	if constexpr(Dims >= 2) { offset[1] = GENERATE(Catch::Generators::range(size_t{0}, num_tiles[1])) * tile_width; }
	if constexpr(Dims >= 3) { offset[2] = GENERATE(Catch::Generators::range(size_t{0}, num_tiles[2])) * tile_width; }

	CAPTURE(buffer_range);
	CAPTURE(tile_range);
	CAPTURE(offset);

	auto& q = dim_device_queue_fixture<Dims>::get_device_queue();

	sycl::buffer<int, Dims> buf{buffer_range};
	q.submit([&](sycl::handler& cgh) { cgh.fill(buf.template get_access<sycl::access_mode::discard_write>(cgh), -1); });

	q.submit([&](sycl::handler& cgh) {
		const auto acc = buf.template get_access<sycl::access_mode::read_write>(cgh, tile_range, offset);
		const auto buf_range = buf.get_range();
		cgh.parallel_for<bind_kernel_name<accessor_range_test_kernel<Dims>>>(tile_range, [=](const sycl::id<Dims> rel_index) {
			const auto abs_index = offset + rel_index;
			int value = 0;
			for(int d = 0; d < Dims; ++d) {
				value = 10 * value + static_cast<int>(abs_index[d]);
			}
			ranged_sycl_access(acc, buf_range, rel_index) += 1 + value;
		});
	});

	std::vector<int> expected;
	expected.reserve(buffer_range.size());
	test_utils::for_each_in_range(buffer_range, [&](const id<Dims> index) {
		bool inside_tile = true;
		int value = 0;
		for(int d = 0; d < Dims; ++d) {
			inside_tile &= index[d] >= offset[d] && index[d] < offset[d] + tile_width;
			value = 10 * value + static_cast<int>(index[d]);
		}
		expected.push_back(inside_tile ? value : -1);
	});

	std::vector<int> actual(buf.get_range().size());
	q.submit([&](sycl::handler& cgh) { cgh.copy(buf.template get_access<sycl::access::mode::read>(cgh), actual.data()); }).wait_and_throw();

	CHECK(actual == expected);
}


template <access_mode, bool>
class access_test_kernel;

#if CELERITY_WORKAROUND(DPCPP) || CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // target::gobal_buffer is now target::device, but only for very recent versions of DPC++
constexpr auto sycl_target_device = cl::sycl::access::target::global_buffer;
#pragma GCC diagnostic pop
#else
constexpr auto sycl_target_device = cl::sycl::access::target::device;
#endif

template <access_mode AccessMode, bool UsingPlaceholderAccessor>
static auto make_device_accessor(sycl::buffer<int, 1>& buf, sycl::handler& cgh, const subrange<1>& sr) {
	if constexpr(UsingPlaceholderAccessor) {
		sycl::accessor<int, 1, AccessMode, sycl_target_device, sycl::access::placeholder::true_t> acc{buf, sr.range, sr.offset};
		cgh.require(acc);
		return acc;
	} else {
		return buf.get_access<AccessMode>(cgh, sr.range, sr.offset);
	}
}

template <access_mode AccessMode, bool UsingPlaceholderAccessor>
static void test_access(sycl::queue& q, sycl::buffer<int, 1>& test_buf, const subrange<1>& sr) {
	CAPTURE(AccessMode);
	CAPTURE(UsingPlaceholderAccessor);

	bool verified = false;
	sycl::buffer<bool> verify_buf{&verified, 1};
	q.submit([&](sycl::handler& cgh) {
		 const auto test_acc = make_device_accessor<AccessMode, UsingPlaceholderAccessor>(test_buf, cgh, sr);
		 const auto verify_acc = verify_buf.get_access<access_mode::write>(cgh);
		 cgh.parallel_for<access_test_kernel<AccessMode, UsingPlaceholderAccessor>>(range<1>{1}, [=](sycl::item<1>) { //
			 verify_acc[0] = test_acc.get_range() == sr.range;
		 });
	 }).wait_and_throw();

	sycl::host_accessor verify_acc{verify_buf};
	CHECK(verify_acc[0]);
};

// We artificially allocate unit-sized SYCL buffers for ComputeCpp and DPC++ to work around the lack of zero-sized accessor support. Once this test succeeds,
// we can get rid of the workaround.
TEST_CASE_METHOD(test_utils::device_queue_fixture, "SYCL can access empty buffers natively", "[sycl][accessor][!mayfail]") {
	sycl::buffer<int> buf{zero_range};

	auto& queue = get_device_queue().get_sycl_queue();
	const auto sr = subrange<1>{0, 0};

	SECTION("Using regular accessors") {
		test_access<access_mode::discard_write, false>(queue, buf, sr);
		test_access<access_mode::read_write, false>(queue, buf, sr);
		test_access<access_mode::read, false>(queue, buf, sr);
	}

	SECTION("Using placeholder accessors") {
		test_access<access_mode::discard_write, true>(queue, buf, sr);
		test_access<access_mode::read_write, true>(queue, buf, sr);
		test_access<access_mode::read, true>(queue, buf, sr);
	}
}

TEST_CASE_METHOD(test_utils::device_queue_fixture, "SYCL can access empty buffers through device_buffer_storage", "[sycl][accessor]") {
	detail::device_buffer_storage<int, 1> storage{zero_range, get_device_queue().get_sycl_queue()};
	CHECK(range_cast<1>(storage.get_range()) == zero_range);

	auto& queue = get_device_queue().get_sycl_queue();
	auto& buf = storage.get_device_buffer();
	const auto requested_sr = subrange<1>{buf.get_range()[0], 0}; // offset == backing buffer range just to mess with things
	const auto effective_sr = detail::get_effective_sycl_accessor_subrange({}, requested_sr);

	SECTION("Using regular accessors") {
		test_access<access_mode::discard_write, false>(queue, buf, effective_sr);
		test_access<access_mode::read_write, false>(queue, buf, effective_sr);
		test_access<access_mode::read, false>(queue, buf, effective_sr);
	}

	SECTION("Using placeholder accessors") {
		test_access<access_mode::discard_write, true>(queue, buf, effective_sr);
		test_access<access_mode::read_write, true>(queue, buf, effective_sr);
		test_access<access_mode::read, true>(queue, buf, effective_sr);
	}
}

#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS

// If this test fails, celerity can't reliably support reductions on the user's combination of backend and hardware
TEST_CASE_METHOD(test_utils::device_queue_fixture, "SYCL has working simple scalar reductions", "[sycl][reductions]") {
	const size_t N = GENERATE(64, 512, 1024, 4096);
	CAPTURE(N);

	sycl::buffer<int> buf{1};

	get_device_queue().get_sycl_queue().submit([&](sycl::handler& cgh) {
		cgh.parallel_for(range<1>{N}, sycl::reduction(buf, cgh, sycl::plus<int>{}, sycl::property::reduction::initialize_to_identity{}),
		    [](auto, auto& r) { r.combine(1); });
	});

	sycl::host_accessor acc{buf};
	CHECK(static_cast<size_t>(acc[0]) == N);
}

#endif

TEST_CASE("SYCL implements by-value equality-comparison of device information", "[sycl][device-selection][!mayfail]") {
	constexpr static auto get_devices = [] {
		auto devs = sycl::device::get_devices();
		std::sort(devs.begin(), devs.end(), [](const sycl::device& lhs, const sycl::device& rhs) {
			const auto lhs_vendor_id = lhs.get_info<sycl::info::device::vendor_id>(), rhs_vendor_id = rhs.get_info<sycl::info::device::vendor_id>();
			const auto lhs_name = lhs.get_info<sycl::info::device::name>(), rhs_name = rhs.get_info<sycl::info::device::name>();
			if(lhs_vendor_id < rhs_vendor_id) return true;
			if(lhs_vendor_id > rhs_vendor_id) return false;
			return lhs_name < rhs_name;
		});
		return devs;
	};

	constexpr static auto get_platforms = [] {
		const auto devs = get_devices();
		std::vector<sycl::platform> pfs;
		for(const auto& d : devs) {
			pfs.push_back(d.get_platform());
		}
		return pfs;
	};

	SECTION("for sycl::device") {
		const auto first = get_devices();
		const auto second = get_devices();
		CHECK(first == second);
	}

	SECTION("for sycl::platforms") {
		const auto first = get_platforms();
		const auto second = get_platforms();
		CHECK(first == second);
	}
}

} // namespace celerity::detail
