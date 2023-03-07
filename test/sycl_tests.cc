#include "test_utils.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

namespace celerity::detail {

template <int Dims>
class dim_device_queue_fixture : public test_utils::device_queue_fixture {};

template <access_mode, bool>
class access_test_kernel;

#if CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 7)
constexpr auto sycl_target_device = cl::sycl::access::target::global_buffer;
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
