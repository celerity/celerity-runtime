#include "test_utils.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>


using namespace celerity;
using namespace celerity::detail;

template <access_mode, bool>
class access_test_kernel;

template <access_mode AccessMode, bool UsingPlaceholderAccessor>
static auto make_device_accessor(sycl::buffer<int, 1>& buf, sycl::handler& cgh, const subrange<1>& sr) {
	if constexpr(UsingPlaceholderAccessor) {
		sycl::accessor<int, 1, AccessMode, sycl::target::device, sycl::access::placeholder::true_t> acc{buf, sr.range, sr.offset};
		cgh.require(acc);
		return acc;
	} else {
		return buf.get_access<AccessMode>(cgh, sr.range, sr.offset);
	}
}

// If this test fails, celerity can't reliably support reductions on the user's combination of backend and hardware
TEST_CASE_METHOD(test_utils::sycl_queue_fixture, "SYCL has working simple scalar reductions", "[sycl][reductions]") {
	const size_t N = GENERATE(64, 512, 1024, 4096);
	CAPTURE(N);

	const auto buf = sycl::malloc_host<int>(1, get_sycl_queue());
	*buf = 99; // SYCL reduction must overwrite this, not include it in the reduction result

	get_sycl_queue()
	    .submit([&](sycl::handler& cgh) {
		    cgh.parallel_for(sycl::nd_range<1>{N, 64}, // ND-range: DPC++ e330855 (May 7, 2024) on CUDA will run out of registers for the default WG size
		        sycl::reduction(buf, sycl::plus<int>{}, sycl::property::reduction::initialize_to_identity{}), [](auto, auto& r) { r.combine(1); });
	    })
	    .wait();

	CHECK(static_cast<size_t>(*buf) == N);

	sycl::free(buf, get_sycl_queue());
}

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
