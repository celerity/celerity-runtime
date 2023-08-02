#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <celerity.h>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

// NOTE: We currently test backends on a "best effort" basis, that is we test whatever configuration(s) are available.
// To exhaustively test all backend configurations we'd need full control over the hardware and software (SYCL implementation)
// as well as a script to generate the necessary build configurations.

template <int Dims>
void write_linear_ids(size_t* ptr, const range<Dims>& buffer_range) {
	const auto br3 = range_cast<3>(buffer_range);
	for(size_t k = 0; k < br3[0]; ++k) {
		for(size_t j = 0; j < br3[1]; ++j) {
			for(size_t i = 0; i < br3[2]; ++i) {
				const auto linear_id = get_linear_index(buffer_range, id_cast<Dims>(id<3>(k, j, i)));
				ptr[linear_id] = linear_id;
			}
		}
	}
}

template <int Dims>
void verify_copied_linear_ids(const size_t* host_buf, const range<Dims>& source_range, const id<Dims>& source_offset, const range<Dims>& target_range,
    const id<Dims>& target_offset, const range<Dims>& copy_range) {
	const auto cr3 = range_cast<3>(copy_range);
	for(size_t k = 0; k < cr3[0]; ++k) {
		for(size_t j = 0; j < cr3[1]; ++j) {
			for(size_t i = 0; i < cr3[2]; ++i) {
				const auto src_idx = source_offset + id_cast<Dims>(id<3>(k, j, i));
				const auto tgt_idx = target_offset + id_cast<Dims>(id<3>(k, j, i));
				REQUIRE_LOOP(host_buf[get_linear_index(target_range, tgt_idx)] == get_linear_index(source_range, src_idx));
			}
		}
	}
}

template <int Dims>
struct copy_parameters {
	range<Dims> source_range = test_utils::truncate_range<Dims>({5, 7, 11});
	range<Dims> target_range = test_utils::truncate_range<Dims>({13, 17, 19});
	range<Dims> copy_range = test_utils::truncate_range<Dims>({2, 4, 8});
	id<Dims> source_offset = test_utils::truncate_id<Dims>({2, 2, 2});
	id<Dims> target_offset = test_utils::truncate_id<Dims>({3, 5, 7});
};

template <int Dims>
std::vector<size_t> copy_to_host(sycl::queue& q, const size_t* device_ptr, const range<Dims>& buffer_range) {
	std::vector<size_t> result(buffer_range.size());
	q.memcpy(result.data(), device_ptr, buffer_range.size() * sizeof(size_t));
	q.wait_and_throw();
	return result;
}

enum class copy_test_type { intra_device, inter_device, host_to_device, device_to_host };

struct host_or_device {
	std::optional<sycl::queue> queue;
	size_t* ptr = nullptr;

	void malloc(const size_t count) {
		if(queue.has_value()) {
			ptr = sycl::malloc_device<size_t>(count, *queue);
		} else {
			ptr = static_cast<size_t*>(std::malloc(count * sizeof(size_t))); // NOLINT cppcoreguidelines-no-malloc
		}
	}

	template <typename KernelName, typename Fn>
	void run(const Fn fn) {
		if(queue.has_value()) {
			queue->single_task<KernelName>([=]() { fn(); });
			queue->wait_and_throw();
		} else {
			fn();
		}
	}

	~host_or_device() {
		if(ptr != nullptr) {
			if(queue.has_value()) {
				sycl::free(ptr, *queue);
			} else {
				std::free(ptr); // NOLINT cppcoreguidelines-no-malloc
			}
		}
	}
};

std::pair<host_or_device, host_or_device> select_source_and_target(const copy_test_type test_type, const sycl::platform& platform) {
	const auto devices = platform.get_devices();
	if(devices.empty()) { throw std::runtime_error(fmt::format("Platform {} has no devices", platform.get_info<sycl::info::platform::name>())); }
	switch(test_type) {
	case copy_test_type::intra_device: {
		return std::pair{host_or_device{sycl::queue{devices[0]}}, host_or_device{sycl::queue{devices[0]}}};
	}
	case copy_test_type::inter_device: {
		if(devices.size() < 2) {
			throw std::runtime_error(fmt::format("Platform {} has less than two devices", platform.get_info<sycl::info::platform::name>()));
		}
		return std::pair{host_or_device{sycl::queue{devices[0]}}, host_or_device{sycl::queue{devices[1]}}};
	}
	case copy_test_type::host_to_device: {
		return std::pair{host_or_device{std::nullopt}, host_or_device{sycl::queue{devices[0]}}};
	}
	case copy_test_type::device_to_host: {
		return std::pair{host_or_device{sycl::queue{devices[0]}}, host_or_device{std::nullopt}};
	}
	}
}

template <int Dims>
struct copy_kernel {};

TEMPLATE_TEST_CASE_SIG("memcpy_strided_device allows to copy between the same device, different devices on the same platform and between host and device",
    "[backend]", ((int Dims), Dims), 1, 2, 3) {
	const copy_parameters<Dims> cp;
	const size_t platform_id = GENERATE(Catch::Generators::range(size_t(0), sycl::platform::get_platforms().size()));
	const auto test_type = GENERATE(copy_test_type::intra_device, copy_test_type::inter_device, copy_test_type::host_to_device, copy_test_type::device_to_host);
	CAPTURE(platform_id, test_type);

	const auto platform = sycl::platform::get_platforms()[platform_id];
	CAPTURE(platform.get_info<sycl::info::platform::name>());

	if(platform.get_info<sycl::info::platform::name>().substr(0, 5) == "Intel" && test_type == copy_test_type::inter_device) {
		SKIP("Inter-GPU copy appears to currently be broken on Intel OpenCL / Level Zero");
	}

	auto [src, tgt] = ([&] {
		try {
			return select_source_and_target(test_type, platform);
		} catch(std::runtime_error& e) {
			SKIP(e.what());
			abort(); // Will never be reached, just to shut up compiler about not all control paths returning a value
		}
	})();

	src.malloc(cp.source_range.size());
	tgt.malloc(cp.target_range.size());

	src.template run<copy_kernel<Dims>>([ptr = src.ptr, cp = cp]() { write_linear_ids(ptr, cp.source_range); });

	const auto get_a_queue = [](host_or_device& a, host_or_device& b) -> sycl::queue& {
		if(a.queue.has_value()) return *a.queue;
		return *b.queue;
	};

	// Note: This may also be the generic backend
	SECTION("using automatically selected backend") {
		backend::memcpy_strided_device(
		    get_a_queue(src, tgt), src.ptr, tgt.ptr, sizeof(size_t), cp.source_range, cp.source_offset, cp.target_range, cp.target_offset, cp.copy_range);
	}

	SECTION("using generic backend") {
		backend_detail::backend_operations<backend::type::generic>::memcpy_strided_device(
		    get_a_queue(src, tgt), src.ptr, tgt.ptr, sizeof(size_t), cp.source_range, cp.source_offset, cp.target_range, cp.target_offset, cp.copy_range);
	}

	const auto host_buf = copy_to_host(get_a_queue(tgt, src), tgt.ptr, cp.target_range);
	verify_copied_linear_ids(host_buf.data(), cp.source_range, cp.source_offset, cp.target_range, cp.target_offset, cp.copy_range);
}