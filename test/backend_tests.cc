#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "backend/backend.h"
#include "backend/queue.h"
#include "nd_memory.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;


enum class copy_direction { host_to_device, device_to_host, device_to_peer, device_to_itself };

template <>
struct Catch::StringMaker<copy_direction> {
	static std::string convert(const copy_direction value) {
		switch(value) {
		case copy_direction::host_to_device: return "host-to-device";
		case copy_direction::device_to_host: return "device-to-host";
		case copy_direction::device_to_peer: return "device-to-peer";
		case copy_direction::device_to_itself: return "device-to-itself";
		default: return "unknown";
		}
	}
};

template <>
struct Catch::StringMaker<backend::type> {
	static std::string convert(const backend::type value) { return std::string(backend::get_name(value)); }
};

TEMPLATE_TEST_CASE_SIG("backend_queue::nd_copy works correctly on all source- and destination layouts ", "[backend]", ((int Dims), Dims), 0, 1, 2, 3) {
	const auto backend = GENERATE(values({backend::type::generic, backend::type::cuda}));
	if(!backend_detail::is_enabled_v<backend::type::cuda>) SKIP("CUDA backend not available");
	CAPTURE(backend);

	// device discovery - we need at least one to run anything and two to run device-to-peer tests
	std::vector<backend::device_config> backend_devices;
	device_id next_did = 0;
	memory_id next_device_mid = first_device_memory_id;
	for(const auto& device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
		if(backend == backend::type::generic || backend::get_effective_type(device) == backend) {
			backend_devices.push_back(backend::device_config{next_did++, next_device_mid++, device});
		}
	}
	if(backend_devices.empty()) { SKIP("No devices available for backend"); }

	// device_to_itself is used for buffer resizes, and device_to_peer for coherence (if the backend supports it)
	const auto direction =
	    GENERATE(values({copy_direction::host_to_device, copy_direction::device_to_host, copy_direction::device_to_peer, copy_direction::device_to_itself}));
	CAPTURE(direction);

	memory_id source_mid;
	memory_id dest_mid;
	switch(direction) {
	case copy_direction::host_to_device:
		source_mid = host_memory_id;
		dest_mid = backend_devices[0].native_memory;
		break;
	case copy_direction::device_to_host:
		source_mid = backend_devices[0].native_memory;
		dest_mid = host_memory_id;
		break;
	case copy_direction::device_to_itself:
		source_mid = backend_devices[0].native_memory;
		dest_mid = backend_devices[0].native_memory;
		break;
	case copy_direction::device_to_peer:
		if(backend_devices.size() < 2) SKIP("Not enough devices available to test peer-to-peer copy");
		if(!backend::enable_copy_between_peer_memories(backend_devices[0].sycl_device, backend_devices[1].sycl_device)) {
			SKIP("Peer-to-peer copy not supported by devices");
		}
		source_mid = backend_devices[0].native_memory;
		dest_mid = backend_devices[1].native_memory;
		break;
	default: throw std::runtime_error("Unknown test type");
	}
	CAPTURE(source_mid, dest_mid);

	// generate random boundaries before and after copy range in every dimension
	int source_shift[3];
	int dest_shift[3];
	if constexpr(Dims > 0) { source_shift[0] = GENERATE(values({-2, 0, 2})), dest_shift[0] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 1) { source_shift[1] = GENERATE(values({-2, 0, 2})), dest_shift[1] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 2) { source_shift[2] = GENERATE(values({-2, 0, 2})), dest_shift[2] = GENERATE(values({-2, 0, 2})); }

	const auto copy_box = box<Dims>(test_utils::truncate_id<Dims>({2, 2, 2}), test_utils::truncate_range<Dims>({5, 6, 7}));
	CAPTURE(copy_box);

	box<Dims> source_box;
	box<Dims> dest_box;
	{
		id<Dims> source_min;
		id<Dims> source_max;
		id<Dims> dest_min;
		id<Dims> dest_max;
		for(int i = 0; i < Dims; ++i) {
			source_min[i] = std::max(0, source_shift[i]);
			source_max[i] = copy_box.get_max()[i] + std::max(0, -source_shift[i]);
			dest_min[i] = std::max(0, dest_shift[i]);
			dest_max[i] = copy_box.get_max()[i] + std::max(0, -dest_shift[i]);
		}
		source_box = box<Dims>(source_min, source_max);
		dest_box = box<Dims>(dest_min, dest_max);
	}
	CAPTURE(source_box, dest_box);

	// generate the source pattern in user memory
	std::vector<int> source_template(source_box.get_area());
	std::iota(source_template.begin(), source_template.end(), 1);

	// reference is nd_copy_host (tested in nd_memory_tests)
	std::vector<int> expected_dest(dest_box.get_area());
	copy_region_host(source_template.data(), expected_dest.data(), box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

	// use a helper SYCL queues to init allocations and copy between user and source/dest memories
	sycl::queue source_sycl_queue(backend_devices[0].sycl_device);
	sycl::queue dest_sycl_queue(backend_devices[direction == copy_direction::device_to_peer ? 1 : 0].sycl_device);

	const auto backend_queue = backend::make_queue(backend, backend_devices);

	const auto source_base = backend_queue->alloc(source_mid, source_box.get_area() * sizeof(int), alignof(int));
	source_sycl_queue.submit([&](sycl::handler& cgh) { cgh.memcpy(source_base, source_template.data(), source_template.size() * sizeof(int)); }).wait();

	const auto dest_base = backend_queue->alloc(dest_mid, dest_box.get_area() * sizeof(int), alignof(int));
	dest_sycl_queue.submit([&](sycl::handler& cgh) { cgh.memset(dest_base, 0, dest_box.get_area() * sizeof(int)); }).wait();

	const auto copy_event = backend_queue->copy_region(
	    source_mid, dest_mid, source_base, dest_base, box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));
	while(!copy_event.is_complete()) {} // busy-wait

	std::vector<int> actual_dest(dest_box.get_area());
	dest_sycl_queue.submit([&](sycl::handler& cgh) { cgh.memcpy(actual_dest.data(), dest_base, actual_dest.size() * sizeof(int)); }).wait();

	CHECK(actual_dest == expected_dest);

	backend_queue->free(source_mid, source_base);
	backend_queue->free(dest_mid, dest_base);
}
