#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "backend/sycl_backend.h"
#include "nd_memory.h"

#include "copy_test_utils.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

// backend_*() functions here dispatch _host / _device member functions based on whether a device id is provided or not

void* backend_alloc(backend& backend, const std::optional<device_id>& device, const size_t size, const size_t alignment) {
	return test_utils::await(device.has_value() ? backend.enqueue_device_alloc(*device, size, alignment) : backend.enqueue_host_alloc(size, alignment));
}

void backend_free(backend& backend, const std::optional<device_id>& device, void* const ptr) {
	test_utils::await(device.has_value() ? backend.enqueue_device_free(*device, ptr) : backend.enqueue_host_free(ptr));
}

void backend_copy(backend& backend, const std::optional<device_id>& source_device, const std::optional<device_id>& dest_device, const void* const source_base,
    void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) {
	if(source_device.has_value() || dest_device.has_value()) {
		auto device = source_device.has_value() ? *source_device : *dest_device;
		test_utils::await(backend.enqueue_device_copy(device, 0, source_base, dest_base, source_box, dest_box, copy_region, elem_size));
	} else {
		test_utils::await(backend.enqueue_host_copy(0, source_base, dest_base, source_box, dest_box, copy_region, elem_size));
	}
}

/// For extracting hydration results
template <target Target>
struct mock_accessor {
	hydration_id hid;
	std::optional<closure_hydrator::accessor_info> info;

	explicit mock_accessor(hydration_id hid) : hid(hid) {}
	mock_accessor(const mock_accessor& other) : hid(other.hid) { copy_and_hydrate(other); }
	mock_accessor(mock_accessor&&) = delete;
	mock_accessor& operator=(const mock_accessor& other) { hid = other.hid, copy_and_hydrate(other); }
	mock_accessor& operator=(mock_accessor&&) = delete;
	~mock_accessor() = default;

	void copy_and_hydrate(const mock_accessor& other) {
		if(!info.has_value() && detail::closure_hydrator::is_available() && detail::closure_hydrator::get_instance().is_hydrating()) {
			info = detail::closure_hydrator::get_instance().get_accessor_info<Target>(hid);
		}
	}
};

std::vector<sycl::device> select_devices_for_backend(sycl_backend_type type) {
	// device discovery - we need at least one to run anything and two to run device-to-peer tests
	const auto all_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	std::vector<sycl::device> backend_devices;
	std::copy_if(all_devices.begin(), all_devices.end(), std::back_inserter(backend_devices),
	    [=](const sycl::device& device) { return utils::contains(sycl_backend_enumerator{}.compatible_backends(device), type); });
	return backend_devices;
}

std::tuple<sycl_backend_type, std::unique_ptr<backend>, std::vector<sycl::device>> generate_backends_with_devices(bool enable_profiling = false) {
	const auto backend_type = GENERATE(test_utils::from_vector(sycl_backend_enumerator{}.available_backends()));
	auto sycl_devices = select_devices_for_backend(backend_type);
	CAPTURE(backend_type, sycl_devices);

	if(sycl_devices.empty()) { SKIP("No devices available for backend"); }
	auto backend = make_sycl_backend(backend_type, sycl_devices, enable_profiling);
	return {backend_type, std::move(backend), std::move(sycl_devices)};
}

bool accessor_info_equal(const closure_hydrator::accessor_info& lhs, const closure_hydrator::accessor_info& rhs) {
	bool equal = lhs.ptr == rhs.ptr && lhs.allocated_box_in_buffer == rhs.allocated_box_in_buffer && lhs.accessed_box_in_buffer == rhs.accessed_box_in_buffer;
	CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(equal &= lhs.out_of_bounds_indices == rhs.out_of_bounds_indices;)
	return equal;
}

TEST_CASE("debug allocations are host- and device-accessible", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const auto debug_ptr = static_cast<int*>(backend->debug_alloc(sizeof(int)));
	*debug_ptr = 1;
	sycl::queue(sycl_devices[0], sycl::property::queue::in_order{}).single_task([=]() { *debug_ptr += 1; }).wait();
	CHECK(*debug_ptr == 2);
	backend->debug_free(debug_ptr);
}

TEST_CASE("backend allocations are properly aligned", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	constexpr size_t size = 1024;
	constexpr size_t sycl_max_alignment = 64; // See SYCL spec 4.14.2.6

	const auto host_ptr = backend_alloc(*backend, std::nullopt, size, sycl_max_alignment);
	CHECK(reinterpret_cast<uintptr_t>(host_ptr) % sycl_max_alignment == 0);
	backend_free(*backend, std::nullopt, host_ptr);

	for(device_id did = 0; did < sycl_devices.size(); ++did) {
		CAPTURE(did);
		const auto device_ptr = backend_alloc(*backend, did, size, sycl_max_alignment);
		CHECK(reinterpret_cast<uintptr_t>(device_ptr) % sycl_max_alignment == 0);
		backend_free(*backend, did, device_ptr);
	}
}

TEST_CASE("backend allocations are pattern-filled in debug builds", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	sycl::queue sycl_queue(sycl_devices[0], sycl::property::queue::in_order{});

	constexpr size_t size = 1024;
	const std::vector<uint8_t> expected(size, sycl_backend_detail::uninitialized_memory_pattern);

	for(const auto did : std::initializer_list<std::optional<device_id>>{std::nullopt, device_id(0)}) {
		CAPTURE(did);
		const auto ptr = backend_alloc(*backend, did, 1024, 1);
		std::vector<uint8_t> contents(size);
		sycl_queue.memcpy(contents.data(), ptr, size).wait();
		CHECK(contents == expected);
		backend_free(*backend, did, ptr);
	}
#else
	SKIP("Not in a debug build");
#endif
}

TEST_CASE("host task lambdas are hydrated and invoked with the correct parameters", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const mock_accessor<target::host_task> acc1(hydration_id(1));
	const mock_accessor<target::host_task> acc2(hydration_id(2));
	const std::vector<closure_hydrator::accessor_info> accessor_infos{
	    {reinterpret_cast<void*>(0x1337), box<3>{{1, 2, 3}, {4, 5, 6}},
	        box<3>{{0, 1, 2}, {7, 8, 9}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x69420))},
	    {reinterpret_cast<void*>(0x7331), box<3>{{3, 2, 1}, {6, 5, 4}},
	        box<3>{{2, 1, 0}, {9, 8, 7}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x1230))}};

	constexpr size_t lane = 0;
	const box<3> execution_range({1, 2, 3}, {4, 5, 6});
	const auto collective_comm = reinterpret_cast<const communicator*>(0x42000);

	int value = 1;

	// no accessors
	test_utils::await(backend->enqueue_host_task(
	    lane,
	    [&](const box<3>& b, const communicator* c) {
		    CHECK(b == execution_range);
		    CHECK(c == collective_comm);
		    value += 1;
	    },
	    {}, execution_range, collective_comm));

	// yes accessors
	test_utils::await(backend->enqueue_host_task(
	    lane,
	    [&, acc1, acc2](const box<3>& b, const communicator* c) {
		    REQUIRE(acc1.info.has_value());
		    REQUIRE(acc2.info.has_value());
		    CHECK(accessor_info_equal(*acc1.info, accessor_infos[0]));
		    CHECK(accessor_info_equal(*acc2.info, accessor_infos[1]));
		    CHECK(b == execution_range);
		    CHECK(c == collective_comm);
		    value += 1;
	    },
	    accessor_infos, execution_range, collective_comm));

	CHECK(value == 3);
}

TEST_CASE("host tasks in a single lane execute in-order", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	constexpr size_t lane = 0;

	std::optional<std::thread::id> first_thread_id;
	const auto first_fn = [&](const box<3>&, const communicator*) {
		first_thread_id = std::this_thread::get_id();
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	};
	const auto first = backend->enqueue_host_task(lane, first_fn, {}, box_cast<3>(box<0>()), nullptr);

	std::optional<std::thread::id> second_thread_id;
	const auto second_fn = [&](const box<3>&, const communicator* /* collective_comm */) {
		CHECK(first.is_complete());
		second_thread_id = std::this_thread::get_id();
	};
	const auto second = backend->enqueue_host_task(lane, second_fn, {}, box_cast<3>(box<0>()), nullptr);

	for(;;) {
		if(second.is_complete()) {
			CHECK(first.is_complete());
			break;
		}
	}

	REQUIRE(first_thread_id.has_value());
	REQUIRE(second_thread_id.has_value());
	CHECK(*first_thread_id == *second_thread_id);
}

TEST_CASE("device kernel command groups are hydrated and invoked with the correct parameters", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const mock_accessor<target::device> acc1(hydration_id(1));
	const mock_accessor<target::device> acc2(hydration_id(2));
	const std::vector<closure_hydrator::accessor_info> accessor_infos{
	    {reinterpret_cast<void*>(0x1337), box<3>{{1, 2, 3}, {4, 5, 6}},
	        box<3>{{0, 1, 2}, {7, 8, 9}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x69420))},
	    {reinterpret_cast<void*>(0x7331), box<3>{{3, 2, 1}, {6, 5, 4}},
	        box<3>{{2, 1, 0}, {9, 8, 7}} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, reinterpret_cast<oob_bounding_box*>(0x1230))}};

	constexpr size_t lane = 0;
	const box<3> execution_range({1, 2, 3}, {4, 5, 6});
	const std::vector<void*> reduction_ptrs{nullptr, reinterpret_cast<void*>(1337)};

	const auto value_ptr = static_cast<int*>(backend->debug_alloc(sizeof(int)));

	for(device_id did = 0; did < sycl_devices.size(); ++did) {
		*value_ptr = 1;

		// no accessors
		test_utils::await(backend->enqueue_device_kernel(
		    did, lane,
		    [&](sycl::handler& cgh, const box<3>& b, const std::vector<void*>& r) {
			    CHECK(b == execution_range);
			    CHECK(r == reduction_ptrs);
			    cgh.single_task([=] { *value_ptr += 1; });
		    },
		    {}, execution_range, reduction_ptrs));

		// yes accessors
		test_utils::await(backend->enqueue_device_kernel(
		    did, lane,
		    [&, acc1, acc2](sycl::handler& cgh, const box<3>& b, const std::vector<void*>& r) {
			    REQUIRE(acc1.info.has_value());
			    REQUIRE(acc2.info.has_value());
			    CHECK(accessor_info_equal(*acc1.info, accessor_infos[0]));
			    CHECK(accessor_info_equal(*acc2.info, accessor_infos[1]));
			    CHECK(b == execution_range);
			    CHECK(r == reduction_ptrs);
			    cgh.single_task([=] { *value_ptr += 1; });
		    },
		    accessor_infos, execution_range, reduction_ptrs));

		CHECK(*value_ptr == 3);
	}

	backend->debug_free(value_ptr);
}

TEST_CASE("device kernels in a single lane execute in-order", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	const auto dummy = static_cast<volatile int*>(backend->debug_alloc(sizeof(int)));
	*dummy = 0;

	constexpr size_t lane = 0;

	const auto first = backend->enqueue_device_kernel(device_id(0), lane,
	    [=](sycl::handler& cgh, const box<3>&, const std::vector<void*>&) {
		    cgh.single_task([=] {
			    // busy "wait" - takes ~10ms on AdaptiveCpp debug build with RTX 3090
			    for(int i = 0; i < 100'000; ++i) {
				    *dummy = i;
			    }
		    });
	    },
	    {}, box_cast<3>(box<0>()), {});

	const auto second = backend->enqueue_device_kernel(
	    device_id(0), lane, [=](sycl::handler& cgh, const box<3>&, const std::vector<void*>&) { cgh.single_task([=] {}); }, {}, box_cast<3>(box<0>()), {});

	for(;;) {
		if(second.is_complete()) {
			CHECK(first.is_complete());
			break;
		}
	}

	backend->debug_free(const_cast<int*>(dummy));
}

TEST_CASE("backend copies work correctly on all source- and destination layouts", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices();
	CAPTURE(backend_type, sycl_devices);

	// "device to itself" is used for buffer resizes, and "device to peer" for coherence (if the backend supports it)
	const auto direction = GENERATE(values<std::string>({"host to host", "host to device", "device to host", "device to peer", "device to itself"}));
	CAPTURE(direction);

	std::optional<device_id> source_did; // host memory if nullopt
	std::optional<device_id> dest_did;   // host memory if nullopt
	if(direction == "host to device") {
		dest_did = device_id(0);
	} else if(direction == "device to host") {
		source_did = device_id(0);
	} else if(direction == "device to itself") {
		source_did = device_id(0);
		dest_did = device_id(0);
	} else if(direction == "device to peer") {
		const auto& system = backend->get_system_info();
		if(system.devices.size() < 2) { SKIP("Not enough devices available to test peer-to-peer copy"); }
		if(system.devices[0].native_memory < first_device_memory_id || system.devices[1].native_memory < first_device_memory_id
		    || system.devices[0].native_memory == system.devices[1].native_memory) {
			SKIP("Available devices do not report disjoint, dedicated memories");
		}
		if(!system.memories[system.devices[0].native_memory].copy_peers.test(system.devices[1].native_memory)) {
			SKIP("Available devices do not support peer-to-peer copy");
		}
		source_did = device_id(0);
		dest_did = device_id(1);
	} else if(direction != "host to host") {
		FAIL("Unknown test type");
	}
	CAPTURE(source_did, dest_did);

	// use a helper SYCL queue to init allocations and copy between user and source/dest memories
	sycl::queue source_sycl_queue(sycl_devices[0], sycl::property::queue::in_order{});
	sycl::queue dest_sycl_queue(sycl_devices[direction == "device to peer" ? 1 : 0], sycl::property::queue::in_order{});

	const auto source_base = backend_alloc(*backend, source_did, test_utils::copy_test_max_range.size() * sizeof(int), alignof(int));
	const auto dest_base = backend_alloc(*backend, dest_did, test_utils::copy_test_max_range.size() * sizeof(int), alignof(int));

	// generate the source pattern in user memory
	std::vector<int> source_template(test_utils::copy_test_max_range.size());
	std::iota(source_template.begin(), source_template.end(), 1);

	// use a loop instead of GENERATE() to avoid re-instantiating the backend and re-allocating device memory on each iteration (very expensive!)
	for(const auto& [source_box, dest_box, copy_box] : test_utils::generate_copy_test_layouts()) {
		CAPTURE(source_box, dest_box, copy_box);
		REQUIRE(all_true(source_box.get_range() <= test_utils::copy_test_max_range));
		REQUIRE(all_true(dest_box.get_range() <= test_utils::copy_test_max_range));

		// reference is nd_copy_host (tested in nd_memory_tests)
		std::vector<int> expected_dest(dest_box.get_area());
		nd_copy_host(source_template.data(), expected_dest.data(), box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

		source_sycl_queue.memcpy(source_base, source_template.data(), source_box.get_area() * sizeof(int)).wait();
		dest_sycl_queue.memset(dest_base, 0, dest_box.get_area() * sizeof(int)).wait();

		backend_copy(
		    *backend, source_did, dest_did, source_base, dest_base, box_cast<3>(source_box), box_cast<3>(dest_box), box_cast<3>(copy_box), sizeof(int));

		std::vector<int> actual_dest(dest_box.get_area());
		dest_sycl_queue.memcpy(actual_dest.data(), dest_base, actual_dest.size() * sizeof(int)).wait();

		REQUIRE(actual_dest == expected_dest);
	}

	backend_free(*backend, source_did, source_base);
	backend_free(*backend, dest_did, dest_base);
}

TEST_CASE("SYCL backend enumerator classifies backends correctly", "[backend]") {
	CHECK_FALSE(sycl_backend_enumerator().is_specialized(sycl_backend_type::generic));
	CHECK(sycl_backend_enumerator().is_specialized(sycl_backend_type::cuda));
	CHECK(sycl_backend_enumerator().get_priority(sycl_backend_type::cuda) > sycl_backend_enumerator().get_priority(sycl_backend_type::generic));
}

TEST_CASE("backends report execution time iff profiling is enabled", "[backend]") {
	test_utils::allow_backend_fallback_warnings();

	const auto enable_profiling = static_cast<bool>(GENERATE(values({0, 1})));
	const auto [backend_type, backend, sycl_devices] = generate_backends_with_devices(enable_profiling);
	CAPTURE(backend_type, sycl_devices);

	const auto dummy_ptr = static_cast<volatile int*>(backend->debug_alloc(sizeof(int)));
	const size_t host_device_alloc_size = 4096;
	const std::vector<uint8_t> user_alloc(4096);
	const auto host_ptr = test_utils::await(backend->enqueue_host_alloc(host_device_alloc_size, 1));
	const auto device_ptr = test_utils::await(backend->enqueue_device_alloc(device_id(0), host_device_alloc_size, 1));

	async_event event;

	SECTION("on device kernels") {
		*dummy_ptr = 0;
		event = backend->enqueue_device_kernel(device_id(0), /* lane */ 0,
		    [=](sycl::handler& cgh, const box<3>&, const std::vector<void*>&) {
			    cgh.single_task([=] {
				    // busy "wait" - takes ~1ms on AdaptiveCpp debug build with RTX 3090
				    for(int i = 0; i < 100'000; ++i) {
					    *dummy_ptr = i;
				    }
			    });
		    },
		    {}, box_cast<3>(box<0>()), {});
	}

	SECTION("on host tasks") {
		event = backend->enqueue_host_task(
		    /* lane */ 0, [&](const box<3>&, const communicator*) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }, {}, box_cast<3>(box<0>()),
		    nullptr);
	}

	const auto unit_box = box_cast<3>(box<0>());

	SECTION("on host copies") {
		event = backend->enqueue_host_copy(/* lane */ 0, user_alloc.data(), host_ptr, unit_box, unit_box, unit_box, host_device_alloc_size);
	}

	SECTION("on device copies") {
		event = backend->enqueue_device_copy(device_id(0), /* lane */ 0, host_ptr, device_ptr, unit_box, unit_box, unit_box, host_device_alloc_size);
	}

	test_utils::await(event);

	const auto time = event.get_native_execution_time();
	REQUIRE(time.has_value() == enable_profiling);

	if(enable_profiling) { CHECK(time.value() > std::chrono::nanoseconds(0)); }

	test_utils::await(backend->enqueue_device_free(device_id(0), device_ptr));
	test_utils::await(backend->enqueue_host_free(host_ptr));
	backend->debug_free(const_cast<int*>(dummy_ptr));
}
