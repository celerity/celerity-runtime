#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "test_utils.h"

using dt = sycl::info::device_type;
using namespace celerity;
using namespace celerity::detail;

struct mock_platform;

struct type_and_name {
	dt type;
	std::string name;
};

enum class mock_backend { foo, bar, qux };

struct mock_device {
	mock_device() : mock_device({0, dt::gpu, "", nullptr, {}}) {}

	mock_device(size_t id, mock_platform& platform, dt type) : mock_device({id, type, fmt::format("Mock device {}", id), &platform, {}}){};

	mock_device(size_t id, mock_platform& platform, const type_and_name& tan) : mock_device({id, tan.type, tan.name, &platform, {}}) {}

	bool operator==(const mock_device& other) const { return other.m_pimpl == m_pimpl; }

	mock_platform& get_platform() const {
		assert(m_pimpl->platform != nullptr);
		return *m_pimpl->platform;
	}

	template <typename Param>
	auto get_info() const {
		if constexpr(std::is_same_v<Param, sycl::info::device::name>) { return m_pimpl->name; }
		if constexpr(std::is_same_v<Param, sycl::info::device::device_type>) { return m_pimpl->type; }
	}

	dt get_type() const { return m_pimpl->type; }

	size_t get_id() const { return m_pimpl->id; }

	mock_backend get_backend() const;

	bool has(const sycl::aspect aspect) const { return m_pimpl->aspects.count(aspect) > 0 ? m_pimpl->aspects.at(aspect) : true; }

	void set_aspect(const sycl::aspect aspect, const bool value) { m_pimpl->aspects[aspect] = value; }

  private:
	struct impl {
		size_t id;
		dt type;
		std::string name;
		mock_platform* platform;
		std::unordered_map<sycl::aspect, bool> aspects;
	};
	std::shared_ptr<impl> m_pimpl; // Use PIMPL for reference semantics

	mock_device(impl i) : m_pimpl(std::make_shared<impl>(i)) {}
};

struct mock_platform {
	mock_platform(size_t id, std::optional<std::string> name) : m_id(id), m_name(name.has_value() ? std::move(*name) : fmt::format("Mock platform {}", id)) {}

	template <typename... Args>
	auto create_devices(Args... args) {
		std::array<mock_device, sizeof...(args)> new_devices = {mock_device(m_next_device_id++, *this, args)...};
		m_devices.insert(m_devices.end(), new_devices.begin(), new_devices.end());
		return new_devices;
	}

	std::vector<mock_device> get_devices(dt type = dt::all) const {
		if(type != dt::all) {
			std::vector<mock_device> devices_with_type;
			for(auto device : m_devices) {
				if(device.get_type() == type) { devices_with_type.emplace_back(device); }
			}
			return devices_with_type;
		}
		return m_devices;
	}

	template <typename Param>
	std::string get_info() const {
		return m_name;
	}

	bool operator==(const mock_platform& other) const { return other.m_id == m_id; }
	bool operator!=(const mock_platform& other) const { return !(*this == other); }

	size_t get_id() const { return m_id; }

	mock_backend get_backend() const { return m_backend; }
	void set_backend(mock_backend backend) { m_backend = backend; }

  private:
	size_t m_id;
	std::string m_name;
	size_t m_next_device_id = 0;
	std::vector<mock_device> m_devices;
	mock_backend m_backend = mock_backend::foo;
};

mock_backend mock_device::get_backend() const { return m_pimpl->platform->get_backend(); }

template <typename... Args>
auto create_mock_platforms(Args... args) {
	size_t next_id = 0;
	return std::array<mock_platform, sizeof...(args)>{mock_platform(next_id++, args)...};
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "check_required_device_aspects throws if a device does not support required aspects", "[device-selection]") {
	mock_device device;
	CHECK_NOTHROW(check_required_device_aspects(device));

	// Note: This assumes that the following checks are performed in reverse order within check_required_device_aspects

	device.set_aspect(sycl::aspect::usm_host_allocations, false);
	CHECK_THROWS_WITH(check_required_device_aspects(device), "device does not support USM host allocations");

	device.set_aspect(sycl::aspect::usm_device_allocations, false);
	CHECK_THROWS_WITH(check_required_device_aspects(device), "device does not support USM device allocations");
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices prefers user-specified device list", "[device-selection]") {
	celerity::test_utils::allow_max_log_level(celerity::detail::log_level::warn);

	config cfg(nullptr, nullptr);
	auto [mp] = create_mock_platforms(std::nullopt);

	CHECK_THROWS_WITH(
	    pick_devices(cfg, std::vector<mock_device>{}, std::vector<mock_platform>{mp}), "Device selection failed: The user-provided list of devices is empty");

	const auto devices = mp.create_devices(dt::gpu, dt::gpu, dt::cpu, dt::accelerator);
	const auto selected = pick_devices(cfg, std::vector<mock_device>{devices.begin(), devices.end()}, std::vector<mock_platform>{mp});
	CHECK(selected == std::vector<mock_device>{devices.begin(), devices.end()});
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices requires user-specified devices to have the same backend", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp_1, mp_2] = create_mock_platforms(std::nullopt, std::nullopt);
	auto [md_1] = mp_1.create_devices(dt::gpu);
	auto [md_2] = mp_2.create_devices(dt::gpu);

	CHECK_NOTHROW(pick_devices(cfg, std::vector<mock_device>{md_1, md_2}, std::vector<mock_platform>{mp_1, mp_2}));

	mp_1.set_backend(mock_backend::foo);
	mp_2.set_backend(mock_backend::bar);
	CHECK_THROWS_WITH(pick_devices(cfg, std::vector<mock_device>{md_1, md_2}, std::vector<mock_platform>{mp_1, mp_2}),
	    "Device selection failed: The user-provided list of devices contains devices from different backends");
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices throws if a user-specified devices does not support required aspects", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp] = create_mock_platforms(std::nullopt);
	auto [md_1, md_2] = mp.create_devices(dt::gpu, dt::gpu);
	md_2.set_aspect(sycl::aspect::usm_device_allocations, false);
	CHECK_THROWS_WITH(pick_devices(cfg, std::vector<mock_device>{md_1, md_2}, std::vector<mock_platform>{mp}),
	    "Device selection failed: Device 1 in user-provided list of devices caused error: device does not support USM device allocations");
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices selects the largest subset of GPUs that share the same backend", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_1.create_devices(dt::gpu);
	auto [md_2, md_3, md_4] = mp_2.create_devices(dt::gpu, dt::gpu, dt::gpu);
	mp_3.create_devices(dt::gpu, dt::gpu);

	mp_1.set_backend(mock_backend::foo);
	mp_2.set_backend(mock_backend::bar);
	mp_3.set_backend(mock_backend::qux);

	const auto selected = pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected == std::vector<mock_device>{md_2, md_3, md_4});
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices falls back to other device types if no GPUs are available", "[device-selection]") {
	celerity::test_utils::allow_max_log_level(celerity::detail::log_level::warn);

	config cfg(nullptr, nullptr);
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_1.create_devices(dt::cpu);
	auto [md_2, md_3, md_4] = mp_2.create_devices(dt::accelerator, dt::cpu, dt::host);
	mp_3.create_devices(dt::cpu, dt::accelerator);

	mp_1.set_backend(mock_backend::foo);
	mp_2.set_backend(mock_backend::bar);
	mp_3.set_backend(mock_backend::qux);

	const auto selected_1 = pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected_1 == std::vector<mock_device>{md_2, md_3, md_4});

	// Once there is a GPU however, it takes precedence
	auto [md_9] = mp_3.create_devices(dt::gpu);
	const auto selected_2 = pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected_2 == std::vector<mock_device>{md_9});
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices only considers devices that support required aspects", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto type = GENERATE(dt::gpu, dt::cpu);
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_1.create_devices(type);
	auto [md_2, md_3, md_4] = mp_2.create_devices(type, type, type);
	auto [md_5, md_6] = mp_3.create_devices(type, type);

	mp_1.set_backend(mock_backend::foo);
	mp_2.set_backend(mock_backend::bar);
	mp_3.set_backend(mock_backend::qux);

	md_2.set_aspect(sycl::aspect::usm_device_allocations, false);
	md_3.set_aspect(sycl::aspect::usm_device_allocations, false);

	const auto selected = pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected == std::vector<mock_device>{md_5, md_6});
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices supports passing a device selector function", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp_1, mp_2] = create_mock_platforms(std::nullopt, std::nullopt);
	auto [md_1] = mp_1.create_devices(dt::gpu);
	auto [md_2] = mp_2.create_devices(dt::gpu);

	mp_1.set_backend(mock_backend::foo);
	mp_2.set_backend(mock_backend::bar);

	const auto device_idx = GENERATE(0, 1);
	CAPTURE(device_idx);
	const auto to_select = std::array<mock_device, 2>{md_1, md_2}[device_idx];

	auto device_selector = [to_select](const mock_device& d) { return d == to_select ? 2 : 1; };
	auto selected = pick_devices(cfg, device_selector, std::vector<mock_platform>{mp_1, mp_2});
	CHECK(selected == std::vector<mock_device>{to_select});
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices selects the subset of devices with the largest cumulative selector score sharing the same backend",
    "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	auto [md_1] = mp_1.create_devices(dt::gpu);
	auto [md_2, md_3, md_4] = mp_2.create_devices(dt::cpu, dt::gpu, dt::accelerator);
	auto [md_5, md_6] = mp_3.create_devices(dt::host, dt::cpu);

	mp_1.set_backend(mock_backend::foo);
	mp_2.set_backend(mock_backend::bar);
	mp_3.set_backend(mock_backend::qux);

	const auto ignore_md_4 = GENERATE(true, false);
	const auto md_6_no_usm = GENERATE(true, false);

	md_6.set_aspect(sycl::aspect::usm_device_allocations, !md_6_no_usm);

	const std::vector<std::vector<int>> scores = {
	    {/* md_1 */ 50}, {/* md_2 */ 10, /* md_3 */ 20, /* md_4 */ ignore_md_4 ? -1 : 30}, {/* md_5 */ 30, /* md_6 */ 40}};
	const auto selector = [&scores](const mock_device& d) { return scores[d.get_platform().get_id()][d.get_id()]; };

	const auto selected = pick_devices(cfg, selector, std::vector<mock_platform>{mp_1, mp_2, mp_3});

	if(ignore_md_4) {
		if(md_6_no_usm) {
			CHECK(selected == std::vector<mock_device>{md_1});
		} else {
			CHECK(selected == std::vector<mock_device>{md_5, md_6});
		}
	} else {
		if(md_6_no_usm) {
			CHECK(selected == std::vector<mock_device>{md_2, md_3, md_4});
		} else {
			CHECK(selected == std::vector<mock_device>{md_5, md_6});
		}
	}
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices does not consider devices with a negative selector score", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp] = create_mock_platforms(std::nullopt);
	auto [md_1, md_2, md_3] = mp.create_devices(dt::gpu, dt::gpu, dt::gpu);

	auto selector = [md_2 = md_2](const mock_device& d) { return d.get_id() == md_2.get_id() ? -1 : 1; };
	const auto selected = pick_devices(cfg, selector, std::vector<mock_platform>{mp});
	CHECK(selected == std::vector<mock_device>{md_1, md_3});
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices throws if no devices are available", "[device-selection]") {
	config cfg(nullptr, nullptr);
	auto [mp] = create_mock_platforms(std::nullopt);

	SECTION("from the start") {
		const auto selector = [](const mock_device&) { return -1; };
		CHECK_THROWS_WITH(pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp}), "Device selection failed: No devices available");
		CHECK_THROWS_WITH(pick_devices(cfg, selector, std::vector<mock_platform>{mp}), "Device selection failed: No devices available");
	}

	SECTION("if all are ignored due to missing aspects") {
		auto [md_1] = mp.create_devices(dt::gpu);
		md_1.set_aspect(sycl::aspect::usm_device_allocations, false);
		CHECK_THROWS_WITH(pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp}), "Device selection failed: No eligible devices found");
		auto [md_2] = mp.create_devices(dt::cpu);
		md_2.set_aspect(sycl::aspect::usm_device_allocations, false);
		CHECK_THROWS_WITH(pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp}), "Device selection failed: No eligible devices found");
	}

	SECTION("if all are discarded by selector") {
		auto [md_1] = mp.create_devices(dt::gpu);
		const auto selector = [](const mock_device&) { return -1; };
		CHECK_THROWS_WITH(pick_devices(cfg, selector, std::vector<mock_platform>{mp}), "Device selection failed: No eligible devices found");
	}
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices attempts to evenly distributed devices if there is more than one local node", "[device-selection]") {
	struct distribution {
		size_t num_devices;
		std::vector<size_t> devices_per_node;
	};

	const std::vector<distribution> distributions = {
	    {1, {1}},             //
	    {2, {1, 1}},          //
	    {3, {2, 1}},          //
	    {4, {2, 2}},          //
	    {5, {2, 2, 1}},       //
	    {7, {3, 2, 2}},       //
	    {9, {2, 2, 2, 2, 1}}, //
	    {13, {3, 3, 3, 2, 2}} //
	};

	celerity::test_utils::allow_max_log_level(celerity::detail::log_level::warn);

	config cfg(nullptr, nullptr);

	for(const auto& dist : distributions) {
		auto [mp] = create_mock_platforms(std::nullopt);
		for(size_t i = 0; i < dist.num_devices; ++i) {
			mp.create_devices(dt::gpu);
		}

		std::unordered_set<std::pair<size_t, size_t>, utils::pair_hash> unique_devices;
		for(size_t i = 0; i < dist.devices_per_node.size(); ++i) {
			host_config h_cfg{dist.devices_per_node.size(), i};
			config_testspy::set_mock_host_cfg(cfg, h_cfg);
			const auto selected = pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp});
			REQUIRE_LOOP(selected.size() == dist.devices_per_node[i]);
			for(const auto& d : selected) {
				const auto unique_id = std::pair{d.get_platform().get_id(), d.get_id()};
				REQUIRE_LOOP(unique_devices.count(unique_id) == 0);
				unique_devices.insert(unique_id);
			}
		}
	}

	CHECK(test_utils::log_contains_exact(log_level::warn, "Celerity detected more than one node (MPI rank) on this host, which is not recommended. Will "
	                                                      "attempt to distribute local devices evenly across nodes."));
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices distributes devices in round-robin fashion if there are fewer than local nodes", "[device-selection]") {
	auto [mp] = create_mock_platforms(std::nullopt);
	auto devices = mp.create_devices(dt::gpu, dt::gpu, dt::gpu);

	const size_t num_ranks = GENERATE(4, 5, 6);

	test_utils::allow_max_log_level(log_level::warn);

	config cfg(nullptr, nullptr);

	for(size_t i = 0; i < num_ranks; ++i) {
		host_config h_cfg{num_ranks, i};
		config_testspy::set_mock_host_cfg(cfg, h_cfg);
		const auto selected = pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp});
		REQUIRE(selected.size() == 1);
		CHECK(selected[0] == devices[i % 3]);
		CHECK(test_utils::log_contains_exact(
		    log_level::warn, fmt::format("Found fewer devices (3) than local nodes ({}), multiple nodes will use the same device(s).", num_ranks)));
	}
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices prints device and platform information", "[device-selection]") {
	config cfg(nullptr, nullptr);
	mock_platform mp(68, "My platform");
	auto mds = mp.create_devices(type_and_name{dt::gpu, "My first device"}, type_and_name{dt::gpu, "My second device"});

	SECTION("when devices are provided by user") {
		pick_devices(cfg, std::vector<mock_device>{mds[0], mds[1]}, std::vector<mock_platform>{mp});
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform 'My platform', device 'My first device' (specified by user)"));
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform 'My platform', device 'My second device' (specified by user)"));
	}

	SECTION("when automatically selecting a device") {
		pick_devices(cfg, auto_select_devices{}, std::vector<mock_platform>{mp});
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform 'My platform', device 'My first device' (automatically selected)"));
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform 'My platform', device 'My second device' (automatically selected)"));
	}

	SECTION("when a device selector is provided") {
		const auto selector = [mds](const mock_device&) { return 100; };
		pick_devices(cfg, selector, std::vector<mock_platform>{mp});
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform 'My platform', device 'My first device' (via user-provided selector)"));
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform 'My platform', device 'My second device' (via user-provided selector)"));
	}
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "pick_devices prints information about device backend", "[device-selection]") {
	config cfg(nullptr, nullptr);

	const auto devices = sycl::device::get_devices();
	std::optional<sycl::device> generic_device;
	std::optional<sycl::device> cuda_device;

	for(const auto& d : devices) {
		if(backend::get_type(d) == backend::type::cuda) {
			cuda_device = d;
		} else {
			generic_device = d;
		}
	}

	SECTION("warns when using generic backend") {
		if(!generic_device.has_value()) {
			SKIP("No generic device available");
		} else {
			celerity::test_utils::allow_max_log_level(celerity::detail::log_level::warn);
			pick_devices(cfg, std::vector<sycl::device>{*generic_device}, std::vector<sycl::platform>{});
			CHECK(celerity::test_utils::log_contains_substring(celerity::detail::log_level::warn,
			    fmt::format("No backend specialization available for selected platform '{}', falling back to generic. Performance may be degraded.",
			        generic_device->get_platform().get_info<sycl::info::platform::name>())));
		}
	}

	SECTION("informs user when using specialized backend") {
		if(!cuda_device.has_value() || !backend_detail::is_enabled_v<backend::type::cuda>) {
			SKIP("No CUDA device available or CUDA backend not enabled");
		} else {
			pick_devices(cfg, std::vector<sycl::device>{*cuda_device}, std::vector<sycl::platform>{});
			CHECK(celerity::test_utils::log_contains_substring(celerity::detail::log_level::debug,
			    fmt::format("Using CUDA backend for selected platform '{}'.", cuda_device->get_platform().get_info<sycl::info::platform::name>())));
		}
	}

	SECTION("warns when specialized backend is not enabled") {
		if(!cuda_device.has_value() || backend_detail::is_enabled_v<backend::type::cuda>) {
			SKIP("No CUDA device available or CUDA backend is enabled");
		} else {
			celerity::test_utils::allow_max_log_level(celerity::detail::log_level::warn);
			pick_devices(cfg, std::vector<sycl::device>{*cuda_device}, std::vector<sycl::platform>{});
			CHECK(celerity::test_utils::log_contains_substring(celerity::detail::log_level::warn,
			    fmt::format("Selected platform '{}' is compatible with specialized CUDA backend, but it has not been compiled.",
			        cuda_device->get_platform().get_info<sycl::info::platform::name>())));
		}
	}
}
