#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "select_devices.h"
#include "test_utils.h"

using dt = sycl::info::device_type;
using namespace celerity;
using namespace celerity::detail;

struct mock_platform;

struct type_and_name {
	dt type;
	std::string name;
};

enum class mock_device_backend { foo, bar, qux };

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

	mock_device_backend get_backend() const;

	bool has(const sycl::aspect aspect) const { return m_pimpl->aspects.count(aspect) > 0 ? m_pimpl->aspects.at(aspect) : true; }

	void set_aspect(const sycl::aspect aspect, const bool value) { m_pimpl->aspects[aspect] = value; }

	size_t hash() const { return std::hash<impl*>{}(m_pimpl.get()); }

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

template <>
struct std::hash<mock_device> {
	size_t operator()(const mock_device& dev) const { return dev.hash(); }
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

	mock_device_backend get_backend() const { return m_backend; }
	void set_backend(mock_device_backend backend) { m_backend = backend; }

  private:
	size_t m_id;
	std::string m_name;
	size_t m_next_device_id = 0;
	std::vector<mock_device> m_devices;
	mock_device_backend m_backend = mock_device_backend::foo;
};

mock_device_backend mock_device::get_backend() const { return m_pimpl->platform->get_backend(); }

template <typename... Args>
auto create_mock_platforms(Args... args) {
	size_t next_id = 0;
	return std::array<mock_platform, sizeof...(args)>{mock_platform(next_id++, args)...};
}

TEST_CASE("check_required_device_aspects throws if a device does not support required aspects", "[device-selection]") {
	mock_device device;
	CHECK_NOTHROW(check_required_device_aspects(device));

	// Note: This assumes that the following checks are performed in reverse order within check_required_device_aspects

	device.set_aspect(sycl::aspect::usm_host_allocations, false);
	CHECK_THROWS_WITH(check_required_device_aspects(device), "device does not support USM host allocations");

	device.set_aspect(sycl::aspect::usm_device_allocations, false);
	CHECK_THROWS_WITH(check_required_device_aspects(device), "device does not support USM device allocations");
}

TEST_CASE("select_devices prefers user-specified device list", "[device-selection]") {
	test_utils::allow_max_log_level(log_level::warn);

	const host_config h_cfg{1, 0};
	auto [mp] = create_mock_platforms(std::nullopt);

	CHECK_THROWS_WITH(select_devices(h_cfg, std::vector<mock_device>{}, std::vector<mock_platform>{mp}),
	    "Device selection failed: The user-provided list of devices is empty");

	const auto devices = mp.create_devices(dt::gpu, dt::gpu, dt::cpu, dt::accelerator);
	const auto selected = select_devices(h_cfg, std::vector<mock_device>{devices.begin(), devices.end()}, std::vector<mock_platform>{mp});
	CHECK(selected == std::vector<mock_device>{devices.begin(), devices.end()});
}

TEST_CASE("select_devices requires user-specified devices to have the same backend", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp_1, mp_2] = create_mock_platforms(std::nullopt, std::nullopt);
	auto [md_1] = mp_1.create_devices(dt::gpu);
	auto [md_2] = mp_2.create_devices(dt::gpu);

	CHECK_NOTHROW(select_devices(h_cfg, std::vector<mock_device>{md_1, md_2}, std::vector<mock_platform>{mp_1, mp_2}));

	mp_1.set_backend(mock_device_backend::foo);
	mp_2.set_backend(mock_device_backend::bar);
	CHECK_THROWS_WITH(select_devices(h_cfg, std::vector<mock_device>{md_1, md_2}, std::vector<mock_platform>{mp_1, mp_2}),
	    "Device selection failed: The user-provided list of devices contains devices from different backends");
}

TEST_CASE("select_devices throws if a user-specified devices does not support required aspects", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp] = create_mock_platforms(std::nullopt);
	auto [md_1, md_2] = mp.create_devices(dt::gpu, dt::gpu);
	md_2.set_aspect(sycl::aspect::usm_device_allocations, false);
	CHECK_THROWS_WITH(select_devices(h_cfg, std::vector<mock_device>{md_1, md_2}, std::vector<mock_platform>{mp}),
	    "Device selection failed: Device 1 in user-provided list of devices caused error: device does not support USM device allocations");
}

TEST_CASE("select_devices selects the largest subset of GPUs that share the same backend", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_1.create_devices(dt::gpu);
	auto [md_2, md_3, md_4] = mp_2.create_devices(dt::gpu, dt::gpu, dt::gpu);
	mp_3.create_devices(dt::gpu, dt::gpu);

	mp_1.set_backend(mock_device_backend::foo);
	mp_2.set_backend(mock_device_backend::bar);
	mp_3.set_backend(mock_device_backend::qux);

	const auto selected = select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected == std::vector<mock_device>{md_2, md_3, md_4});
}

TEST_CASE("select_devices falls back to other device types if no GPUs are available", "[device-selection]") {
	test_utils::allow_max_log_level(log_level::warn);

	const host_config h_cfg{1, 0};
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_1.create_devices(dt::cpu);
	auto [md_2, md_3, md_4] = mp_2.create_devices(dt::accelerator, dt::cpu, dt::host);
	mp_3.create_devices(dt::cpu, dt::accelerator);

	mp_1.set_backend(mock_device_backend::foo);
	mp_2.set_backend(mock_device_backend::bar);
	mp_3.set_backend(mock_device_backend::qux);

	const auto selected_1 = select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected_1 == std::vector<mock_device>{md_2, md_3, md_4});

	// Once there is a GPU however, it takes precedence
	auto [md_9] = mp_3.create_devices(dt::gpu);
	const auto selected_2 = select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected_2 == std::vector<mock_device>{md_9});
}

TEST_CASE("select_devices only considers devices that support required aspects", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto type = GENERATE(dt::gpu, dt::cpu);
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_1.create_devices(type);
	auto [md_2, md_3, md_4] = mp_2.create_devices(type, type, type);
	auto [md_5, md_6] = mp_3.create_devices(type, type);

	mp_1.set_backend(mock_device_backend::foo);
	mp_2.set_backend(mock_device_backend::bar);
	mp_3.set_backend(mock_device_backend::qux);

	md_2.set_aspect(sycl::aspect::usm_device_allocations, false);
	md_3.set_aspect(sycl::aspect::usm_device_allocations, false);

	const auto selected = select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp_1, mp_2, mp_3});
	CHECK(selected == std::vector<mock_device>{md_5, md_6});
}

TEST_CASE("select_devices supports passing a device selector function", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp_1, mp_2] = create_mock_platforms(std::nullopt, std::nullopt);
	auto [md_1] = mp_1.create_devices(dt::gpu);
	auto [md_2] = mp_2.create_devices(dt::gpu);

	mp_1.set_backend(mock_device_backend::foo);
	mp_2.set_backend(mock_device_backend::bar);

	const auto device_idx = GENERATE(0, 1);
	CAPTURE(device_idx);
	const auto to_select = std::array<mock_device, 2>{md_1, md_2}[device_idx];

	auto device_selector = [to_select](const mock_device& d) { return d == to_select ? 2 : 1; };
	auto selected = select_devices(h_cfg, device_selector, std::vector<mock_platform>{mp_1, mp_2});
	CHECK(selected == std::vector<mock_device>{to_select});
}

TEST_CASE("select_devices selects the subset of devices with the largest cumulative selector score sharing the same backend", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp_1, mp_2, mp_3] = create_mock_platforms(std::nullopt, std::nullopt, std::nullopt);
	auto [md_1] = mp_1.create_devices(dt::gpu);
	auto [md_2, md_3, md_4] = mp_2.create_devices(dt::cpu, dt::gpu, dt::accelerator);
	auto [md_5, md_6] = mp_3.create_devices(dt::host, dt::cpu);

	mp_1.set_backend(mock_device_backend::foo);
	mp_2.set_backend(mock_device_backend::bar);
	mp_3.set_backend(mock_device_backend::qux);

	const auto ignore_md_4 = GENERATE(true, false);
	const auto md_6_no_usm = GENERATE(true, false);

	md_6.set_aspect(sycl::aspect::usm_device_allocations, !md_6_no_usm);

	const std::vector<std::vector<int>> scores = {
	    {/* md_1 */ 50}, {/* md_2 */ 10, /* md_3 */ 20, /* md_4 */ ignore_md_4 ? -1 : 30}, {/* md_5 */ 30, /* md_6 */ 40}};
	const auto selector = [&scores](const mock_device& d) { return scores[d.get_platform().get_id()][d.get_id()]; };

	const auto selected = select_devices(h_cfg, selector, std::vector<mock_platform>{mp_1, mp_2, mp_3});

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

TEST_CASE("select_devices does not consider devices with a negative selector score", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp] = create_mock_platforms(std::nullopt);
	auto [md_1, md_2, md_3] = mp.create_devices(dt::gpu, dt::gpu, dt::gpu);

	auto selector = [md_2 = md_2](const mock_device& d) { return d.get_id() == md_2.get_id() ? -1 : 1; };
	const auto selected = select_devices(h_cfg, selector, std::vector<mock_platform>{mp});
	CHECK(selected == std::vector<mock_device>{md_1, md_3});
}

TEST_CASE("select_devices throws if no devices are available", "[device-selection]") {
	const host_config h_cfg{1, 0};
	auto [mp] = create_mock_platforms(std::nullopt);

	SECTION("from the start") {
		const auto selector = [](const mock_device&) { return -1; };
		CHECK_THROWS_WITH(select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp}), "Device selection failed: No devices available");
		CHECK_THROWS_WITH(select_devices(h_cfg, selector, std::vector<mock_platform>{mp}), "Device selection failed: No devices available");
	}

	SECTION("if all are ignored due to missing aspects") {
		auto [md_1] = mp.create_devices(dt::gpu);
		md_1.set_aspect(sycl::aspect::usm_device_allocations, false);
		CHECK_THROWS_WITH(select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp}), "Device selection failed: No eligible devices found");
		auto [md_2] = mp.create_devices(dt::cpu);
		md_2.set_aspect(sycl::aspect::usm_device_allocations, false);
		CHECK_THROWS_WITH(select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp}), "Device selection failed: No eligible devices found");
	}

	SECTION("if all are discarded by selector") {
		auto [md_1] = mp.create_devices(dt::gpu);
		const auto selector = [](const mock_device&) { return -1; };
		CHECK_THROWS_WITH(select_devices(h_cfg, selector, std::vector<mock_platform>{mp}), "Device selection failed: No eligible devices found");
	}
}

TEST_CASE("select_devices attempts to evenly distributed devices if there is more than one local node", "[device-selection]") {
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

	test_utils::allow_max_log_level(log_level::warn);

	for(const auto& dist : distributions) {
		auto [mp] = create_mock_platforms(std::nullopt);
		for(size_t i = 0; i < dist.num_devices; ++i) {
			mp.create_devices(dt::gpu);
		}

		std::unordered_set<std::pair<size_t, size_t>, utils::pair_hash> unique_devices;
		for(size_t i = 0; i < dist.devices_per_node.size(); ++i) {
			const host_config h_cfg{dist.devices_per_node.size(), i};
			const auto selected = select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp});
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

TEST_CASE("select_devices distributes devices in round-robin fashion if there are fewer than local nodes", "[device-selection]") {
	auto [mp] = create_mock_platforms(std::nullopt);
	auto devices = mp.create_devices(dt::gpu, dt::gpu, dt::gpu);

	const size_t num_ranks = GENERATE(4, 5, 6);

	test_utils::allow_max_log_level(log_level::warn);

	for(size_t i = 0; i < num_ranks; ++i) {
		const host_config h_cfg{num_ranks, i};
		const auto selected = select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp});
		REQUIRE(selected.size() == 1);
		CHECK(selected[0] == devices[i % 3]);
		CHECK(test_utils::log_contains_exact(
		    log_level::warn, fmt::format("Found fewer devices (3) than local nodes ({}), multiple nodes will use the same device(s).", num_ranks)));
	}
}

TEST_CASE("select_devices prints device and platform information", "[device-selection]") {
	const host_config h_cfg{1, 0};
	mock_platform mp(68, "My platform");
	auto mds = mp.create_devices(type_and_name{dt::gpu, "My first device"}, type_and_name{dt::gpu, "My second device"});

	SECTION("when devices are provided by user") {
		select_devices(h_cfg, std::vector<mock_device>{mds[0], mds[1]}, std::vector<mock_platform>{mp});
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform \"My platform\", device \"My first device\" as D0 (specified by user)"));
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform \"My platform\", device \"My second device\" as D1 (specified by user)"));
	}

	SECTION("when automatically selecting a device") {
		select_devices(h_cfg, auto_select_devices{}, std::vector<mock_platform>{mp});
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform \"My platform\", device \"My first device\" as D0 (automatically selected)"));
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform \"My platform\", device \"My second device\" as D1 (automatically selected)"));
	}

	SECTION("when a device selector is provided") {
		const auto selector = [mds](const mock_device&) { return 100; };
		select_devices(h_cfg, selector, std::vector<mock_platform>{mp});
		CHECK(test_utils::log_contains_exact(log_level::info, "Using platform \"My platform\", device \"My first device\" as D0 (via user-provided selector)"));
		CHECK(
		    test_utils::log_contains_exact(log_level::info, "Using platform \"My platform\", device \"My second device\" as D1 (via user-provided selector)"));
	}
}

enum class mock_backend_type { generic1, generic2, specialized1, specialized2 };

template <>
struct fmt::formatter<mock_backend_type> : fmt::formatter<std::string_view> {
	format_context::iterator format(const mock_backend_type type, format_context& ctx) const {
		const auto repr = [=]() -> std::string_view {
			switch(type) {
			case mock_backend_type::generic1: return "generic#1";
			case mock_backend_type::generic2: return "generic#2";
			case mock_backend_type::specialized1: return "SPECIALIZED#1";
			case mock_backend_type::specialized2: return "SPECIALIZED#2";
			default: utils::unreachable();
			}
		}();
		return std::copy(repr.begin(), repr.end(), ctx.out());
	}
};

struct mock_backend_enumerator {
	using backend_type = mock_backend_type;
	using device_type = mock_device;

	std::vector<backend_type> available;
	std::unordered_map<mock_device, std::vector<backend_type>> compatible;

	std::vector<backend_type> compatible_backends(const mock_device& device) const { return compatible.at(device); }

	std::vector<backend_type> available_backends() const { return available; }

	bool is_specialized(backend_type type) const { return type == mock_backend_type::specialized1 || type == mock_backend_type::specialized2; }

	int get_priority(backend_type type) const {
		switch(type) {
		case mock_backend_type::generic1: return 1;
		case mock_backend_type::generic2: return 0;
		case mock_backend_type::specialized1: return 3;
		case mock_backend_type::specialized2: return 2;
		default: utils::unreachable();
		}
	}
};

TEST_CASE("select_backend picks highest-priority available specialized backend", "[device-selection]") {
	test_utils::allow_max_log_level(log_level::warn);

	mock_platform platform(0, "platform");
	std::vector<mock_device> devices{
	    mock_device(0, platform, type_and_name{sycl::info::device_type::gpu, "gpu0"}),
	    mock_device(1, platform, type_and_name{sycl::info::device_type::gpu, "gpu1"}),
	};
	const mock_backend_enumerator enumerator{{mock_backend_type::generic1, mock_backend_type::generic2, mock_backend_type::specialized2},
	    {
	        {devices.at(0), {mock_backend_type::generic1, mock_backend_type::specialized1, mock_backend_type::specialized2}},
	        {devices.at(1), {mock_backend_type::generic1, mock_backend_type::specialized1, mock_backend_type::specialized2}},
	    }};

	auto backend = select_backend(enumerator, devices);
	CHECK(backend == mock_backend_type::specialized2);
	CHECK(test_utils::log_contains_exact(log_level::warn,
	    fmt::format("All selected devices are compatible with specialized {} backend, but it has not been compiled. Performance may be degraded.",
	        mock_backend_type::specialized1)));
	CHECK(test_utils::log_contains_exact(log_level::debug, fmt::format("Using {} backend for the selected devices.", mock_backend_type::specialized2)));
}

TEST_CASE("select_backend picks highest-priority available generic backend if there is no common specialization", "[device-selection]") {
	test_utils::allow_max_log_level(log_level::warn);

	mock_platform platform(0, "platform");
	std::vector<mock_device> devices{
	    mock_device(0, platform, type_and_name{sycl::info::device_type::gpu, "gpu0"}),
	    mock_device(1, platform, type_and_name{sycl::info::device_type::gpu, "gpu1"}),
	};
	const mock_backend_enumerator enumerator{
	    {mock_backend_type::generic1, mock_backend_type::generic2, mock_backend_type::specialized1, mock_backend_type::specialized2},
	    {
	        {devices.at(0), {mock_backend_type::generic1, mock_backend_type::generic2, mock_backend_type::specialized1}},
	        {devices.at(1), {mock_backend_type::generic1, mock_backend_type::generic2, mock_backend_type::specialized2}},
	    }};

	auto backend = select_backend(enumerator, devices);
	CHECK(backend == mock_backend_type::generic1);
	CHECK(test_utils::log_contains_exact(
	    log_level::warn, fmt::format("No common backend specialization available for all selected devices, falling back to {}. Performance may be degraded.",
	                         mock_backend_type::generic1)));
}

TEST_CASE("select_backend picks a generic backend if no compatible specialization was compiled", "[device-selection]") {
	test_utils::allow_max_log_level(log_level::warn);

	mock_platform platform(0, "platform");
	std::vector<mock_device> devices{
	    mock_device(0, platform, type_and_name{sycl::info::device_type::gpu, "gpu0"}),
	    mock_device(1, platform, type_and_name{sycl::info::device_type::gpu, "gpu1"}),
	};
	const mock_backend_enumerator enumerator{{mock_backend_type::generic1, mock_backend_type::generic2},
	    {
	        {devices.at(0), {mock_backend_type::generic2, mock_backend_type::specialized1, mock_backend_type::specialized2}},
	        {devices.at(1), {mock_backend_type::generic2, mock_backend_type::specialized1, mock_backend_type::specialized2}},
	    }};

	auto backend = select_backend(enumerator, devices);
	CHECK(backend == mock_backend_type::generic2);
	CHECK(test_utils::log_contains_exact(log_level::warn,
	    fmt::format("All selected devices are compatible with specialized {} backend, but it has not been compiled. Performance may be degraded.",
	        mock_backend_type::specialized1)));
	CHECK(test_utils::log_contains_exact(log_level::warn,
	    fmt::format("All selected devices are compatible with specialized {} backend, but it has not been compiled. Performance may be degraded.",
	        mock_backend_type::specialized2)));
	CHECK(test_utils::log_contains_exact(
	    log_level::warn, fmt::format("No common backend specialization available for all selected devices, falling back to {}. Performance may be degraded.",
	                         mock_backend_type::generic2)));
}
