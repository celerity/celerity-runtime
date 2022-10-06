#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "log_test_utils.h"
#include "test_utils.h"

using dt = sycl::info::device_type;

struct mock_platform;

struct type_and_name {
	dt type;
	std::string name;
};

struct mock_device {
	mock_device() : m_id(0), m_type(dt::gpu), m_platform(nullptr) {}

	mock_device(size_t id, mock_platform& platform, dt type) : mock_device(id, platform, {type, fmt::format("Mock device {}", id)}){};

	mock_device(size_t id, mock_platform& platform, const type_and_name& tan) : m_id(id), m_type(tan.type), m_name(tan.name), m_platform(&platform) {}

	bool operator==(const mock_device& other) const { return other.m_id == m_id; }

	mock_platform& get_platform() const {
		assert(m_platform != nullptr);
		return *m_platform;
	}

#if CELERITY_WORKAROUND(HIPSYCL) || CELERITY_WORKAROUND(COMPUTECPP) // old API: device enum
	template <sycl::info::device Param>
	auto get_info() const {
		if constexpr(Param == sycl::info::device::name) { return m_name; }
		if constexpr(Param == sycl::info::device::device_type) { return m_type; }
	}
#else // new API: device tag type
	template <typename Param>
	auto get_info() const {
		if constexpr(std::is_same_v<Param, sycl::info::device::name>) { return m_name; }
		if constexpr(std::is_same_v<Param, sycl::info::device::device_type>) { return m_type; }
	}
#endif

	dt get_type() const { return m_type; }

	size_t get_id() const { return m_id; }

  private:
	size_t m_id;
	dt m_type;
	std::string m_name;
	mock_platform* m_platform;
};

struct mock_platform_factory {
  public:
	template <typename... Args>
	auto create_platforms(Args... args) {
		return std::array<mock_platform, sizeof...(args)>{mock_platform(m_next_id++, args)...};
	}

  private:
	size_t m_next_id = 0;
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

#if CELERITY_WORKAROUND(HIPSYCL) || CELERITY_WORKAROUND(COMPUTECPP) // old API: platform enum
	template <sycl::info::platform Param>
#else // new API: platform tag type
	template <typename Param>
#endif
	std::string get_info() const {
		return m_name;
	}

	bool operator==(const mock_platform& other) const { return other.m_id == m_id; }
	bool operator!=(const mock_platform& other) const { return !(*this == other); }

	size_t get_id() const { return m_id; }

  private:
	size_t m_id;
	std::string m_name;
	size_t m_next_device_id = 0;
	std::vector<mock_device> m_devices;
};

namespace celerity::detail {
struct config_testspy {
	static void set_mock_device_cfg(config& cfg, const device_config& d_cfg) { cfg.m_device_cfg = d_cfg; }
	static void set_mock_host_cfg(config& cfg, const host_config& h_cfg) { cfg.m_host_cfg = h_cfg; }
};
} // namespace celerity::detail

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device prefers user specified device pointer", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);
	mock_platform_factory mpf;

	auto [mp] = mpf.create_platforms(std::nullopt);
	auto md = mp.create_devices(dt::gpu)[0];

	auto device = pick_device(cfg, md, std::vector<mock_platform>{mp});
	CHECK(device == md);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture,
    "pick_device automatically selects a gpu device if available, otherwise falls back to the first device available", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);
	mock_platform_factory mpf;

	auto dv_type_1 = GENERATE(as<dt>(), dt::gpu, dt::accelerator, dt::cpu, dt::custom, dt::host);
	CAPTURE(dv_type_1);

	auto dv_type_2 = GENERATE(as<dt>(), dt::gpu, dt::accelerator, dt::cpu, dt::custom, dt::host);
	CAPTURE(dv_type_2);

	auto [mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt);

	auto md_1 = mp_1.create_devices(dv_type_1)[0];
	auto md_2 = mp_2.create_devices(dv_type_2)[0];

	auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_1, mp_2});
	if(dv_type_1 == dt::gpu || (dv_type_1 != dt::gpu && dv_type_2 != dt::gpu)) {
		CHECK(device == md_1);
	} else {
		CHECK(device == md_2);
	}
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device selects device using CELERITY_DEVICES", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);
	mock_platform_factory mpf;

	auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu, dt::gpu);
	auto md = mp_1.create_devices(dt::gpu, dt::gpu, dt::cpu)[1];

	celerity::detail::device_config d_cfg{1, 1};
	celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

	auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1});
	CHECK(device == md);
}

TEST_CASE_METHOD(
    celerity::test_utils::mpi_fixture, "pick_device attempts to select a unique device from a single platform for each local node", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);

	SECTION("preferring GPUs over other device types") {
		mock_platform_factory mpf;

		const size_t node_count = 4;
		const size_t local_rank = 3;

		auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
		mp_0.create_devices(dt::cpu);
		auto md = mp_1.create_devices(dt::gpu, dt::gpu, dt::gpu, dt::gpu)[local_rank];
		mp_2.create_devices(dt::accelerator, dt::accelerator, dt::accelerator, dt::accelerator);

		celerity::detail::host_config h_cfg{node_count, local_rank};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1, mp_2});
		CHECK(device == md);
	}

	SECTION("falling back to other device types when an insufficient number of GPUs is available") {
		mock_platform_factory mpf;

		const size_t node_count = 4;
		const size_t local_rank = 2;

		auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(dt::gpu, dt::gpu, dt::gpu);
		auto md = mp_2.create_devices(dt::accelerator, dt::accelerator, dt::accelerator, dt::accelerator)[local_rank];

		celerity::detail::host_config h_cfg{node_count, local_rank};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1, mp_2});
		CHECK(device == md);
	}

	SECTION("falling back to a single GPU for all nodes if an insufficient number of GPUs and other device types is available") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
		mp_0.create_devices(dt::cpu);
		auto md = mp_1.create_devices(dt::gpu)[0];
		mp_2.create_devices(dt::accelerator, dt::accelerator, dt::accelerator);

		const size_t node_count = 4;
		const size_t local_rank = GENERATE(0, 3);

		celerity::detail::host_config h_cfg{node_count, local_rank};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1, mp_2});
		CHECK(device == md);
	}

	SECTION("falling back to a single device of any type for all nodes if an insufficient number of GPUs or other device types is available") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, std::nullopt);
		auto md = mp_0.create_devices(dt::cpu)[0];
		mp_1.create_devices(dt::accelerator, dt::accelerator, dt::accelerator);

		const size_t node_count = 4;
		const size_t local_rank = GENERATE(0, 3);

		celerity::detail::host_config h_cfg{node_count, local_rank};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1});
		CHECK(device == md);
	}
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device prints expected info/warn messages", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);
	SECTION("when device pointer is specified by user") {
		mock_platform tp(68, "My platform");
		auto td = tp.create_devices(dt::gpu, type_and_name{dt::gpu, "My device"}, dt::gpu)[1];

		celerity::test_utils::log_capture lc;
		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'My platform', device 'My device' (specified by user)"));
	}

	SECTION("when CELERITY_DEVICES is set") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, "My platform");
		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(dt::gpu, type_and_name{dt::gpu, "My device"});

		celerity::detail::device_config d_cfg{1, 1};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

		celerity::test_utils::log_capture lc;
		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1});
		CHECK_THAT(lc.get_log(),
		    Catch::Matchers::ContainsSubstring("Using platform 'My platform', device 'My device' (set by CELERITY_DEVICES: platform 1, device 1)"));
	}

	SECTION("when automatically selecting a device") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, "My platform");
		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(type_and_name{dt::gpu, "My device"}, dt::gpu);

		celerity::test_utils::log_capture lc;
		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1});
		CHECK_THAT(
		    lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'My platform', device 'My device' (automatically selected platform 1, device 0)"));
	}

	SECTION("when it can't find a platform with a sufficient number of GPUs") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(dt::gpu, dt::gpu, dt::gpu);
		mp_2.create_devices(dt::accelerator, dt::accelerator, dt::accelerator, dt::accelerator);

		const size_t node_count = 4;
		const size_t local_rank = 3;

		celerity::detail::host_config h_cfg{node_count, local_rank};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		celerity::test_utils::log_capture lc(spdlog::level::warn);
		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1, mp_2});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 GPU devices, and CELERITY_DEVICES not set"));
	}

	SECTION("when it can't find a platform with a sufficient number of devices of any type") {
		mock_platform_factory mpf;
		auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);

		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(dt::gpu);
		mp_2.create_devices(dt::accelerator, dt::accelerator, dt::accelerator);

		const size_t node_count = 4;
		const size_t local_rank = 3;

		celerity::detail::host_config h_cfg{node_count, local_rank};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		celerity::test_utils::log_capture lc(spdlog::level::warn);
		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1, mp_2});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 devices, and CELERITY_DEVICES not set"));
	}

	SECTION("when CELERITY_DEVICES contains an invalid platform id") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, std::nullopt);
		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(dt::gpu, dt::gpu);

		celerity::detail::device_config d_cfg{3, 0};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);
		CHECK_THROWS_WITH(pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1}),
		    "Invalid platform id 3: Only 2 platforms available");
	}

	SECTION("when CELERITY_DEVICES contains an invalid device id") {
		mock_platform_factory mpf;

		auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, std::nullopt);
		mp_0.create_devices(dt::cpu);
		mp_1.create_devices(dt::gpu, dt::gpu);

		celerity::detail::device_config d_cfg{1, 5};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

		CHECK_THROWS_WITH(pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{mp_0, mp_1}),
		    "Invalid device id 5: Only 2 devices available on platform 1");
	}

	SECTION("when no device was selected") {
		CHECK_THROWS_WITH(
		    pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{}), "Automatic device selection failed: No device available");
	}
}

// The following test doesn't work with ComputeCpp backend, since the == operator behaves differently
#if !CELERITY_WORKAROUND(COMPUTECPP)
TEST_CASE_METHOD(celerity::test_utils::runtime_fixture, "pick_device supports passing a device selector function", "[device-selection]") {
	std::vector<sycl::device> devices = sycl::device::get_devices();
	if(devices.size() < 2) {
		WARN("Platforms must have 2 or more devices!");
		return;
	}

	auto device_idx = GENERATE(0, 1);
	CAPTURE(device_idx);
	sycl::device device = devices[device_idx];
	CAPTURE(device);

	auto device_selector = [device](const sycl::device& d) -> int { return d == device ? 2 : 1; };

	celerity::distr_queue q(device_selector);

	auto& dq = celerity::detail::runtime::get_instance().get_device_queue();
	CHECK(dq.get_sycl_queue().get_device() == device);
}
#endif

TEST_CASE("pick_device correctly selects according to device selector score", "[device-selection]") {
	mock_platform_factory mpf;

	auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu);
	mp_1.create_devices(dt::gpu, dt::gpu, dt::gpu, dt::gpu);
	auto md = mp_2.create_devices(dt::accelerator, dt::accelerator, dt::accelerator, dt::accelerator)[1];

	auto device_selector = [md](const mock_device& d) -> int { return d == md ? 2 : 1; };

	celerity::detail::config cfg(nullptr, nullptr);
	auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{mp_0, mp_1, mp_2});
	CHECK(device == md);
}

TEST_CASE("pick_device selects a unique device for each local node according to device selector score", "[device-selection]") {
	mock_platform_factory mpf;

	auto [mp_0, mp_1] = mpf.create_platforms(std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu);
	mp_1.create_devices(dt::gpu, dt::gpu, dt::gpu);

	mock_platform mp_2(2, "My platform");
	auto md = mp_2.create_devices(dt::gpu, dt::gpu, dt::accelerator, dt::accelerator, dt::accelerator, type_and_name{dt::accelerator, "My device"})[5];

	const size_t node_count = 4;
	const size_t local_rank = 3;

	celerity::detail::host_config h_cfg{node_count, local_rank};
	celerity::detail::config cfg(nullptr, nullptr);
	celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

	auto device_selector = [](const mock_device& d) -> int { return d.get_type() == dt::accelerator ? 2 : 1; };

	celerity::test_utils::log_capture lc;
	auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{mp_0, mp_1, mp_2});
	CHECK_THAT(
	    lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'My platform', device 'My device' (device selector specified: platform 2, device 3)"));
	CHECK(device == md);
}

TEST_CASE("pick_device selects the highest scoring device for all nodes if an insufficient number of total devices is available", "[device-selection]") {
	mock_platform_factory mpf;

	auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu);
	mp_1.create_devices(dt::gpu);
	auto md = mp_2.create_devices(dt::accelerator)[0];

	const size_t node_count = 4;
	const size_t local_rank = 3;

	celerity::detail::host_config h_cfg{node_count, local_rank};
	celerity::detail::config cfg(nullptr, nullptr);
	celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);
	auto device_selector = [md](const mock_device& d) -> int { return d == md ? 2 : 1; };

	celerity::test_utils::log_capture lc(spdlog::level::warn);
	auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{mp_0, mp_1, mp_2});
	CHECK_THAT(
	    lc.get_log(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 devices that match the specified device selector"));
	CHECK(device == md);
}

TEST_CASE("pick_device warns when highest scoring devices span multiple platforms", "[device-selection]") {
	mock_platform_factory mpf;

	auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu);
	mp_1.create_devices(dt::accelerator, dt::gpu, dt::gpu, dt::gpu);
	mp_2.create_devices(dt::gpu, dt::accelerator, dt::accelerator, dt::accelerator);

	const size_t node_count = 4;
	const size_t local_rank = GENERATE(0, 3);

	celerity::detail::host_config h_cfg{node_count, local_rank};
	celerity::detail::config cfg(nullptr, nullptr);
	celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

	auto device_selector = [](const mock_device& d) -> int { return d.get_type() == dt::accelerator ? 2 : 1; };

	celerity::test_utils::log_capture lc(spdlog::level::warn);
	auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{mp_0, mp_1, mp_2});
	INFO("Platform id" << device.get_platform().get_id() << " device id " << device.get_id());
	CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Selected devices are of different type and/or do not belong to the same platform"));

	if(local_rank == 0) {
		CHECK(device.get_platform() == mp_1);
	} else {
		CHECK(device.get_platform() == mp_2);
	}
}

TEST_CASE("pick_device warns when highest scoring devices are of different types", "[device-selection]") {
	mock_platform_factory mpf;

	auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu);
	auto md = mp_1.create_devices(dt::accelerator, dt::gpu, dt::gpu, dt::gpu)[0];
	mp_2.create_devices(dt::accelerator, dt::gpu, dt::gpu, dt::gpu);

	const size_t node_count = 4;
	const size_t local_rank = 0;

	celerity::detail::host_config h_cfg{node_count, local_rank};
	celerity::detail::config cfg(nullptr, nullptr);
	celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

	auto device_selector = [](const mock_device& d) -> int { return d.get_type() == dt::accelerator ? 2 : 1; };

	celerity::test_utils::log_capture lc(spdlog::level::warn);
	auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{mp_0, mp_1, mp_2});
	INFO("Platform id" << device.get_platform().get_id() << " device id " << device.get_id());
	CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Selected devices are of different type and/or do not belong to the same platform"));
	CHECK(device == md);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device does not consider devices with a negative selector score", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);
	mock_platform_factory mpf;

	auto [mp_0, mp_1, mp_2] = mpf.create_platforms(std::nullopt, std::nullopt, std::nullopt);
	mp_0.create_devices(dt::cpu);
	mp_1.create_devices(dt::gpu, dt::gpu, dt::gpu);
	mp_2.create_devices(dt::gpu);

	const size_t node_count = 4;
	const size_t local_rank = 2;

	celerity::detail::host_config h_cfg{node_count, local_rank};
	celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

	auto device_selector = [](const mock_device& d) -> int { return d.get_type() == dt::accelerator ? 1 : -1; };
	CHECK_THROWS_WITH(
	    pick_device(cfg, device_selector, std::vector<mock_platform>{mp_0, mp_1, mp_2}), "Device selection with device selector failed: No device available");
}
