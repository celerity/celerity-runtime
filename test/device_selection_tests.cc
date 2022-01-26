#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"
#include "spdlog/sinks/ostream_sink.h"
#include "test_utils.h"
#include <celerity.h>

struct mock_platform;
struct mock_device {
	mock_device() : platform(nullptr), id(0), type(sycl::info::device_type::gpu) {}

	mock_device(size_t id, mock_platform& platform, sycl::info::device_type type = sycl::info::device_type::gpu) : platform(&platform), id(id), type(type) {}

	mock_platform& get_platform() const { return *platform; }

	template <sycl::info::device Param>
	auto get_info() const {
		if constexpr(Param == sycl::info::device::name) { return name; }
		if constexpr(Param == sycl::info::device::device_type) { return type; }
	}

	bool operator==(const mock_device& other) const { return other.id == id; }

	sycl::info::device_type get_type() const { return type; }

	size_t get_id() { return id; }

  private:
	mock_platform* platform;
	std::string name = "bar";
	size_t id;
	sycl::info::device_type type;
};
struct mock_platform {
	mock_platform(size_t id) : id(id) {}

	void set_devices(std::vector<mock_device> devices) { this->devices = devices; }

	std::vector<mock_device> get_devices(sycl::info::device_type type = sycl::info::device_type::all) const {
		if(type != sycl::info::device_type::all) {
			std::vector<mock_device> devices_with_type;
			for(auto device : devices) {
				if(device.get_type() == type) { devices_with_type.emplace_back(device); }
			}
			return devices_with_type;
		} else
			return devices;
	}

	template <sycl::info::platform Param>
	std::string get_info() const {
		return name;
	}

	void set_info(std::string name) { this->name = name; }

	bool operator!=(const mock_platform& other) const { return other.id != id; }

	size_t get_id() { return id; }

  private:
	std::vector<mock_device> devices;
	size_t id;
	std::string name = "foo";
};


namespace celerity::detail {
struct config_testspy {
	static void set_mock_device_cfg(config& cfg, const device_config& d_cfg) { cfg.device_cfg = d_cfg; }
	static void set_mock_host_cfg(config& cfg, const host_config& h_cfg) { cfg.host_cfg = h_cfg; }
};
} // namespace celerity::detail

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device prefers user specified device pointer", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);

	mock_platform tp(68);
	mock_device td(42, tp);
	tp.set_devices({td});

	auto device = pick_device(cfg, td, std::vector<mock_platform>{tp});
	CHECK(device == td);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture,
    "pick_device automatically selects a gpu device if available and otherwise falls back to the first device available", "[device-selection]") {
	celerity::detail::config cfg(nullptr, nullptr);

	using device_t = sycl::info::device_type;

	auto dv_type_1 = GENERATE(as<sycl::info::device_type>(), device_t::gpu, device_t::accelerator, device_t::cpu, device_t::custom, device_t::host);
	CAPTURE(dv_type_1);

	mock_platform tp_1(0);
	mock_device td_1(0, tp_1, dv_type_1);
	tp_1.set_devices({td_1});

	auto dv_type_2 = GENERATE(as<sycl::info::device_type>(), device_t::gpu, device_t::accelerator, device_t::cpu, device_t::custom, device_t::host);
	CAPTURE(dv_type_2);

	mock_platform tp_2(1);
	mock_device td_2(1, tp_2, dv_type_2);
	tp_2.set_devices({td_2});

	auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_1, tp_2});
	if(dv_type_1 == device_t::gpu || (dv_type_1 != device_t::gpu && dv_type_2 != device_t::gpu)) {
		CHECK(device == td_1);
	} else {
		CHECK(device == td_2);
	}
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device selects device using CELERITY_DEVICES", "[device-selection][device-cfg]") {
	celerity::detail::config cfg(nullptr, nullptr);

	mock_platform tp_0(0);
	mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
	tp_0.set_devices({td_1});

	mock_platform tp_1(1);
	mock_device td_2(0, tp_1, sycl::info::device_type::gpu);
	mock_device td_3(1, tp_1, sycl::info::device_type::gpu);
	tp_1.set_devices({td_2, td_3});

	celerity::detail::device_config d_cfg{tp_1.get_id(), td_3.get_id()};
	celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

	auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1});
	CHECK(device == td_3);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture,
    "pick_device selects a GPU for each local_rank or falls back to any type of sufficient device for all ranks", "[device-selection][host-cfg]") {
	celerity::detail::config cfg(nullptr, nullptr);

	SECTION("pick_device unique GPU per node") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4, td_5});

		size_t node_count = 4;
		size_t local_rank = 2;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1});
		CHECK(device == td_4);
	}

	SECTION("pick_device prefers unique GPU over other devices") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4, td_5});

		mock_platform tp_2(1);
		mock_device td_6(5, tp_2, sycl::info::device_type::accelerator);
		mock_device td_7(6, tp_2, sycl::info::device_type::accelerator);
		mock_device td_8(7, tp_2, sycl::info::device_type::accelerator);
		mock_device td_9(8, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_6, td_7, td_8, td_9});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK(device == td_5);
	}

	SECTION("pick_device falls back to other devices with insufficient GPUs") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4});

		mock_platform tp_2(2);
		mock_device td_5(4, tp_2, sycl::info::device_type::accelerator);
		mock_device td_6(5, tp_2, sycl::info::device_type::accelerator);
		mock_device td_7(6, tp_2, sycl::info::device_type::accelerator);
		mock_device td_8(7, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_5, td_6, td_7, td_8});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK(device == td_8);
	}

	SECTION("pick_device prefers the first available GPU with insufficient GPUs and other devices") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2});

		mock_platform tp_2(2);
		mock_device td_3(2, tp_2, sycl::info::device_type::accelerator);
		mock_device td_4(3, tp_2, sycl::info::device_type::accelerator);
		mock_device td_5(4, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_3, td_4, td_5});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK(device == td_2);
	}

	SECTION("pick_device prefers the first available device(any) with no GPUs") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::accelerator);
		mock_device td_3(2, tp_1, sycl::info::device_type::accelerator);
		mock_device td_4(3, tp_1, sycl::info::device_type::accelerator);
		tp_1.set_devices({td_2, td_3, td_4});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1});
		CHECK(device == td_1);
	}
}

class log_capture {
  public:
	log_capture(spdlog::level::level_enum level = spdlog::level::trace) {
		auto logger = spdlog::default_logger();
		auto ostream_info_sink = std::make_shared<spdlog::sinks::ostream_sink_st>(oss);
		ostream_info_sink->set_level(level);
		logger->sinks().push_back(ostream_info_sink);
	}

	~log_capture() {
		auto logger = spdlog::default_logger();
		// TODO: Assert that no other sink has been pushed in the meantime
		logger->sinks().pop_back();
	}

	std::string get_log() { return oss.str(); }

  private:
	std::ostringstream oss;
};

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device prints expected info/warn messages", "[device-selection][msg]") {
	celerity::detail::config cfg(nullptr, nullptr);
	SECTION("device_pointer is specified by the user") {
		log_capture lc;
		mock_platform tp(68);
		mock_device td(42, tp);
		tp.set_devices({{5, tp}, {7, tp}, {9, tp}});

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'foo', device 'bar' (specified by user)"));
	}

	SECTION("CELERITY_DEVICE is set by the user") {
		log_capture lc;
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(0, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(1, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3});

		celerity::detail::device_config d_cfg{td_3.get_id(), tp_1.get_id()};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'foo', device 'bar' (set by CELERITY_DEVICES: platform 1, device 1)"));
	}


	SECTION("pick_device selects a gpu/any per node automaticaly") {
		log_capture lc;
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(0, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(1, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3});

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'foo', device 'bar' (automatically selected platform 1, device 0)"));
	}

	SECTION("pick_device can't find any platform with sufficient GPUs") {
		log_capture lc{spdlog::level::warn};
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(0, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(1, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(2, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4});

		mock_platform tp_2(1);
		mock_device td_5(0, tp_2, sycl::info::device_type::accelerator);
		mock_device td_6(1, tp_2, sycl::info::device_type::accelerator);
		mock_device td_7(2, tp_2, sycl::info::device_type::accelerator);
		mock_device td_8(3, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_5, td_6, td_7, td_8});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 GPU devices, and CELERITY_DEVICES not set"));
	}

	SECTION("pick_device can't find any platform with any type of sufficient device") {
		log_capture lc(spdlog::level::warn);
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(0, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2});

		mock_platform tp_2(2);
		mock_device td_3(0, tp_2, sycl::info::device_type::accelerator);
		mock_device td_4(1, tp_2, sycl::info::device_type::accelerator);
		mock_device td_5(2, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_3, td_4, td_5});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 devices, and CELERITY_DEVICES not set"));
	}

	SECTION("CELERITY_DEVICE is set with invalid platform id") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(3);
		mock_device td_2(0, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(1, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3});

		celerity::detail::device_config d_cfg{tp_1.get_id(), td_3.get_id()};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);
		CHECK_THROWS_WITH(pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1}),
		    "Invalid platform id 3: Only 2 platforms available");
	}

	SECTION("CELERITY_DEVICE is set with invalid device id") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});

		mock_platform tp_1(1);
		mock_device td_2(4, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(5, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3});

		celerity::detail::device_config d_cfg{tp_1.get_id(), td_3.get_id()};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

		CHECK_THROWS_WITH(pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{tp_0, tp_1}),
		    "Invalid device id 5: Only 2 devices available on platform 1");
	}

	SECTION("pick_device couldn't find any device") {
		CHECK_THROWS_WITH(
		    pick_device(cfg, celerity::detail::auto_select_device{}, std::vector<mock_platform>{}), "Automatic device selection failed: No device available");
	}
}

// The following test doesn't work with ComputeCpp backend, since the == operator behaves differently
#if !defined(WORKAROUND_COMPUTECPP)
TEST_CASE_METHOD(celerity::test_utils::runtime_fixture,
    "runtime::init/distr_queue provides an overloaded constructor with device selector, testing sycl::device", "[distr_queue][ctor][sycl]") {
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

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "runtime::init/distr_queue provides an overloaded constructor with device selector, testing mock_device",
    "[device-selection][ctor][mock-host-cfg]") {
	celerity::detail::config cfg(nullptr, nullptr);

	SECTION("pick_device prefers a particular device over all") {
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});
		tp_0.set_info("foo_0");

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4, td_5});
		tp_1.set_info("foo_1");

		mock_platform tp_2(2);
		mock_device td_6(5, tp_2, sycl::info::device_type::accelerator);
		mock_device td_7(6, tp_2, sycl::info::device_type::accelerator);
		mock_device td_8(7, tp_2, sycl::info::device_type::accelerator);
		mock_device td_9(8, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_6, td_7, td_8, td_9});
		tp_2.set_info("foo_2");

		auto device_selector = [td_7](const mock_device& d) -> int { return d == td_7 ? 2 : 1; };

		auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK(device == td_7);
	}

	SECTION("pick_device prefers a group of devices") {
		log_capture lc;
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});
		tp_0.set_info("foo_0");

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4});
		tp_1.set_info("foo_1");

		mock_platform tp_2(2);
		mock_device td_5(4, tp_2, sycl::info::device_type::gpu);
		mock_device td_6(5, tp_2, sycl::info::device_type::gpu);
		mock_device td_7(6, tp_2, sycl::info::device_type::accelerator);
		mock_device td_8(7, tp_2, sycl::info::device_type::accelerator);
		mock_device td_9(8, tp_2, sycl::info::device_type::accelerator);
		mock_device td_10(9, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_5, td_6, td_7, td_8, td_9, td_10});
		tp_2.set_info("foo_2");

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device_selector = [](const mock_device& d) -> int { return d.get_type() == sycl::info::device_type::accelerator ? 2 : 1; };

		auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Using platform 'foo_2', device 'bar' (device selector specified: platform 2, device 3)"));
		CHECK(device == td_10);
	}

	SECTION("pick_device prefers prioritised device with selector with insufficient devices") {
		log_capture lc(spdlog::level::warn);
		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});
		tp_0.set_info("foo_0");

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2});
		tp_1.set_info("foo_1");

		mock_platform tp_2(2);
		mock_device td_3(2, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_3});
		tp_2.set_info("foo_2");

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);
		auto device_selector = [td_3](const mock_device& d) -> int { return d == td_3 ? 2 : 1; };

		auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK_THAT(
		    lc.get_log(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 devices that match the specified device selector"));
		CHECK(device == td_3);
	}

	SECTION("pick_device can choose devices across platform with warnings") {
		log_capture lc(spdlog::level::warn);

		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});
		tp_0.set_info("foo_0");

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::accelerator);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4, td_5});
		tp_1.set_info("foo_1");

		mock_platform tp_2(2);
		mock_device td_6(5, tp_2, sycl::info::device_type::gpu);
		mock_device td_7(6, tp_2, sycl::info::device_type::accelerator);
		mock_device td_8(7, tp_2, sycl::info::device_type::accelerator);
		mock_device td_9(8, tp_2, sycl::info::device_type::accelerator);
		tp_2.set_devices({td_6, td_7, td_8, td_9});
		tp_2.set_info("foo_2");

		size_t node_count = 4;
		size_t local_rank = 2;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device_selector = [](const mock_device& d) -> int { return d.get_type() == sycl::info::device_type::accelerator ? 2 : 1; };

		auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		INFO("Platform id" << device.get_platform().get_id() << " device id " << device.get_id());
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Selected devices are of different type and/or do not belong to the same platform"));
		CHECK(device == td_8);
	}

	SECTION("pick_device can choose different types of devices with warnings") {
		log_capture lc(spdlog::level::warn);

		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});
		tp_0.set_info("foo_0");

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::accelerator);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4, td_5});
		tp_1.set_info("foo_1");

		mock_platform tp_2(2);
		mock_device td_6(5, tp_2, sycl::info::device_type::accelerator);
		mock_device td_7(6, tp_2, sycl::info::device_type::gpu);
		mock_device td_8(7, tp_2, sycl::info::device_type::gpu);
		mock_device td_9(8, tp_2, sycl::info::device_type::gpu);
		tp_2.set_devices({td_6, td_7, td_8, td_9});
		tp_2.set_info("foo_2");

		size_t node_count = 4;
		size_t local_rank = 0;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device_selector = [](const mock_device& d) -> int { return d.get_type() == sycl::info::device_type::accelerator ? 2 : 1; };

		auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		INFO("Platform id" << device.get_platform().get_id() << " device id " << device.get_id());
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Selected devices are of different type and/or do not belong to the same platform"));
		CHECK(device == td_2);
	}

	SECTION("pick_device can choose different types of devices with insufficient devices in platforms with warnings") {
		log_capture lc(spdlog::level::warn);

		mock_platform tp_0(0);
		mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
		tp_0.set_devices({td_1});
		tp_0.set_info("foo_0");

		mock_platform tp_1(1);
		mock_device td_2(1, tp_1, sycl::info::device_type::accelerator);
		mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
		mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
		mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
		tp_1.set_devices({td_2, td_3, td_4, td_5});
		tp_1.set_info("foo_1");

		mock_platform tp_2(2);
		mock_device td_6(5, tp_2, sycl::info::device_type::accelerator);
		mock_device td_7(6, tp_2, sycl::info::device_type::gpu);
		mock_device td_8(7, tp_2, sycl::info::device_type::gpu);
		tp_2.set_devices({td_6, td_7, td_8});
		tp_2.set_info("foo_2");

		size_t node_count = 4;
		size_t local_rank = 1;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device_selector = [](const mock_device& d) -> int { return d.get_type() == sycl::info::device_type::accelerator ? 2 : 1; };

		auto device = pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		INFO("Platform id" << device.get_platform().get_id() << " device id " << device.get_id());
		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring("Selected devices are of different type and/or do not belong to the same platform"));
		CHECK(device == td_6);
	}
}

TEST_CASE_METHOD(
    celerity::test_utils::mpi_fixture, "pick_device does not consider devices with a negative selector score", "[device-selection][msg][negative]") {
	celerity::detail::config cfg(nullptr, nullptr);
	log_capture lc;

	mock_platform tp_0(0);
	mock_device td_1(0, tp_0, sycl::info::device_type::cpu);
	tp_0.set_devices({td_1});
	tp_0.set_info("foo_0");

	mock_platform tp_1(1);
	mock_device td_2(1, tp_1, sycl::info::device_type::gpu);
	mock_device td_3(2, tp_1, sycl::info::device_type::gpu);
	mock_device td_4(3, tp_1, sycl::info::device_type::gpu);
	mock_device td_5(4, tp_1, sycl::info::device_type::gpu);
	tp_1.set_devices({td_2, td_3, td_4});
	tp_1.set_info("foo_1");

	mock_platform tp_2(2);
	mock_device td_7(5, tp_2, sycl::info::device_type::gpu);
	tp_2.set_devices({td_7});
	tp_2.set_info("foo_2");

	size_t node_count = 4;
	size_t local_rank = 2;
	size_t local_num_cpus = 1;

	celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
	celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

	auto device_selector = [](const mock_device& d) -> int { return d.get_type() == sycl::info::device_type::accelerator ? 1 : -1; };

	CHECK_THROWS_WITH(
	    pick_device(cfg, device_selector, std::vector<mock_platform>{tp_0, tp_1, tp_2}), "Device selection with device selector failed: No device available");
}
