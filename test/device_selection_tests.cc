#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"
#include "spdlog/sinks/ostream_sink.h"
#include "test_utils.h"
#include <celerity.h>

struct mock_platform;
struct mock_device {
	mock_device() : id(0), type(cl::sycl::info::device_type::gpu) {}

	mock_device(size_t id, cl::sycl::info::device_type type = cl::sycl::info::device_type::gpu) : id(id), type(type) {}

	mock_platform get_platform() const;

	template <cl::sycl::info::device Param>
	std::string get_info() const {
		return "bar";
	}

	bool operator==(const mock_device& other) const { return other.id == id; }

	cl::sycl::info::device_type get_type() const { return type; }

	size_t get_id() { return id; }

  private:
	size_t id;
	cl::sycl::info::device_type type;
};
struct mock_platform {
	// TODO: These devices should somehow have this platform as their platform (?)
	mock_platform(size_t id, std::vector<mock_device> devices) : devices(std::move(devices)), id(id) {}

	std::vector<mock_device> get_devices(cl::sycl::info::device_type type = cl::sycl::info::device_type::all) const {
		if(type != cl::sycl::info::device_type::all) {
			std::vector<mock_device> devices_with_type;
			for(auto device : devices) {
				if(device.get_type() == type) { devices_with_type.emplace_back(device); }
			}
			return devices_with_type;
		} else
			return devices;
	}

	template <cl::sycl::info::platform Param>
	std::string get_info() const {
		return "foo";
	}

	size_t get_id() { return id; }

  private:
	std::vector<mock_device> devices;
	size_t id;
};

// TODO: Device should know its associated platform and return it from here
mock_platform mock_device::get_platform() const {
	return {15, {}}; // Setting random platform ID for now
}


namespace celerity::detail {
struct config_testspy {
	static void set_mock_device_cfg(config& cfg, const device_config& d_cfg) { cfg.device_cfg = d_cfg; }
	static void set_mock_host_cfg(config& cfg, const host_config& h_cfg) { cfg.host_cfg = h_cfg; }
};
} // namespace celerity::detail

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device prefers user specified device pointer", "[device-selection][one]") {
	celerity::detail::config cfg(nullptr, nullptr);

	mock_device td(42);
	mock_platform tp(68, {{5}, {7}, {9}});

	auto device = pick_device(cfg, &td, std::vector<mock_platform>{tp});
	CHECK(device == td);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture,
    "pick_device automatically selects a gpu device if available and otherwise falls back to the first device available", "[device-selection][gen]") {
	celerity::detail::config cfg(nullptr, nullptr);

	mock_device* td = nullptr;
	using device_t = cl::sycl::info::device_type;

	auto dv_type_1 = GENERATE(as<cl::sycl::info::device_type>(), device_t::gpu, device_t::accelerator, device_t::cpu, device_t::custom, device_t::host);
	CAPTURE(dv_type_1);

	mock_device td_1(0, dv_type_1);
	mock_platform tp_1(0, {td_1});

	auto dv_type_2 = GENERATE(as<cl::sycl::info::device_type>(), device_t::gpu, device_t::accelerator, device_t::cpu, device_t::custom, device_t::host);
	CAPTURE(dv_type_2);

	mock_device td_2(0, dv_type_2);
	mock_platform tp_2(1, {td_2});

	auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_1, tp_2});
	std::vector<mock_device> devices;
	if(dv_type_1 == device_t::gpu || (dv_type_1 != device_t::gpu && dv_type_2 != device_t::gpu)) {
		devices = tp_1.get_devices();
	} else {
		devices = tp_2.get_devices();
	}
	CHECK(device == devices[0]);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device selects device using CELERITY_DEVICES", "[device-selection][device-cfg]") {
	celerity::detail::config cfg(nullptr, nullptr);
	mock_device* td = nullptr;

	mock_device td_1(0, cl::sycl::info::device_type::cpu);
	mock_platform tp_0(0, {td_1});

	mock_device td_2(0, cl::sycl::info::device_type::gpu);
	mock_device td_3(1, cl::sycl::info::device_type::gpu);
	mock_platform tp_1(1, {td_2, td_3});

	celerity::detail::device_config d_cfg{tp_1.get_id(), td_3.get_id()};
	celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

	auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1});
	std::vector<mock_device> devices = tp_1.get_devices();
	CHECK(device == devices[1]);
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture,
    "pick_device selects a GPU for each local_rank or falls back to any type of sufficient device for all ranks", "[device-selection][host-cfg]") {
	celerity::detail::config cfg(nullptr, nullptr);
	mock_device* td = nullptr;

	SECTION("pick_device unique GPU per node") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_device td_4(2, cl::sycl::info::device_type::gpu);
		mock_device td_5(3, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3, td_4, td_5});

		size_t node_count = 4;
		size_t local_rank = 2;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1});
		std::vector<mock_device> devices = tp_1.get_devices();
		CHECK(device == devices[2]);
	}

	SECTION("pick_device prefers unique GPU over other devices") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_device td_4(2, cl::sycl::info::device_type::gpu);
		mock_device td_5(3, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3, td_4, td_5});

		mock_device td_6(0, cl::sycl::info::device_type::accelerator);
		mock_device td_7(1, cl::sycl::info::device_type::accelerator);
		mock_device td_8(2, cl::sycl::info::device_type::accelerator);
		mock_device td_9(3, cl::sycl::info::device_type::accelerator);
		mock_platform tp_2(1, {td_6, td_7, td_8, td_9});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		std::vector<mock_device> devices = tp_1.get_devices();
		CHECK(device == devices[3]);
	}

	SECTION("pick_device falls back to other devices with insufficient GPUs") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_device td_4(2, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3, td_4});

		mock_device td_5(0, cl::sycl::info::device_type::accelerator);
		mock_device td_6(1, cl::sycl::info::device_type::accelerator);
		mock_device td_7(2, cl::sycl::info::device_type::accelerator);
		mock_device td_8(3, cl::sycl::info::device_type::accelerator);
		mock_platform tp_2(1, {td_5, td_6, td_7, td_8});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		std::vector<mock_device> devices = tp_2.get_devices();
		CHECK(device == devices[3]);
	}

	SECTION("pick_device prefers the first available GPU with insufficient GPUs and other devices") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2});

		mock_device td_5(0, cl::sycl::info::device_type::accelerator);
		mock_device td_6(1, cl::sycl::info::device_type::accelerator);
		mock_device td_7(2, cl::sycl::info::device_type::accelerator);
		mock_platform tp_2(1, {td_5, td_6, td_7});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		std::vector<mock_device> devices = tp_1.get_devices();
		CHECK(device == devices[0]);
	}

	SECTION("pick_device prefers the first available device(any) with no GPUs") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_5(0, cl::sycl::info::device_type::accelerator);
		mock_device td_6(1, cl::sycl::info::device_type::accelerator);
		mock_device td_7(2, cl::sycl::info::device_type::accelerator);
		mock_platform tp_2(1, {td_5, td_6, td_7});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_2});
		std::vector<mock_device> devices = tp_2.get_devices();
		CHECK(device == devices[0]);
	}
}

TEST_CASE_METHOD(celerity::test_utils::mpi_fixture, "pick_device prints expected info/warn messages", "[device-selection][msg]") {
	std::ostringstream oss;
	auto logger = spdlog::default_logger();
	auto ostream_info_sink = std::make_shared<spdlog::sinks::ostream_sink_st>(oss);
	ostream_info_sink->set_level(spdlog::level::info);
	logger->sinks().push_back(ostream_info_sink);

	celerity::detail::config cfg(nullptr, nullptr);
	SECTION("device_pointer is specified by the user") {
		mock_device td(42);
		mock_platform tp(68, {{5}, {7}, {9}});

		auto device = pick_device(cfg, &td, std::vector<mock_platform>{tp});
		CHECK_THAT(oss.str(), Catch::Matchers::ContainsSubstring("Using platform 'foo', device 'bar' (specified by user)"));
		oss.str("");
	}

	mock_device* td = nullptr;
	SECTION("CELERITY_DEVICE is set by the user") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3});

		celerity::detail::device_config d_cfg{td_3.get_id(), tp_1.get_id()};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1});
		CHECK_THAT(oss.str(), Catch::Matchers::ContainsSubstring("Using platform 'foo', device 'bar' (set by CELERITY_DEVICES: platform 1, device 1)"));
		oss.str("");
	}


	SECTION("pick_device selects a gpu/any per node automaticaly") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3});

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1});
		CHECK_THAT(oss.str(), Catch::Matchers::ContainsSubstring("Using platform 'foo', device 'bar' (automatically selected platform 1, device 0)"));
		oss.str("");
	}

	std::ostringstream _oss;
	auto ostream_warn_sink = std::make_shared<spdlog::sinks::ostream_sink_st>(_oss);
	ostream_warn_sink->set_level(spdlog::level::warn);
	logger->sinks().push_back(ostream_warn_sink);
	SECTION("pick_device can't find any platform with sufficient GPUs") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_device td_4(2, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3, td_4});

		mock_device td_5(0, cl::sycl::info::device_type::accelerator);
		mock_device td_6(1, cl::sycl::info::device_type::accelerator);
		mock_device td_7(2, cl::sycl::info::device_type::accelerator);
		mock_device td_8(3, cl::sycl::info::device_type::accelerator);
		mock_platform tp_2(1, {td_5, td_6, td_7, td_8});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK_THAT(_oss.str(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 GPU devices, and CELERITY_DEVICES not set"));
		_oss.str("");
	}

	SECTION("pick_device can't find any platform with any type of sufficient device") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2});

		mock_device td_5(0, cl::sycl::info::device_type::accelerator);
		mock_device td_6(1, cl::sycl::info::device_type::accelerator);
		mock_device td_7(2, cl::sycl::info::device_type::accelerator);
		mock_platform tp_2(1, {td_5, td_6, td_7});

		size_t node_count = 4;
		size_t local_rank = 3;
		size_t local_num_cpus = 1;

		celerity::detail::host_config h_cfg{node_count, local_rank, local_num_cpus};
		celerity::detail::config_testspy::set_mock_host_cfg(cfg, h_cfg);

		auto device = pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1, tp_2});
		CHECK_THAT(_oss.str(), Catch::Matchers::ContainsSubstring("No suitable platform found that can provide 4 devices, and CELERITY_DEVICES not set"));
		_oss.str("");
	}

	SECTION("CELERITY_DEVICE is set with invalid platform id") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(0, cl::sycl::info::device_type::gpu);
		mock_device td_3(1, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(3, {td_2, td_3});

		celerity::detail::device_config d_cfg{tp_1.get_id(), td_3.get_id()};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);
		CHECK_THROWS_WITH(pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1}), "Invalid platform id 3: Only 2 platforms available");
	}

	SECTION("CELERITY_DEVICE is set with invalid device id") {
		mock_device td_1(0, cl::sycl::info::device_type::cpu);
		mock_platform tp_0(0, {td_1});

		mock_device td_2(4, cl::sycl::info::device_type::gpu);
		mock_device td_3(5, cl::sycl::info::device_type::gpu);
		mock_platform tp_1(1, {td_2, td_3});

		celerity::detail::device_config d_cfg{tp_1.get_id(), td_3.get_id()};
		celerity::detail::config_testspy::set_mock_device_cfg(cfg, d_cfg);

		CHECK_THROWS_WITH(pick_device(cfg, td, std::vector<mock_platform>{tp_0, tp_1}), "Invalid device id 5: Only 2 devices available on platform 1");
	}

	SECTION("pick_device couldn't find any device") {
		CHECK_THROWS_WITH(pick_device(cfg, td, std::vector<mock_platform>{}), "Automatic device selection failed: No device available");
	}
}