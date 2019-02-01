#include "device_queue.h"

#include <CL/sycl.hpp>

#include "workaround.h"

namespace celerity {
namespace detail {

	void device_queue::init(config& cfg, task_manager* task_mngr, cl::sycl::device* user_device) {
		assert(sycl_queue == nullptr);
		assert(task_mngr != nullptr);
		this->task_mngr = task_mngr;
		const auto forced_wg_size_cfg = cfg.get_forced_work_group_size();
		forced_work_group_size = forced_wg_size_cfg != boost::none ? *forced_wg_size_cfg : 0;
		const auto profiling_cfg = cfg.get_enable_device_profiling();
		device_profiling_enabled = profiling_cfg != boost::none && *profiling_cfg;
		if(device_profiling_enabled) { queue_logger.info("Device profiling enabled."); }

		// TODO: Do we need a queue on master nodes? (Only for single-node execution?)
		cl::sycl::property_list props = ([&]() {
#if !WORKAROUND(HIPSYCL, 0)
			if(device_profiling_enabled) { return cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()}; }
#endif
			return cl::sycl::property_list{};
		})(); // IIFE
		const auto handle_exceptions = [this](cl::sycl::exception_list el) { this->handle_async_exceptions(el); };
		auto device = pick_device(cfg, user_device);
		sycl_queue = std::make_unique<cl::sycl::queue>(device, handle_exceptions, props);
	}

	cl::sycl::device device_queue::pick_device(config& cfg, cl::sycl::device* user_device) const {
		cl::sycl::device device;
		std::string how_selected = "automatically selected";
		if(user_device != nullptr) {
			device = *user_device;
			how_selected = "specified by user";
		} else {
			const auto device_cfg = cfg.get_device_config();
			if(device_cfg != boost::none) {
				how_selected = fmt::format("set by CELERITY_DEVICES: platform {}, device {}", device_cfg->platform_id, device_cfg->device_id);
				const auto platforms = cl::sycl::platform::get_platforms();
				queue_logger.trace("{} platforms available", platforms.size());
				if(device_cfg->platform_id >= platforms.size()) {
					throw std::runtime_error(fmt::format("Invalid platform id {}: Only {} platforms available", device_cfg->platform_id, platforms.size()));
				}
				const auto devices = platforms[device_cfg->platform_id].get_devices();
				if(device_cfg->device_id >= devices.size()) {
					throw std::runtime_error(fmt::format(
					    "Invalid device id {}: Only {} devices available on platform {}", device_cfg->device_id, devices.size(), device_cfg->platform_id));
				}
				device = devices[device_cfg->device_id];
			} else {
				queue_logger.warn("CELERITY_DEVICES not set");

				// Just use the first device available
				const auto devices = cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);
				if(devices.empty()) { throw std::runtime_error("Automatic device selection failed: No GPU device available"); }
				device = devices[0];
			}
		}

		const auto platform_name = device.get_platform().get_info<cl::sycl::info::platform::name>();
		const auto device_name = device.get_info<cl::sycl::info::device::name>();
#if WORKAROUND(COMPUTECPP, 1, 0, 5)
		// The names returned by ComputeCpp seem to contain an additional null byte,
		// which causes problems (log files get interpreted as binary data etc), so we chop it off.
		queue_logger.info("Using platform '{}', device '{}' ({})", platform_name.substr(0, platform_name.size() - 1),
		    device_name.substr(0, device_name.size() - 1), how_selected);
#else
		queue_logger.info("Using platform '{}', device '{}' ({})", platform_name, device_name, how_selected);
#endif

		return device;
	}

	void device_queue::handle_async_exceptions(cl::sycl::exception_list el) const {
		for(auto& e : el) {
			try {
				std::rethrow_exception(e);
			} catch(cl::sycl::exception& e) {
				// TODO: We'd probably want to abort execution here
				queue_logger.error("SYCL asynchronous exception: {}", e.what());
			}
		}
	}


} // namespace detail
} // namespace celerity
