#include "device_queue.h"

#include <CL/sycl.hpp>

#include "log.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	void device_queue::init(const config& cfg, cl::sycl::device* user_device) {
		assert(sycl_queue == nullptr);
		const auto profiling_cfg = cfg.get_enable_device_profiling();
		device_profiling_enabled = profiling_cfg != std::nullopt && *profiling_cfg;
		if(device_profiling_enabled) { CELERITY_INFO("Device profiling enabled."); }

		const auto props = device_profiling_enabled ? cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()} : cl::sycl::property_list{};
		const auto handle_exceptions = cl::sycl::async_handler{[this](cl::sycl::exception_list el) { this->handle_async_exceptions(el); }};
		auto device = pick_device(cfg, user_device);
		sycl_queue = std::make_unique<cl::sycl::queue>(device, handle_exceptions, props);
	}

	cl::sycl::device device_queue::pick_device(const config& cfg, cl::sycl::device* user_device) const {
		cl::sycl::device device;
		std::string how_selected = "automatically selected";
		if(user_device != nullptr) {
			device = *user_device;
			how_selected = "specified by user";
		} else {
			const auto device_cfg = cfg.get_device_config();
			if(device_cfg != std::nullopt) {
				how_selected = fmt::format("set by CELERITY_DEVICES: platform {}, device {}", device_cfg->platform_id, device_cfg->device_id);
				const auto platforms = cl::sycl::platform::get_platforms();
				CELERITY_DEBUG("{} platforms available", platforms.size());
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
				const auto host_cfg = cfg.get_host_config();

				const auto try_find_device_per_node = [&host_cfg, &device, &how_selected](cl::sycl::info::device_type type) {
					// Try to find a platform that can provide a unique device for each node.
					const auto platforms = cl::sycl::platform::get_platforms();
					for(size_t i = 0; i < platforms.size(); ++i) {
						auto&& platform = platforms[i];
						const auto devices = platform.get_devices(type);
						if(devices.size() >= host_cfg.node_count) {
							how_selected = fmt::format("automatically selected platform {}, device {}", i, host_cfg.local_rank);
							device = devices[host_cfg.local_rank];
							return true;
						}
					}
					return false;
				};

				const auto try_find_one_device = [&device](cl::sycl::info::device_type type) {
					const auto devices = cl::sycl::device::get_devices(type);
					if(!devices.empty()) {
						device = devices[0];
						return true;
					}
					return false;
				};

				// Try to find a unique GPU per node.
				if(!try_find_device_per_node(cl::sycl::info::device_type::gpu)) {
					// Try to find a unique device (of any type) per node.
					if(try_find_device_per_node(cl::sycl::info::device_type::all)) {
						CELERITY_WARN("No suitable platform found that can provide {} GPU devices, and CELERITY_DEVICES not set", host_cfg.node_count);
					} else {
						CELERITY_WARN("No suitable platform found that can provide {} devices, and CELERITY_DEVICES not set", host_cfg.node_count);
						// Just use the first available device. Prefer GPUs, but settle for anything.
						if(!try_find_one_device(cl::sycl::info::device_type::gpu) && !try_find_one_device(cl::sycl::info::device_type::all)) {
							throw std::runtime_error("Automatic device selection failed: No device available");
						}
					}
				}
			}
		}

		const auto platform_name = device.get_platform().get_info<cl::sycl::info::platform::name>();
		const auto device_name = device.get_info<cl::sycl::info::device::name>();
		CELERITY_INFO("Using platform '{}', device '{}' ({})", platform_name, device_name, how_selected);

		return device;
	}

	void device_queue::handle_async_exceptions(cl::sycl::exception_list el) const {
		for(auto& e : el) {
			try {
				std::rethrow_exception(e);
			} catch(cl::sycl::exception& e) {
				CELERITY_ERROR("SYCL asynchronous exception: {}. Terminating.", e.what());
				std::terminate();
			}
		}
	}


} // namespace detail
} // namespace celerity
