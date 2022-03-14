#pragma once

#include <memory>

#include <CL/sycl.hpp>

#include "config.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	class task;

	/**
	 * The @p device_queue wraps the actual SYCL queue and is used to submit kernels.
	 */
	class device_queue {
	  public:
		/**
		 * @brief Initializes the @p device_queue, selecting an appropriate device in the process.
		 *
		 * @param cfg The configuration is used to select the appropriate SYCL device.
		 * @param user_device Optionally a device can be provided, which will take precedence over any configuration.
		 */
		void init(const config& cfg, cl::sycl::device* user_device);

		/**
		 * @brief Executes the kernel associated with task @p ctsk over the chunk @p chnk.
		 */
		template <typename Fn>
		cl::sycl::event submit(Fn&& fn) {
			auto evt = sycl_queue->submit([fn = std::forward<Fn>(fn)](cl::sycl::handler& sycl_handler) { fn(sycl_handler); });
#if WORKAROUND_HIPSYCL
			// hipSYCL does not guarantee that command groups are actually scheduled until an explicit await operation, which we cannot insert without
			// blocking the executor loop (See https://github.com/illuhad/hipSYCL/issues/599). Instead, we explicitly flush the queue to be able to continue
			// using our polling-based approach.
			hipsycl::rt::application::dag().flush_async();
#endif
			return evt;
		}

		/**
		 * @brief Waits until all currently submitted operations have completed.
		 */
		void wait() { sycl_queue->wait_and_throw(); }

		/**
		 * @brief Returns whether device profiling is enabled.
		 */
		bool is_profiling_enabled() const { return device_profiling_enabled; }

		cl::sycl::queue& get_sycl_queue() const {
			assert(sycl_queue != nullptr);
			return *sycl_queue;
		}

	  private:
		std::unique_ptr<cl::sycl::queue> sycl_queue;
		bool device_profiling_enabled = false;

		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

	template <typename DeviceT, typename PlatformT>
	DeviceT pick_device(const config& cfg, DeviceT* user_device, const std::vector<PlatformT>& platforms) {
		DeviceT device;
		std::string how_selected = "automatically selected";
		if(user_device != nullptr) {
			device = *user_device;
			how_selected = "specified by user";
		} else {
			const auto device_cfg = cfg.get_device_config();
			if(device_cfg != std::nullopt) {
				how_selected = fmt::format("set by CELERITY_DEVICES: platform {}, device {}", device_cfg->platform_id, device_cfg->device_id);
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

				const auto try_find_device_per_node = [&host_cfg, &device, &how_selected, &platforms](cl::sycl::info::device_type type) {
					// Try to find a platform that can provide a unique device for each node.
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

				const auto try_find_one_device = [&device, &platforms](cl::sycl::info::device_type type) {
					for(auto& p : platforms) {
						for(auto& d : p.get_devices(type)) {
							device = d;
							return true;
						}
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

		const auto platform_name = device.get_platform().template get_info<cl::sycl::info::platform::name>();
		const auto device_name = device.template get_info<cl::sycl::info::device::name>();
		CELERITY_INFO("Using platform '{}', device '{}' ({})", platform_name, device_name, how_selected);

		return device;
	}

} // namespace detail
} // namespace celerity
