#pragma once

#include <algorithm>
#include <memory>
#include <type_traits>
#include <variant>

#include <CL/sycl.hpp>

#include "config.h"
#include "legacy_backend/backend.h"
#include "log.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	struct auto_select_device {};
	using device_selector = std::function<int(const sycl::device&)>;
	using device_or_selector = std::variant<auto_select_device, sycl::device, device_selector>;

	class task;

	struct device_allocation {
		void* ptr = nullptr;
		size_t size_bytes = 0;
	};

	class allocation_error : public std::runtime_error {
	  public:
		allocation_error(const std::string& msg) : std::runtime_error(msg) {}
	};

	/**
	 * The @p device_queue wraps the actual SYCL queue and is used to submit kernels.
	 */
	class device_queue {
	  public:
		/**
		 * @brief Initializes the @p device_queue, selecting an appropriate device in the process.
		 *
		 * @param cfg The configuration is used to select the appropriate SYCL device.
		 * @param user_device_or_selector Optionally a device (which will take precedence over any configuration) or a device selector can be provided.
		 */
		void init(const config& cfg, const device_or_selector& user_device_or_selector);

		/**
		 * @brief Executes the kernel associated with task @p ctsk over the chunk @p chnk.
		 */
		template <typename Fn>
		cl::sycl::event submit(Fn&& fn) {
			auto evt = m_sycl_queue->submit([fn = std::forward<Fn>(fn)](cl::sycl::handler& sycl_handler) { fn(sycl_handler); });
#if CELERITY_WORKAROUND(HIPSYCL)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
			// hipSYCL does not guarantee that command groups are actually scheduled until an explicit await operation, which we cannot insert without
			// blocking the executor loop (see https://github.com/illuhad/hipSYCL/issues/599). Instead, we explicitly flush the queue to be able to continue
			// using our polling-based approach.
			m_sycl_queue->get_context().hipSYCL_runtime()->dag().flush_async();
#pragma GCC diagnostic pop
#endif
			return evt;
		}

		template <typename T>
		[[nodiscard]] device_allocation malloc(const size_t count) {
			const size_t size_bytes = count * sizeof(T);
			assert(m_sycl_queue != nullptr);
			assert(m_global_mem_allocated_bytes + size_bytes < m_global_mem_total_size_bytes);
			CELERITY_DEBUG("Allocating {} bytes on device", size_bytes);
			T* ptr = nullptr;
			try {
				ptr = sycl::aligned_alloc_device<T>(alignof(T), count, *m_sycl_queue);
			} catch(sycl::exception& e) {
				CELERITY_CRITICAL("sycl::aligned_alloc_device failed with exception: {}", e.what());
				ptr = nullptr;
			}
			if(ptr == nullptr) {
				throw allocation_error(fmt::format("Allocation of {} bytes failed; likely out of memory. Currently allocated: {} out of {} bytes.",
				    count * sizeof(T), m_global_mem_allocated_bytes, m_global_mem_total_size_bytes));
			}
			m_global_mem_allocated_bytes += size_bytes;
			return device_allocation{ptr, size_bytes};
		}

		void free(device_allocation alloc) {
			assert(m_sycl_queue != nullptr);
			assert(alloc.size_bytes <= m_global_mem_allocated_bytes);
			assert(alloc.ptr != nullptr || alloc.size_bytes == 0);
			CELERITY_DEBUG("Freeing {} bytes on device", alloc.size_bytes);
			if(alloc.size_bytes != 0) { sycl::free(alloc.ptr, *m_sycl_queue); }
			m_global_mem_allocated_bytes -= alloc.size_bytes;
		}

		size_t get_global_memory_total_size_bytes() const { return m_global_mem_total_size_bytes; }

		size_t get_global_memory_allocated_bytes() const { return m_global_mem_allocated_bytes; }

		/**
		 * @brief Waits until all currently submitted operations have completed.
		 */
		void wait() { m_sycl_queue->wait_and_throw(); }

		/**
		 * @brief Returns whether device profiling is enabled.
		 */
		bool is_profiling_enabled() const { return m_device_profiling_enabled; }

		cl::sycl::queue& get_sycl_queue() const {
			assert(m_sycl_queue != nullptr);
			return *m_sycl_queue;
		}

	  private:
		size_t m_global_mem_total_size_bytes = 0;
		size_t m_global_mem_allocated_bytes = 0;
		std::unique_ptr<cl::sycl::queue> m_sycl_queue;
		bool m_device_profiling_enabled = false;

		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

	// Try to find a platform that can provide a unique device for each node using a device selector.
	template <typename DeviceT, typename PlatformT, typename SelectorT>
	bool try_find_device_per_node(
	    std::string& how_selected, DeviceT& device, const std::vector<PlatformT>& platforms, const host_config& host_cfg, SelectorT selector) {
		std::vector<std::tuple<DeviceT, size_t>> devices_with_platform_idx;
		for(size_t i = 0; i < platforms.size(); ++i) {
			auto&& platform = platforms[i];
			for(auto device : platform.get_devices()) {
				if(selector(device) == -1) { continue; }
				devices_with_platform_idx.emplace_back(device, i);
			}
		}

		std::stable_sort(devices_with_platform_idx.begin(), devices_with_platform_idx.end(),
		    [selector](const auto& a, const auto& b) { return selector(std::get<0>(a)) > selector(std::get<0>(b)); });
		bool same_platform = true;
		bool same_device_type = true;
		if(devices_with_platform_idx.size() >= host_cfg.node_count) {
			auto [device_from_platform, idx] = devices_with_platform_idx[0];
			const auto platform = device_from_platform.get_platform();
			const auto device_type = device_from_platform.template get_info<sycl::info::device::device_type>();

			for(size_t i = 1; i < host_cfg.node_count; ++i) {
				auto [device_from_platform, idx] = devices_with_platform_idx[i];
				if(device_from_platform.get_platform() != platform) { same_platform = false; }
				if(device_from_platform.template get_info<sycl::info::device::device_type>() != device_type) { same_device_type = false; }
			}

			if(!same_platform || !same_device_type) { CELERITY_WARN("Selected devices are of different type and/or do not belong to the same platform"); }

			auto [selected_device_from_platform, selected_idx] = devices_with_platform_idx[host_cfg.local_rank];
			how_selected = fmt::format("device selector specified: platform {}, device {}", selected_idx, host_cfg.local_rank);
			device = selected_device_from_platform;
			return true;
		}

		return false;
	}

	// Try to find a platform that can provide a unique device for each node.
	template <typename DeviceT, typename PlatformT>
	bool try_find_device_per_node(
	    std::string& how_selected, DeviceT& device, const std::vector<PlatformT>& platforms, const host_config& host_cfg, sycl::info::device_type type) {
		for(size_t i = 0; i < platforms.size(); ++i) {
			auto&& platform = platforms[i];
			std::vector<DeviceT> platform_devices;

			platform_devices = platform.get_devices(type);
			if(platform_devices.size() >= host_cfg.node_count) {
				how_selected = fmt::format("automatically selected platform {}, device {}", i, host_cfg.local_rank);
				device = platform_devices[host_cfg.local_rank];
				return true;
			}
		}

		return false;
	}

	template <typename DeviceT, typename PlatformT, typename SelectorT>
	bool try_find_one_device(
	    std::string& how_selected, DeviceT& device, const std::vector<PlatformT>& platforms, const host_config& host_cfg, SelectorT selector) {
		std::vector<DeviceT> platform_devices;
		for(auto& p : platforms) {
			auto p_devices = p.get_devices();
			platform_devices.insert(platform_devices.end(), p_devices.begin(), p_devices.end());
		}

		std::stable_sort(platform_devices.begin(), platform_devices.end(), [selector](const auto& a, const auto& b) { return selector(a) > selector(b); });
		if(!platform_devices.empty()) {
			if(selector(platform_devices[0]) == -1) { return false; }
			device = platform_devices[0];
			return true;
		}

		return false;
	};

	template <typename DeviceT, typename PlatformT>
	bool try_find_one_device(
	    std::string& how_selected, DeviceT& device, const std::vector<PlatformT>& platforms, const host_config& host_cfg, sycl::info::device_type type) {
		for(auto& p : platforms) {
			for(auto& d : p.get_devices(type)) {
				device = d;
				return true;
			}
		}

		return false;
	};


	template <typename DevicePtrOrSelector, typename PlatformT>
	auto pick_device(const config& cfg, const DevicePtrOrSelector& user_device_or_selector, const std::vector<PlatformT>& platforms) {
		using DeviceT = typename decltype(std::declval<PlatformT&>().get_devices())::value_type;

		constexpr bool user_device_provided = std::is_same_v<DevicePtrOrSelector, DeviceT>;
		constexpr bool device_selector_provided = std::is_invocable_r_v<int, DevicePtrOrSelector, DeviceT>;
		constexpr bool auto_select = std::is_same_v<auto_select_device, DevicePtrOrSelector>;
		static_assert(
		    user_device_provided ^ device_selector_provided ^ auto_select, "pick_device requires either a device, a selector, or the auto_select_device tag");

		DeviceT device;
		std::string how_selected = "automatically selected";
		if constexpr(user_device_provided) {
			device = user_device_or_selector;
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

				if constexpr(!device_selector_provided) {
					// Try to find a unique GPU per node.
					if(!try_find_device_per_node(how_selected, device, platforms, host_cfg, sycl::info::device_type::gpu)) {
						if(try_find_device_per_node(how_selected, device, platforms, host_cfg, sycl::info::device_type::all)) {
							CELERITY_WARN("No suitable platform found that can provide {} GPU devices, and CELERITY_DEVICES not set", host_cfg.node_count);
						} else {
							CELERITY_WARN("No suitable platform found that can provide {} devices, and CELERITY_DEVICES not set", host_cfg.node_count);
							// Just use the first available device. Prefer GPUs, but settle for anything.
							if(!try_find_one_device(how_selected, device, platforms, host_cfg, sycl::info::device_type::gpu)
							    && !try_find_one_device(how_selected, device, platforms, host_cfg, sycl::info::device_type::all)) {
								throw std::runtime_error("Automatic device selection failed: No device available");
							}
						}
					}
				} else {
					// Try to find a unique device per node using a selector.
					if(!try_find_device_per_node(how_selected, device, platforms, host_cfg, user_device_or_selector)) {
						CELERITY_WARN("No suitable platform found that can provide {} devices that match the specified device selector, and "
						              "CELERITY_DEVICES not set",
						    host_cfg.node_count);
						// Use the first available device according to the selector, but fails if no such device is found.
						if(!try_find_one_device(how_selected, device, platforms, host_cfg, user_device_or_selector)) {
							throw std::runtime_error("Device selection with device selector failed: No device available");
						}
					}
				}
			}
		}

		const auto platform_name = device.get_platform().template get_info<sycl::info::platform::name>();
		const auto device_name = device.template get_info<sycl::info::device::name>();
		CELERITY_INFO("Using platform '{}', device '{}' ({})", platform_name, device_name, how_selected);

		if constexpr(std::is_same_v<DeviceT, sycl::device>) {
			if(legacy_backend::get_effective_type(device) == legacy_backend::type::generic) {
				if(legacy_backend::get_type(device) == legacy_backend::type::unknown) {
					CELERITY_WARN("No backend specialization available for selected platform '{}', falling back to generic. Performance may be degraded.",
					    device.get_platform().template get_info<sycl::info::platform::name>());
				} else {
					CELERITY_WARN(
					    "Selected platform '{}' is compatible with specialized {} backend, but it has not been compiled. Performance may be degraded.",
					    device.get_platform().template get_info<sycl::info::platform::name>(), legacy_backend::get_name(legacy_backend::get_type(device)));
				}
			} else {
				CELERITY_DEBUG("Using {} backend for selected platform '{}'.", legacy_backend::get_name(legacy_backend::get_effective_type(device)),
				    device.get_platform().template get_info<sycl::info::platform::name>());
			}
		}

		return device;
	}

} // namespace detail
} // namespace celerity
