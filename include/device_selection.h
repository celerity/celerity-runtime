#pragma once

#include <algorithm>
#include <functional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "backend/backend.h"
#include "config.h"
#include "log.h"

#include <sycl/sycl.hpp>

namespace celerity::detail {

// TODO these are required by distr_queue.h, but we don't want to pull all include dependencies of the pick_devices implementation into user code!
struct auto_select_devices {};
using device_selector = std::function<int(const sycl::device&)>;
using devices_or_selector = std::variant<auto_select_devices, std::vector<sycl::device>, device_selector>;

template <typename DeviceT>
void check_required_device_aspects(const DeviceT& device) {
	if(!device.has(sycl::aspect::usm_device_allocations)) { throw std::runtime_error("device does not support USM device allocations"); }
	// NOTE: We don't need host allocations strictly speaking, only used for convenience.
	if(!device.has(sycl::aspect::usm_host_allocations)) { throw std::runtime_error("device does not support USM host allocations"); }
}

template <typename DevicesOrSelector, typename PlatformT>
auto pick_devices(const config& cfg, const DevicesOrSelector& user_devices_or_selector, const std::vector<PlatformT>& platforms) {
	using DeviceT = typename decltype(std::declval<PlatformT&>().get_devices())::value_type;
	using BackendT = decltype(std::declval<DeviceT&>().get_backend());

	constexpr bool user_devices_provided = std::is_same_v<DevicesOrSelector, std::vector<DeviceT>>;
	constexpr bool device_selector_provided = std::is_invocable_r_v<int, DevicesOrSelector, DeviceT>;
	constexpr bool auto_select = std::is_same_v<auto_select_devices, DevicesOrSelector>;
	static_assert(user_devices_provided ^ device_selector_provided ^ auto_select,
	    "pick_device requires either a list of devices, a selector, or the auto_select_devices tag");

	std::vector<DeviceT> selected_devices;
	std::string how_selected;

	if(cfg.get_host_config().node_count > 1) {
		CELERITY_WARN("Celerity detected more than one node (MPI rank) on this host, which is not recommended. Will attempt to distribute local devices evenly "
		              "across nodes.");
	}

	if constexpr(user_devices_provided) {
		const auto devices = user_devices_or_selector;
		if(devices.empty()) { throw std::runtime_error("Device selection failed: The user-provided list of devices is empty"); }
		auto backend = devices[0].get_backend();
		for(size_t i = 0; i < devices.size(); ++i) {
			if(devices[i].get_backend() != backend) {
				throw std::runtime_error("Device selection failed: The user-provided list of devices contains devices from different backends");
			}
			try {
				check_required_device_aspects(devices[i]);
			} catch(std::runtime_error& e) {
				throw std::runtime_error(fmt::format("Device selection failed: Device {} in user-provided list of devices caused error: {}", i, e.what()));
			}
		}
		selected_devices = devices;
		how_selected = "specified by user";
	} else {
		if(std::all_of(platforms.cbegin(), platforms.cend(), [](auto& p) { return p.get_devices().empty(); })) {
			throw std::runtime_error("Device selection failed: No devices available");
		}

		const auto select_all = [platforms](auto& selector) {
			std::unordered_map<BackendT, std::vector<std::pair<DeviceT, size_t>>> scored_devices_by_backend;
			for(size_t i = 0; i < platforms.size(); ++i) {
				const auto devices = platforms[i].get_devices(sycl::info::device_type::all);
				for(size_t j = 0; j < devices.size(); ++j) {
					try {
						check_required_device_aspects(devices[j]);
					} catch(std::runtime_error& e) {
						CELERITY_TRACE("Ignoring device {} on platform {}: {}", j, i, e.what());
						continue;
					}
					const auto score = selector(devices[j]);
					if(score < 0) continue;
					scored_devices_by_backend[devices[j].get_backend()].push_back(std::pair{devices[j], score});
				}
			}
			size_t max_score = 0;
			std::vector<DeviceT> max_score_devices;
			for(auto& [backend, scored_devices] : scored_devices_by_backend) {
				size_t sum_score = 0;
				std::vector<DeviceT> devices;
				for(auto& [d, score] : scored_devices) {
					sum_score += score;
					devices.push_back(d);
				}
				if(sum_score > max_score) {
					max_score = sum_score;
					max_score_devices = std::move(devices);
				}
			}
			return max_score_devices;
		};

		if constexpr(device_selector_provided) {
			how_selected = "via user-provided selector";
			selected_devices = select_all(user_devices_or_selector);
		} else {
			how_selected = "automatically selected";
			// First try to find eligible GPUs
			const auto selector = [](const DeviceT& d) {
				return d.template get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu ? 1 : -1;
			};
			selected_devices = select_all(selector);
			if(selected_devices.empty()) {
				// If none were found, fall back to other device types
				const auto selector = [](const DeviceT& d) { return 1; };
				selected_devices = select_all(selector);
			}
		}

		if(selected_devices.empty()) { throw std::runtime_error("Device selection failed: No eligible devices found"); }
	}

	// When running with more than one local node, attempt to distribute devices evenly
	if(cfg.get_host_config().node_count > 1) {
		if(selected_devices.size() >= cfg.get_host_config().node_count) {
			const size_t quotient = selected_devices.size() / cfg.get_host_config().node_count;
			const size_t remainder = selected_devices.size() % cfg.get_host_config().node_count;

			const auto rank = cfg.get_host_config().local_rank;
			const size_t offset = rank < remainder ? rank * (quotient + 1) : remainder * (quotient + 1) + (rank - remainder) * quotient;
			const size_t count = rank < remainder ? quotient + 1 : quotient;

			std::vector<DeviceT> subset{selected_devices.begin() + offset, selected_devices.begin() + offset + count};
			selected_devices = std::move(subset);
		} else {
			CELERITY_WARN("Found fewer devices ({}) than local nodes ({}), multiple nodes will use the same device(s).", selected_devices.size(),
			    cfg.get_host_config().node_count);
			selected_devices = {selected_devices[cfg.get_host_config().local_rank % selected_devices.size()]};
		}
	}

	for(auto& device : selected_devices) {
		const auto platform_name = device.get_platform().template get_info<sycl::info::platform::name>();
		const auto device_name = device.template get_info<sycl::info::device::name>();
		CELERITY_INFO("Using platform '{}', device '{}' ({})", platform_name, device_name, how_selected);
	}

	if constexpr(std::is_same_v<DeviceT, sycl::device>) {
		const auto device = selected_devices[0]; // For now all devices must share the same backend
		if(backend::get_effective_type(device) == backend::type::generic) {
			if(backend::get_type(device) == backend::type::unknown) {
				CELERITY_WARN("No backend specialization available for selected platform '{}', falling back to generic. Performance may be degraded.",
				    device.get_platform().template get_info<sycl::info::platform::name>());
			} else {
				CELERITY_WARN("Selected platform '{}' is compatible with specialized {} backend, but it has not been compiled. Performance may be degraded.",
				    device.get_platform().template get_info<sycl::info::platform::name>(), backend::get_name(backend::get_type(device)));
			}
		} else {
			CELERITY_DEBUG("Using {} backend for selected platform '{}'.", backend::get_name(backend::get_effective_type(device)),
			    device.get_platform().template get_info<sycl::info::platform::name>());
		}
	}

	return selected_devices;
}

} // namespace celerity::detail