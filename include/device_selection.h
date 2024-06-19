#pragma once

#include <functional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

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
	if(!device.has(sycl::aspect::usm_host_allocations)) { throw std::runtime_error("device does not support USM host allocations"); }
}

template <typename DevicesOrSelector, typename PlatformT>
auto pick_devices(const host_config& cfg, const DevicesOrSelector& user_devices_or_selector, const std::vector<PlatformT>& platforms) {
	using DeviceT = typename decltype(std::declval<PlatformT&>().get_devices())::value_type;
	using BackendT = decltype(std::declval<DeviceT&>().get_backend());

	constexpr bool user_devices_provided = std::is_same_v<DevicesOrSelector, std::vector<DeviceT>>;
	constexpr bool device_selector_provided = std::is_invocable_r_v<int, DevicesOrSelector, DeviceT>;
	constexpr bool auto_select = std::is_same_v<auto_select_devices, DevicesOrSelector>;
	static_assert(user_devices_provided ^ device_selector_provided ^ auto_select,
	    "pick_device requires either a list of devices, a selector, or the auto_select_devices tag");

	std::vector<DeviceT> selected_devices;
	std::string how_selected;

	if(cfg.node_count > 1) {
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
						CELERITY_TRACE("Ignoring platform {} \"{}\", device {} \"{}\": {}", i, platforms[i].template get_info<sycl::info::platform::name>(), j,
						    devices[j].template get_info<sycl::info::device::name>(), e.what());
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
	if(cfg.node_count > 1) {
		if(selected_devices.size() >= cfg.node_count) {
			const size_t quotient = selected_devices.size() / cfg.node_count;
			const size_t remainder = selected_devices.size() % cfg.node_count;

			const auto rank = cfg.local_rank;
			const size_t offset = rank < remainder ? rank * (quotient + 1) : remainder * (quotient + 1) + (rank - remainder) * quotient;
			const size_t count = rank < remainder ? quotient + 1 : quotient;

			std::vector<DeviceT> subset{selected_devices.begin() + offset, selected_devices.begin() + offset + count};
			selected_devices = std::move(subset);
		} else {
			CELERITY_WARN(
			    "Found fewer devices ({}) than local nodes ({}), multiple nodes will use the same device(s).", selected_devices.size(), cfg.node_count);
			selected_devices = {selected_devices[cfg.local_rank % selected_devices.size()]};
		}
	}

	for(device_id did = 0; did < selected_devices.size(); ++did) {
		const auto platform_name = selected_devices[did].get_platform().template get_info<sycl::info::platform::name>();
		const auto device_name = selected_devices[did].template get_info<sycl::info::device::name>();
		CELERITY_INFO("Using platform \"{}\", device \"{}\" as D{} ({})", platform_name, device_name, did, how_selected);
	}

	return selected_devices;
}

/*
template<typename T>
concept BackendEnumerator = requires(const T &a) {
    typename T::backend_type;
    typename T::device_type;
    {a.compatible_backends(std::declval<typename T::device_type>)} -> std::same_as<std::vector<T::backend_type>>;
    {a.available_backends()} -> std::same_as<std::vector<T::backend_type>>;
    {a.is_specialized(std::declval<T::backend_type>())} -> std::same_as<bool>;
    {a.get_priority(std::declval<T::backend_type>())} -> std::same_as<int>;
};
*/

template <typename BackendEnumerator>
inline auto select_backend(const BackendEnumerator& enumerator, const std::vector<typename BackendEnumerator::device_type>& devices) {
	using backend_type = typename BackendEnumerator::backend_type;

	const auto available_backends = enumerator.available_backends();

	std::vector<backend_type> common_backends;
	for(auto& device : devices) {
		auto device_backends = enumerator.compatible_backends(device);
		common_backends = common_backends.empty() ? std::move(device_backends) : utils::set_intersection(common_backends, device_backends);
	}

	assert(!common_backends.empty());
	std::sort(common_backends.begin(), common_backends.end(),
	    [&](const backend_type lhs, const backend_type rhs) { return enumerator.get_priority(lhs) > enumerator.get_priority(rhs); });

	for(const auto backend : common_backends) {
		const auto is_specialized = enumerator.is_specialized(backend);
		if(utils::contains(available_backends, backend)) {
			if(is_specialized) {
				CELERITY_DEBUG("Using {} backend for the selected devices.", backend);
			} else {
				CELERITY_WARN("No common backend specialization available for all selected devices, falling back to {}. Performance may be degraded.", backend);
			}
			return backend;
		} else if(is_specialized) {
			CELERITY_WARN(
			    "All selected devices are compatible with specialized {} backend, but it has not been compiled. Performance may be degraded.", backend);
		}
	}
	utils::panic("no compatible backend available");
}

} // namespace celerity::detail
