#include "local_devices.h"

namespace celerity::detail {

void local_devices::init(const config& cfg) {
	assert(!m_is_initialized);
	m_is_initialized = true;

	if(cfg.get_host_config().node_count > 1) {
		CELERITY_WARN("Celerity detected more than one MPI rank on this host. Running in legacy mode (one device per rank).");
		sycl::device legacy_mode_device = celerity::detail::pick_device(cfg, auto_select_device{}, sycl::platform::get_platforms());

		m_device_queues.emplace_back(0, get_memory_id(0));
		m_device_queues.back().init(cfg, legacy_mode_device);
		return;
	}

	// NOCOMMIT We're simply selecting all GPUs for now
	const auto all_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	for(device_id did = 0; did < all_devices.size(); ++did) {
		auto& device = all_devices[did];
		m_device_queues.emplace_back(did, get_memory_id(did));
		m_device_queues.back().init(cfg, device);

		const auto platform_name = device.get_platform().template get_info<sycl::info::platform::name>();
		const auto device_name = device.template get_info<sycl::info::device::name>();
		CELERITY_INFO("Device {}: '{}', device '{}'", did, platform_name, device_name);
	}
}

void local_devices::wait_all() {
	for(auto& dq : m_device_queues) {
		dq.wait();
	}
	m_host_queue.wait();
}

} // namespace celerity::detail