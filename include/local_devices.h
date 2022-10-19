#pragma once

#include "device_queue.h"
#include "host_queue.h"
#include "types.h"

namespace celerity::detail {

class config;

// This is a very simplistic initial implementation
// TODO: It would be neat if we could model host/RAM as just another device / memory.
class local_devices {
  public:
	void init(const config& cfg);

	host_queue& get_host_queue() { return m_host_queue; }

	size_t num_compute_devices() const { return m_device_queues.size(); }

	size_t num_memories() const { return num_compute_devices() + 1; }

	device_queue& get_device_queue(const device_id did) {
		assert(did < m_device_queues.size());
		return m_device_queues[did];
	}

	memory_id get_host_memory_id() const { return 0; }

	memory_id get_memory_id(const device_id did) const { return did + 1; }

	device_queue& get_close_device_queue(const memory_id mid) { return m_device_queues[mid - 1]; }

	void wait_all();

  private:
	host_queue m_host_queue;
	std::vector<device_queue> m_device_queues;
};

} // namespace celerity::detail