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

	host_queue& get_host_queue() {
		assert(m_is_initialized);
		return m_host_queue;
	}

	size_t num_compute_devices() const {
		assert(m_is_initialized);
		return m_device_queues.size();
	}

	size_t num_memories() const {
		assert(m_is_initialized);
		return num_compute_devices() + 1;
	}

	device_queue& get_device_queue(const device_id did) {
		assert(m_is_initialized);
		assert(did < m_device_queues.size());
		return m_device_queues[did];
	}

	memory_id get_host_memory_id() const { return 0; }

	memory_id get_memory_id(const device_id did) const { return did + 1; }

	/**
	 * Returns a device that is "close" to the given memory.
	 *
	 * Since we don't support shared memory yet, this currently just returns the main memory for each device.
	 */
	device_queue& get_close_device_queue(const memory_id mid) {
		assert(m_is_initialized);
		return m_device_queues[mid - 1];
	}

	void wait_all();

  private:
	[[maybe_unused]] bool m_is_initialized = false;
	host_queue m_host_queue{get_host_memory_id()};
	std::vector<device_queue> m_device_queues;
};

} // namespace celerity::detail