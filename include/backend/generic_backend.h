#pragma once

#include "ranges.h"

#include "backend/queue.h"


namespace celerity::detail::backend {

class generic_queue : public queue {
  public:
	explicit generic_queue(const std::vector<device_config>& devices);

	void* alloc(memory_id where, size_t size, size_t alignment) override;

	void free(memory_id where, void* allocation) override;

	async_event copy_region(memory_id source_mid, memory_id dest_mid, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override;

	async_event launch_kernel(
	    device_id did, const device_kernel_launcher& launcher, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

  private:
	std::unordered_map<device_id, sycl::queue> m_device_queues;
	std::unordered_map<memory_id, sycl::queue> m_memory_queues;
};

} // namespace celerity::detail::backend
