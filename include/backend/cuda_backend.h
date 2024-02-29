#pragma once

#include "backend/queue.h"


namespace celerity::detail::backend {

class cuda_queue final : public queue {
  public:
	using cuda_device_id = int;

	explicit cuda_queue(const std::vector<device_config>& devices);
	~cuda_queue() override;

	void init() override;

	void* alloc(memory_id where, size_t size, size_t alignment) override;

	void free(memory_id where, void* allocation) override;

	async_event copy_region(memory_id source_mid, memory_id dest_mid, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) override;

	async_event launch_kernel(
	    device_id did, const device_kernel_launcher& launcher, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail::backend
