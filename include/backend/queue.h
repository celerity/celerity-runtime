#pragma once

#include "async_event.h"
#include "launcher.h"
#include "ranges.h"
#include "types.h"

#include <vector>

namespace celerity::detail::backend {

class sycl_event final : public async_event_base {
  public:
	sycl_event() = default;
	sycl_event(std::vector<sycl::event> wait_list);

	bool is_complete() const override;

  private:
	mutable std::vector<sycl::event> m_incomplete;
};

struct device_config {
	detail::device_id device_id = 0;
	memory_id native_memory = first_device_memory_id;
	sycl::device sycl_device;
};

class queue {
  public:
	queue() = default;
	queue(const queue&) = delete;
	queue& operator=(const queue&) = delete;
	virtual ~queue() = default;

	virtual void init() {}

	virtual void* alloc(memory_id where, size_t size, size_t alignment) = 0;

	virtual void free(memory_id where, void* allocation) = 0;

	virtual async_event copy_region(memory_id source_mid, memory_id dest_mid, const void* const source_base, void* const dest_base, const box<3>& source_box,
	    const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) = 0;

	virtual async_event launch_kernel(
	    device_id did, const device_kernel_launcher& launcher, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) = 0;
};

async_event launch_sycl_kernel(
    sycl::queue& queue, const device_kernel_launcher& launcher, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs);

void flush_sycl_queue(sycl::queue& queue);

void handle_sycl_errors(const sycl::exception_list& errors);

#if CELERITY_DETAIL_ENABLE_DEBUG
inline constexpr std::byte uninitialized_memory_pattern = std::byte(0xff); // floats and doubles filled with this pattern show up as "-nan"
#endif

} // namespace celerity::detail::backend
