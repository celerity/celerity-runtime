#pragma once

#include "async_event.h"
#include "closure_hydrator.h"
#include "launcher.h"
#include "nd_memory.h"
#include "types.h"

#include <vector>

#include <sycl/sycl.hpp>

namespace celerity::detail {

class communicator;
struct system_info;

/// The backend is responsible for asynchronously allocating device- and device-accessible host memory, copying data between host and device allocations, and
/// launching host tasks and device kernels. Asynchronous work must be explicitly assigned to in-order queue ids as assigned by `out_of_order_engine`.
class backend {
  public:
	backend() = default;
	backend(const backend&) = delete;
	backend(backend&&) = delete;
	backend& operator=(const backend&) = delete;
	backend& operator=(backend&&) = delete;
	virtual ~backend() = default;

	/// Returns metadata about the system as it appears to the backend implementation.
	virtual const system_info& get_system_info() const = 0;

	/// Performs (possibly latency-intensive) backend initialization. Separate from the constructor to allow this function to be called from a different thread.
	virtual void init() = 0;

	/// Synchronously allocates device-accessible host memory. This is slow and meant for debugging purposes only.
	virtual void* debug_alloc(size_t size) = 0;

	/// Synchronously frees device-accessible host memory. This is slow and meant for debugging purposes only.
	virtual void debug_free(void* ptr) = 0;

	/// Schedules the allocation of device-accessible host memory with the specified size and alignment. The operation will complete in-order with respect to
	/// any other asynchronous `alloc` and `free` operation on the same backend.
	virtual async_event enqueue_host_alloc(size_t size, size_t alignment) = 0;

	/// Schedules the allocation of device memory with the specified size and alignment. The operation will complete in-order with respect to any other
	/// asynchronous `alloc` and `free` operation on the same backend.
	virtual async_event enqueue_device_alloc(device_id memory_device, size_t size, size_t alignment) = 0;

	/// Schedules the release of memory allocated via `enqueue_host_alloc`. The operation will complete in-order with respect to any other asynchronous `alloc`
	/// and `free` operation on the same backend.
	virtual async_event enqueue_host_free(void* ptr) = 0;

	/// Schedules the release of memory allocated via `enqueue_device_alloc`. The operation will complete in-order with respect to any other asynchronous
	/// `alloc` and `free` operation on the same backend.
	virtual async_event enqueue_device_free(device_id memory_device, void* ptr) = 0;

	/// Enqueues the asynchronous execution of a host task in a background thread identified by `host_lane`. The operation will complete in-order with respect
	/// to any other asynchronous host operation on `host_lane`.
	virtual async_event enqueue_host_task(size_t host_lane, const host_task_launcher& launcher, std::vector<closure_hydrator::accessor_info> accessor_infos,
	    const box<3>& execution_range, const communicator* collective_comm) = 0;

	/// Enqueues the asynchronous execution of a kernel in an in-order device queue identified by `device` and `device_lane`. The operation will complete
	/// in-order with respect to any other asynchronous device operation on `device` and `device_lane`.
	virtual async_event enqueue_device_kernel(device_id device, size_t device_lane, const device_kernel_launcher& launcher,
	    std::vector<closure_hydrator::accessor_info> accessor_infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) = 0;

	/// Enqueues an n-dimensional copy between two host allocations (both either device-accessible or user-allocated). The operation will complete
	/// in-order with respect to any other asynchronous host operation on `host_lane`.
	virtual async_event enqueue_host_copy(size_t host_lane, const void* source_base, void* dest_base, const region_layout& source_layout,
	    const region_layout& dest_layout, const region<3>& copy_region, size_t elem_size) = 0;

	/// Enqueues an n-dimensional copy between two device-accessible allocations (at least one device-native). The operation will complete in-order with respect
	/// to any other asynchronous device operation on `device` and `device_lane`.
	virtual async_event enqueue_device_copy(device_id device, size_t device_lane, const void* source_base, void* dest_base, const region_layout& source_layout,
	    const region_layout& dest_layout, const region<3>& copy_region, size_t elem_size) = 0;

	/// Check internal queues and panic if any asynchronous errors occurred.
	virtual void check_async_errors() = 0;
};

} // namespace celerity::detail
