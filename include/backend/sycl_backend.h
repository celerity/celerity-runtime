#pragma once

#include "async_event.h"

#include "backend/backend.h"

#include <fmt/format.h>

namespace celerity::detail::sycl_backend_detail {

class sycl_event final : public async_event_impl {
  public:
	sycl_event() = default;
	sycl_event(sycl::event last, bool enable_profiling) : m_first(enable_profiling ? std::optional(last) : std::nullopt), m_last(std::move(last)) {}
	sycl_event(std::optional<sycl::event> first, sycl::event last) : m_first(std::move(first)), m_last(std::move(last)) {}

	bool is_complete() override;

	std::optional<std::chrono::nanoseconds> get_native_execution_time() override;

  private:
	std::optional<sycl::event> m_first; // set iff profiling is enabled - can be a copy of m_last.
	sycl::event m_last;
};

/// Ensure that all operations previously submitted to the SYCL queue begin executing even when not explicitly awaited.
void flush(sycl::queue& queue);

#if CELERITY_DETAIL_ENABLE_DEBUG
inline constexpr uint8_t uninitialized_memory_pattern = 0xff; // floats and doubles filled with this pattern show up as "-nan"
#endif

} // namespace celerity::detail::sycl_backend_detail

namespace celerity::detail {

/// Backend implementation which sources all allocations from SYCL and dispatches device kernels to SYCL in-order queues.
///
/// This abstract class implements all `backend` functions except copies, which not subject to platform-dependent specialization.
class sycl_backend : public backend {
  public:
	explicit sycl_backend(const std::vector<sycl::device>& devices, bool enable_profiling);
	sycl_backend(const sycl_backend&) = delete;
	sycl_backend(sycl_backend&&) = delete;
	sycl_backend& operator=(const sycl_backend&) = delete;
	sycl_backend& operator=(sycl_backend&&) = delete;
	~sycl_backend() override;

	const system_info& get_system_info() const override;

	void init() override;

	void* debug_alloc(size_t size) override;

	void debug_free(void* ptr) override;

	async_event enqueue_host_alloc(size_t size, size_t alignment) override;

	async_event enqueue_device_alloc(device_id device, size_t size, size_t alignment) override;

	async_event enqueue_host_free(void* ptr) override;

	async_event enqueue_device_free(device_id device, void* ptr) override;

	async_event enqueue_host_task(size_t host_lane, const host_task_launcher& launcher, std::vector<closure_hydrator::accessor_info> accessor_infos,
	    const box<3>& execution_range, const communicator* collective_comm) override;

	async_event enqueue_device_kernel(device_id device, size_t device_lane, const device_kernel_launcher& launcher,
	    std::vector<closure_hydrator::accessor_info> accessor_infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) override;

	async_event enqueue_host_copy(size_t host_lane, const void* const source_base, void* const dest_base, const region_layout& source_layout,
	    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) override;

	void check_async_errors() override;

  protected:
	sycl::queue& get_device_queue(device_id device, size_t lane);

	system_info& get_system_info(); // mutable system_info is filled by sycl_cuda_backend constructor

	bool is_profiling_enabled() const;

  private:
	struct impl;
	std::unique_ptr<impl> m_impl;
};

/// Generic implementation of `sycl_backend` providing a fallback implementation for device copies that might be inefficient in the 2D / 3D case.
class sycl_generic_backend final : public sycl_backend {
  public:
	sycl_generic_backend(const std::vector<sycl::device>& devices, bool enable_profiling);

	async_event enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
	    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) override;
};

#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
/// CUDA specialized implementation of `sycl_backend` that uses native CUDA operations for 2D / 3D copies.
class sycl_cuda_backend final : public sycl_backend {
  public:
	sycl_cuda_backend(const std::vector<sycl::device>& devices, bool enable_profiling);

	async_event enqueue_device_copy(device_id device, size_t device_lane, const void* const source_base, void* const dest_base,
	    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) override;
};
#endif

/// We differentiate between non-specialized and specialized Celerity SYCL backends.
enum class sycl_backend_type { generic, cuda };

/// Enumerates the SYCL backends devices are compatible with and that Celerity has been compiled with.
/// This type implements the (nameless) concept accepted by `pick_devices`.
struct sycl_backend_enumerator {
	using backend_type = sycl_backend_type;
	using device_type = sycl::device;

	/// Lists the backend types a device is compatible with, even if Celerity has not been compiled with that backend.
	std::vector<backend_type> compatible_backends(const sycl::device& device) const;

	/// Lists the backend types Celerity has been compiled with.
	std::vector<backend_type> available_backends() const;

	/// Queries whether a given backend type is specialized (for diagnostics only).
	bool is_specialized(backend_type type) const;

	/// Returns a priority value for each backend type, where the highest-priority compatible backend type should offer the best performance.
	int get_priority(backend_type type) const;
};

/// Creates a SYCL backend instance of the specified type with the devices listed. Requires that Celerity has been compiled with the given backend and all
/// devices are compatible with it. If `enable_profiling` is true, events for asynchronous operations will report native execution times.
std::unique_ptr<backend> make_sycl_backend(const sycl_backend_type type, const std::vector<sycl::device>& devices, bool enable_profiling);

} // namespace celerity::detail
