#include "backend/sycl_backend.h"

#include "log.h"
#include "nd_memory.h"
#include "ranges.h"
#include "tracy.h"
#include "types.h"

namespace celerity::detail::sycl_backend_detail {

void nd_copy_device_generic_async(sycl::queue& queue, const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const box<3>& copy_box, const size_t elem_size, bool enable_profiling, std::optional<sycl::event>& first, sycl::event& last) //
{
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	const auto layout = layout_nd_copy(source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
	    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size);
	for_each_contiguous_chunk(layout, [&](const size_t chunk_offset_in_source, const size_t chunk_offset_in_dest, const size_t chunk_size) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("sycl::submit", Orange2);
		// first, last: We remember the first and last submission event to report completion time spanning the entire region copy
		last = queue.memcpy(
		    static_cast<std::byte*>(dest_base) + chunk_offset_in_dest, static_cast<const std::byte*>(source_base) + chunk_offset_in_source, chunk_size);
		if(enable_profiling && !first.has_value()) { first = last; }
	});
}

async_event nd_copy_device_generic(sycl::queue& queue, const void* const source_base, void* const dest_base, const region_layout& source_layout,
    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size, const bool enable_profiling) //
{
	// We remember the first and last submission event to report completion time spanning the entire region copy.
	std::optional<sycl::event> first; // set only if profiling is enabled
	sycl::event last;

	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&queue, elem_size, enable_profiling, &first, &last](
	        const void* const source, void* const dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_device_generic_async(queue, source, dest, source_box, dest_box, copy_box, elem_size, enable_profiling, first, last);
	    },
	    [&queue, enable_profiling, &first, &last](const void* const source, void* const dest, size_t size_bytes) {
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED("sycl::submit", Orange2);
		    last = queue.memcpy(dest, source, size_bytes);
		    if(enable_profiling) { first = last; }
	    });

	flush(queue);
	return make_async_event<sycl_event>(std::move(first), std::move(last));
}

} // namespace celerity::detail::sycl_backend_detail

namespace celerity::detail {

sycl_generic_backend::sycl_generic_backend(const std::vector<sycl::device>& devices, bool enable_profiling) : sycl_backend(devices, enable_profiling) {
	if(devices.size() > 1) { CELERITY_DEBUG("Generic backend does not support peer memory access, device-to-device copies will be staged in host memory"); }
}

async_event sycl_generic_backend::enqueue_device_copy(const device_id device, const size_t device_lane, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) //
{
	auto& queue = get_device_queue(device, device_lane);
	return sycl_backend_detail::nd_copy_device_generic(
	    queue, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
}

} // namespace celerity::detail
