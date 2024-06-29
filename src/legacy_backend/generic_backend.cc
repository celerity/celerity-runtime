#include "legacy_backend/generic_backend.h"

#include "ranges.h"

namespace celerity::detail::legacy_backend_detail {

void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<0>& /* source_range */,
    const id<0>& /* source_offset */, const range<0>& /* target_range */, const id<0>& /* target_offset */, const range<0>& /* copy_range */) {
	auto evt = queue.memcpy(target_base_ptr, source_base_ptr, elem_size);
	evt.wait();
}

void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<1>& source_range,
    const id<1>& source_offset, const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range) {
	const size_t line_size = elem_size * copy_range[0];
	auto evt = queue.memcpy(static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size);
	evt.wait();
}

// TODO Optimize for contiguous copies?
void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<2>& source_range,
    const id<2>& source_offset, const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range) {
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	const size_t line_size = elem_size * copy_range[1];
	std::vector<sycl::event> wait_list{copy_range[0]};
	for(size_t i = 0; i < copy_range[0]; ++i) {
		auto e = queue.memcpy(static_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
		    static_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
		wait_list[i] = std::move(e);
	}
	sycl::event::wait(wait_list);
}

// TODO Optimize for contiguous copies?
void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<3>& source_range,
    const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range) {
	// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
	const auto source_base_offset =
	    get_linear_index(source_range, source_offset) - get_linear_index(range<2>{source_range[1], source_range[2]}, id<2>{source_offset[1], source_offset[2]});
	const auto target_base_offset =
	    get_linear_index(target_range, target_offset) - get_linear_index(range<2>{target_range[1], target_range[2]}, id<2>{target_offset[1], target_offset[2]});

	for(size_t i = 0; i < copy_range[0]; ++i) {
		const auto* const source_ptr = static_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
		auto* const target_ptr = static_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
		memcpy_strided_device_generic(queue, source_ptr, target_ptr, elem_size, range<2>{source_range[1], source_range[2]},
		    id<2>{source_offset[1], source_offset[2]}, range<2>{target_range[1], target_range[2]}, id<2>{target_offset[1], target_offset[2]},
		    range<2>{copy_range[1], copy_range[2]});
	}
}

} // namespace celerity::detail::legacy_backend_detail
