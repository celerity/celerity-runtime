#pragma once

#include "grid.h"
#include "ranges.h"

#include <string.h>

namespace celerity::detail {

template <typename F>
void for_each_linear_slice_in_nd_copy(
    const range<3>& source_range, const range<3>& dest_range, const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range, F&& f) //
{
	assert(all_true(offset_in_source + copy_range <= source_range));
	assert(all_true(offset_in_dest + copy_range <= dest_range));

	if(copy_range.size() == 0) return;

	// Find the first (slowest) dimension in which we can do linear copies. We can do better than relying on range::get_effective_dims() by recognizing that we
	// can treat multiple "rows" in the copy range as one contiguous entity as long as they fully span both the source and destination ranges.
	int linear_dim = 0;
	for(int d = 1; d < 3; ++d) {
		if(source_range[d] != copy_range[d] || dest_range[d] != copy_range[d]) { linear_dim = d; }
	}

	// There might be a more elegant way to cover all three nesting levels, but I haven't come up with one yet.
	switch(linear_dim) {
	case 0: {
		const size_t linear_size = copy_range[0] * copy_range[1] * copy_range[2];
		f(get_linear_index(source_range, offset_in_source), get_linear_index(dest_range, offset_in_dest), linear_size);
		break;
	}
	case 1: {
		const size_t linear_size = copy_range[1] * copy_range[2];
		auto index_in_source = offset_in_source;
		auto index_in_dest = offset_in_dest;
		for(size_t i = 0; i < copy_range[0]; ++i) {
			index_in_source[0] = offset_in_source[0] + i;
			index_in_dest[0] = offset_in_dest[0] + i;
			f(get_linear_index(source_range, index_in_source), get_linear_index(dest_range, index_in_dest), linear_size);
		}
		break;
	}
	case 2: {
		const size_t linear_size = copy_range[2];
		auto index_in_source = offset_in_source;
		auto index_in_dest = offset_in_dest;
		for(size_t i = 0; i < copy_range[0]; ++i) {
			index_in_source[0] = offset_in_source[0] + i;
			index_in_dest[0] = offset_in_dest[0] + i;
			for(size_t j = 0; j < copy_range[1]; ++j) {
				index_in_source[1] = offset_in_source[1] + j;
				index_in_dest[1] = offset_in_dest[1] + j;
				f(get_linear_index(source_range, index_in_source), get_linear_index(dest_range, index_in_dest), linear_size);
			}
		}
		break;
	}
	default: assert(!"unreachable");
	}
}

inline void nd_copy_host(const void* const source_base, void* const dest_base, const range<3>& source_range, const range<3>& dest_range,
    const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size) //
{
	for_each_linear_slice_in_nd_copy(source_range, dest_range, offset_in_source, offset_in_dest, copy_range,
	    [&](const size_t linear_offset_in_source, const size_t linear_offset_in_dest, const size_t linear_size) {
		    memcpy(static_cast<std::byte*>(dest_base) + linear_offset_in_dest * elem_size,
		        static_cast<const std::byte*>(source_base) + linear_offset_in_source * elem_size, linear_size * elem_size);
	    });
}

inline void copy_region_host(const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box,
    const region<3>& copy_region, const size_t elem_size) //
{
	for(const auto& copy_box : copy_region.get_boxes()) {
		assert(source_box.covers(copy_box));
		assert(dest_box.covers(copy_box));
		nd_copy_host(source_base, dest_base, source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
		    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size);
	}
}

} // namespace celerity::detail