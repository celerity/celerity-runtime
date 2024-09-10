#pragma once

#include "grid.h"
#include "ranges.h"
#include "utils.h"

#include <cstring>
#include <variant>


namespace celerity::detail {

/// A region is linearized in the given memory location by densely concatenating the content of all boxes.
struct linearized_layout {
	size_t offset_bytes = 0;

	explicit linearized_layout(const size_t offset_bytes) : offset_bytes(offset_bytes) {}

	friend bool operator==(const linearized_layout& lhs, const linearized_layout& rhs) { return lhs.offset_bytes == rhs.offset_bytes; }
	friend bool operator!=(const linearized_layout& lhs, const linearized_layout& rhs) { return !(lhs == rhs); }
};

/// A region is represented by the placement of n-dimensional boxes within an n-dimensional allocation.
struct strided_layout {
	box<3> allocation;

	explicit strided_layout(const box<3>& allocation) : allocation(allocation) {}

	friend bool operator==(const strided_layout& lhs, const strided_layout& rhs) { return lhs.allocation == rhs.allocation; }
	friend bool operator!=(const strided_layout& lhs, const strided_layout& rhs) { return !(lhs == rhs); }
};

/// Layout for source and destination of a copy operation.
using region_layout = std::variant<strided_layout, linearized_layout>;

/// Describes a box-shaped copy operation between two box-shaped allocations in terms of linear offsets and strides.
struct nd_copy_layout {
	struct stride_dimension {
		size_t source_stride = 1; ///< by how many bytes to advance the source pointer after one step in this dimension.
		size_t dest_stride = 1;   ///< by how many bytes to advance the destination pointer after one step in this dimension.
		size_t count = 1;         ///< how many iterations to perform in this dimension.

		friend bool operator==(const stride_dimension& lhs, const stride_dimension& rhs) {
			return lhs.source_stride == rhs.source_stride && lhs.dest_stride == rhs.dest_stride && lhs.count == rhs.count;
		}

		friend bool operator!=(const stride_dimension& lhs, const stride_dimension& rhs) { return !(lhs == rhs); }
	};

	size_t offset_in_source = 0; ///< offset in the source allocation, in bytes, of the first chunk to copy.
	size_t offset_in_dest = 0;   ///< offset in the destination allocation, in bytes, of the first chunk to copy.
	int num_complex_strides = 0; ///< number of strides which are not contiguous in either source or destination.
	stride_dimension strides[2]; ///< in the 3D / 2-complex case, strides[0] is the outer and strides[1] the inner stride.
	size_t contiguous_size = 0;  ///< number of contiguous bytes in the last dimension.

	friend bool operator==(const nd_copy_layout& lhs, const nd_copy_layout& rhs) {
		return lhs.offset_in_source == rhs.offset_in_source && lhs.offset_in_dest == rhs.offset_in_dest && lhs.contiguous_size == rhs.contiguous_size
		       && lhs.num_complex_strides == rhs.num_complex_strides && lhs.strides[0] == rhs.strides[0] && lhs.strides[1] == rhs.strides[1];
	}

	friend bool operator!=(const nd_copy_layout& lhs, const nd_copy_layout& rhs) { return !(lhs == rhs); }
};

/// Computes the minimum number of complex strides required to describe a box-shaped copy between box-shaped allocations.
inline nd_copy_layout layout_nd_copy(const range<3>& source_range, const range<3>& dest_range, const id<3>& offset_in_source, const id<3>& offset_in_dest,
    const range<3>& copy_range, size_t elem_size) {
	assert(all_true(offset_in_source + copy_range <= source_range));
	assert(all_true(offset_in_dest + copy_range <= dest_range));

	if(copy_range.size() == 0) return {};

	nd_copy_layout layout;
	layout.offset_in_source = get_linear_index(source_range, offset_in_source) * elem_size;
	layout.offset_in_dest = get_linear_index(dest_range, offset_in_dest) * elem_size;
	layout.contiguous_size = elem_size;
	size_t* current_range = &layout.contiguous_size; // we first maximize the contiguous range, then the range of the 0th / 1st stride.
	size_t next_source_stride = elem_size;
	size_t next_dest_stride = elem_size;
	bool contiguous = true; // when false, the next-higher non-trivial dimension must insert a stride
	for(int d = 2; d >= 0; --d) {
		if(!contiguous && copy_range[d] != 1) { // if range is 1, we can postpone (and possibly avoid) inserting a stride
			++layout.num_complex_strides;
			layout.strides[1] = layout.strides[0];
			layout.strides[0] = {next_source_stride, next_dest_stride, 1};
			current_range = &layout.strides[0].count;
			contiguous = true;
		}
		next_source_stride *= source_range[d];
		next_dest_stride *= dest_range[d];
		*current_range *= copy_range[d];
		if(source_range[d] != copy_range[d] || dest_range[d] != copy_range[d]) { contiguous = false; }
	}

	return layout;
}

/// For every contiguous chunk in an nd-copy, invoke `f(byte_offset_in_source, byte_offset_in_dest, chunk_bytes)`.
template <typename F>
inline void for_each_contiguous_chunk(const nd_copy_layout& layout, F&& f) {
	if(layout.contiguous_size == 0) return;

	// nd_copy_layout is defined such that we can ignore num_complex_strides in this loop and will perform exactly one iteration for every non-complex stride.
	size_t source_offset_0 = layout.offset_in_source;
	size_t dest_offset_0 = layout.offset_in_dest;
	for(size_t i = 0; i < layout.strides[0].count; ++i) {
		size_t source_offset_1 = source_offset_0;
		size_t dest_offset_1 = dest_offset_0;
		for(size_t j = 0; j < layout.strides[1].count; ++j) {
			f(source_offset_1, dest_offset_1, layout.contiguous_size);
			source_offset_1 += layout.strides[1].source_stride;
			dest_offset_1 += layout.strides[1].dest_stride;
		}
		source_offset_0 += layout.strides[0].source_stride;
		dest_offset_0 += layout.strides[0].dest_stride;
	}
}

/// From allocation `source_base` spanning `source_box` to allocation `dest_base` spanning `dest_box`, copy `copy_region` elements of `elem_size` bytes.
template <typename /* void(const void*, void*, box<3>, box<3>, box<3>, size_t) */ BoxCopyFn, typename /* void(const void*, void*, size_t) */ LinearCopyFn>
void dispatch_nd_region_copy(const void* const source_base, void* const dest_base, const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size, BoxCopyFn&& box_copy, LinearCopyFn&& linear_copy) //
{
	if(std::holds_alternative<strided_layout>(source_layout) && std::holds_alternative<strided_layout>(dest_layout)) {
		const auto& source_box = std::get<strided_layout>(source_layout).allocation;
		const auto& dest_box = std::get<strided_layout>(dest_layout).allocation;
		for(const auto& copy_box : copy_region.get_boxes()) {
			box_copy(source_base, dest_base, source_box, dest_box, copy_box);
		}
	} else if(std::holds_alternative<strided_layout>(source_layout) && std::holds_alternative<linearized_layout>(dest_layout)) {
		const auto& source_box = std::get<strided_layout>(source_layout).allocation;
		auto dest_cursor = utils::offset(dest_base, std::get<linearized_layout>(dest_layout).offset_bytes);
		for(const auto& copy_box : copy_region.get_boxes()) {
			box_copy(source_base, dest_cursor, source_box, copy_box, copy_box);
			dest_cursor = utils::offset(dest_cursor, copy_box.get_area() * elem_size);
		}
	} else if(std::holds_alternative<linearized_layout>(source_layout) && std::holds_alternative<strided_layout>(dest_layout)) {
		const auto& dest_box = std::get<strided_layout>(dest_layout).allocation;
		auto source_cursor = utils::offset(source_base, std::get<linearized_layout>(source_layout).offset_bytes);
		for(const auto& copy_box : copy_region.get_boxes()) {
			box_copy(source_cursor, dest_base, copy_box, dest_box, copy_box);
			source_cursor = utils::offset(source_cursor, copy_box.get_area() * elem_size);
		}
	} else /* fast path: both linearized */ {
		assert(std::holds_alternative<linearized_layout>(source_layout) && std::holds_alternative<linearized_layout>(dest_layout));
		const auto source_cursor = utils::offset(source_base, std::get<linearized_layout>(source_layout).offset_bytes);
		const auto dest_cursor = utils::offset(dest_base, std::get<linearized_layout>(dest_layout).offset_bytes);
		linear_copy(source_cursor, dest_cursor, copy_region.get_area() * elem_size);
	}
}

/// From allocation `source_base` sized `source_range` starting at `offset_in_source`, to allocation `dest_base` sized `dest_range` starting at to
/// `offset_in_dest`, copy `copy_range` elements of `elem_size` bytes.
inline void nd_copy_host(const void* const source_base, void* const dest_base, const range<3>& source_range, const range<3>& dest_range,
    const id<3>& offset_in_source, const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size) //
{
	const auto layout = layout_nd_copy(source_range, dest_range, offset_in_source, offset_in_dest, copy_range, elem_size);
	for_each_contiguous_chunk(layout, [&](const size_t chunk_offset_in_source, const size_t chunk_offset_in_dest, const size_t chunk_size) {
		memcpy(static_cast<std::byte*>(dest_base) + chunk_offset_in_dest, static_cast<const std::byte*>(source_base) + chunk_offset_in_source, chunk_size);
	});
}

/// From allocation `source_base` spanning `source_box` to allocation `dest_base` spanning `dest_box`, copy `copy_box` elements of `elem_size` bytes.
inline void nd_copy_host(
    const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size) //
{
	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));
	nd_copy_host(source_base, dest_base, source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
	    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size);
}

/// From allocation `source_base` spanning `source_box` to allocation `dest_base` spanning `dest_box`, copy `copy_region` elements of `elem_size` bytes.
inline void nd_copy_host(const void* const source_base, void* const dest_base, const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size) //
{
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [elem_size](const void* source, void* dest, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_host(source, dest, source_box, dest_box, copy_box, elem_size);
	    },
	    [](const void* const source, void* const dest, const size_t size_bytes) { memcpy(dest, source, size_bytes); });
}

} // namespace celerity::detail
