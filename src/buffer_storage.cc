#include "buffer_storage.h"

namespace celerity {
namespace detail {

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const celerity::range<0>& /* source_range */,
	    const celerity::id<0>& /* source_offset */, const celerity::range<0>& /* target_range */, const celerity::id<0>& /* target_offset */,
	    const celerity::range<0>& /* copy_range */) {
		std::memcpy(target_base_ptr, source_base_ptr, elem_size);
	}

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const celerity::range<1>& source_range,
	    const celerity::id<1>& source_offset, const celerity::range<1>& target_range, const celerity::id<1>& target_offset,
	    const celerity::range<1>& copy_range) {
		const size_t line_size = elem_size * copy_range[0];
		std::memcpy(static_cast<std::byte*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
		    static_cast<const std::byte*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size);
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const celerity::range<2>& source_range,
	    const celerity::id<2>& source_offset, const celerity::range<2>& target_range, const celerity::id<2>& target_offset,
	    const celerity::range<2>& copy_range) {
		const size_t line_size = elem_size * copy_range[1];
		const auto source_base_offset = get_linear_index(source_range, source_offset);
		const auto target_base_offset = get_linear_index(target_range, target_offset);
		for(size_t i = 0; i < copy_range[0]; ++i) {
			std::memcpy(static_cast<std::byte*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
			    static_cast<const std::byte*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
		}
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const celerity::range<3>& source_range,
	    const celerity::id<3>& source_offset, const celerity::range<3>& target_range, const celerity::id<3>& target_offset,
	    const celerity::range<3>& copy_range) {
		// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
		const auto source_base_offset = get_linear_index(source_range, source_offset)
		                                - get_linear_index(celerity::range<2>{source_range[1], source_range[2]}, {source_offset[1], source_offset[2]});
		const auto target_base_offset = get_linear_index(target_range, target_offset)
		                                - get_linear_index(celerity::range<2>{target_range[1], target_range[2]}, {target_offset[1], target_offset[2]});
		for(size_t i = 0; i < copy_range[0]; ++i) {
			const auto* const source_ptr =
			    static_cast<const std::byte*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
			auto* const target_ptr = static_cast<std::byte*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
			memcpy_strided_host(source_ptr, target_ptr, elem_size, celerity::range<2>{source_range[1], source_range[2]}, {source_offset[1], source_offset[2]},
			    {target_range[1], target_range[2]}, {target_offset[1], target_offset[2]}, {copy_range[1], copy_range[2]});
		}
	}

	void linearize_subrange(const void* source_base_ptr, void* target_ptr, size_t elem_size, const range<3>& source_range, const subrange<3>& copy_sr) {
		assert((id_cast<3>(copy_sr.offset) < id_cast<3>(source_range)) == celerity::id<3>(1, 1, 1));
		assert((id_cast<3>(copy_sr.offset + copy_sr.range) <= id_cast<3>(source_range)) == celerity::id<3>(1, 1, 1));

		if(source_range[2] <= 1) {
			if(source_range[1] <= 1) {
				memcpy_strided_host(source_base_ptr, target_ptr, elem_size, range_cast<1>(source_range), range_cast<1>(copy_sr.offset),
				    range_cast<1>(copy_sr.range), celerity::id<1>(0), range_cast<1>(copy_sr.range));
			} else {
				memcpy_strided_host(source_base_ptr, target_ptr, elem_size, range_cast<2>(source_range), range_cast<2>(copy_sr.offset),
				    range_cast<2>(copy_sr.range), celerity::id<2>(0, 0), range_cast<2>(copy_sr.range));
			}
		} else {
			memcpy_strided_host(
			    source_base_ptr, target_ptr, elem_size, range_cast<3>(source_range), copy_sr.offset, copy_sr.range, celerity::id<3>(0, 0, 0), copy_sr.range);
		}
	}

} // namespace detail
} // namespace celerity
