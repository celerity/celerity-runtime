#include "buffer_storage.h"

namespace celerity {
namespace detail {
	size_t get_linear_index(const cl::sycl::range<1>& buffer_range, const cl::sycl::id<1>& index) { return index[0]; }

	size_t get_linear_index(const cl::sycl::range<2>& buffer_range, const cl::sycl::id<2>& index) { return index[0] * buffer_range[1] + index[1]; }

	size_t get_linear_index(const cl::sycl::range<3>& buffer_range, const cl::sycl::id<3>& index) {
		return index[0] * buffer_range[1] * buffer_range[2] + index[1] * buffer_range[2] + index[2];
	}

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<1>& source_range,
	    const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range, const cl::sycl::id<1>& target_offset,
	    const cl::sycl::range<1>& copy_range) {
		const size_t line_size = elem_size * copy_range[0];
		std::memcpy(reinterpret_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
		    reinterpret_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size);
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<2>& source_range,
	    const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range, const cl::sycl::id<2>& target_offset,
	    const cl::sycl::range<2>& copy_range) {
		const size_t line_size = elem_size * copy_range[1];
		const auto source_base_offset = get_linear_index(source_range, source_offset);
		const auto target_base_offset = get_linear_index(target_range, target_offset);
		for(size_t i = 0; i < copy_range[0]; ++i) {
			std::memcpy(reinterpret_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
			    reinterpret_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
		}
	}

	// TODO Optimize for contiguous copies?
	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<3>& source_range,
	    const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range, const cl::sycl::id<3>& target_offset,
	    const cl::sycl::range<3>& copy_range) {
		// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
		const auto source_base_offset =
		    get_linear_index(source_range, source_offset) - get_linear_index({source_range[1], source_range[2]}, {source_offset[1], source_offset[2]});
		const auto target_base_offset =
		    get_linear_index(target_range, target_offset) - get_linear_index({target_range[1], target_range[2]}, {target_offset[1], target_offset[2]});
		for(size_t i = 0; i < copy_range[0]; ++i) {
			const auto source_ptr = reinterpret_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
			const auto target_ptr = reinterpret_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
			memcpy_strided(source_ptr, target_ptr, elem_size, {source_range[1], source_range[2]}, {source_offset[1], source_offset[2]},
			    {target_range[1], target_range[2]}, {target_offset[1], target_offset[2]}, {copy_range[1], copy_range[2]});
		}
	}

	raw_buffer_data raw_buffer_data::copy(cl::sycl::id<3> offset, cl::sycl::range<3> copy_range) {
		assert((id_cast<3>(offset) < id_cast<3>(range)) == cl::sycl::id<3>(1, 1, 1));
		assert((id_cast<3>(offset + copy_range) <= id_cast<3>(range)) == cl::sycl::id<3>(1, 1, 1));
		raw_buffer_data result(elem_size, range_cast<3>(copy_range));

		if(range[2] == 1) {
			if(range[1] == 1) {
				memcpy_strided(data.get(), result.get_pointer(), elem_size, range_cast<1>(range), range_cast<1>(offset), range_cast<1>(copy_range),
				    cl::sycl::id<1>(0), range_cast<1>(copy_range));
			} else {
				memcpy_strided(data.get(), result.get_pointer(), elem_size, range_cast<2>(range), range_cast<2>(offset), range_cast<2>(copy_range),
				    cl::sycl::id<2>(0, 0), range_cast<2>(copy_range));
			}
		} else {
			memcpy_strided(data.get(), result.get_pointer(), elem_size, range_cast<3>(range), offset, copy_range, cl::sycl::id<3>(0, 0, 0), copy_range);
		}

		return result;
	}
} // namespace detail
} // namespace celerity
