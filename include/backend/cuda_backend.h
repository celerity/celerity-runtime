#pragma once

#include "backend/operations.h"
#include "backend/type.h"
#include "ranges.h"

namespace celerity::detail::backend_detail {

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const celerity::range<0>& source_range, const celerity::id<0>& source_offset, const celerity::range<0>& target_range, const celerity::id<0>& target_offset,
    const celerity::range<0>& copy_range);

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const celerity::range<1>& source_range, const celerity::id<1>& source_offset, const celerity::range<1>& target_range, const celerity::id<1>& target_offset,
    const celerity::range<1>& copy_range);

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const celerity::range<2>& source_range, const celerity::id<2>& source_offset, const celerity::range<2>& target_range, const celerity::id<2>& target_offset,
    const celerity::range<2>& copy_range);

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const celerity::range<3>& source_range, const celerity::id<3>& source_offset, const celerity::range<3>& target_range, const celerity::id<3>& target_offset,
    const celerity::range<3>& copy_range);

template <>
struct backend_operations<backend::type::cuda> {
	template <typename... Args>
	static void memcpy_strided_device(Args&&... args) {
		memcpy_strided_device_cuda(args...);
	}
};

} // namespace celerity::detail::backend_detail
