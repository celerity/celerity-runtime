#pragma once

#include <sycl/sycl.hpp>

#include "backend/operations.h"
#include "backend/type.h"

namespace celerity::detail::backend_detail {

void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const sycl::range<1>& source_range,
    const sycl::id<1>& source_offset, const sycl::range<1>& target_range, const sycl::id<1>& target_offset, const sycl::range<1>& copy_range);

void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const sycl::range<2>& source_range,
    const sycl::id<2>& source_offset, const sycl::range<2>& target_range, const sycl::id<2>& target_offset, const sycl::range<2>& copy_range);

void memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const sycl::range<3>& source_range,
    const sycl::id<3>& source_offset, const sycl::range<3>& target_range, const sycl::id<3>& target_offset, const sycl::range<3>& copy_range);

template <>
struct backend_operations<backend::type::generic> {
	template <typename... Args>
	static void memcpy_strided_device(Args&&... args) {
		memcpy_strided_device_generic(args...);
	}
};

} // namespace celerity::detail::backend_detail
