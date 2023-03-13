#pragma once

#include <sycl/sycl.hpp>

#include "backend/async_event.h"
#include "backend/generic_backend.h"
#include "backend/operations.h"
#include "backend/traits.h"
#include "backend/type.h"

// NOTE: These should not leak any symbols from the backend library (i.e. don't include it in the header)
#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
#include "backend/cuda_backend.h"
#endif

namespace celerity::detail::backend_detail {
template <template <backend::type> typename Template, typename Callback>
auto specialize_for_backend(backend::type type, Callback cb) {
	switch(type) {
	case backend::type::cuda: return cb(Template<backend::type::cuda>{});
	case backend::type::generic: return cb(Template<backend::type::generic>{});
	case backend::type::unknown: [[fallthrough]];
	default: return cb(Template<backend::type::unknown>{});
	}
}
} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

/**
 * Returns the detected backend type for this SYCL device.
 *
 * Returns either a specialized backend or 'unknown', never 'generic'.
 */
type get_type(const sycl::device& device);

/**
 * Returns the effective backend type for this SYCL device, depending on the detected
 * backend type and which backend modules have been compiled.
 *
 * Returns either a specialized backend or 'generic', never 'unknown'.
 */
type get_effective_type(const sycl::device& device);

inline std::string_view get_name(type type) {
	return backend_detail::specialize_for_backend<backend_detail::name>(type, [](auto op) { return decltype(op)::value; });
}

template <int Dims>
async_event memcpy_strided_device(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<Dims>& source_range, const sycl::id<Dims>& source_offset, const sycl::range<Dims>& target_range, const sycl::id<Dims>& target_offset,
    const sycl::range<Dims>& copy_range, void* HACK_backend_context) {
	return backend_detail::specialize_for_backend<backend_detail::backend_operations>(get_effective_type(queue.get_device()), [&](auto op) {
		return decltype(op)::memcpy_strided_device(
		    queue, source_base_ptr, target_base_ptr, elem_size, source_range, source_offset, target_range, target_offset, copy_range, HACK_backend_context);
	});
}

} // namespace celerity::detail::backend