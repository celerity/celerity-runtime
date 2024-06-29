#pragma once

#include <sycl/sycl.hpp>

#include "legacy_backend/generic_backend.h"
#include "legacy_backend/traits.h"
#include "legacy_backend/type.h"

// NOTE: These should not leak any symbols from the backend library (i.e. don't include it in the header)
#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
#include "legacy_backend/cuda_backend.h"
#endif

// Helper function to instantiate `Template` (during compile time) based on the backend type (a runtime value).
namespace celerity::detail::legacy_backend_detail {
template <template <legacy_backend::type> typename Template, typename Callback>
auto specialize_for_backend(legacy_backend::type type, Callback cb) {
	switch(type) {
	case legacy_backend::type::cuda: return cb(Template<legacy_backend::type::cuda>{});
	case legacy_backend::type::generic: return cb(Template<legacy_backend::type::generic>{});
	case legacy_backend::type::unknown: [[fallthrough]];
	default: return cb(Template<legacy_backend::type::unknown>{});
	}
}
} // namespace celerity::detail::legacy_backend_detail

namespace celerity::detail::legacy_backend {

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
	return legacy_backend_detail::specialize_for_backend<legacy_backend_detail::name>(type, [](auto op) { return decltype(op)::value; });
}

template <int Dims>
void memcpy_strided_device(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<Dims>& source_range,
    const id<Dims>& source_offset, const range<Dims>& target_range, const id<Dims>& target_offset, const range<Dims>& copy_range) {
	legacy_backend_detail::specialize_for_backend<legacy_backend_detail::backend_operations>(get_effective_type(queue.get_device()), [&](auto op) {
		decltype(op)::memcpy_strided_device(
		    queue, source_base_ptr, target_base_ptr, elem_size, source_range, source_offset, target_range, target_offset, copy_range);
	});
}

} // namespace celerity::detail::legacy_backend
