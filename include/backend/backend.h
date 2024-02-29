#pragma once

#include <sycl/sycl.hpp>

#include "backend/queue.h"
#include "backend/traits.h"
#include "backend/type.h"

// TODO this is only used for `get_name` - maybe we can get rid of it?
// Helper function to instantiate `Template` (during compile time) based on the backend type (a runtime value).
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

[[nodiscard]] bool enable_copy_between_peer_memories(sycl::device& a, sycl::device& b);

inline std::string_view get_name(type type) {
	return backend_detail::specialize_for_backend<backend_detail::name>(type, [](auto op) { return decltype(op)::value; });
}

std::unique_ptr<queue> make_queue(type t, const std::vector<device_config>& devices);

} // namespace celerity::detail::backend
