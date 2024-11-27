#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>


namespace celerity::detail {

using memory_scope_ut = std::underlying_type_t<sycl::memory_scope>;

} // namespace celerity::detail

namespace celerity {

using sycl::property_list;

namespace property {
	using sycl::property::no_init;
}

namespace property::reduction {
	using sycl::property::reduction::initialize_to_identity;
}

inline const property::no_init no_init;

// We do not expose memory_scope::system because Celerity does not expose USM or multi-GPU kernel launches
enum class memory_scope : detail::memory_scope_ut {
	work_item = static_cast<detail::memory_scope_ut>(sycl::memory_scope::work_item),
	sub_group = static_cast<detail::memory_scope_ut>(sycl::memory_scope::sub_group),
	work_group = static_cast<detail::memory_scope_ut>(sycl::memory_scope::work_group),
	device = static_cast<detail::memory_scope_ut>(sycl::memory_scope::device)
};

inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;

template <typename T>
using decorated_global_ptr = sycl::global_ptr<T>;

template <typename T>
using decorated_local_ptr = sycl::local_ptr<T>;

} // namespace celerity
