#pragma once

#include "workaround.h"

#include <CL/sycl.hpp>

namespace celerity {

namespace detail {

	struct read_only_tag_t {};
	struct read_write_tag_t {};
	struct write_only_tag_t {};
	struct read_only_host_task_tag_t {};
	struct read_write_host_task_tag_t {};
	struct write_only_host_task_tag_t {};

#if !WORKAROUND_COMPUTECPP // no memory_scope
	using memory_scope_ut = std::underlying_type_t<cl::sycl::memory_scope>;
#endif

} // namespace detail

using access_mode = cl::sycl::access::mode;

enum class target {
	device,
	host_task,
};

inline constexpr detail::read_only_tag_t read_only;
inline constexpr detail::read_write_tag_t read_write;
inline constexpr detail::write_only_tag_t write_only;
inline constexpr detail::read_only_host_task_tag_t read_only_host_task;
inline constexpr detail::read_write_host_task_tag_t read_write_host_task;
inline constexpr detail::write_only_host_task_tag_t write_only_host_task;

namespace property {
#if WORKAROUND_COMPUTECPP
	struct no_init : cl::sycl::detail::property_base {
		no_init() : cl::sycl::detail::property_base(static_cast<cl::sycl::detail::property_enum>(0)) {}
	};
#else
	using no_init = cl::sycl::property::no_init;
#endif
} // namespace property

inline const property::no_init no_init;

// We do not expose memory_scope::system because Celerity does not expose USM or multi-GPU kernel launches
#if WORKAROUND_COMPUTECPP // no memory_scope
enum class memory_scope { work_item, sub_group, work_group, device };
#else
enum class memory_scope : detail::memory_scope_ut {
	work_item = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::work_item),
	sub_group = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::sub_group),
	work_group = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::work_group),
	device = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::device)
};
#endif

inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;

template <typename T>
using decorated_global_ptr = cl::sycl::global_ptr<T>;

template <typename T>
using decorated_local_ptr = cl::sycl::local_ptr<T>;

} // namespace celerity
