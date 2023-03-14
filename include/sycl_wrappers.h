#pragma once

#include "workaround.h"

#include <CL/sycl.hpp>

namespace celerity {

using access_mode = cl::sycl::access::mode;

enum class target {
	device,
	host_task,
};

} // namespace celerity

namespace celerity::detail {

template <access_mode Mode, access_mode NoInitMode, target Target>
struct access_tag {};

using memory_scope_ut = std::underlying_type_t<cl::sycl::memory_scope>;

} // namespace celerity::detail

namespace celerity {

inline constexpr detail::access_tag<access_mode::read, access_mode::read, target::device> read_only;
inline constexpr detail::access_tag<access_mode::write, access_mode::discard_write, target::device> write_only;
inline constexpr detail::access_tag<access_mode::read_write, access_mode::discard_read_write, target::device> read_write;
inline constexpr detail::access_tag<access_mode::read, access_mode::read, target::host_task> read_only_host_task;
inline constexpr detail::access_tag<access_mode::write, access_mode::discard_write, target::host_task> write_only_host_task;
inline constexpr detail::access_tag<access_mode::read_write, access_mode::discard_read_write, target::host_task> read_write_host_task;

using cl::sycl::property_list;

namespace property {
#if CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 8)
	struct no_init : cl::sycl::detail::property_base {
		no_init() : cl::sycl::detail::property_base(static_cast<cl::sycl::detail::property_enum>(0)) {}
	};
#else
	using cl::sycl::property::no_init;
#endif

#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
	namespace reduction {
		using cl::sycl::property::reduction::initialize_to_identity;
	}
#endif
} // namespace property

inline const property::no_init no_init;

// We do not expose memory_scope::system because Celerity does not expose USM or multi-GPU kernel launches
enum class memory_scope : detail::memory_scope_ut {
	work_item = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::work_item),
	sub_group = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::sub_group),
	work_group = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::work_group),
	device = static_cast<detail::memory_scope_ut>(cl::sycl::memory_scope::device)
};

inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;

template <typename T>
using decorated_global_ptr = cl::sycl::global_ptr<T>;

template <typename T>
using decorated_local_ptr = cl::sycl::local_ptr<T>;
} // namespace celerity
