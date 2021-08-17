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

} // namespace detail

using access_mode = cl::sycl::access::mode;

enum class target : std::underlying_type_t<cl::sycl::access::target> {
	device = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::global_buffer),
	host_task = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::host_buffer),
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

} // namespace celerity
