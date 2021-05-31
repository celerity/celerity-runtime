/**
 * @file
 * This whole file is just a 'compatibility' layer for ComputeCPP 2.5.0 until they provide support for the SYCL 2020 features specified below.
 */
#pragma once

#include "workaround.h"

#if WORKAROUND_COMPUTECPP

#include <CL/sycl.hpp>


namespace cl::sycl {
using access_mode = cl::sycl::access::mode;
enum class target {
	device = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::global_buffer),
	host_task = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::host_buffer),
	global_buffer = device,
	constant_buffer = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::constant_buffer),
	local = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::local),
	host_buffer = static_cast<std::underlying_type_t<cl::sycl::access::target>>(cl::sycl::access::target::host_buffer)
};

namespace detail {

	template <cl::sycl::target Target>
	constexpr cl::sycl::access::target ccpp_target_2_acc() {
		return static_cast<cl::sycl::access::target>(Target);
	}


	template <cl::sycl::access::target Target>
	constexpr cl::sycl::target ccpp_acc_2_target() {
		return static_cast<cl::sycl::target>(Target);
	}

	struct read_only_tag_t {};
	struct read_write_tag_t {};
	struct write_only_tag_t {};
	struct read_only_host_task_tag_t {};
	struct read_write_host_task_tag_t {};
	struct write_only_host_task_tag_t {};

} // namespace detail

inline constexpr detail::read_only_tag_t read_only;
inline constexpr detail::read_write_tag_t read_write;
inline constexpr detail::write_only_tag_t write_only;
inline constexpr detail::read_only_host_task_tag_t read_only_host_task;
inline constexpr detail::read_write_host_task_tag_t read_write_host_task;
inline constexpr detail::write_only_host_task_tag_t write_only_host_task;

namespace property {
	struct no_init : detail::property_base {
		no_init() : detail::property_base(static_cast<detail::property_enum>(0)) {}
	};
} // namespace property
inline property::no_init no_init;

}; // namespace cl::sycl

#endif