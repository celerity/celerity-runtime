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

using cl::sycl::id;
using cl::sycl::range;

// We re-implement nd_range to un-deprecate kernel offsets
template <int Dims = 1>
class nd_range {
  public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	nd_range(cl::sycl::nd_range<Dims> s_range)
	    : global_range(s_range.get_global_range()), local_range(s_range.get_local_range()), offset(s_range.get_offset()) {}
#pragma GCC diagnostic pop

	nd_range(range<Dims> global_range, range<Dims> local_range, id<Dims> offset = {}) : global_range(global_range), local_range(local_range), offset(offset) {
#ifndef __SYCL_DEVICE_ONLY__
		for(int d = 0; d < Dims; ++d) {
			if(local_range[d] == 0 || global_range[d] % local_range[d] != 0) { throw std::invalid_argument("global_range is not divisible by local_range"); }
		}
#endif
	}

	operator cl::sycl::nd_range<Dims>() const { return cl::sycl::nd_range<Dims>{global_range, local_range, offset}; }

	range<Dims> get_global_range() const { return global_range; }
	range<Dims> get_local_range() const { return local_range; }
	range<Dims> get_group_range() const { return global_range / local_range; }
	id<Dims> get_offset() const { return offset; }

	friend bool operator==(const nd_range& lhs, const nd_range& rhs) {
		return lhs.global_range == rhs.global_range && lhs.local_range == rhs.local_range && lhs.offset == rhs.offset;
	}

	friend bool operator!=(const nd_range& lhs, const nd_range& rhs) { return !(lhs == rhs); }

  private:
	range<Dims> global_range;
	range<Dims> local_range;
	id<Dims> offset;
};

// Non-templated deduction guides allow construction of nd_range from range initializer lists like so: nd_range{{1, 2}, {3, 4}}
// ... except, currently, for ComputeCpp which uses an outdated Clang (TODO)
nd_range(range<1> global_range, range<1> local_range, id<1> offset)->nd_range<1>;
nd_range(range<1> global_range, range<1> local_range)->nd_range<1>;
nd_range(range<2> global_range, range<2> local_range, id<2> offset)->nd_range<2>;
nd_range(range<2> global_range, range<2> local_range)->nd_range<2>;
nd_range(range<3> global_range, range<3> local_range, id<3> offset)->nd_range<3>;
nd_range(range<3> global_range, range<3> local_range)->nd_range<3>;

} // namespace celerity
