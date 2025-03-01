#pragma once

#include "expert_mapper.h"
#include "grid.h"
#include "hint.h"
#include "range_mapper.h"
#include "ranges.h"
#include "reduction.h"
#include "task_geometry.h"
#include "types.h"

#include <functional>
#include <memory>
#include <optional>
#include <source_location>
#include <variant>
#include <vector>

#include <matchbox.hh>
#include <sycl/sycl.hpp>


namespace celerity {

class handler;

} // namespace celerity

namespace celerity::detail {

class communicator;

// FIXME: We need the same thing but w/o template parameter
struct custom_task_geometry_desc {
	int dimensions = 0;
	range<3> global_size;
	id<3> global_offset;
	range<3> local_size; // TODO: Here and for the other types - why do we even store this?
	                     // => For basic geometry we implicitly store this in granularity, and
	                     // the task launcher already knows the local size
	std::vector<geometry_chunk> assigned_chunks;

	template <int Dims>
	explicit(false) custom_task_geometry_desc(custom_task_geometry<Dims> geo)
	    : dimensions(Dims), global_size(geo.global_size), global_offset(geo.global_offset), local_size(geo.local_size),
	      assigned_chunks(std::move(geo.assigned_chunks)) {}

	template <int Dims>
	explicit(false) custom_task_geometry_desc(nd_custom_task_geometry<Dims> geo)
	    : dimensions(Dims), global_size(geo.global_size), global_offset(geo.global_offset), local_size(geo.local_size),
	      assigned_chunks(std::move(geo.assigned_chunks)) {}

	bool operator==(const custom_task_geometry_desc&) const = default;
};

// TODO: Or "automatic" task geometry?
struct basic_task_geometry {
	int dimensions = 0;
	range<3> global_size{1, 1, 1};
	id<3> global_offset;
	range<3> local_size{1, 1, 1};
	range<3> granularity{1, 1, 1}; ///< Like local_size, but potentially further constrained

	bool operator==(const basic_task_geometry&) const = default;
};

using task_geometry = std::variant<basic_task_geometry, custom_task_geometry_desc>;

inline int get_dimensions(const task_geometry& geo) {
	return matchbox::match(geo, [](auto& g) { return g.dimensions; });
}
inline range<3> get_global_size(const task_geometry& geo) {
	return matchbox::match(geo, [](auto& g) { return g.global_size; });
}
inline id<3> get_global_offset(const task_geometry& geo) {
	return matchbox::match(geo, [](auto& g) { return g.global_offset; });
}
inline range<3> get_local_size(const task_geometry& geo) {
	return matchbox::match(geo, [](auto& g) { return g.local_size; });
}

struct buffer_access {
	buffer_id bid = -1;
	access_mode mode = access_mode::read;
	std::variant<std::unique_ptr<range_mapper_base>, expert_mapper> range_mapper; // NOCOMMIT Naming
	bool is_replicated = false;
};

struct host_object_effect {
	host_object_id hoid = -1;
	experimental::side_effect_order order = experimental::side_effect_order::sequential;
};

using device_kernel_launcher = std::function<void(sycl::handler& sycl_cgh, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs)>;
using host_task_launcher = std::function<void(const range<3>& global_range, const box<3>& execution_range, const communicator* collective_comm)>;
using command_group_launcher = std::variant<device_kernel_launcher, host_task_launcher>;

// TODO: Move elsewhere (and into public namespace!)
// TODO: Should it be "any" or "all"?
// TODO: Should we also distinguish between intra node host and device, and within same device? Former is tricky because of p2p emulation
enum class data_movement_scope { any, inter_node, intra_node };
enum class allocation_scope { any, host, device };

// TODO: Move elsewhere
struct performance_assertions {
	task_id tid;                 // HACK
	std::string task_debug_name; // HACK

	bool assert_no_data_movement = false;
	data_movement_scope assert_no_data_movement_scope = data_movement_scope::any;
	std::source_location assert_no_data_movement_source_loc;

	bool assert_no_allocations = false;
	allocation_scope assert_no_allocations_scope = allocation_scope::any;
	std::source_location assert_no_allocations_source_loc;
};

/// Captures the raw contents of a command group function (CGF) invocation as recorded by `celerity::handler`.
/// This is passed on to `task_manager`, which validates the structure and turns it into a TDAG node.
struct raw_command_group {
	std::optional<detail::task_type> task_type; ///< nullopt until a kernel or host task has been submitted
	std::vector<buffer_access> buffer_accesses;
	std::vector<host_object_effect> side_effects;
	std::vector<reduction_info> reductions;
	std::optional<detail::collective_group_id> collective_group_id;
	std::optional<task_geometry> geometry;
	std::optional<detail::command_group_launcher> launcher;
	std::optional<std::string> task_name;
	std::vector<std::unique_ptr<detail::hint_base>> hints;
	performance_assertions perf_assertions;
};

class task_promise {
  public:
	task_promise() = default;
	task_promise(const task_promise&) = delete;
	task_promise(task_promise&&) = delete;
	task_promise& operator=(const task_promise&) = delete;
	task_promise& operator=(task_promise&&) = delete;
	virtual ~task_promise() = default;

	virtual void fulfill() = 0;
	virtual allocation_id get_user_allocation_id() = 0; // TODO move to struct task instead
};

} // namespace celerity::detail
