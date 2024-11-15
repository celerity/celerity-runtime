#pragma once

#include "expert_mapper.h"
#include "grid.h"
#include "hint.h"
#include "range_mapper.h"
#include "ranges.h"
#include "reduction.h"
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

// NOTES ON TASK GEOMETRIES:
// - Whatever the frontend interface will look like, in the end we will have to pass a list of chunks to the task manager, CGGEN and IGGEN
// - Chunks may or may not be assigned to nodes and devices
//		- At first only support fully assigned
// - Via notes on expert mapper: In theory we don't need to know what the remote chunks are at all. We only need to know which peer
//   requires data from us. Hmm...
//		=> WELL: Only if we don't use ANY normal range mappers
//		=> Make sure to detect this case, i.e., if someone only wants to specify local chunks, they have to somehow explicitly state that there
//         are no remote chunks (or not use normal ranger mappers). Otherwise throw an error.
// - Q: What happens when passing a geometry w/ device assignments into a host_task?
// - Q: How do we specify ND-range kernels?
// - Q: Should we allow the same chunk / overlapping chunks to be executed on multiple nodes? Not sure what the implications would be, let's say no for now.
//		=> PROBLEM: I think we need it for the stencil optimization proposed by Peter. BUT: How do we then identify chunks for e.g. data access or scratch
//         buffers?! Do we need a "chunk id" after all? UGH.
// - Q: Do we always have to have a "global" domain in which chunks exist?
//		- If not, we couldn't define global indices - is that a problem?
//		  Or rather, each chunk would have its own separate index space. In a sense those chunks would be overlapping.
// - Q: If a custom geometry needs to have a global size, how does that interact with algebraic operations (e.g. removing a boundary from a domain).
// 		=> Maybe the global size should be implicitly computed from the bounding box of all chunks...?
// 		   Would this bounding box always start at 0,0? Would anything else make sense..?

// TODO API: Naming
struct geometry_chunk {
	// TODO API: Subrange or box? Or both?
	subrange<3> sr;
	detail::node_id nid;
	std::optional<detail::device_id> did;

	bool operator==(const geometry_chunk&) const = default;
};

// TODO API: Naming
// TODO API: This should probably use <Dims> ranges/ids
// TODO: Overlap with basic_task_geometry
template <int Dims = 1>
struct custom_task_geometry {
	range<3> global_size{1, 1, 1};
	id<3> global_offset;
	range<3> local_size{1, 1, 1}; // FIXME: Figure out how nd-range kernels work w/ custom geometries

	std::vector<geometry_chunk> assigned_chunks;
};

class handler;

} // namespace celerity

namespace celerity::detail {

class communicator;

// FIXME: We need the same thing but w/o template parameter
struct custom_task_geometry_desc {
	int dimensions = 0;
	range<3> global_size;
	id<3> global_offset;
	range<3> local_size;
	std::vector<geometry_chunk> assigned_chunks;

	template <int Dims>
	explicit(false) custom_task_geometry_desc(custom_task_geometry<Dims> geo)
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

// TODO: Move elsewhere
struct performance_assertions {
	task_id tid;                 // HACK
	std::string task_debug_name; // HACK

	bool assert_no_data_movement = false;
	std::source_location assert_no_data_movement_source_loc;

	bool assert_no_allocations = false;
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
