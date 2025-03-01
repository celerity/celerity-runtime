#pragma once

#include "grid.h"
#include "types.h"

#include <optional>
#include <vector>

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
	// TODO: Have to publicly expose box
	detail::box<3> box;
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
	id<3> global_offset; // NOCOMMIT TODO: Why even have this? => Same reason we still need global_size - interop w/ range mappers
	                     // BUT DO WE actually need global_size/offset? Currently BAM applies RMs to all chunks and computes union
	                     // Using global_size/offset would only be correct if we have a proper partitioning of the domain, no gaps / overlaps
	range<3> local_size{1, 1, 1};

	std::vector<geometry_chunk> assigned_chunks;
};

// TODO API: Naming
// TODO: Also, instead of making a separate public type, we could have something like custom_task_geometry::as_nd_range()
template <int Dims = 1>
struct nd_custom_task_geometry {
	range<3> global_size{1, 1, 1};
	id<3> global_offset; // NOCOMMIT TODO: Why even have this?
	range<3> local_size{1, 1, 1};

	// NOCOMMIT TODO: Verify that chunks are divisible by local size
	std::vector<geometry_chunk> assigned_chunks;
};

} // namespace celerity
