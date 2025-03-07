#pragma once

#include "dense_map.h"
#include "types.h"

#include <bitset>
#include <cstddef>


namespace celerity::detail {

/// Clang as of 17.0.0 does not expose std::hardware_destructive_interference_size because it has issues around -march / -mcpu flags among others
/// (see discussion at https://discourse.llvm.org/t/rfc-c-17-hardware-constructive-destructive-interference-size/48674).
/// To keep it simple we conservatively pick an alignment of 128 bytes to avoid false sharing across all relevant architectures.
/// Aarch64 and PowerPC64 have 128-byte cache lines; and x86_64 prefetches 64-byte cache lines in pairs starting from Sandy Bridge
/// (see https://github.com/crossbeam-rs/crossbeam/blob/e7b5922e/crossbeam-utils/src/cache_padded.rs for a detailed enumeration by architecture).
constexpr size_t hardware_destructive_interference_size = 128;

/// Memory id for (unpinned) host memory allocated for or by the user. This memory id is assumed for pointers passed for buffer host-initialization and for the
/// explicit user-side allocation of a buffer_snapshot that is performed before a buffer fence.
inline constexpr memory_id user_memory_id = 0;

/// Memory id for (pinned) host memory that the executor will obtain from the backend for buffer allocations and staging buffers.
inline constexpr memory_id host_memory_id = 1;

/// Memory id for the first device-native memory, if any.
inline constexpr memory_id first_device_memory_id = 2;

static constexpr size_t max_num_memories = 64;
using memory_mask = std::bitset<max_num_memories>;

/// Information about a single device in the local system.
struct device_info {
	/// Before accessing any memory on a device, instruction_graph_generator will prepare a corresponding allocation on its `native_memory`. Multiple
	/// devices can share the same native memory. No attempts at reading from peer or shared memory to elide copies are currently made, but could be in the
	/// future.
	memory_id native_memory = -1;
};

/// Information about a single memory in the local system.
struct memory_info {
	/// This mask contains a 1-bit for every memory_id that the associated backend queue can copy data from or to directly. instruction_graph_generator
	/// expects this mapping to be reflexive, i.e. `system_info::memories[a].copy_peers[b] == system_info::memories[b].copy_peers[a]`.
	/// Further, copies must always be possible between `host_memory_id` and `user_memory_id` as well as between `host_memory_id` and every other memory.
	/// instruction_graph_generator will create a staging copy in host memory if data must be transferred between two memories that are not copy peers.
	memory_mask copy_peers;
};

/// All information about the local system that influences the generated instruction graph.
struct system_info {
	dense_map<device_id, device_info> devices;
	dense_map<memory_id, memory_info> memories;
};

} // namespace celerity::detail
