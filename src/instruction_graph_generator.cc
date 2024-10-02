#include "instruction_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "grid.h"
#include "instruction_graph.h"
#include "recorders.h"
#include "region_map.h"
#include "split.h"
#include "system_info.h"
#include "task.h"
#include "task_manager.h"
#include "tracy.h"
#include "types.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace celerity::detail::instruction_graph_generator_detail {

/// Helper for split_into_communicator_compatible_boxes().
void split_into_communicator_compatible_boxes_recurse(
    box_vector<3>& compatible_boxes, const box<3>& send_box, id<3> min, id<3> max, const int compatible_starting_from_dim, const int dim) {
	assert(dim <= compatible_starting_from_dim);
	const auto& full_box_min = send_box.get_min();
	const auto& full_box_max = send_box.get_max();

	if(dim == compatible_starting_from_dim) {
		// There are no incompatible strides in faster dimensions, so we simply split along this dimension into communicator_max_coordinate-sized chunks
		for(min[dim] = full_box_min[dim]; min[dim] < full_box_max[dim]; min[dim] += communicator_max_coordinate) {
			max[dim] = std::min(full_box_max[dim], min[dim] + communicator_max_coordinate);
			compatible_boxes.emplace_back(min, max);
		}
	} else {
		// A fast dimension (> 0) has incompatible strides - we can't do better than iterating over the slow dimension
		for(min[dim] = full_box_min[dim]; min[dim] < full_box_max[dim]; ++min[dim]) {
			max[dim] = min[dim] + 1;
			split_into_communicator_compatible_boxes_recurse(compatible_boxes, send_box, min, max, compatible_starting_from_dim, dim + 1);
		}
	}
}

/// Computes a decomposition of `send_box` into boxes that are compatible with every communicator. Note that this takes `buffer_range` rather than
/// `allocation_range` as a parameter, because a box that is compatible with the sender allocation might not be compatible with the allocation on the receiver
/// side (which we don't know anything about), but splitting for `buffer_range` will guarantee both.
///
/// On the MPI side, we implement buffer transfers as peer-to-peer operations with sub-array datatypes. These are implemented with 32-bit signed integer
/// strides, so transfers on buffers with a range > 2^31 in any dimension might have to be split to work around the coordinate limit by adjusting the base
/// pointer to using the stride as an offset. For 1D buffers this will no practical performance consequences because of the implied transfer sizes, but in
/// degenerate higher-dimensional cases we might end up transferring individual buffer elements per send instruction.
///
/// TODO The proper solution to this is to apply the host-staging mechanism (which is currently in place for copies between devices that don't support peer
/// access) to sends / receives as well.
box_vector<3> split_into_communicator_compatible_boxes(const range<3>& buffer_range, const box<3>& send_box) {
	assert(box(subrange<3>(zeros, buffer_range)).covers(send_box));

	int compatible_starting_from_dim = 0;
	for(int d = 1; d < 3; ++d) {
		if(buffer_range[d] > communicator_max_coordinate) { compatible_starting_from_dim = d; }
	}

	// There are pathological patterns like single-element columns in a 2D buffer with y-extent > communicator_max_coordinate that generate a huge number of
	// individual transfers with a small payload each. This might take very long and / or derail the instruction graph generator, so we log a warning before
	// computing the actual set.
	size_t max_compatible_box_area = 1;
	size_t min_num_compatible_boxes = 1;
	for(int d = 0; d < 3; ++d) {
		(d < compatible_starting_from_dim ? min_num_compatible_boxes : max_compatible_box_area) *= send_box.get_range()[d];
	}
	if(min_num_compatible_boxes > 256 && max_compatible_box_area < 65536) {
		CELERITY_WARN("Celerity is generating an excessive amount of small transfers to keep strides representable as 32-bit integers for MPI compatibility. "
		              "This might be very slow and / or exhaust system memory. Consider transposing your buffer layout to remedy this.");
	}

	box_vector<3> compatible_boxes;
	split_into_communicator_compatible_boxes_recurse(compatible_boxes, send_box, send_box.get_min(), send_box.get_max(), compatible_starting_from_dim, 0);
	return compatible_boxes;
}

/// Determines whether two boxes are either overlapping or touching on edges (not corners). This means on 2-connectivity for 1d boxes, 4-connectivity for 2d
/// boxes and 6-connectivity for 3d boxes. For 0-dimensional boxes, always returns true.
template <int Dims>
bool boxes_edge_connected(const box<Dims>& box1, const box<Dims>& box2) {
	if(box1.empty() || box2.empty()) return false;

	// boxes can be 2/4/6 connected by either fully overlapping, or by overlapping in Dims-1 dimensions and touching (a.max[d] == b.min[d]) in the remaining one
	bool disconnected = false;
	int n_dims_touching = 0;
	for(int d = 0; d < Dims; ++d) {
		// compute the intersection but without normalizing the box to distinguish the "disconnected" from the "touching" case
		const auto min = std::max(box1.get_min()[d], box2.get_min()[d]);
		const auto max = std::min(box1.get_max()[d], box2.get_max()[d]);
		if(min < max) {
			// boxes overlap in this dimension
		} else if(min == max) {
			n_dims_touching += 1;
		} else /* min > max */ {
			disconnected = true;
		}
	}
	return !disconnected && n_dims_touching <= 1;
}

// explicit instantiations for tests
template bool boxes_edge_connected(const box<0>& box1, const box<0>& box2);
template bool boxes_edge_connected(const box<1>& box1, const box<1>& box2);
template bool boxes_edge_connected(const box<2>& box1, const box<2>& box2);
template bool boxes_edge_connected(const box<3>& box1, const box<3>& box2);

/// Subdivide a region into connected partitions (where connectivity is established by `boxes_edge_connected`) and return the bounding box of each partition.
/// Note that the returned boxes are not necessarily disjoint event through the partitions always are.
///
/// This logic is employed to find connected subregions in a pending-receive that might be satisfied by a peer through a single send operation and thus requires
/// a contiguous backing allocation.
template <int Dims>
box_vector<Dims> connected_subregion_bounding_boxes(const region<Dims>& region) {
	auto boxes = region.get_boxes();
	auto begin = boxes.begin();
	auto end = boxes.end();
	box_vector<Dims> bounding_boxes;
	while(begin != end) {
		auto connected_end = std::next(begin);
		auto connected_bounding_box = *begin; // optimization: skip connectivity checks if bounding box is disconnected
		for(; connected_end != end; ++connected_end) {
			const auto next_connected = std::find_if(connected_end, end, [&](const auto& candidate) {
				return boxes_edge_connected(connected_bounding_box, candidate)
				       && std::any_of(begin, connected_end, [&](const auto& box) { return boxes_edge_connected(candidate, box); });
			});
			if(next_connected == end) break;
			connected_bounding_box = bounding_box(connected_bounding_box, *next_connected);
			std::swap(*next_connected, *connected_end);
		}
		bounding_boxes.push_back(connected_bounding_box);
		begin = connected_end;
	}
	return bounding_boxes;
}

// explicit instantiations for tests
template box_vector<0> connected_subregion_bounding_boxes(const region<0>& region);
template box_vector<1> connected_subregion_bounding_boxes(const region<1>& region);
template box_vector<2> connected_subregion_bounding_boxes(const region<2>& region);
template box_vector<3> connected_subregion_bounding_boxes(const region<3>& region);

/// Iteratively replaces each pair of overlapping boxes by their bounding box such that in the modified set of boxes, no two boxes overlap, but all original
/// input boxes are covered.
///
/// When a kernel or host task has multiple accessors into a single allocation, each must be backed by a contiguous allocation. This makes it necessary to
/// contiguously allocate the bounding box of all overlapping accesses.
template <int Dims>
void merge_overlapping_bounding_boxes(box_vector<Dims>& boxes) {
restart:
	for(auto first = boxes.begin(); first != boxes.end(); ++first) {
		const auto last = std::remove_if(std::next(first), boxes.end(), [&](const auto& box) {
			const auto overlapping = !box_intersection(*first, box).empty();
			if(overlapping) { *first = bounding_box(*first, box); }
			return overlapping;
		});
		if(last != boxes.end()) {
			boxes.erase(last, boxes.end());
			goto restart; // NOLINT(cppcoreguidelines-avoid-goto)
		}
	}
}

/// In a set of potentially overlapping regions, removes the overlap between any two regions {A, B} by replacing them with {A ∖ B, A ∩ B, B ∖ A}.
///
/// This is used when generating copy and receive-instructions such that every data item is only copied or received once, but the following device kernels /
/// host tasks have their dependencies satisfied as soon as their subset of input data is available.
template <int Dims>
void symmetrically_split_overlapping_regions(std::vector<region<Dims>>& regions) {
	for(size_t i = 0; i < regions.size(); ++i) {
		for(size_t j = i + 1; j < regions.size(); ++j) {
			auto intersection = region_intersection(regions[i], regions[j]);
			if(!intersection.empty()) {
				// merely shrinking regions will not introduce new intersections downstream, so we do not need to restart the loop
				regions[i] = region_difference(regions[i], intersection);
				regions[j] = region_difference(regions[j], intersection);
				regions.push_back(std::move(intersection));
			}
		}
	}
	// if any of the intersections above are actually subsets, we will end up with empty regions
	regions.erase(std::remove_if(regions.begin(), regions.end(), std::mem_fn(&region<Dims>::empty)), regions.end());
}

// explicit instantiations for tests
template void symmetrically_split_overlapping_regions(std::vector<region<0>>& regions);
template void symmetrically_split_overlapping_regions(std::vector<region<1>>& regions);
template void symmetrically_split_overlapping_regions(std::vector<region<2>>& regions);
template void symmetrically_split_overlapping_regions(std::vector<region<3>>& regions);

/// Returns whether an iterator range of instruction pointers is topologically sorted, i.e. sequential execution would satisfy all internal dependencies.
template <typename Iterator>
bool is_topologically_sorted(Iterator begin, Iterator end) {
	for(auto check = begin; check != end; ++check) {
		for(const auto dep : (*check)->get_dependencies()) {
			if(std::find_if(std::next(check), end, [dep](const auto& node) { return node->get_id() == dep; }) != end) return false;
		}
	}
	return true;
}

/// Starting from `first` (inclusive), selects the next memory_id which is set in `location`.
memory_id next_location(const memory_mask& location, memory_id first) {
	for(size_t i = 0; i < max_num_memories; ++i) {
		const memory_id mem = (first + i) % max_num_memories;
		if(location[mem]) { return mem; }
	}
	utils::panic("data is requested to be read, but not located in any memory");
}

/// Maintains a set of concurrent instructions that are accessing a subrange of a buffer allocation.
/// Instruction pointers are ordered by id to allow equality comparision on the internal vector structure.
class access_front {
  public:
	enum mode { none, allocate, read, write };

	access_front() = default;
	explicit access_front(const mode mode) : m_mode(mode) {}
	explicit access_front(instruction* const instr, const mode mode) : m_instructions{instr}, m_mode(mode) {}

	void add_instruction(instruction* const instr) {
		// we insert instructions as soon as they are generated, so inserting at the end keeps the vector sorted
		m_instructions.push_back(instr);
		assert(std::is_sorted(m_instructions.begin(), m_instructions.end(), instruction_id_less()));
	}

	[[nodiscard]] access_front apply_epoch(instruction* const epoch) const {
		const auto first_retained = std::upper_bound(m_instructions.begin(), m_instructions.end(), epoch, instruction_id_less());
		const auto last_retained = m_instructions.end();

		// only include the new epoch in the access front if it in fact subsumes another instruction
		if(first_retained == m_instructions.begin()) return *this;

		access_front pruned(m_mode);
		pruned.m_instructions.resize(1 + static_cast<size_t>(std::distance(first_retained, last_retained)));
		pruned.m_instructions.front() = epoch;
		std::copy(first_retained, last_retained, std::next(pruned.m_instructions.begin()));
		assert(std::is_sorted(pruned.m_instructions.begin(), pruned.m_instructions.end(), instruction_id_less()));
		return pruned;
	};

	const gch::small_vector<instruction*>& get_instructions() const { return m_instructions; }
	mode get_mode() const { return m_mode; }

	friend bool operator==(const access_front& lhs, const access_front& rhs) { return lhs.m_instructions == rhs.m_instructions && lhs.m_mode == rhs.m_mode; }
	friend bool operator!=(const access_front& lhs, const access_front& rhs) { return !(lhs == rhs); }

  private:
	gch::small_vector<instruction*> m_instructions; // ordered by id to allow equality comparison
	mode m_mode = none;
};

/// Per-allocation state for a single buffer. This is where we track last-writer instructions and access fronts.
struct buffer_allocation_state {
	allocation_id aid;
	detail::box<3> box;                                ///< in buffer coordinates
	region_map<access_front> last_writers;             ///< in buffer coordinates
	region_map<access_front> last_concurrent_accesses; ///< in buffer coordinates

	explicit buffer_allocation_state(const allocation_id aid, alloc_instruction* const ainstr /* optional: null for user allocations */,
	    const detail::box<3>& allocated_box, const range<3>& buffer_range)
	    : aid(aid), box(allocated_box), //
	      last_writers(allocated_box, ainstr != nullptr ? access_front(ainstr, access_front::allocate) : access_front()),
	      last_concurrent_accesses(allocated_box, ainstr != nullptr ? access_front(ainstr, access_front::allocate) : access_front()) {}

	/// Add `instr` to the active set of concurrent reads, or replace the current access front if the last access was not a read.
	void track_concurrent_read(const region<3>& region, instruction* const instr) {
		if(region.empty()) return;
		for(auto& [box, front] : last_concurrent_accesses.get_region_values(region)) {
			if(front.get_mode() == access_front::read) {
				front.add_instruction(instr);
			} else {
				front = access_front(instr, access_front::read);
			}
			last_concurrent_accesses.update_region(box, front);
		}
	}

	/// Replace the current access front with a write. The write is treated as "atomic" in the sense that there is never a second, concurrent write operation
	/// happening simultaneously. This is true for all writes except those from device kernels and host tasks, which might specify overlapping write-accessors.
	void track_atomic_write(const region<3>& region, instruction* const instr) {
		if(region.empty()) return;
		last_writers.update_region(region, access_front(instr, access_front::write));
		last_concurrent_accesses.update_region(region, access_front(instr, access_front::write));
	}

	/// Replace the current access front with an empty write-front. This is done in preparation of writes from device kernels and host tasks.
	void begin_concurrent_writes(const region<3>& region) {
		if(region.empty()) return;
		last_writers.update_region(region, access_front(access_front::write));
		last_concurrent_accesses.update_region(region, access_front(access_front::write));
	}

	/// Add an instruction to the current set of concurrent writes. This is used to track writes from device kernels and host tasks and requires
	/// begin_concurrent_writes to be called beforehand. Multiple concurrent writes will only occur when a task declares overlapping writes and
	/// overlapping-write detection is disabled via the error policy. In order to still produce an executable (albeit racy instruction graph) in that case, we
	/// track multiple last-writers for the same buffer element.
	void track_concurrent_write(const region<3>& region, instruction* const instr) {
		if(region.empty()) return;
		for(auto& [box, front] : last_writers.get_region_values(region)) {
			assert(front.get_mode() == access_front::write && "must call begin_concurrent_writes first");
			front.add_instruction(instr);
			last_writers.update_region(box, front);
			last_concurrent_accesses.update_region(box, front);
		}
	}

	/// Replace all tracked instructions that older than `epoch` with `epoch`.
	void apply_epoch(instruction* const epoch) {
		last_writers.apply_to_values([epoch](const access_front& front) { return front.apply_epoch(epoch); });
		last_concurrent_accesses.apply_to_values([epoch](const access_front& front) { return front.apply_epoch(epoch); });
	}
};

/// Per-memory state for a single buffer. Dependencies and last writers are tracked on the contained allocations.
struct buffer_memory_state {
	// TODO bound the number of allocations per buffer in order to avoid runaway tracking overhead (similar to horizons)
	// TODO evaluate if it ever makes sense to use a region_map here, or if we're better off expecting very few allocations and sticking to a vector here
	std::vector<buffer_allocation_state> allocations; // disjoint

	const buffer_allocation_state& get_allocation(const allocation_id aid) const {
		const auto it = std::find_if(allocations.begin(), allocations.end(), [=](const buffer_allocation_state& a) { return a.aid == aid; });
		assert(it != allocations.end());
		return *it;
	}

	buffer_allocation_state& get_allocation(const allocation_id aid) { return const_cast<buffer_allocation_state&>(std::as_const(*this).get_allocation(aid)); }

	/// Returns the (unique) allocation covering `box` if such an allocation exists, otherwise nullptr.
	const buffer_allocation_state* find_contiguous_allocation(const box<3>& box) const {
		const auto it = std::find_if(allocations.begin(), allocations.end(), [&](const buffer_allocation_state& a) { return a.box.covers(box); });
		return it != allocations.end() ? &*it : nullptr;
	}

	buffer_allocation_state* find_contiguous_allocation(const box<3>& box) {
		return const_cast<buffer_allocation_state*>(std::as_const(*this).find_contiguous_allocation(box));
	}

	/// Returns the (unique) allocation covering `box`.
	const buffer_allocation_state& get_contiguous_allocation(const box<3>& box) const {
		const auto alloc = find_contiguous_allocation(box);
		assert(alloc != nullptr);
		return *alloc;
	}

	buffer_allocation_state& get_contiguous_allocation(const box<3>& box) {
		return const_cast<buffer_allocation_state&>(std::as_const(*this).get_contiguous_allocation(box));
	}

	bool is_allocated_contiguously(const box<3>& box) const { return find_contiguous_allocation(box) != nullptr; }

	/// Replace all tracked instructions that older than `epoch` with `epoch`.
	void apply_epoch(instruction* const epoch) {
		for(auto& alloc : allocations) {
			alloc.apply_epoch(epoch);
		}
	}
};

/// State for a single buffer.
struct buffer_state {
	/// Tracks a pending non-reduction await-push that will be compiled into a receive_instructions as soon as a command reads from its region.
	struct region_receive {
		task_id consumer_tid;
		region<3> received_region;
		box_vector<3> required_contiguous_allocations;

		region_receive(const task_id consumer_tid, region<3> received_region, box_vector<3> required_contiguous_allocations)
		    : consumer_tid(consumer_tid), received_region(std::move(received_region)),
		      required_contiguous_allocations(std::move(required_contiguous_allocations)) {}
	};

	/// Tracks a pending reduction await-push, which will be compiled into gather_receive_instructions and reduce_instructions when read from.
	struct gather_receive {
		task_id consumer_tid;
		reduction_id rid;
		box<3> gather_box;

		gather_receive(const task_id consumer_tid, const reduction_id rid, const box<3> gather_box)
		    : consumer_tid(consumer_tid), rid(rid), gather_box(gather_box) {}
	};

	std::string debug_name;
	celerity::range<3> range;
	size_t elem_size;  ///< in bytes
	size_t elem_align; ///< in bytes

	/// Per-memory and per-allocation state of this buffer.
	dense_map<memory_id, buffer_memory_state> memories;

	/// Contains a mask for every memory_id that either was written to by the original-producer instruction or that has already been made coherent previously.
	region_map<memory_mask> up_to_date_memories;

	/// Tracks the instruction that initially produced each buffer element on the local node - this can be a kernel, host task, region-receive or
	/// reduce-instruction, or - in case of a host-initialized buffer - an epoch. It is different from `buffer_allocation_state::last_writers` in that it never
	/// contains copy instructions. Copy- and send source regions are split on their original producer instructions to facilitate computation-communication
	/// overlap when different producers finish at different times.
	region_map<instruction*> original_writers;

	/// Tracks the memory to which the original_writer instruction wrote each buffer element. `original_write_memories[box]` is meaningless when
	/// `up_to_date_memories[box]` is empty (i.e. the buffer is not up-to-date on the local node due to being uninitialized or await-pushed without being
	/// consumed).
	region_map<memory_id> original_write_memories;

	// We store pending receives (await push regions) in a vector instead of a region map since we must process their entire regions en-bloc rather than on
	// a per-element basis.
	std::vector<region_receive> pending_receives;
	std::vector<gather_receive> pending_gathers;

	explicit buffer_state(const celerity::range<3>& range, const size_t elem_size, const size_t elem_align, const size_t n_memories)
	    : range(range), elem_size(elem_size), elem_align(elem_align), memories(n_memories), up_to_date_memories(range), original_writers(range),
	      original_write_memories(range) {}

	void track_original_write(const region<3>& region, instruction* const instr, const memory_id mid) {
		original_writers.update_region(region, instr);
		original_write_memories.update_region(region, mid);
		up_to_date_memories.update_region(region, memory_mask().set(mid));
	}

	/// Replace all tracked instructions that are older than `epoch` with `epoch`.
	void apply_epoch(instruction* const epoch) {
		for(auto& memory : memories) {
			memory.apply_epoch(epoch);
		}
		original_writers.apply_to_values([epoch](instruction* const instr) { return instr != nullptr && instr->get_id() < epoch->get_id() ? epoch : instr; });

		// This is an opportune point to verify that all await-pushes are fully consumed by the commands that require them.
		// TODO Assumes that the command graph generator issues await-pushes immediately before the commands that need them.
		assert(pending_receives.empty());
		assert(pending_gathers.empty());
	}
};

struct host_object_state {
	bool owns_instance; ///< If true, `destroy_host_object_instruction` will be emitted when `destroy_host_object` is called (false for `host_object<T&/void>`)
	instruction* last_side_effect; ///< Host tasks with side effects will be serialized wrt/ the host object.

	explicit host_object_state(const bool owns_instance, instruction* const last_epoch) : owns_instance(owns_instance), last_side_effect(last_epoch) {}

	/// If the last side-effect instruction was older than `epoch`, replaces it with `epoch`.
	void apply_epoch(instruction* const epoch) {
		if(last_side_effect != nullptr && last_side_effect->get_id() < epoch->get_id()) { last_side_effect = epoch; }
	}
};

struct collective_group_state {
	/// Collective host tasks will be serialized wrt/ the collective group to ensure that the user can freely submit MPI collectives on their communicator.
	instruction* last_collective_operation;

	explicit collective_group_state(instruction* const last_host_task) : last_collective_operation(last_host_task) {}

	/// If the last host-task instruction was older than `epoch`, replaces it with `epoch`.
	void apply_epoch(instruction* const epoch) {
		if(last_collective_operation != nullptr && last_collective_operation->get_id() < epoch->get_id()) { last_collective_operation = epoch; }
	}
};

struct staging_allocation {
	allocation_id aid = null_allocation_id;
	size_t size_bytes = 0;
	size_t align_bytes = 1;
	access_front last_accesses;

	void apply_epoch(instruction* const epoch) { last_accesses = last_accesses.apply_epoch(epoch); }
};

/// `allocation_id`s are "namespaced" to their memory ID, so we maintain the next `raw_allocation_id` for each memory separately.
struct memory_state {
	raw_allocation_id next_raw_aid = 1; // 0 is reserved for null_allocation_id

	// On host memory, we maintain a pool of staging allocations for host-staged device-to-device copies.
	std::vector<staging_allocation> staging_allocation_pool;

	void apply_epoch(instruction* const epoch) {
		for(auto& alloc : staging_allocation_pool) {
			alloc.apply_epoch(epoch);
		}
	}
};

/// We submit the set of instructions and pilots generated within a call to compile() en-bloc to relieve contention on the executor queue lock. To collect all
/// instructions that are generated in the call stack without polluting internal state, we pass a `batch&` output parameter to any function that transitively
/// generates instructions or pilots.
struct batch { // NOLINT(cppcoreguidelines-special-member-functions) (do not complain about the asserting destructor)
	std::vector<const instruction*> generated_instructions;
	std::vector<outbound_pilot> generated_pilots;

	/// The base priority of a batch adds to the priority per instruction type to transitively prioritize dependencies of important instructions.
	int base_priority = 0;

#ifndef NDEBUG
	~batch() {
		if(std::uncaught_exceptions() == 0) { assert(generated_instructions.empty() && generated_pilots.empty() && "unflushed batch detected"); }
	}
#endif
};

// We assign instruction priorities heuristically by deciding on a batch::base_priority at the beginning of a compile_* function and offsetting it by the
// instruction-type specific values defined here. This aims to perform low-latency submits of long-running instructions first to maximize overlap.
// clang-format off
template <typename Instruction> constexpr int instruction_type_priority = 0; // higher means more urgent
template <> constexpr int instruction_type_priority<free_instruction> = -1; // only free when forced to - nothing except an epoch or horizon will depend on this
template <> constexpr int instruction_type_priority<alloc_instruction> = 1; // allocations are synchronous and slow, so we postpone them as much as possible
template <> constexpr int instruction_type_priority<await_receive_instruction> = 2;
template <> constexpr int instruction_type_priority<split_receive_instruction> = 2;
template <> constexpr int instruction_type_priority<receive_instruction> = 2;
template <> constexpr int instruction_type_priority<send_instruction> = 2;
template <> constexpr int instruction_type_priority<fence_instruction> = 3;
template <> constexpr int instruction_type_priority<host_task_instruction> = 4; // we expect kernel launches to have low latency but comparatively long run time
template <> constexpr int instruction_type_priority<device_kernel_instruction> = 4; 
template <> constexpr int instruction_type_priority<epoch_instruction> = 5; // epochs and horizons are low-latency and stop the task buffers from reaching capacity
template <> constexpr int instruction_type_priority<horizon_instruction> = 5;
template <> constexpr int instruction_type_priority<copy_instruction> = 6; // stalled device-to-device copies can block kernel execution on peer devices
// clang-format on

/// A chunk of a task's execution space that will be assigned to a device (or the host) and thus knows which memory its instructions will operate on.
struct localized_chunk {
	detail::memory_id memory_id = host_memory_id;
	std::optional<detail::device_id> device_id;
	box<3> execution_range;
};

/// Transient state for a node-local eager reduction that is emitted around kernels that explicitly include a reduction. Tracks the gather-allocation that is
/// created early to rescue the current buffer value across the kernel invocation in case the reduction is not `initialize_to_identity` and the command graph
/// designates the local node to be the reduction initializer.
struct local_reduction {
	bool include_local_buffer_value = false; ///< If true local node is the one to include its current version of the existing buffer in the reduction.
	size_t first_kernel_chunk_offset = 0;    ///< If the local reduction includes the current buffer value, we add an additional reduction-chunk at the front.
	size_t num_input_chunks = 1;             ///< One per participating local chunk, plus one if the local node includes the current buffer value.
	size_t chunk_size_bytes = 0;
	allocation_id gather_aid = null_allocation_id;
	alloc_instruction* gather_alloc_instr = nullptr;
};

/// Maps instruction DAG types to their record type.
template <typename Instruction>
using record_type_for_t = utils::type_switch_t<Instruction, clone_collective_group_instruction(clone_collective_group_instruction_record),
    alloc_instruction(alloc_instruction_record), free_instruction(free_instruction_record), copy_instruction(copy_instruction_record),
    device_kernel_instruction(device_kernel_instruction_record), host_task_instruction(host_task_instruction_record), send_instruction(send_instruction_record),
    receive_instruction(receive_instruction_record), split_receive_instruction(split_receive_instruction_record),
    await_receive_instruction(await_receive_instruction_record), gather_receive_instruction(gather_receive_instruction_record),
    fill_identity_instruction(fill_identity_instruction_record), reduce_instruction(reduce_instruction_record), fence_instruction(fence_instruction_record),
    destroy_host_object_instruction(destroy_host_object_instruction_record), horizon_instruction(horizon_instruction_record),
    epoch_instruction(epoch_instruction_record)>;

class generator_impl {
  public:
	generator_impl(const task_manager& tm, size_t num_nodes, node_id local_nid, const system_info& system, instruction_graph& idag,
	    instruction_graph_generator::delegate* dlg, instruction_recorder* recorder, const instruction_graph_generator::policy_set& policy);

	void notify_buffer_created(buffer_id bid, const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid = null_allocation_id);
	void notify_buffer_debug_name_changed(buffer_id bid, const std::string& name);
	void notify_buffer_destroyed(buffer_id bid);
	void notify_host_object_created(host_object_id hoid, bool owns_instance);
	void notify_host_object_destroyed(host_object_id hoid);
	void compile(const abstract_command& cmd);

  private:
	inline static const box<3> scalar_reduction_box{zeros, ones};

	// construction parameters (immutable)
	instruction_graph* m_idag;
	const task_manager* m_tm; // TODO commands should reference tasks by pointer, not id - then we wouldn't need this member.
	size_t m_num_nodes;
	node_id m_local_nid;
	system_info m_system;
	instruction_graph_generator::delegate* m_delegate;
	instruction_recorder* m_recorder;
	instruction_graph_generator::policy_set m_policy;

	instruction_id m_next_instruction_id = 0;
	message_id m_next_message_id = 0;

	instruction* m_last_horizon = nullptr;
	instruction* m_last_epoch = nullptr; // set once the initial epoch instruction is generated in the constructor

	/// The set of all instructions that are not yet depended upon by other instructions. These are collected by collapse_execution_front_to() as part of
	/// horizon / epoch generation.
	std::unordered_set<instruction_id> m_execution_front;

	dense_map<memory_id, memory_state> m_memories;
	std::unordered_map<buffer_id, buffer_state> m_buffers;
	std::unordered_map<host_object_id, host_object_state> m_host_objects;
	std::unordered_map<collective_group_id, collective_group_state> m_collective_groups;

	/// The instruction executor maintains a mapping of allocation_id -> USM pointer. For IDAG-managed memory, these entries are deleted after executing a
	/// `free_instruction`, but since user allocations are not deallocated by us, we notify the executor on each horizon or epoch via the `instruction_garbage`
	/// struct about entries that will no longer be used and can therefore be collected. We include user allocations for buffer fences immediately after
	/// emitting the fence, and buffer host-initialization user allocations after the buffer has been destroyed.
	std::vector<allocation_id> m_unreferenced_user_allocations;

	/// True if a recorder is present and create() will call the `record_with` lambda passed as its last parameter.
	bool is_recording() const { return m_recorder != nullptr; }

	allocation_id new_allocation_id(const memory_id mid);

	template <typename Instruction, typename... CtorParamsAndRecordWithFn, size_t... CtorParamIndices, size_t RecordWithFnIndex>
	Instruction* create_internal(batch& batch, const std::tuple<CtorParamsAndRecordWithFn...>& ctor_args_and_record_with,
	    std::index_sequence<CtorParamIndices...> /* ctor_param_indices*/, std::index_sequence<RecordWithFnIndex> /* record_with_fn_index */);

	/// Create an instruction, insert it into the IDAG and the current execution front, and record it if a recorder is present.
	///
	/// Invoke as
	/// ```
	/// create<instruction-type>(instruction-ctor-params...,
	///         [&](const auto record_debug_info) { return record_debug_info(instruction-record-additional-ctor-params)})
	/// ```
	template <typename Instruction, typename... CtorParamsAndRecordWithFn>
	Instruction* create(batch& batch, CtorParamsAndRecordWithFn&&... ctor_args_and_record_with);

	message_id create_outbound_pilot(batch& batch, node_id target, const transfer_id& trid, const box<3>& box);

	/// Inserts a graph dependency and removes `to` form the execution front (if present). The `record_origin` is debug information.
	void add_dependency(instruction* const from, instruction* const to, const instruction_dependency_origin record_origin);

	void add_dependencies_on_access_front(
	    instruction* const accessing_instruction, const access_front& front, const instruction_dependency_origin origin_for_read_write_front);

	void add_dependencies_on_last_writers(instruction* const accessing_instruction, buffer_allocation_state& allocation, const region<3>& region);

	/// Add dependencies to the last writer of a region, and track the instruction as the new last (concurrent) reader.
	void perform_concurrent_read_from_allocation(instruction* const reading_instruction, buffer_allocation_state& allocation, const region<3>& region);

	void add_dependencies_on_last_concurrent_accesses(instruction* const accessing_instruction, buffer_allocation_state& allocation, const region<3>& region,
	    const instruction_dependency_origin origin_for_read_write_front);

	/// Add dependencies to the last concurrent accesses of a region, and track the instruction as the new last (unique) writer.
	void perform_atomic_write_to_allocation(instruction* const writing_instruction, buffer_allocation_state& allocation, const region<3>& region);

	/// Replace all tracked instructions that older than `epoch` with `epoch`.
	void apply_epoch(instruction* const epoch);

	/// Add dependencies from `horizon_or_epoch` to all instructions in `m_execution_front` and clear the set.
	void collapse_execution_front_to(instruction* const horizon_or_epoch);

	/// Create a new host allocation for copy staging, or re-use a cached staging allocation whose last access is older than the current epoch.
	staging_allocation& acquire_staging_allocation(batch& current_batch, memory_id mid, size_t size_bytes, size_t align_bytes);

	/// Free all cached staging allocations allocated so far.
	void free_all_staging_allocations(batch& current_batch);

	/// Ensure that all boxes in `required_contiguous_boxes` have a contiguous allocation on `mid`.
	/// Re-allocation of one buffer on one memory never interacts with other buffers or other memories backing the same buffer, this function can be called
	/// in any order of allocation requirements without generating additional dependencies.
	void allocate_contiguously(batch& batch, buffer_id bid, memory_id mid, box_vector<3>&& required_contiguous_boxes);

	/// Insert one or more receive instructions in order to fulfil a pending receive, making the received data available in host_memory_id. This may entail
	/// receiving a region that is larger than the union of all regions read.
	void commit_pending_region_receive_to_host_memory(
	    batch& batch, buffer_id bid, const buffer_state::region_receive& receives, const std::vector<region<3>>& concurrent_reads);

	/// Insert coherence copy instructions where necessary to all specified regions coherent on their respective memories. Requires the necessary allocations in
	/// `dest_mid` to already be present. We deliberately allow overlapping read-regions to avoid aggregated copies introducing synchronization points between
	/// otherwise independent instructions.
	void establish_coherence_between_buffer_memories(
	    batch& current_batch, const buffer_id bid, dense_map<memory_id, std::vector<region<3>>>& concurrent_reads_from_memory);

	/// Issue instructions to create any collective group required by a task.
	void create_task_collective_groups(batch& command_batch, const task& tsk);

	/// Split a tasks local execution range (given by execution_command) into chunks according to device configuration and a possible oversubscription hint.
	std::vector<localized_chunk> split_task_execution_range(const execution_command& ecmd, const task& tsk);

	/// Detect overlapping writes between local chunks of a task and report it according to m_policy.
	void report_task_overlapping_writes(const task& tsk, const std::vector<localized_chunk>& concurrent_chunks) const;

	/// Allocate memory, apply any pending receives, and issue resize- and coherence copies to prepare all buffer memories for a task's execution.
	void satisfy_task_buffer_requirements(batch& batch, buffer_id bid, const task& tsk, const subrange<3>& local_execution_range, bool is_reduction_initializer,
	    const std::vector<localized_chunk>& concurrent_chunks_after_split);

	/// Create a gather allocation and optionally save the current buffer value before creating partial reduction results in any kernel.
	local_reduction prepare_task_local_reduction(
	    batch& command_batch, const reduction_info& rinfo, const execution_command& ecmd, const task& tsk, const size_t num_concurrent_chunks);

	/// Combine any partial reduction results computed by local chunks and write it to buffer host memory.
	void finish_task_local_reduction(batch& command_batch, const local_reduction& red, const reduction_info& rinfo, const execution_command& ecmd,
	    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks);

	/// Launch a device kernel for each local chunk of a task, passing the relevant buffer allocations in place of accessors and reduction descriptors.
	instruction* launch_task_kernel(batch& command_batch, const execution_command& ecmd, const task& tsk, const localized_chunk& chunk);

	/// Add dependencies for all buffer accesses and reductions of a task, then update tracking structures accordingly.
	void perform_task_buffer_accesses(
	    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks, const std::vector<instruction*>& command_instructions);

	/// If a task has side effects, serialize it with respect to the last task that shares a host object.
	void perform_task_side_effects(
	    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks, const std::vector<instruction*>& command_instructions);

	/// If a task is part of a collective group, serialize it with respect to the last host task in that group.
	void perform_task_collective_operations(
	    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks, const std::vector<instruction*>& command_instructions);

	void compile_execution_command(batch& batch, const execution_command& ecmd);
	void compile_push_command(batch& batch, const push_command& pcmd);
	void defer_await_push_command(const await_push_command& apcmd);
	void compile_reduction_command(batch& batch, const reduction_command& rcmd);
	void compile_fence_command(batch& batch, const fence_command& fcmd);
	void compile_horizon_command(batch& batch, const horizon_command& hcmd);
	void compile_epoch_command(batch& batch, const epoch_command& ecmd);

	/// Passes all instructions and outbound pilots that have been accumulated in `batch` to the delegate (if any). Called after compiling a command, creating
	/// or destroying a buffer or host object, and also in our constructor for the creation of the initial epoch.
	void flush_batch(batch&& batch);

	std::string print_buffer_debug_label(buffer_id bid) const;
};

generator_impl::generator_impl(const task_manager& tm, const size_t num_nodes, const node_id local_nid, const system_info& system, instruction_graph& idag,
    instruction_graph_generator::delegate* const dlg, instruction_recorder* const recorder, const instruction_graph_generator::policy_set& policy)
    : m_idag(&idag), m_tm(&tm), m_num_nodes(num_nodes), m_local_nid(local_nid), m_system(system), m_delegate(dlg), m_recorder(recorder), m_policy(policy),
      m_memories(m_system.memories.size()) //
{
#ifndef NDEBUG
	assert(m_system.memories.size() <= max_num_memories);
	assert(std::all_of(
	    m_system.devices.begin(), m_system.devices.end(), [&](const device_info& device) { return device.native_memory < m_system.memories.size(); }));
	for(memory_id mid_a = 0; mid_a < m_system.memories.size(); ++mid_a) {
		assert(m_system.memories[mid_a].copy_peers[mid_a]);
		for(memory_id mid_b = mid_a + 1; mid_b < m_system.memories.size(); ++mid_b) {
			assert(m_system.memories[mid_a].copy_peers[mid_b] == m_system.memories[mid_b].copy_peers[mid_a]
			       && "system_info::memories::copy_peers must be reflexive");
		}
	}
#endif

	batch init_epoch_batch;
	m_idag->begin_epoch(task_manager::initial_epoch_task);
	const auto init_epoch = create<epoch_instruction>(init_epoch_batch, task_manager::initial_epoch_task, epoch_action::none, instruction_garbage{},
	    [](const auto& record_debug_info) { record_debug_info(command_id(0 /* or so we assume */)); });
	m_last_epoch = init_epoch;
	flush_batch(std::move(init_epoch_batch));

	// The root collective group already exists in the runtime, but we must still equip it with a meaningful last_host_task.
	m_collective_groups.emplace(root_collective_group_id, init_epoch);
}

void generator_impl::notify_buffer_created(
    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, allocation_id user_aid) //
{
	const auto [iter, inserted] =
	    m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(range, elem_size, elem_align, m_system.memories.size()));
	assert(inserted);

	if(user_aid != null_allocation_id) {
		// The buffer was host-initialized through a user-specified pointer, which we consider a fully coherent allocation in user_memory_id.
		assert(user_aid.get_memory_id() == user_memory_id);
		const box entire_buffer = subrange({}, range);

		auto& buffer = iter->second;
		auto& memory = buffer.memories[user_memory_id];
		auto& allocation = memory.allocations.emplace_back(user_aid, nullptr /* alloc_instruction */, entire_buffer, buffer.range);

		allocation.track_atomic_write(entire_buffer, m_last_epoch);
		buffer.track_original_write(entire_buffer, m_last_epoch, user_memory_id);
	}
}

void generator_impl::notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name) { m_buffers.at(bid).debug_name = name; }

void generator_impl::notify_buffer_destroyed(const buffer_id bid) {
	const auto iter = m_buffers.find(bid);
	assert(iter != m_buffers.end());
	auto& buffer = iter->second;

	batch free_batch;
	for(memory_id mid = 0; mid < buffer.memories.size(); ++mid) {
		auto& memory = buffer.memories[mid];
		if(mid == user_memory_id) {
			// When the buffer is gone, we can also drop the user allocation from executor tracking (there currently are either 0 or 1 of them).
			for(const auto& user_alloc : memory.allocations) {
				m_unreferenced_user_allocations.push_back(user_alloc.aid);
			}
		} else {
			for(auto& allocation : memory.allocations) {
				const auto free_instr = create<free_instruction>(free_batch, allocation.aid, [&](const auto& record_debug_info) {
					record_debug_info(allocation.box.get_area() * buffer.elem_size, buffer_allocation_record{bid, buffer.debug_name, allocation.box});
				});
				add_dependencies_on_last_concurrent_accesses(free_instr, allocation, allocation.box, instruction_dependency_origin::allocation_lifetime);
				// no need to modify the access front - we're removing the buffer altogether!
			}
		}
	}
	flush_batch(std::move(free_batch));

	m_buffers.erase(iter);
}

void generator_impl::notify_host_object_created(const host_object_id hoid, const bool owns_instance) {
	assert(m_host_objects.count(hoid) == 0);
	m_host_objects.emplace(std::piecewise_construct, std::tuple(hoid), std::tuple(owns_instance, m_last_epoch));
	// The host object is created in "userspace" and no instructions need to be emitted.
}

void generator_impl::notify_host_object_destroyed(const host_object_id hoid) {
	const auto iter = m_host_objects.find(hoid);
	assert(iter != m_host_objects.end());
	auto& obj = iter->second;

	if(obj.owns_instance) { // this is false for host_object<T&> and host_object<void>
		batch destroy_batch;
		const auto destroy_instr = create<destroy_host_object_instruction>(destroy_batch, hoid, [](const auto& record_debug_info) { record_debug_info(); });
		add_dependency(destroy_instr, obj.last_side_effect, instruction_dependency_origin::side_effect);
		flush_batch(std::move(destroy_batch));
	}

	m_host_objects.erase(iter);
}

allocation_id generator_impl::new_allocation_id(const memory_id mid) {
	assert(mid < m_memories.size());
	assert(mid != user_memory_id && "user allocation ids are not managed by the instruction graph generator");
	return allocation_id(mid, m_memories[mid].next_raw_aid++);
}

template <typename Instruction, typename... CtorParamsAndRecordWithFn, size_t... CtorParamIndices, size_t RecordWithFnIndex>
Instruction* generator_impl::create_internal(batch& batch, const std::tuple<CtorParamsAndRecordWithFn...>& ctor_args_and_record_with,
    std::index_sequence<CtorParamIndices...> /* ctor_param_indices*/, std::index_sequence<RecordWithFnIndex> /* record_with_fn_index */) {
	const auto iid = m_next_instruction_id++;
	const auto priority = batch.base_priority + instruction_type_priority<Instruction>;
	auto unique_instr = std::make_unique<Instruction>(iid, priority, std::get<CtorParamIndices>(ctor_args_and_record_with)...);
	const auto instr = unique_instr.get(); // we need to access the raw pointer after moving unique_ptr

	m_idag->push_instruction(std::move(unique_instr));
	m_execution_front.insert(iid);
	batch.generated_instructions.push_back(instr);

	if(is_recording()) {
		const auto& record_with = std::get<RecordWithFnIndex>(ctor_args_and_record_with);
#ifndef NDEBUG
		bool recorded = false;
#endif
		record_with([&](auto&&... debug_info) {
			m_recorder->record_instruction(
			    std::make_unique<record_type_for_t<Instruction>>(std::as_const(*instr), std::forward<decltype(debug_info)>(debug_info)...));
#ifndef NDEBUG
			recorded = true;
#endif
		});
		assert(recorded && "record_debug_info() not called within recording functor");
	}

	return instr;
}

template <typename Instruction, typename... CtorParamsAndRecordWithFn>
Instruction* generator_impl::create(batch& batch, CtorParamsAndRecordWithFn&&... ctor_args_and_record_with) {
	constexpr auto n_args = sizeof...(CtorParamsAndRecordWithFn);
	static_assert(n_args > 0);
	return create_internal<Instruction>(batch, std::forward_as_tuple(std::forward<CtorParamsAndRecordWithFn>(ctor_args_and_record_with)...),
	    std::make_index_sequence<n_args - 1>(), std::index_sequence<n_args - 1>());
}

message_id generator_impl::create_outbound_pilot(batch& current_batch, const node_id target, const transfer_id& trid, const box<3>& box) {
	// message ids (IDAG equivalent of MPI send/receive tags) tie send / receive instructions to their respective pilots.
	const message_id msgid = m_next_message_id++;
	const outbound_pilot pilot{target, pilot_message{msgid, trid, box}};
	current_batch.generated_pilots.push_back(pilot);
	if(is_recording()) { m_recorder->record_outbound_pilot(pilot); }
	return msgid;
}

void generator_impl::add_dependency(instruction* const from, instruction* const to, const instruction_dependency_origin record_origin) {
	from->add_dependency(to->get_id());
	if(is_recording()) { m_recorder->record_dependency(instruction_dependency_record(to->get_id(), from->get_id(), record_origin)); }
	m_execution_front.erase(to->get_id());
}

void generator_impl::add_dependencies_on_access_front(
    instruction* const accessing_instruction, const access_front& front, const instruction_dependency_origin origin_for_read_write_front) //
{
	const auto record_origin = front.get_mode() == access_front::allocate ? instruction_dependency_origin::allocation_lifetime : origin_for_read_write_front;
	for(const auto writer : front.get_instructions()) {
		add_dependency(accessing_instruction, writer, record_origin);
	}
}

void generator_impl::add_dependencies_on_last_writers(instruction* const accessing_instruction, buffer_allocation_state& allocation, const region<3>& region) {
	for(const auto& [box, front] : allocation.last_writers.get_region_values(region)) {
		add_dependencies_on_access_front(accessing_instruction, front, instruction_dependency_origin::read_from_allocation);
	}
}

void generator_impl::perform_concurrent_read_from_allocation(
    instruction* const reading_instruction, buffer_allocation_state& allocation, const region<3>& region) //
{
	add_dependencies_on_last_writers(reading_instruction, allocation, region);
	allocation.track_concurrent_read(region, reading_instruction);
}

void generator_impl::add_dependencies_on_last_concurrent_accesses(instruction* const accessing_instruction, buffer_allocation_state& allocation,
    const region<3>& region, const instruction_dependency_origin origin_for_read_write_front) //
{
	for(const auto& [box, front] : allocation.last_concurrent_accesses.get_region_values(region)) {
		add_dependencies_on_access_front(accessing_instruction, front, origin_for_read_write_front);
	}
}

void generator_impl::perform_atomic_write_to_allocation(instruction* const writing_instruction, buffer_allocation_state& allocation, const region<3>& region) {
	add_dependencies_on_last_concurrent_accesses(writing_instruction, allocation, region, instruction_dependency_origin::write_to_allocation);
	allocation.track_atomic_write(region, writing_instruction);
}

void generator_impl::apply_epoch(instruction* const epoch) {
	for(auto& memory : m_memories) {
		memory.apply_epoch(epoch);
	}
	for(auto& [_, buffer] : m_buffers) {
		buffer.apply_epoch(epoch);
	}
	for(auto& [_, host_object] : m_host_objects) {
		host_object.apply_epoch(epoch);
	}
	for(auto& [_, collective_group] : m_collective_groups) {
		collective_group.apply_epoch(epoch);
	}
	m_last_epoch = epoch;
}

void generator_impl::collapse_execution_front_to(instruction* const horizon_or_epoch) {
	for(const auto iid : m_execution_front) {
		if(iid == horizon_or_epoch->get_id()) continue;
		// we can't use instruction_graph_generator::add_dependency since it modifies the m_execution_front which we're iterating over here
		horizon_or_epoch->add_dependency(iid);
		if(is_recording()) {
			m_recorder->record_dependency(instruction_dependency_record(iid, horizon_or_epoch->get_id(), instruction_dependency_origin::execution_front));
		}
	}
	m_execution_front.clear();
	m_execution_front.insert(horizon_or_epoch->get_id());
}

staging_allocation& generator_impl::acquire_staging_allocation(batch& current_batch, const memory_id mid, const size_t size_bytes, const size_t align_bytes) {
	assert(size_bytes % align_bytes == 0);
	auto& memory = m_memories[mid];

	for(auto& alloc : memory.staging_allocation_pool) {
		// We attempt to re-use allocations made with exactly the same parameters. Staging allocations are implicitly treated as free when their last use
		// disappears behind a new epoch. This guarantees that staging allocation re-use never introduces new dependencies in the graph.
		if(alloc.size_bytes == size_bytes && alloc.align_bytes == align_bytes && alloc.last_accesses.get_instructions().size() == 1
		    && alloc.last_accesses.get_instructions().front() == m_last_epoch) {
			return alloc; // last use of this allocation is already epoch-serialized with the current batch, so no new unintended serialization will occur
		}
	}

	const auto aid = new_allocation_id(mid);
	const auto alloc_instr = create<alloc_instruction>(current_batch, aid, size_bytes, align_bytes, //
	    [&](const auto& record_debug_info) { record_debug_info(alloc_instruction_record::alloc_origin::staging, std::nullopt, std::nullopt); });
	add_dependency(alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);
	memory.staging_allocation_pool.push_back({aid, size_bytes, align_bytes, access_front(alloc_instr, access_front::allocate)});
	return memory.staging_allocation_pool.back();
}

void generator_impl::free_all_staging_allocations(batch& current_batch) {
	for(auto& memory : m_memories) {
		for(const auto& alloc : memory.staging_allocation_pool) {
			const auto size_bytes = alloc.size_bytes;                                  // permit lambda capture
			const auto free_instr = create<free_instruction>(current_batch, alloc.aid, //
			    [&](const auto& record_debug_info) { record_debug_info(size_bytes, std::nullopt); });
			add_dependencies_on_access_front(free_instr, alloc.last_accesses, instruction_dependency_origin::allocation_lifetime);
		}
		memory.staging_allocation_pool.clear();
	}
}

void generator_impl::allocate_contiguously(batch& current_batch, const buffer_id bid, const memory_id mid, box_vector<3>&& required_contiguous_boxes) //
{
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("iggen::allocate", Teal);

	if(required_contiguous_boxes.empty()) return;

	auto& buffer = m_buffers.at(bid);
	auto& memory = buffer.memories[mid];

	assert(std::all_of(required_contiguous_boxes.begin(), required_contiguous_boxes.end(),
	    [&](const box<3>& box) { return !box.empty() && detail::box<3>::full_range(buffer.range).covers(box); }));

	if(std::all_of(required_contiguous_boxes.begin(), required_contiguous_boxes.end(), //
	       [&](const box<3>& box) { return memory.is_allocated_contiguously(box); })) {
		return;
	}

	// We currently only ever *grow* the allocation of buffers on each memory, which means that we must merge (re-size) existing allocations that overlap with
	// but do not fully contain one of the required contiguous boxes. *Overlapping* here strictly means having a non-empty intersection; two allocations whose
	// boxes merely touch can continue to co-exist
	auto&& contiguous_boxes_after_realloc = std::move(required_contiguous_boxes);
	for(auto& alloc : memory.allocations) {
		contiguous_boxes_after_realloc.push_back(alloc.box);
	}
	merge_overlapping_bounding_boxes(contiguous_boxes_after_realloc);

	// Allocations that are now fully contained in (but not equal to) one of the newly contiguous bounding boxes will be freed at the end of the reallocation
	// step, because we currently disallow overlapping allocations for simplicity. These will function as sources for resize-copies below.
	const auto resize_from_begin = std::partition(memory.allocations.begin(), memory.allocations.end(), [&](const buffer_allocation_state& allocation) {
		return std::find(contiguous_boxes_after_realloc.begin(), contiguous_boxes_after_realloc.end(), allocation.box) != contiguous_boxes_after_realloc.end();
	});
	const auto resize_from_end = memory.allocations.end();

	// Derive the set of new boxes to allocate by removing all existing boxes from the set of contiguous boxes.
	auto&& new_alloc_boxes = std::move(contiguous_boxes_after_realloc);
	const auto last_new_allocation = std::remove_if(new_alloc_boxes.begin(), new_alloc_boxes.end(),
	    [&](auto& box) { return std::any_of(memory.allocations.begin(), memory.allocations.end(), [&](auto& alloc) { return alloc.box == box; }); });
	new_alloc_boxes.erase(last_new_allocation, new_alloc_boxes.end());
	assert(!new_alloc_boxes.empty()); // otherwise we would have returned early

	// Opportunistically merge connected boxes to keep the number of allocations and the tracking overhead low. This will not introduce artificial
	// synchronization points because resize-copies are still rooted on the original last-writers.
	// TODO consider over-allocating to avoid future reallocations, i.e. by using bounding boxes of boxes that have a common boundary but are not "connected" in
	// the sense that they can simply be merged).
	merge_connected_boxes(new_alloc_boxes);

	// We collect new allocations in a vector *separate* from memory.allocations as to not invalidate iterators (and to avoid resize-copying from them).
	std::vector<buffer_allocation_state> new_allocations;
	new_allocations.reserve(new_alloc_boxes.size());

	// Create new allocations and initialize them via resize-copies if necessary.
	for(const auto& new_box : new_alloc_boxes) {
		const auto aid = new_allocation_id(mid);
		const auto alloc_instr =
		    create<alloc_instruction>(current_batch, aid, new_box.get_area() * buffer.elem_size, buffer.elem_align, [&](const auto& record_debug_info) {
			    record_debug_info(alloc_instruction_record::alloc_origin::buffer, buffer_allocation_record{bid, buffer.debug_name, new_box}, std::nullopt);
		    });
		add_dependency(alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);

		auto& new_alloc = new_allocations.emplace_back(aid, alloc_instr, new_box, buffer.range);

		// Since allocations don't overlap, we copy from those that are about to be freed
		for(auto source_it = resize_from_begin; source_it != resize_from_end; ++source_it) {
			auto& resize_source_alloc = *source_it;

			// Only copy those boxes to the new allocation that are still up-to-date in the old allocation. The caller of allocate_contiguously should remove
			// any region from up_to_date_memories that they intend to discard / overwrite immediately to avoid dead resize copies.
			// TODO investigate a garbage-collection heuristic that omits these copies if there are other up-to-date memories and we do not expect the region to
			// be read again on this memory.
			const auto full_copy_box = box_intersection(new_alloc.box, resize_source_alloc.box);
			if(full_copy_box.empty()) continue; // not every previous allocation necessarily intersects with every new allocation

			box_vector<3> live_copy_boxes;
			for(const auto& [copy_box, location] : buffer.up_to_date_memories.get_region_values(full_copy_box)) {
				if(location.test(mid)) { live_copy_boxes.push_back(copy_box); }
			}
			// even if allocations intersect, the entire intersection might be overwritten by the task that requested reallocation - in which case the caller
			// would have reset up_to_date_memories for the corresponding elements
			if(live_copy_boxes.empty()) continue;

			region<3> live_copy_region(std::move(live_copy_boxes));
			const auto copy_instr = create<copy_instruction>(current_batch, resize_source_alloc.aid, new_alloc.aid, strided_layout(resize_source_alloc.box),
			    strided_layout(new_alloc.box), live_copy_region, buffer.elem_size,
			    [&](const auto& record_debug_info) { record_debug_info(copy_instruction_record::copy_origin::resize, bid, buffer.debug_name); });

			perform_concurrent_read_from_allocation(copy_instr, resize_source_alloc, live_copy_region);
			perform_atomic_write_to_allocation(copy_instr, new_alloc, live_copy_region);
		}
	}

	// Free old allocations now that all required resize-copies have been issued.
	// TODO consider keeping old allocations around until their box is written to (or at least until the end of the current instruction batch) in order to
	// resolve "buffer-locking" anti-dependencies
	for(auto it = resize_from_begin; it != resize_from_end; ++it) {
		auto& old_alloc = *it;

		const auto free_instr = create<free_instruction>(current_batch, old_alloc.aid, [&](const auto& record_debug_info) {
			record_debug_info(old_alloc.box.get_area() * buffer.elem_size, buffer_allocation_record{bid, buffer.debug_name, old_alloc.box});
		});
		add_dependencies_on_last_concurrent_accesses(free_instr, old_alloc, old_alloc.box, instruction_dependency_origin::allocation_lifetime);
	}

	// TODO garbage-collect allocations that are not up-to-date and not written to in this task

	memory.allocations.erase(resize_from_begin, memory.allocations.end());
	memory.allocations.insert(memory.allocations.end(), std::make_move_iterator(new_allocations.begin()), std::make_move_iterator(new_allocations.end()));
}

void generator_impl::commit_pending_region_receive_to_host_memory(
    batch& current_batch, const buffer_id bid, const buffer_state::region_receive& receive, const std::vector<region<3>>& concurrent_reads) //
{
	const auto trid = transfer_id(receive.consumer_tid, bid, no_reduction_id);

	// For simplicity of the initial IDAG implementation, we choose to receive directly into host-buffer allocations. This saves us from juggling
	// staging-buffers, but comes at a price in performance since the communicator needs to linearize and de-linearize transfers from and to regions that have
	// non-zero strides within their host allocation.
	//
	// TODO 1) maintain staging allocations and move (de-)linearization to the device in order to profit from higher memory bandwidths
	//      2) explicitly support communicators that can send and receive directly to and from device memory (NVIDIA GPUDirect RDMA)

	auto& buffer = m_buffers.at(bid);
	auto& host_memory = buffer.memories[host_memory_id];

	std::vector<buffer_allocation_state*> allocations;
	for(const auto& min_contiguous_box : receive.required_contiguous_allocations) {
		// The caller (aka satisfy_task_buffer_requirements) must ensure that all received boxes are allocated contiguously
		auto& alloc = host_memory.get_contiguous_allocation(min_contiguous_box);
		if(std::find(allocations.begin(), allocations.end(), &alloc) == allocations.end()) { allocations.push_back(&alloc); }
	}

	for(const auto alloc : allocations) {
		const auto region_received_into_alloc = region_intersection(alloc->box, receive.received_region);
		std::vector<region<3>> independent_await_regions;
		for(const auto& read_region : concurrent_reads) {
			const auto await_region = region_intersection(read_region, region_received_into_alloc);
			if(!await_region.empty()) { independent_await_regions.push_back(await_region); }
		}
		assert(!independent_await_regions.empty());

		// Ensure that receive-instructions inserted for concurrent readers are themselves concurrent.
		symmetrically_split_overlapping_regions(independent_await_regions);

		if(independent_await_regions.size() > 1) {
			// If there are multiple concurrent readers requiring different parts of the received region, we emit independent await_receive_instructions so as
			// to not introduce artificial synchronization points (and facilitate computation-communication overlap). Since the (remote) sender might still
			// choose to perform the entire transfer en-bloc, we must inform the receive_arbiter of the target allocation and the full transfer region via a
			// split_receive_instruction.
			const auto split_recv_instr = create<split_receive_instruction>(current_batch, trid, region_received_into_alloc, alloc->aid, alloc->box,
			    buffer.elem_size, [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });

			// We add dependencies to the split_receive_instruction as if it were a writer, but update the last_writers only at the await_receive_instruction.
			// The actual write happens somewhere in-between these instructions as orchestrated by the receive_arbiter, so no other access must depend on
			// split_receive_instruction directly.
			add_dependencies_on_last_concurrent_accesses(
			    split_recv_instr, *alloc, region_received_into_alloc, instruction_dependency_origin::write_to_allocation);

			for(const auto& await_region : independent_await_regions) {
				const auto await_instr = create<await_receive_instruction>(
				    current_batch, trid, await_region, [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });

				add_dependency(await_instr, split_recv_instr, instruction_dependency_origin::split_receive);

				alloc->track_atomic_write(await_region, await_instr);
				buffer.original_writers.update_region(await_region, await_instr);
			}
		} else {
			// A receive_instruction is equivalent to a spit_receive_instruction followed by a single await_receive_instruction, but (as the common case) has
			// less tracking overhead in the instruction graph.
			const auto recv_instr = create<receive_instruction>(current_batch, trid, region_received_into_alloc, alloc->aid, alloc->box, buffer.elem_size,
			    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });

			perform_atomic_write_to_allocation(recv_instr, *alloc, region_received_into_alloc);
			buffer.original_writers.update_region(region_received_into_alloc, recv_instr);
		}
	}

	buffer.original_write_memories.update_region(receive.received_region, host_memory_id);
	buffer.up_to_date_memories.update_region(receive.received_region, memory_mask().set(host_memory_id));
}

/// Multi-dimensional host <-> device copies become a severe performance bottleneck on Nvidia devices when the contiguous chunk size is small and the number of
/// chunks is large (on the order of 1-10 MiB/s copy throughput vs. the available 25 GB/s host memory bandwidth). This function heuristically decides if it is
/// beneficial to replace the nD host <-> device copy with an nD <-> 1d device (de)linearization step that enables a follow-up fast 1d host <-> device copy.
bool should_linearize_copy_region(const memory_id alloc_mid, const box<3>& alloc_box, const region<3>& copy_region, const size_t elem_size) {
	constexpr size_t max_linearized_region_bytes = 64 << 20; // 64 MiB - limit the device memory consumed for staging
	constexpr size_t max_chunk_size_to_linearize = 64;

	if(alloc_mid < first_device_memory_id) return false;
	if(copy_region.get_area() * elem_size > max_linearized_region_bytes) return false;

	size_t min_discontinuous_chunk_size_bytes = std::numeric_limits<size_t>::max();
	for(const auto& copy_box : copy_region.get_boxes()) {
		const auto linearization =
		    layout_nd_copy(alloc_box.get_range(), copy_box.get_range(), copy_box.get_offset() - alloc_box.get_offset(), zeros, copy_box.get_range(), elem_size);
		if(linearization.num_complex_strides > 0) {
			min_discontinuous_chunk_size_bytes = std::min(min_discontinuous_chunk_size_bytes, linearization.contiguous_size);
		}
	}
	return min_discontinuous_chunk_size_bytes < max_chunk_size_to_linearize;
}

void generator_impl::establish_coherence_between_buffer_memories(
    batch& current_batch, const buffer_id bid, dense_map<memory_id, std::vector<region<3>>>& concurrent_reads_from_memory) //
{
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("iggen::coherence", Red2);

	auto& buffer = m_buffers.at(bid);

	// (1) Examine what regions need to be copied between memories locally to satisfy all reads. Regions within `concurrent_reads_from_memory` are already split
	// on consumer instructions, meaning that we will not introduce artificial dependencies between multiple parallel producer-consumer chains as long as we do
	// not attempt to merge between those separate regions. We further split unsatisfied regions by original writer as well as writer and reader allocation
	// boxes in order to produce copy regions that can be serviced by individual `copy_instruction`s.

	// In the fast path, regions can be copied directly from producer to consumer memory.
	std::unordered_map<std::pair<memory_id, memory_id>, std::vector<region<3>>, utils::pair_hash> concurrent_direct_copies;

	// Some hardware setups require staging device-to-device copies through (pinned) host memory. Instead of keying by source and destination memories like for
	// direct copies, we re-examine both in (3) to make sure we only copy to host once in case of a 1:n device-to-device broadcast.
	std::unordered_map<memory_id, std::vector<region<3>>> concurrently_host_staged_copies;

	// Instead of planning / creating instructions directly, collect region vectors so (2) can remove overlaps with symmetrically_split_overlapping_regions.
	for(memory_id dest_mid = 0; dest_mid < concurrent_reads_from_memory.size(); ++dest_mid) {
		for(auto& dest_region : concurrent_reads_from_memory[dest_mid]) {
			// up_to_date_memories is a memory_mask, so regions that are up-to-date on one memory can still end up being enumerated as disjoint boxes.
			// We therefore merge them using a map keyed by original writers and their memories before constructing the final copy regions.
			std::unordered_map<std::pair<memory_id, instruction_id>, box_vector<3>, utils::pair_hash> source_boxes_by_writer;
			for(const auto& [box, up_to_date_mids] : buffer.up_to_date_memories.get_region_values(dest_region)) {
				if(up_to_date_mids.any() /* gracefully handle uninitialized read */ && !up_to_date_mids.test(dest_mid)) {
					for(const auto& [source_memory_box, source_mid] : buffer.original_write_memories.get_region_values(box)) {
						for(const auto& [source_memory_writer_box, original_writer] : buffer.original_writers.get_region_values(source_memory_box)) {
							source_boxes_by_writer[{source_mid, original_writer->get_id()}].push_back(source_memory_writer_box);
						}
					}
				}
			}
			for(auto& [source, source_boxes] : source_boxes_by_writer) {
				const auto& [source_mid, _] = source;
				const region source_region(std::move(source_boxes));

				for(auto& source_alloc : buffer.memories[source_mid].allocations) {
					const auto source_alloc_region = region_intersection(source_region, source_alloc.box);
					if(source_alloc_region.empty()) continue;

					for(auto& dest_alloc : buffer.memories[dest_mid].allocations) {
						auto copy_region = region_intersection(source_alloc_region, dest_alloc.box);
						if(copy_region.empty()) continue;

						if(m_system.memories[source_mid].copy_peers.test(dest_mid)) {
							concurrent_direct_copies[{source_mid, dest_mid}].push_back(std::move(copy_region));
						} else {
							concurrently_host_staged_copies[source_mid].push_back(std::move(copy_region));
						}
					}
				}
			}
		}
	}

	// (2) Plan an abstract tree of copy operations necessary to establish full coherence. Staged or source-linearized copies will manifest as proper
	// instruction trees rather than chains in case of broadcast-like producer-consumer patterns. The explicit planning structure avoids the introduction of
	// temporary region maps to track dependencies across staging allocations by exploiting the fact that (1) results in a full producer-consumer split, meaning
	// that every read from a staging allocation is guaranteed to depend on exactly one writer.

	/// Abstract plan for satisfying all copies from a single buffer allocation and region.
	struct copy_plan {
		/// Data is strided in a persistent buffer (box) allocation.
		struct in_buffer {
			allocation_id aid;
			explicit in_buffer(const allocation_id aid) : aid(aid) {}
		};

		/// Data is linearized in a temporary staging allocation.
		struct staged {
			memory_id mid = 0;
			size_t offset_bytes = 0;
			explicit staged(const memory_id mid, const size_t offset_bytes) : mid(mid), offset_bytes(offset_bytes) {}
		};

		using location = std::variant<in_buffer, staged>;

		/// A node in the copy tree.
		struct hop {
			copy_plan::location location;
			std::vector<hop> next;

			hop(const copy_plan::location loc) : location(loc) {}
			hop& chain(const copy_plan::location& loc) & { return next.emplace_back(loc); }
		};

		detail::region<3> region;
		hop source;

		copy_plan(const detail::region<3>& region, const location& source) : region(region), source(source) {}
		hop& chain(const copy_plan::location& loc) & { return source.chain(loc); }
	};

	std::vector<copy_plan> planned_copies;

	// (2a) Plan all direct, non-staged copies. All such copies are concurrent.

	for(auto& [source_dest_mid, concurrent_copies] : concurrent_direct_copies) {
		const auto [source_mid, dest_mid] = source_dest_mid;
		assert(source_mid != dest_mid);

		symmetrically_split_overlapping_regions(concurrent_copies); // ensure copy regions are disjoint

		for(const auto& copy_region : concurrent_copies) {
			// We split by source / dest allocations above, so source / dest allocations are unique
			auto& source_alloc = buffer.memories[source_mid].get_contiguous_allocation(bounding_box(copy_region));
			auto& dest_alloc = buffer.memories[dest_mid].get_contiguous_allocation(bounding_box(copy_region));

			planned_copies.emplace_back(copy_region, copy_plan::in_buffer(source_alloc.aid)).chain(copy_plan::in_buffer(dest_alloc.aid));
		}
	}

	// (2b) Plan host-staged copy instruction chains where necessary, and heuristically decide whether a strided source or destination should be (de)linearized
	// on the device. Staged copies are all fully concurrent with direct copies from (2a).

	dense_map<memory_id, staging_allocation*> staging_allocs; // empty when concurrently_host_staged_copies.empty()
	if(!concurrently_host_staged_copies.empty()) {
		dense_map<memory_id, size_t> staging_allocation_sizes_bytes(m_memories.size());
		const auto stage_alignment_bytes = std::lcm(buffer.elem_align, hardware_destructive_interference_size);
		const auto get_region_size_bytes = [&](const region<3>& region) { return utils::ceil(region.get_area() * buffer.elem_size, stage_alignment_bytes); };

		for(auto& [source_mid, concurrent_regions] : concurrently_host_staged_copies) {
			symmetrically_split_overlapping_regions(concurrent_regions); // ensure copy regions are disjoint

			for(const auto& region : concurrent_regions) { // iterations are independent
				// We split by source / dest allocations above, so source / dest allocations are unique
				auto& source_alloc = buffer.memories[source_mid].get_contiguous_allocation(bounding_box(region));

				// Begin at the strided source buffer allocation
				auto stage_source_hop = &planned_copies.emplace_back(region, copy_plan::in_buffer(source_alloc.aid)).source;

				if(should_linearize_copy_region(source_mid, source_alloc.box, region, buffer.elem_size)) {
					// Add a linearized hop in source (device) memory
					stage_source_hop = &stage_source_hop->chain(copy_plan::staged(source_mid, staging_allocation_sizes_bytes[source_mid]));
					staging_allocation_sizes_bytes[source_mid] += get_region_size_bytes(region);
				}

				// Add the linearized staging hop in host memory
				const auto host_stage_hop = &stage_source_hop->chain(copy_plan::staged(host_memory_id, staging_allocation_sizes_bytes[host_memory_id]));
				staging_allocation_sizes_bytes[host_memory_id] += get_region_size_bytes(region);

				// There can be multiple destinations in case of a broadcast pattern
				for(memory_id dest_mid = first_device_memory_id; dest_mid < concurrent_reads_from_memory.size(); ++dest_mid) {
					// The region will appear in concurrently_staged_copies if it is outdated on any destination in (1), so we need to query up_to_date_memories
					// a second time to make sure we don't create unnecessary copies (especially ones back to the source in case of an all-read).
					const auto boxes_up_to_date = buffer.up_to_date_memories.get_region_values(region);
					// Because up_to_date_memories maps to a memory_mask, we can end up with multiple boxes, but they must all agree on dest_mid.
					assert(!boxes_up_to_date.empty() && std::all_of(boxes_up_to_date.begin(), boxes_up_to_date.end(), [&](const auto& box_and_mids) {
						return box_and_mids.second.test(dest_mid) == boxes_up_to_date.front().second.test(dest_mid);
					}));
					if(boxes_up_to_date.front().second.test(dest_mid)) continue;

					for(const auto& dest_memory_region : concurrent_reads_from_memory[dest_mid]) {
						if(region_intersection(dest_memory_region, region).empty()) continue;
						assert(region_difference(region, dest_memory_region).empty()); // we split by dest_mid above

						auto& dest_alloc = buffer.memories[dest_mid].get_contiguous_allocation(bounding_box(region));

						auto unstage_source_hop = host_stage_hop;
						if(should_linearize_copy_region(dest_mid, dest_alloc.box, region, buffer.elem_size)) {
							// Add a linearized hop in dest (device) memory
							unstage_source_hop = &host_stage_hop->chain(copy_plan::staged(dest_mid, staging_allocation_sizes_bytes[dest_mid]));
							staging_allocation_sizes_bytes[dest_mid] += get_region_size_bytes(region);
						}

						// Finish the chain in the final strided dest buffer allocation
						unstage_source_hop->chain(copy_plan::in_buffer{dest_alloc.aid});
					}
				}
			}
		}

		// Staging allocation sizes are now final, allocate
		staging_allocs.resize(m_memories.size());
		for(memory_id mid = 0; mid < m_memories.size(); ++mid) {
			if(staging_allocation_sizes_bytes[mid] == 0) continue;
			staging_allocs[mid] = &acquire_staging_allocation(current_batch, mid, staging_allocation_sizes_bytes[mid], stage_alignment_bytes);
		}
	}

	// (3) Recursively traverse each copy_plan to generate all copy instructions and their dependencies.

	using copy_location_metadata = std::tuple<allocation_id, buffer_allocation_state* /* optional */, region_layout>;

	// Looks up metadata from now-allocated buffers and staging space for use in execute_copy_plan_recursive
	const auto get_copy_location_metadata = [&](const copy_plan::location& location) {
		return matchbox::match<copy_location_metadata>(
		    location,
		    [&](const copy_plan::in_buffer& in_buffer) {
			    auto& alloc = buffer.memories[in_buffer.aid.get_memory_id()].get_allocation(in_buffer.aid);
			    return std::tuple(in_buffer.aid, &alloc, strided_layout(alloc.box));
		    },
		    [&](const copy_plan::staged& staged) {
			    assert(staging_allocs[staged.mid] != nullptr);
			    return std::tuple(staging_allocs[staged.mid]->aid, nullptr, linearized_layout(staged.offset_bytes));
		    });
	};

	// Tracks the full read-front for each used staging allocation while staging_allocation::last_accesses still points to the last instruction
	// (= alloc_instruction or the last effective epoch) to avoid incorrectly chaining copies that read from the same allocation.
	dense_map<memory_id, access_front> reads_from_staging_allocs(staging_allocs.size(), access_front(access_front::read));

	// Inserts one copy between two hops, and then recurses to complete the subtree of its destination.
	// The lambda is passed into itself as the last generic parameter to permit recursion.
	const auto execute_copy_plan_recursive = [&](const region<3>& region, const copy_plan::hop& source_hop, const copy_plan::hop& dest_hop,
	                                             instruction* const source_copy_instr, const auto& execute_copy_plan_recursive) -> void {
		const auto [dest_aid, dest_buffer_alloc, dest_layout] = get_copy_location_metadata(dest_hop.location);
		const auto [source_aid, source_buffer_alloc, source_layout] = get_copy_location_metadata(source_hop.location);

		const auto copy_instr = create<copy_instruction>(
		    current_batch, source_aid, dest_aid, source_layout, dest_layout, region, buffer.elem_size, [&](const auto& record_debug_info) {
			    const auto origin = std::holds_alternative<copy_plan::staged>(dest_hop.location) ? copy_instruction_record::copy_origin::staging
			                                                                                     : copy_instruction_record::copy_origin::coherence;
			    record_debug_info(origin, bid, buffer.debug_name);
		    });

		if(source_buffer_alloc != nullptr) {
			perform_concurrent_read_from_allocation(copy_instr, *source_buffer_alloc, region);
		} else /* source is staged */ {
			reads_from_staging_allocs[source_aid.get_memory_id()].add_instruction(copy_instr);
		}

		if(source_copy_instr != nullptr) { add_dependency(copy_instr, source_copy_instr, instruction_dependency_origin::read_from_allocation); }

		if(dest_buffer_alloc != nullptr) {
			perform_atomic_write_to_allocation(copy_instr, *dest_buffer_alloc, region);
		} else /* dest is staged */ {
			auto& stage = *staging_allocs[dest_aid.get_memory_id()];
			// ensure that copy_instr transitively depends on the alloc_instruction of the staging allocation
			add_dependencies_on_access_front(copy_instr, stage.last_accesses, instruction_dependency_origin::write_to_allocation);
		}

		for(const auto& next_dest_hop : dest_hop.next) {
			execute_copy_plan_recursive(region, dest_hop, next_dest_hop, copy_instr, execute_copy_plan_recursive);
		}
	};

	// Create instructions for all planned copies
	for(const auto& plan : planned_copies) {
		for(const auto& dest_hop : plan.source.next) {
			execute_copy_plan_recursive(plan.region, plan.source, dest_hop, nullptr, execute_copy_plan_recursive);
		}
	}

	// Now that all copies have been created, update all staging allocation access fronts accordingly.
	for(memory_id mid = 0; mid < staging_allocs.size(); ++mid) {
		if(staging_allocs[mid] == nullptr) continue;
		staging_allocs[mid]->last_accesses = std::move(reads_from_staging_allocs[mid]);
	}

	// (4) Update buffer.up_to_date_memories en-bloc, regardless of which copy instructions were actually emitted.

	if(!concurrent_direct_copies.empty() || !concurrently_host_staged_copies.empty()) {
		for(memory_id mid = 0; mid < concurrent_reads_from_memory.size(); ++mid) {
			for(const auto& region : concurrent_reads_from_memory[mid]) {
				for(auto& [box, location] : buffer.up_to_date_memories.get_region_values(region)) {
					buffer.up_to_date_memories.update_region(box, memory_mask(location).set(mid));
				}
			}
		}
	}
}

void generator_impl::create_task_collective_groups(batch& command_batch, const task& tsk) {
	const auto cgid = tsk.get_collective_group_id();
	if(cgid == non_collective_group_id) return;
	if(m_collective_groups.count(cgid) != 0) return;

	// New collective groups are created by cloning the root collective group (aka MPI_COMM_WORLD).
	auto& root_cg = m_collective_groups.at(root_collective_group_id);
	const auto clone_cg_isntr = create<clone_collective_group_instruction>(
	    command_batch, root_collective_group_id, tsk.get_collective_group_id(), [](const auto& record_debug_info) { record_debug_info(); });

	m_collective_groups.emplace(cgid, clone_cg_isntr);

	// Cloning itself is a collective operation and must be serialized as such.
	add_dependency(clone_cg_isntr, root_cg.last_collective_operation, instruction_dependency_origin::collective_group_order);
	root_cg.last_collective_operation = clone_cg_isntr;
}

std::vector<localized_chunk> generator_impl::split_task_execution_range(const execution_command& ecmd, const task& tsk) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("iggen::split_task", Maroon);

	if(tsk.get_execution_target() == execution_target::device && m_system.devices.empty()) { utils::panic("no device on which to execute device kernel"); }

	const bool is_splittable_locally =
	    tsk.has_variable_split() && tsk.get_side_effect_map().empty() && tsk.get_collective_group_id() == non_collective_group_id;
	const auto split = tsk.get_hint<experimental::hints::split_2d>() != nullptr ? split_2d : split_1d;

	const auto command_sr = ecmd.get_execution_range();
	const auto command_chunk = chunk<3>(command_sr.offset, command_sr.range, tsk.get_global_size());

	// As a heuristic to keep inter-device communication to a minimum, we split the execution range twice when oversubscription is active: Once to obtain
	// contiguous chunks per device, and one more (below) to subdivide the ranges on each device (which can help with computation-communication overlap).
	std::vector<chunk<3>> coarse_chunks;
	if(is_splittable_locally && tsk.get_execution_target() == execution_target::device) {
		coarse_chunks = split(command_chunk, tsk.get_granularity(), m_system.devices.size());
	} else {
		coarse_chunks = {command_chunk};
	}

	size_t oversubscribe_factor = 1;
	if(const auto oversubscribe = tsk.get_hint<experimental::hints::oversubscribe>(); oversubscribe != nullptr) {
		// Our local reduction setup uses the normal per-device backing buffer allocation as the reduction output of each device. Since we can't track
		// overlapping allocations at the moment, we have no way of oversubscribing reduction kernels without introducing a data race between multiple "fine
		// chunks" on the final write. This could be solved by creating separate reduction-output allocations for each device chunk and not touching the
		// actual buffer allocation. This is left as *future work* for a general overhaul of reductions.
		if(is_splittable_locally && tsk.get_reductions().empty()) {
			oversubscribe_factor = oversubscribe->get_factor();
		} else if(m_policy.unsafe_oversubscription_error != error_policy::ignore) {
			utils::report_error(m_policy.unsafe_oversubscription_error, "Refusing to oversubscribe {}{}.", print_task_debug_label(tsk),
			    !tsk.get_reductions().empty()                              ? " because it performs a reduction"
			    : !tsk.get_side_effect_map().empty()                       ? " because it has side effects"
			    : tsk.get_collective_group_id() != non_collective_group_id ? " because it participates in a collective group"
			    : !tsk.has_variable_split()                                ? " because its iteration space cannot be split"
			                                                               : "");
		}
	}

	// Split a second time (if oversubscribed) and assign native memory and devices (if the task is a device kernel).
	std::vector<localized_chunk> concurrent_chunks;
	for(size_t coarse_idx = 0; coarse_idx < coarse_chunks.size(); ++coarse_idx) {
		for(const auto& fine_chunk : split(coarse_chunks[coarse_idx], tsk.get_granularity(), oversubscribe_factor)) {
			auto& localized_chunk = concurrent_chunks.emplace_back();
			localized_chunk.execution_range = box(subrange(fine_chunk.offset, fine_chunk.range));
			if(tsk.get_execution_target() == execution_target::device) {
				assert(coarse_idx < m_system.devices.size());
				localized_chunk.memory_id = m_system.devices[coarse_idx].native_memory;
				localized_chunk.device_id = device_id(coarse_idx);
			} else {
				localized_chunk.memory_id = host_memory_id;
			}
		}
	}
	return concurrent_chunks;
}

void generator_impl::report_task_overlapping_writes(const task& tsk, const std::vector<localized_chunk>& concurrent_chunks) const {
	box_vector<3> concurrent_execution_ranges(concurrent_chunks.size(), box<3>());
	std::transform(concurrent_chunks.begin(), concurrent_chunks.end(), concurrent_execution_ranges.begin(),
	    [](const localized_chunk& chunk) { return chunk.execution_range; });

	if(const auto overlapping_writes = detect_overlapping_writes(tsk, concurrent_execution_ranges); !overlapping_writes.empty()) {
		auto error = fmt::format("{} has overlapping writes on N{} in", print_task_debug_label(tsk, true /* title case */), m_local_nid);
		for(const auto& [bid, overlap] : overlapping_writes) {
			fmt::format_to(std::back_inserter(error), " {} {}", print_buffer_debug_label(bid), overlap);
		}
		error += ". Choose a non-overlapping range mapper for this write access or constrain the split via experimental::constrain_split to make the access "
		         "non-overlapping.";
		utils::report_error(m_policy.overlapping_write_error, "{}", error);
	}
}

void generator_impl::satisfy_task_buffer_requirements(batch& current_batch, const buffer_id bid, const task& tsk, const subrange<3>& local_execution_range,
    const bool local_node_is_reduction_initializer, const std::vector<localized_chunk>& concurrent_chunks_after_split) //
{
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("iggen::satisfy_buffer_requirements", ForestGreen);

	assert(!concurrent_chunks_after_split.empty());

	auto& buffer = m_buffers.at(bid);

	dense_map<memory_id, box_vector<3>> required_contiguous_allocations(m_memories.size());

	box_vector<3> accessed_boxes; // which elements are accessed (to figure out applying receives)
	box_vector<3> consumed_boxes; // which elements are accessed with a consuming access (these need to be preserved across resizes)

	const auto& bam = tsk.get_buffer_access_map();
	for(const auto mode : bam.get_access_modes(bid)) {
		const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), local_execution_range, tsk.get_global_size());
		accessed_boxes.append(req.get_boxes());
		if(access::mode_traits::is_consumer(mode)) { consumed_boxes.append(req.get_boxes()); }
	}

	// reductions can introduce buffer reads if they do not initialize_to_identity (but they cannot be split), so we evaluate them first
	assert(std::count_if(tsk.get_reductions().begin(), tsk.get_reductions().end(), [=](const reduction_info& r) { return r.bid == bid; }) <= 1
	       && "task defines multiple reductions on the same buffer");
	const auto reduction = std::find_if(tsk.get_reductions().begin(), tsk.get_reductions().end(), [=](const reduction_info& r) { return r.bid == bid; });
	if(reduction != tsk.get_reductions().end()) {
		for(const auto& chunk : concurrent_chunks_after_split) {
			required_contiguous_allocations[chunk.memory_id].push_back(scalar_reduction_box);
		}
		const auto include_current_value = local_node_is_reduction_initializer && reduction->init_from_buffer;
		if(concurrent_chunks_after_split.size() > 1 || include_current_value) {
			// We insert a host-side reduce-instruction in the multi-chunk scenario; its result will end up in the host buffer allocation.
			// If the user did not specify `initialize_to_identity`, we treat the existing buffer contents as an additional reduction chunk, so we can always
			// perform SYCL reductions with `initialize_to_identity` semantics.
			required_contiguous_allocations[host_memory_id].push_back(scalar_reduction_box);
		}
		accessed_boxes.push_back(scalar_reduction_box);
		if(include_current_value) {
			// scalar_reduction_box will be copied into the local-reduction gather buffer ahead of the kernel instruction
			consumed_boxes.push_back(scalar_reduction_box);
		}
	}

	const region accessed_region(std::move(accessed_boxes));
	const region consumed_region(std::move(consumed_boxes));

	// Boxes that are accessed but not consumed do not need to be preserved across resizes. This set operation is not equivalent to accumulating all
	// non-consumer mode accesses above, since a kernel can have both a read_only and a discard_write access for the same buffer element, and Celerity must
	// treat the overlap as-if it were a read_write access according to the SYCL spec.
	// We maintain a box_vector here because we also add all received boxes, as these are overwritten by a recv_instruction before being read from the kernel.
	box_vector<3> discarded_boxes = region_difference(accessed_region, consumed_region).into_boxes();

	// Collect all pending receives (await-push commands) that we must apply before executing this task.
	std::vector<buffer_state::region_receive> applied_receives;
	{
		const auto first_applied_receive = std::partition(buffer.pending_receives.begin(), buffer.pending_receives.end(),
		    [&](const buffer_state::region_receive& r) { return region_intersection(consumed_region, r.received_region).empty(); });
		const auto last_applied_receive = buffer.pending_receives.end();
		for(auto it = first_applied_receive; it != last_applied_receive; ++it) {
			// we (re) allocate before receiving, but there's no need to preserve previous data at the receive location
			discarded_boxes.append(it->received_region.get_boxes());
			// split_receive_instruction needs contiguous allocations for the bounding boxes of potentially received fragments
			required_contiguous_allocations[host_memory_id].insert(
			    required_contiguous_allocations[host_memory_id].end(), it->required_contiguous_allocations.begin(), it->required_contiguous_allocations.end());
		}

		if(first_applied_receive != last_applied_receive) {
			applied_receives.assign(first_applied_receive, last_applied_receive);
			buffer.pending_receives.erase(first_applied_receive, last_applied_receive);
		}
	}

	if(reduction != tsk.get_reductions().end()) {
		assert(std::all_of(buffer.pending_receives.begin(), buffer.pending_receives.end(), [&](const buffer_state::region_receive& r) {
			return region_intersection(r.received_region, scalar_reduction_box).empty();
		}) && std::all_of(buffer.pending_gathers.begin(), buffer.pending_gathers.end(), [&](const buffer_state::gather_receive& r) {
			return box_intersection(r.gather_box, scalar_reduction_box).empty();
		}) && "buffer has an unprocessed await-push into a region that is going to be used as a reduction output");
	}

	const region discarded_region = region(std::move(discarded_boxes));

	// Detect and report uninitialized reads
	if(m_policy.uninitialized_read_error != error_policy::ignore) {
		box_vector<3> uninitialized_reads;
		const auto locally_required_region = region_difference(consumed_region, discarded_region);
		for(const auto& [box, location] : buffer.up_to_date_memories.get_region_values(locally_required_region)) {
			if(!location.any()) { uninitialized_reads.push_back(box); }
		}
		if(!uninitialized_reads.empty()) {
			// Observing an uninitialized read that is not visible in the TDAG means we have a bug.
			utils::report_error(m_policy.uninitialized_read_error,
			    "Instructions for {} are trying to read {} {}, which is neither found locally nor has been await-pushed before.", print_task_debug_label(tsk),
			    print_buffer_debug_label(bid), detail::region(std::move(uninitialized_reads)));
		}
	}

	// Do not preserve any received or overwritten region across receives or buffer resizes later on: allocate_contiguously will insert resize-copy instructions
	// for all up_to_date regions of allocations that it replaces with larger ones.
	buffer.up_to_date_memories.update_region(discarded_region, memory_mask());

	// Collect chunk-reads by memory to establish local coherence later
	dense_map<memory_id, std::vector<region<3>>> concurrent_reads_from_memory(m_memories.size());
	for(const auto& chunk : concurrent_chunks_after_split) {
		required_contiguous_allocations[chunk.memory_id].append(
		    bam.get_required_contiguous_boxes(bid, tsk.get_dimensions(), chunk.execution_range.get_subrange(), tsk.get_global_size()));

		box_vector<3> chunk_read_boxes;
		for(const auto mode : access::consumer_modes) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), chunk.execution_range.get_subrange(), tsk.get_global_size());
			chunk_read_boxes.append(req.get_boxes());
		}
		if(!chunk_read_boxes.empty()) { concurrent_reads_from_memory[chunk.memory_id].push_back(region(std::move(chunk_read_boxes))); }
	}
	if(local_node_is_reduction_initializer && reduction != tsk.get_reductions().end() && reduction->init_from_buffer) {
		concurrent_reads_from_memory[host_memory_id].emplace_back(scalar_reduction_box);
	}

	// Now that we know all required contiguous allocations, issue any required alloc- and resize-copy instructions
	for(memory_id mid = 0; mid < required_contiguous_allocations.size(); ++mid) {
		allocate_contiguously(current_batch, bid, mid, std::move(required_contiguous_allocations[mid]));
	}

	// Receive all remote data (which overlaps with the accessed region) into host memory
	std::vector<region<3>> all_concurrent_reads;
	for(const auto& reads : concurrent_reads_from_memory) {
		all_concurrent_reads.insert(all_concurrent_reads.end(), reads.begin(), reads.end());
	}
	for(const auto& receive : applied_receives) {
		commit_pending_region_receive_to_host_memory(current_batch, bid, receive, all_concurrent_reads);
	}

	// Create the necessary coherence copy instructions to satisfy all remaining requirements locally.
	establish_coherence_between_buffer_memories(current_batch, bid, concurrent_reads_from_memory);
}

local_reduction generator_impl::prepare_task_local_reduction(
    batch& command_batch, const reduction_info& rinfo, const execution_command& ecmd, const task& tsk, const size_t num_concurrent_chunks) //
{
	const auto [rid_, bid_, reduction_task_includes_buffer_value] = rinfo;
	const auto bid = bid_; // allow capturing in lambda

	auto& buffer = m_buffers.at(bid);

	local_reduction red;
	red.include_local_buffer_value = reduction_task_includes_buffer_value && ecmd.is_reduction_initializer();
	red.first_kernel_chunk_offset = red.include_local_buffer_value ? 1 : 0;
	red.num_input_chunks = red.first_kernel_chunk_offset + num_concurrent_chunks;
	red.chunk_size_bytes = scalar_reduction_box.get_area() * buffer.elem_size;

	// If the reduction only has a single local contribution, we simply accept it as the fully-reduced final value without issuing additional instructions.
	// A local_reduction with num_input_chunks == 1 is treated as a no-op by `finish_task_local_reduction`.
	assert(red.num_input_chunks > 0);
	if(red.num_input_chunks == 1) return red;

	// Create a gather-allocation into which `finish_task_local_reduction` will copy each new partial result, and if the reduction is not
	// initialize_to_identity, we copy the current buffer value before it is being overwritten by the kernel.
	red.gather_aid = new_allocation_id(host_memory_id);
	red.gather_alloc_instr = create<alloc_instruction>(
	    command_batch, red.gather_aid, red.num_input_chunks * red.chunk_size_bytes, buffer.elem_align, [&](const auto& record_debug_info) {
		    record_debug_info(
		        alloc_instruction_record::alloc_origin::gather, buffer_allocation_record{bid, buffer.debug_name, scalar_reduction_box}, red.num_input_chunks);
	    });
	add_dependency(red.gather_alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);

	/// Normally, there is one _reduction chunk_ per _kernel chunk_, unless the local node is the designated reduction initializer and the reduction is not
	/// `initialize_to_identity`, in which case we add an additional _reduction chunk_ for the current buffer value and insert it in the first position of the
	/// local gather allocation.
	if(red.include_local_buffer_value) {
		// The source host allocation is already provided by satisfy_task_buffer_requirements
		auto& source_allocation = buffer.memories[host_memory_id].get_contiguous_allocation(scalar_reduction_box);
		const size_t dest_offset_bytes = 0; // initial value is the first entry

		// copy to local gather space
		const auto current_value_copy_instr = create<copy_instruction>(command_batch, source_allocation.aid, red.gather_aid,
		    strided_layout(source_allocation.box), linearized_layout(dest_offset_bytes), scalar_reduction_box, buffer.elem_size,
		    [&](const auto& record_debug_info) { record_debug_info(copy_instruction_record::copy_origin::gather, bid, buffer.debug_name); });

		add_dependency(current_value_copy_instr, red.gather_alloc_instr, instruction_dependency_origin::allocation_lifetime);
		perform_concurrent_read_from_allocation(current_value_copy_instr, source_allocation, scalar_reduction_box);
	}
	return red;
}

void generator_impl::finish_task_local_reduction(batch& command_batch, const local_reduction& red, const reduction_info& rinfo, const execution_command& ecmd,
    const task& tsk,
    const std::vector<localized_chunk>& concurrent_chunks) //
{
	// If the reduction only has a single contribution, its write is already the final result and does not need to be reduced.
	if(red.num_input_chunks == 1) return;

	const auto [rid, bid_, reduction_task_includes_buffer_value] = rinfo;
	const auto bid = bid_; // allow capturing in lambda

	auto& buffer = m_buffers.at(bid);
	auto& host_memory = buffer.memories[host_memory_id];

	// prepare_task_local_reduction has allocated gather space which preserves the current buffer value when the reduction does not initialize_to_identity
	std::vector<copy_instruction*> gather_copy_instrs;
	gather_copy_instrs.reserve(concurrent_chunks.size());
	for(size_t j = 0; j < concurrent_chunks.size(); ++j) {
		const auto source_mid = concurrent_chunks[j].memory_id;
		auto& source_allocation = buffer.memories[source_mid].get_contiguous_allocation(scalar_reduction_box);

		// Copy local partial result to gather space
		const auto copy_instr = create<copy_instruction>(command_batch, source_allocation.aid, red.gather_aid, strided_layout(source_allocation.box),
		    linearized_layout((red.first_kernel_chunk_offset + j) * buffer.elem_size /* offset */), scalar_reduction_box, buffer.elem_size,
		    [&](const auto& record_debug_info) { record_debug_info(copy_instruction_record::copy_origin::gather, bid, buffer.debug_name); });

		add_dependency(copy_instr, red.gather_alloc_instr, instruction_dependency_origin::allocation_lifetime);
		perform_concurrent_read_from_allocation(copy_instr, source_allocation, scalar_reduction_box);

		gather_copy_instrs.push_back(copy_instr);
	}

	// Insert a local reduce_instruction which reads from the gather buffer and writes to the host-buffer allocation for `scalar_reduction_box`.
	auto& dest_allocation = host_memory.get_contiguous_allocation(scalar_reduction_box);
	const auto reduce_instr =
	    create<reduce_instruction>(command_batch, rid, red.gather_aid, red.num_input_chunks, dest_allocation.aid, [&](const auto& record_debug_info) {
		    record_debug_info(std::nullopt, bid, buffer.debug_name, scalar_reduction_box, reduce_instruction_record::reduction_scope::local);
	    });

	for(auto& copy_instr : gather_copy_instrs) {
		add_dependency(reduce_instr, copy_instr, instruction_dependency_origin::read_from_allocation);
	}
	perform_atomic_write_to_allocation(reduce_instr, dest_allocation, scalar_reduction_box);
	buffer.track_original_write(scalar_reduction_box, reduce_instr, host_memory_id);

	// Free the gather allocation created in `prepare_task_local_reduction`.
	const auto gather_free_instr = create<free_instruction>(
	    command_batch, red.gather_aid, [&](const auto& record_debug_info) { record_debug_info(red.num_input_chunks * red.chunk_size_bytes, std::nullopt); });
	add_dependency(gather_free_instr, reduce_instr, instruction_dependency_origin::allocation_lifetime);
}

instruction* generator_impl::launch_task_kernel(batch& command_batch, const execution_command& ecmd, const task& tsk, const localized_chunk& chunk) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("iggen::launch_kernel", Blue2);

	const auto& bam = tsk.get_buffer_access_map();

	buffer_access_allocation_map allocation_map(bam.get_num_accesses());
	buffer_access_allocation_map reduction_map(tsk.get_reductions().size());
	size_t global_memory_access_estimate_bytes = 0;

	std::vector<buffer_memory_record> buffer_memory_access_map;       // if is_recording()
	std::vector<buffer_reduction_record> buffer_memory_reduction_map; // if is_recording()
	if(is_recording()) {
		buffer_memory_access_map.resize(bam.get_num_accesses());
		buffer_memory_reduction_map.resize(tsk.get_reductions().size());
	}

	// map buffer accesses (hydration_ids) to allocations in chunk-memory
	for(size_t i = 0; i < bam.get_num_accesses(); ++i) {
		const auto [bid, mode] = bam.get_nth_access(i);
		const auto accessed_box = bam.get_requirements_for_nth_access(i, tsk.get_dimensions(), chunk.execution_range.get_subrange(), tsk.get_global_size());
		const auto& buffer = m_buffers.at(bid);
		if(!accessed_box.empty()) {
			const auto& alloc = buffer.memories[chunk.memory_id].get_contiguous_allocation(accessed_box);
			allocation_map[i] = {alloc.aid, alloc.box, accessed_box CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, bid, buffer.debug_name)};
		} else {
			allocation_map[i] = buffer_access_allocation{null_allocation_id, {}, {} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, bid, buffer.debug_name)};
		}
		global_memory_access_estimate_bytes +=
		    (static_cast<size_t>(access::mode_traits::is_producer(mode)) + static_cast<size_t>(access::mode_traits::is_consumer(mode)))
		    * accessed_box.get_area() * buffer.elem_size;
		if(is_recording()) { buffer_memory_access_map[i] = buffer_memory_record{bid, buffer.debug_name}; }
	}

	// map reduction outputs to allocations in chunk-memory
	for(size_t i = 0; i < tsk.get_reductions().size(); ++i) {
		const auto& rinfo = tsk.get_reductions()[i];
		const auto& buffer = m_buffers.at(rinfo.bid);
		const auto& alloc = buffer.memories[chunk.memory_id].get_contiguous_allocation(scalar_reduction_box);
		reduction_map[i] = {alloc.aid, alloc.box, scalar_reduction_box CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, rinfo.bid, buffer.debug_name)};
		global_memory_access_estimate_bytes += chunk.execution_range.get_area() * buffer.elem_size;
		if(is_recording()) { buffer_memory_reduction_map[i] = buffer_reduction_record{rinfo.bid, buffer.debug_name, rinfo.rid}; }
	}

	if(tsk.get_execution_target() == execution_target::device) {
		assert(chunk.execution_range.get_area() > 0);
		assert(chunk.device_id.has_value());
		return create<device_kernel_instruction>(command_batch, *chunk.device_id, tsk.get_launcher<device_kernel_launcher>(), chunk.execution_range,
		    std::move(allocation_map), std::move(reduction_map),
		    global_memory_access_estimate_bytes //
		        CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, tsk.get_type(), tsk.get_id(), tsk.get_debug_name()),
		    [&](const auto& record_debug_info) {
			    record_debug_info(ecmd.get_tid(), ecmd.get_cid(), tsk.get_debug_name(), buffer_memory_access_map, buffer_memory_reduction_map);
		    });
	} else {
		assert(tsk.get_execution_target() == execution_target::host);
		assert(chunk.memory_id == host_memory_id);
		assert(reduction_map.empty());
		// We ignore global_memory_access_estimate_bytes for host tasks because they are typically limited by I/O instead
		return create<host_task_instruction>(command_batch, tsk.get_launcher<host_task_launcher>(), chunk.execution_range, tsk.get_global_size(),
		    std::move(allocation_map),
		    tsk.get_collective_group_id() //
		    CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, tsk.get_type(), tsk.get_id(), tsk.get_debug_name()),
		    [&](const auto& record_debug_info) { record_debug_info(ecmd.get_tid(), ecmd.get_cid(), tsk.get_debug_name(), buffer_memory_access_map); });
	}
}

void generator_impl::perform_task_buffer_accesses(
    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks, const std::vector<instruction*>& command_instructions) //
{
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("iggen::perform_buffer_access", Red3);

	const auto& bam = tsk.get_buffer_access_map();
	if(bam.get_num_accesses() == 0 && tsk.get_reductions().empty()) return;

	// 1. Collect the read-sets and write-sets of all concurrent chunks on all buffers (TODO this is what buffer_access_map should actually return)

	struct read_write_sets {
		region<3> reads;
		region<3> writes;
	};

	std::vector<std::unordered_map<buffer_id, read_write_sets>> concurrent_read_write_sets(concurrent_chunks.size());

	for(const auto bid : bam.get_accessed_buffers()) {
		for(size_t i = 0; i < concurrent_chunks.size(); ++i) {
			read_write_sets rw;
			for(const auto mode : bam.get_access_modes(bid)) {
				const auto req =
				    bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), concurrent_chunks[i].execution_range.get_subrange(), tsk.get_global_size());
				if(access::mode_traits::is_consumer(mode)) { rw.reads = region_union(rw.reads, req); }
				if(access::mode_traits::is_producer(mode)) { rw.writes = region_union(rw.writes, req); }
			}
			concurrent_read_write_sets[i].emplace(bid, std::move(rw));
		}
	}

	for(const auto& rinfo : tsk.get_reductions()) {
		for(size_t i = 0; i < concurrent_chunks.size(); ++i) {
			auto& rw_map = concurrent_read_write_sets[i][rinfo.bid]; // allow default-insert on `bid`
			rw_map.writes = region_union(rw_map.writes, scalar_reduction_box);
		}
	}

	// 2. Insert all true-dependencies for reads and anti-dependencies for writes. We do this en-bloc instead of using `perform_concurrent_read_from_allocation`
	// or `perform_atomic_write_to_allocation` to avoid incorrect dependencies between our concurrent chunks by updating tracking structures too early.

	for(size_t i = 0; i < concurrent_chunks.size(); ++i) {
		for(const auto& [bid, rw] : concurrent_read_write_sets[i]) {
			auto& buffer = m_buffers.at(bid);
			auto& memory = buffer.memories[concurrent_chunks[i].memory_id];

			for(auto& allocation : memory.allocations) {
				add_dependencies_on_last_writers(command_instructions[i], allocation, region_intersection(rw.reads, allocation.box));
				add_dependencies_on_last_concurrent_accesses(
				    command_instructions[i], allocation, region_intersection(rw.writes, allocation.box), instruction_dependency_origin::write_to_allocation);
			}
		}
	}

	// 3. Clear tracking structures for all regions that are being written to. We gracefully handle overlapping writes by treating the set of all conflicting
	// writers as last writers of an allocation.

	for(size_t i = 0; i < concurrent_chunks.size(); ++i) {
		for(const auto& [bid, rw] : concurrent_read_write_sets[i]) {
			assert(command_instructions[i] != nullptr);
			auto& buffer = m_buffers.at(bid);
			for(auto& alloc : buffer.memories[concurrent_chunks[i].memory_id].allocations) {
				alloc.begin_concurrent_writes(region_intersection(alloc.box, rw.writes));
			}
		}
	}

	// 4. Update data locations and last writers resulting from all concurrent reads and overlapping writes

	for(size_t i = 0; i < concurrent_chunks.size(); ++i) {
		for(const auto& [bid, rw] : concurrent_read_write_sets[i]) {
			assert(command_instructions[i] != nullptr);
			auto& buffer = m_buffers.at(bid);

			for(auto& alloc : buffer.memories[concurrent_chunks[i].memory_id].allocations) {
				alloc.track_concurrent_read(region_intersection(alloc.box, rw.reads), command_instructions[i]);
				alloc.track_concurrent_write(region_intersection(alloc.box, rw.writes), command_instructions[i]);
			}
			buffer.track_original_write(rw.writes, command_instructions[i], concurrent_chunks[i].memory_id);
		}
	}
}

void generator_impl::perform_task_side_effects(
    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks, const std::vector<instruction*>& command_instructions) //
{
	if(tsk.get_side_effect_map().empty()) return;

	assert(concurrent_chunks.size() == 1); // splitting instructions with side effects would race
	assert(!concurrent_chunks[0].device_id.has_value());
	assert(concurrent_chunks[0].memory_id == host_memory_id);

	for(const auto& [hoid, order] : tsk.get_side_effect_map()) {
		auto& host_object = m_host_objects.at(hoid);
		if(const auto last_side_effect = host_object.last_side_effect) {
			add_dependency(command_instructions[0], last_side_effect, instruction_dependency_origin::side_effect);
		}
		host_object.last_side_effect = command_instructions[0];
	}
}

void generator_impl::perform_task_collective_operations(
    const task& tsk, const std::vector<localized_chunk>& concurrent_chunks, const std::vector<instruction*>& command_instructions) //
{
	if(tsk.get_collective_group_id() == non_collective_group_id) return;

	assert(concurrent_chunks.size() == 1); //
	assert(!concurrent_chunks[0].device_id.has_value());
	assert(concurrent_chunks[0].memory_id == host_memory_id);

	auto& group = m_collective_groups.at(tsk.get_collective_group_id()); // must be created previously with clone_collective_group_instruction
	add_dependency(command_instructions[0], group.last_collective_operation, instruction_dependency_origin::collective_group_order);
	group.last_collective_operation = command_instructions[0];
}

void generator_impl::compile_execution_command(batch& command_batch, const execution_command& ecmd) {
	const auto& tsk = *m_tm->get_task(ecmd.get_tid());

	// 1. If this is a collective host task, we might need to insert a `clone_collective_group_instruction` which the task instruction is later serialized on.
	create_task_collective_groups(command_batch, tsk);

	// 2. Split the task into local chunks and (in case of a device kernel) assign it to devices
	const auto concurrent_chunks = split_task_execution_range(ecmd, tsk);

	// 3. Detect and report overlapping writes - is not a fatal error to discover one, we always generate an executable (albeit racy) instruction graph
	if(m_policy.overlapping_write_error != error_policy::ignore) { report_task_overlapping_writes(tsk, concurrent_chunks); }

	// 4. Perform all necessary receives, allocations, resize- and coherence copies to provide an appropriate set of buffer allocations and data distribution
	// for all kernels and host tasks of this task. This is done simultaneously for all chunks to optimize the graph and avoid inefficient copy-chains.
	auto accessed_bids = tsk.get_buffer_access_map().get_accessed_buffers();
	for(const auto& rinfo : tsk.get_reductions()) {
		accessed_bids.insert(rinfo.bid);
	}
	for(const auto bid : accessed_bids) {
		satisfy_task_buffer_requirements(command_batch, bid, tsk, ecmd.get_execution_range(), ecmd.is_reduction_initializer(), concurrent_chunks);
	}

	// 5. If the task contains reductions with more than one local input, create the appropriate gather allocations and (if the local node is the designated
	// reduction initializer) copies the current buffer value into the new gather space.
	std::vector<local_reduction> local_reductions(tsk.get_reductions().size());
	for(size_t i = 0; i < local_reductions.size(); ++i) {
		local_reductions[i] = prepare_task_local_reduction(command_batch, tsk.get_reductions()[i], ecmd, tsk, concurrent_chunks.size());
	}

	// 6. Issue instructions to launch all concurrent kernels / host tasks.
	std::vector<instruction*> command_instructions(concurrent_chunks.size());
	for(size_t i = 0; i < concurrent_chunks.size(); ++i) {
		command_instructions[i] = launch_task_kernel(command_batch, ecmd, tsk, concurrent_chunks[i]);
	}

	// 7. Compute dependencies and update tracking data structures
	perform_task_buffer_accesses(tsk, concurrent_chunks, command_instructions);
	perform_task_side_effects(tsk, concurrent_chunks, command_instructions);
	perform_task_collective_operations(tsk, concurrent_chunks, command_instructions);

	// 8. For any reductions with more than one local input, collect partial results and perform the reduction operation in host memory. This is done eagerly to
	// avoid ever having to persist partial reduction states in our buffer tracking.
	for(size_t i = 0; i < local_reductions.size(); ++i) {
		finish_task_local_reduction(command_batch, local_reductions[i], tsk.get_reductions()[i], ecmd, tsk, concurrent_chunks);
	}

	// 9. If any of the instructions have no predecessor, anchor them on the last epoch (this can only happen for chunks without any buffer accesses).
	for(const auto instr : command_instructions) {
		if(instr->get_dependencies().empty()) { add_dependency(instr, m_last_epoch, instruction_dependency_origin::last_epoch); }
	}
}

void generator_impl::compile_push_command(batch& command_batch, const push_command& pcmd) {
	const auto trid = pcmd.get_transfer_id();
	const auto push_box = box(pcmd.get_range());

	// If not all nodes contribute partial results to a global reductions, the remaining ones need to notify their peers that they should not expect any data.
	// This is done by announcing an empty box through the pilot message, but not actually performing a send.
	if(push_box.empty()) {
		assert(trid.rid != no_reduction_id);
		create_outbound_pilot(command_batch, pcmd.get_target(), trid, box<3>());
		return;
	}

	// Prioritize all instructions participating in a "push" to hide the latency of establishing local coherence behind the typically much longer latencies of
	// inter-node communication
	command_batch.base_priority = 10;

	auto& buffer = m_buffers.at(trid.bid);
	auto& host_memory = buffer.memories[host_memory_id];

	// We want to generate the fewest number of send instructions possible without introducing new synchronization points between chunks of the same
	// command that generated the pushed data. This will allow computation-communication overlap, especially in the case of oversubscribed splits.
	dense_map<memory_id, std::vector<region<3>>> concurrent_send_source_regions(host_memory_id + 1); // establish_coherence() takes a dense_map
	auto& concurrent_send_regions = concurrent_send_source_regions[host_memory_id];

	// Since we now send boxes individually, we do not need to allocate the entire push_box contiguously.
	box_vector<3> required_host_allocation;
	{
		std::unordered_map<instruction_id, box_vector<3>> individual_send_boxes;
		for(auto& [box, original_writer] : buffer.original_writers.get_region_values(push_box)) {
			individual_send_boxes[original_writer->get_id()].push_back(box);
			required_host_allocation.push_back(box);
		}
		for(auto& [original_writer, boxes] : individual_send_boxes) {
			concurrent_send_regions.push_back(region(std::move(boxes)));
		}
	}

	allocate_contiguously(command_batch, trid.bid, host_memory_id, std::move(required_host_allocation));
	establish_coherence_between_buffer_memories(command_batch, trid.bid, concurrent_send_source_regions);

	for(const auto& send_region : concurrent_send_regions) {
		for(const auto& full_send_box : send_region.get_boxes()) {
			// Splitting must happen on buffer range instead of host allocation range to ensure boxes are also suitable for the receiver, which might have
			// a differently-shaped backing allocation
			for(const auto& compatible_send_box : split_into_communicator_compatible_boxes(buffer.range, full_send_box)) {
				const message_id msgid = create_outbound_pilot(command_batch, pcmd.get_target(), trid, compatible_send_box);

				auto& allocation = host_memory.get_contiguous_allocation(compatible_send_box); // we allocate_contiguously above

				const auto offset_in_allocation = compatible_send_box.get_offset() - allocation.box.get_offset();
				const auto send_instr = create<send_instruction>(command_batch, pcmd.get_target(), msgid, allocation.aid, allocation.box.get_range(),
				    offset_in_allocation, compatible_send_box.get_range(), buffer.elem_size,
				    [&](const auto& record_debug_info) { record_debug_info(pcmd.get_cid(), trid, buffer.debug_name, compatible_send_box.get_offset()); });

				perform_concurrent_read_from_allocation(send_instr, allocation, compatible_send_box);
			}
		}
	}
}

void generator_impl::defer_await_push_command(const await_push_command& apcmd) {
	// We do not generate instructions for await-push commands immediately upon receiving them; instead, we buffer them and generate
	// recv-instructions as soon as data is to be read by another instruction. This way, we can split the recv instructions and avoid
	// unnecessary synchronization points between chunks that can otherwise profit from a computation-communication overlap.

	const auto& trid = apcmd.get_transfer_id();
	if(is_recording()) { m_recorder->record_await_push_command_id(trid, apcmd.get_cid()); }

	auto& buffer = m_buffers.at(trid.bid);

#ifndef NDEBUG
	for(const auto& receive : buffer.pending_receives) {
		assert((trid.rid != no_reduction_id || receive.consumer_tid != trid.consumer_tid)
		       && "received multiple await-pushes for the same consumer-task, buffer and reduction id");
		assert(region_intersection(receive.received_region, apcmd.get_region()).empty()
		       && "received an await-push command into a previously await-pushed region without an intermediate read");
	}
	for(const auto& gather : buffer.pending_gathers) {
		assert(std::pair(gather.consumer_tid, gather.rid) != std::pair(trid.consumer_tid, gather.rid)
		       && "received multiple await-pushes for the same consumer-task, buffer and reduction id");
		assert(region_intersection(gather.gather_box, apcmd.get_region()).empty()
		       && "received an await-push command into a previously await-pushed region without an intermediate read");
	}
#endif

	if(trid.rid == no_reduction_id) {
		buffer.pending_receives.emplace_back(trid.consumer_tid, apcmd.get_region(), connected_subregion_bounding_boxes(apcmd.get_region()));
	} else {
		assert(apcmd.get_region().get_boxes().size() == 1);
		buffer.pending_gathers.emplace_back(trid.consumer_tid, trid.rid, apcmd.get_region().get_boxes().front());
	}
}

void generator_impl::compile_reduction_command(batch& command_batch, const reduction_command& rcmd) {
	// In a single-node setting, global reductions are no-ops, so no reduction commands should ever be issued
	assert(m_num_nodes > 1 && "received a reduction command in a single-node configuration");

	const auto [rid_, bid_, init_from_buffer] = rcmd.get_reduction_info();
	const auto rid = rid_; // allow capturing in lambda
	const auto bid = bid_; // allow capturing in lambda

	auto& buffer = m_buffers.at(bid);

	const auto gather = std::find_if(buffer.pending_gathers.begin(), buffer.pending_gathers.end(), [&](const buffer_state::gather_receive& g) {
		return g.rid == rid; // assume that g.consumer_tid is correct because there cannot be multiple concurrent reductions for a single task
	});
	assert(gather != buffer.pending_gathers.end() && "received reduction command that is not preceded by an appropriate await-push");
	assert(gather->gather_box == scalar_reduction_box);

	// 1. Create a host-memory allocation to gather the array of partial results

	const auto gather_aid = new_allocation_id(host_memory_id);
	const auto node_chunk_size = gather->gather_box.get_area() * buffer.elem_size;
	const auto gather_alloc_instr = create<
	    alloc_instruction>(command_batch, gather_aid, m_num_nodes * node_chunk_size, buffer.elem_align, [&](const auto& record_debug_info) {
		record_debug_info(alloc_instruction_record::alloc_origin::gather, buffer_allocation_record{bid, buffer.debug_name, gather->gather_box}, m_num_nodes);
	});
	add_dependency(gather_alloc_instr, m_last_epoch, instruction_dependency_origin::last_epoch);

	// 2. Fill the gather space with the reduction identity, so that the gather_receive_command can simply ignore empty boxes sent by peers that do not
	// contribute to the reduction, and we can skip the gather-copy instruction if we ourselves do not contribute a partial result.

	const auto fill_identity_instr =
	    create<fill_identity_instruction>(command_batch, rid, gather_aid, m_num_nodes, [](const auto& record_debug_info) { record_debug_info(); });
	add_dependency(fill_identity_instr, gather_alloc_instr, instruction_dependency_origin::allocation_lifetime);

	// 3. If the local node contributes to the reduction, copy the contribution to the appropriate position in the gather space. Testing `up_to_date_memories`
	// locally is not enough to establish whether there is a local contribution, since the local node might not have participated in the task that initiated the
	// reduction. Instead, we are informed about this condition by the command graph.

	copy_instruction* local_gather_copy_instr = nullptr;
	if(rcmd.has_local_contribution()) {
		const auto contribution_location = buffer.up_to_date_memories.get_region_values(scalar_reduction_box).front().second;
		const auto source_mid = next_location(contribution_location, host_memory_id);
		// if scalar_box is up to date in that memory, it (the single element) must also be contiguous
		auto& source_allocation = buffer.memories[source_mid].get_contiguous_allocation(scalar_reduction_box);

		local_gather_copy_instr = create<copy_instruction>(command_batch, source_allocation.aid, gather_aid, strided_layout(source_allocation.box),
		    linearized_layout(m_local_nid * buffer.elem_size /* offset */), scalar_reduction_box, buffer.elem_size,
		    [&](const auto& record_debug_info) { record_debug_info(copy_instruction_record::copy_origin::gather, bid, buffer.debug_name); });
		add_dependency(local_gather_copy_instr, fill_identity_instr, instruction_dependency_origin::write_to_allocation);
		perform_concurrent_read_from_allocation(local_gather_copy_instr, source_allocation, scalar_reduction_box);
	}

	// 4. Gather remote contributions to the partial result array

	const transfer_id trid(gather->consumer_tid, bid, gather->rid);
	const auto gather_recv_instr = create<gather_receive_instruction>(command_batch, trid, gather_aid, node_chunk_size,
	    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name, gather->gather_box, m_num_nodes); });
	add_dependency(gather_recv_instr, fill_identity_instr, instruction_dependency_origin::write_to_allocation);

	// 5. Perform the global reduction on the host by reading the array of inputs from the gather space and writing to the buffer's host allocation that covers
	// `scalar_reduction_box`.

	allocate_contiguously(command_batch, bid, host_memory_id, {scalar_reduction_box});

	auto& host_memory = buffer.memories[host_memory_id];
	auto& dest_allocation = host_memory.get_contiguous_allocation(scalar_reduction_box);

	const auto reduce_instr = create<reduce_instruction>(command_batch, rid, gather_aid, m_num_nodes, dest_allocation.aid, [&](const auto& record_debug_info) {
		record_debug_info(rcmd.get_cid(), bid, buffer.debug_name, scalar_reduction_box, reduce_instruction_record::reduction_scope::global);
	});
	add_dependency(reduce_instr, gather_recv_instr, instruction_dependency_origin::read_from_allocation);
	if(local_gather_copy_instr != nullptr) { add_dependency(reduce_instr, local_gather_copy_instr, instruction_dependency_origin::read_from_allocation); }
	perform_atomic_write_to_allocation(reduce_instr, dest_allocation, scalar_reduction_box);
	buffer.track_original_write(scalar_reduction_box, reduce_instr, host_memory_id);

	// 6. Free the gather space

	const auto gather_free_instr = create<free_instruction>(
	    command_batch, gather_aid, [&](const auto& record_debug_info) { record_debug_info(m_num_nodes * node_chunk_size, std::nullopt); });
	add_dependency(gather_free_instr, reduce_instr, instruction_dependency_origin::allocation_lifetime);

	buffer.pending_gathers.clear();

	// The associated reducer will be garbage-collected form the executor as we pass the reduction id on via the instruction_garbage member of the next horizon
	// or epoch instruction.
}

void generator_impl::compile_fence_command(batch& command_batch, const fence_command& fcmd) {
	const auto& tsk = *m_tm->get_task(fcmd.get_tid());

	assert(tsk.get_reductions().empty());
	assert(tsk.get_collective_group_id() == non_collective_group_id);

	const auto& bam = tsk.get_buffer_access_map();
	const auto& sem = tsk.get_side_effect_map();
	assert(bam.get_num_accesses() + sem.size() == 1);

	// buffer fences encode their buffer id and subrange through buffer_access_map with a fixed range mapper (which is rather ugly)
	if(bam.get_num_accesses() != 0) {
		const auto bid = *bam.get_accessed_buffers().begin();
		const auto fence_region = bam.get_mode_requirements(bid, access_mode::read, 0, {}, zeros);
		const auto fence_box = !fence_region.empty() ? fence_region.get_boxes().front() : box<3>();

		const auto user_allocation_id = tsk.get_fence_promise()->get_user_allocation_id();
		assert(user_allocation_id != null_allocation_id && user_allocation_id.get_memory_id() == user_memory_id);

		auto& buffer = m_buffers.at(bid);
		copy_instruction* copy_instr = nullptr;
		// gracefully handle empty-range buffer fences
		if(!fence_box.empty()) {
			// We make the host buffer coherent first in order to apply pending await-pushes.
			// TODO this enforces a contiguous host-buffer allocation which may cause unnecessary resizes.
			satisfy_task_buffer_requirements(command_batch, bid, tsk, {}, false /* is_reduction_initializer: irrelevant */,
			    std::vector{localized_chunk{host_memory_id, std::nullopt, box<3>()}} /* local_chunks: irrelevant */);

			auto& host_buffer_allocation = buffer.memories[host_memory_id].get_contiguous_allocation(fence_box);
			copy_instr = create<copy_instruction>(command_batch, host_buffer_allocation.aid, user_allocation_id, strided_layout(host_buffer_allocation.box),
			    strided_layout(fence_box), fence_box, buffer.elem_size,
			    [&](const auto& record_debug_info) { record_debug_info(copy_instruction_record::copy_origin::fence, bid, buffer.debug_name); });

			perform_concurrent_read_from_allocation(copy_instr, host_buffer_allocation, fence_box);
		}

		const auto fence_instr = create<fence_instruction>(command_batch, tsk.get_fence_promise(),
		    [&](const auto& record_debug_info) { record_debug_info(tsk.get_id(), fcmd.get_cid(), bid, buffer.debug_name, fence_box.get_subrange()); });

		if(copy_instr != nullptr) {
			add_dependency(fence_instr, copy_instr, instruction_dependency_origin::read_from_allocation);
		} else {
			// an empty-range buffer fence has no data dependencies but must still be executed to fulfill its promise - attach it to the current epoch.
			add_dependency(fence_instr, m_last_epoch, instruction_dependency_origin::last_epoch);
		}

		// we will just assume that the runtime does not intend to re-use the allocation it has passed
		m_unreferenced_user_allocations.push_back(user_allocation_id);
	}

	// host-object fences encode their host-object id in the task side effect map (which is also very ugly)
	if(!sem.empty()) {
		const auto hoid = sem.begin()->first;

		auto& obj = m_host_objects.at(hoid);
		const auto fence_instr = create<fence_instruction>(
		    command_batch, tsk.get_fence_promise(), [&, hoid = hoid](const auto& record_debug_info) { record_debug_info(tsk.get_id(), fcmd.get_cid(), hoid); });

		add_dependency(fence_instr, obj.last_side_effect, instruction_dependency_origin::side_effect);
		obj.last_side_effect = fence_instr;
	}
}

void generator_impl::compile_horizon_command(batch& command_batch, const horizon_command& hcmd) {
	m_idag->begin_epoch(hcmd.get_tid());
	instruction_garbage garbage{hcmd.get_completed_reductions(), std::move(m_unreferenced_user_allocations)};
	const auto horizon = create<horizon_instruction>(
	    command_batch, hcmd.get_tid(), std::move(garbage), [&](const auto& record_debug_info) { record_debug_info(hcmd.get_cid()); });

	collapse_execution_front_to(horizon);
	if(m_last_horizon != nullptr) { apply_epoch(m_last_horizon); }
	m_last_horizon = horizon;
}

void generator_impl::compile_epoch_command(batch& command_batch, const epoch_command& ecmd) {
	if(ecmd.get_epoch_action() == epoch_action::shutdown) { free_all_staging_allocations(command_batch); }

	m_idag->begin_epoch(ecmd.get_tid());
	instruction_garbage garbage{ecmd.get_completed_reductions(), std::move(m_unreferenced_user_allocations)};
	const auto epoch = create<epoch_instruction>(
	    command_batch, ecmd.get_tid(), ecmd.get_epoch_action(), std::move(garbage), [&](const auto& record_debug_info) { record_debug_info(ecmd.get_cid()); });

	collapse_execution_front_to(epoch);
	apply_epoch(epoch);
	m_last_horizon = nullptr;
}

void generator_impl::flush_batch(batch&& batch) { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) we do move the members of `batch`
	// sanity check: every instruction except the initial epoch must be temporally anchored through at least one dependency
	assert(std::all_of(batch.generated_instructions.begin(), batch.generated_instructions.end(),
	    [](const auto instr) { return instr->get_id() == 0 || !instr->get_dependencies().empty(); }));
	assert(is_topologically_sorted(batch.generated_instructions.begin(), batch.generated_instructions.end()));

	// instructions must be recorded manually after each create<instr>() call; verify that we never flush an unrecorded instruction
	assert(m_recorder == nullptr || std::all_of(batch.generated_instructions.begin(), batch.generated_instructions.end(), [this](const auto instr) {
		return std::find_if(m_recorder->get_instructions().begin(), m_recorder->get_instructions().end(), [=](const auto& rec) {
			return rec->id == instr->get_id();
		}) != m_recorder->get_instructions().end();
	}));

	if(m_delegate != nullptr && (!batch.generated_instructions.empty() || !batch.generated_pilots.empty())) {
		m_delegate->flush(std::move(batch.generated_instructions), std::move(batch.generated_pilots));
	}

#ifndef NDEBUG // ~batch() checks if it has been flushed, which we want to acknowledge even if m_delegate == nullptr
	batch.generated_instructions = {};
	batch.generated_pilots = {};
#endif
}

void generator_impl::compile(const abstract_command& cmd) {
	batch command_batch;
	matchbox::match(
	    cmd,                                                                                    //
	    [&](const execution_command& ecmd) { compile_execution_command(command_batch, ecmd); }, //
	    [&](const push_command& pcmd) { compile_push_command(command_batch, pcmd); },           //
	    [&](const await_push_command& apcmd) { defer_await_push_command(apcmd); },              //
	    [&](const horizon_command& hcmd) { compile_horizon_command(command_batch, hcmd); },     //
	    [&](const epoch_command& ecmd) { compile_epoch_command(command_batch, ecmd); },         //
	    [&](const reduction_command& rcmd) { compile_reduction_command(command_batch, rcmd); }, //
	    [&](const fence_command& fcmd) { compile_fence_command(command_batch, fcmd); }          //
	);
	flush_batch(std::move(command_batch));
}

std::string generator_impl::print_buffer_debug_label(const buffer_id bid) const { return utils::make_buffer_debug_label(bid, m_buffers.at(bid).debug_name); }

} // namespace celerity::detail::instruction_graph_generator_detail

namespace celerity::detail {

instruction_graph_generator::instruction_graph_generator(const task_manager& tm, const size_t num_nodes, const node_id local_nid, const system_info& system,
    instruction_graph& idag, delegate* dlg, instruction_recorder* const recorder, const policy_set& policy)
    : m_impl(new instruction_graph_generator_detail::generator_impl(tm, num_nodes, local_nid, system, idag, dlg, recorder, policy)) {}

instruction_graph_generator::~instruction_graph_generator() = default;

void instruction_graph_generator::notify_buffer_created(
    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_allocation_id) {
	m_impl->notify_buffer_created(bid, range, elem_size, elem_align, user_allocation_id);
}

void instruction_graph_generator::notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name) {
	m_impl->notify_buffer_debug_name_changed(bid, name);
}

void instruction_graph_generator::notify_buffer_destroyed(const buffer_id bid) { m_impl->notify_buffer_destroyed(bid); }

void instruction_graph_generator::notify_host_object_created(const host_object_id hoid, const bool owns_instance) {
	m_impl->notify_host_object_created(hoid, owns_instance);
}

void instruction_graph_generator::notify_host_object_destroyed(const host_object_id hoid) { m_impl->notify_host_object_destroyed(hoid); }

void instruction_graph_generator::compile(const abstract_command& cmd) { m_impl->compile(cmd); }

} // namespace celerity::detail
