#pragma once

#include "grid.h"
#include "launcher.h"
#include "ranges.h"
#include "types.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

#include <gch/small_vector.hpp>
#include <matchbox.hh>


namespace celerity::detail {

class fence_promise;

/// A node in the `instruction_graph`. This is not implemented as an `intrusive_graph` but references its predecessors by id to avoid thread-safety and lifetime
/// issues.
class instruction
    // Accept visitors to enable matchbox::match() on the instruction inheritance hierarchy
    : public matchbox::acceptor<class clone_collective_group_instruction, class alloc_instruction, class free_instruction, class copy_instruction,
          class device_kernel_instruction, class host_task_instruction, class send_instruction, class receive_instruction, class split_receive_instruction,
          class await_receive_instruction, class gather_receive_instruction, class fill_identity_instruction, class reduce_instruction, class fence_instruction,
          class destroy_host_object_instruction, class horizon_instruction, class epoch_instruction> {
  public:
	using edge_set = gch::small_vector<instruction_id>;

	/// If there are multiple instructions eligible for execution, the runtime should chose the one with the highest `priority` first.
	/// This provides a low-level scheduling heuristic to reduce the impact of submission latency and improve concurrency.
	explicit instruction(const instruction_id iid, const int priority) : m_id(iid), m_priority(priority) {}

	instruction_id get_id() const { return m_id; }
	int get_priority() const { return m_priority; }

	const edge_set& get_dependencies() const { return m_dependencies; }

	void add_dependency(const instruction_id iid) {
		// Adding n (unique) dependencies is O(N²) - we don't expect dependency lists to be long enough to warrant a set data structure.
		if(std::none_of(m_dependencies.begin(), m_dependencies.end(), [iid](const instruction_id dep) { return dep == iid; })) {
			m_dependencies.push_back(iid);
		}
	}

  private:
	instruction_id m_id;
	int m_priority;
	edge_set m_dependencies;
};

/// Orders instruction pointers by instruction id.
struct instruction_id_less {
	bool operator()(const instruction* const lhs, const instruction* const rhs) const { return lhs->get_id() < rhs->get_id(); }
	bool operator()(const std::unique_ptr<instruction>& lhs, const std::unique_ptr<instruction>& rhs) const { return lhs->get_id() < rhs->get_id(); }
};

/// Creates a new (MPI) collective group by cloning an existing one. The instuction is issued whenever the first host task on a new collective_group is
/// compiled, and itself is a collective operation.
class clone_collective_group_instruction : public matchbox::implement_acceptor<instruction, clone_collective_group_instruction> {
  public:
	explicit clone_collective_group_instruction(
	    const instruction_id iid, const int priority, const collective_group_id original_cgid, const collective_group_id new_cgid)
	    : acceptor_base(iid, priority), m_original_cgid(original_cgid), m_new_cgid(new_cgid) {}

	collective_group_id get_original_collective_group_id() const { return m_original_cgid; }
	collective_group_id get_new_collective_group_id() const { return m_new_cgid; }

  private:
	collective_group_id m_original_cgid;
	collective_group_id m_new_cgid;
};

/// Allocates a contiguous range of memory, either for use as a backing allocation for a buffer or for other purposes i.e. staging for transfer operations.
class alloc_instruction final : public matchbox::implement_acceptor<instruction, alloc_instruction> {
  public:
	explicit alloc_instruction(const instruction_id iid, const int priority, const allocation_id aid, const size_t size, const size_t alignment)
	    : acceptor_base(iid, priority), m_aid(aid), m_size(size), m_alignment(alignment) {}

	allocation_id get_allocation_id() const { return m_aid; }
	size_t get_size_bytes() const { return m_size; }
	size_t get_alignment_bytes() const { return m_alignment; }

  private:
	allocation_id m_aid;
	size_t m_size;
	size_t m_alignment;
};

/// Returns an allocation made with alloc_instruction to the system.
class free_instruction final : public matchbox::implement_acceptor<instruction, free_instruction> {
  public:
	explicit free_instruction(const instruction_id iid, const int priority, const allocation_id aid) : acceptor_base(iid, priority), m_aid(aid) {}

	allocation_id get_allocation_id() const { return m_aid; }

  private:
	allocation_id m_aid;
};


/// Copies one or more subranges of elements from one allocation to another, potentially between different memories.
class copy_instruction final : public matchbox::implement_acceptor<instruction, copy_instruction> {
  public:
	explicit copy_instruction(const instruction_id iid, const int priority, const allocation_with_offset& source_alloc,
	    const allocation_with_offset& dest_alloc, const box<3>& source_box, const box<3>& dest_box, region<3> copy_region, const size_t elem_size)
	    : acceptor_base(iid, priority), m_source_alloc(source_alloc), m_dest_alloc(dest_alloc), m_source_box(source_box), m_dest_box(dest_box),
	      m_copy_region(std::move(copy_region)), m_elem_size(elem_size) //
	{
		assert(!m_copy_region.empty());
		assert(m_source_box.covers(bounding_box(m_copy_region)));
		assert(m_dest_box.covers(bounding_box(m_copy_region)));
	}

	const allocation_with_offset& get_source_allocation() const { return m_source_alloc; }
	const allocation_with_offset& get_dest_allocation() const { return m_dest_alloc; }
	const box<3>& get_source_box() const { return m_source_box; }
	const box<3>& get_dest_box() const { return m_dest_box; }
	const region<3>& get_copy_region() const { return m_copy_region; }
	size_t get_element_size() const { return m_elem_size; }

  private:
	allocation_with_offset m_source_alloc;
	allocation_with_offset m_dest_alloc;
	box<3> m_source_box;
	box<3> m_dest_box;
	region<3> m_copy_region;
	size_t m_elem_size;
};

/// Description of an accessor or a reduction write in terms of the buffer's backing allocation at that location.
struct buffer_access_allocation {
	detail::allocation_id allocation_id = null_allocation_id;
	box<3> allocated_box_in_buffer;
	box<3> accessed_box_in_buffer;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	buffer_id oob_buffer_id;
	std::string oob_buffer_name;
#endif
};

/// Allocation-equivalent of a buffer_access_map. The runtime hydration and reduction mechanism are keyed by zero-based indices per instruction.
/// We use a std::vector directly instead of a small_vector because entries are > 100 bytes each.
using buffer_access_allocation_map = std::vector<buffer_access_allocation>;

/// Launches a SYCL device kernel on a single device. Bound accessors and reductions are hydrated through buffer_access_allocation_maps.
class device_kernel_instruction final : public matchbox::implement_acceptor<instruction, device_kernel_instruction> {
  public:
	explicit device_kernel_instruction(const instruction_id iid, const int priority, const device_id did, device_kernel_launcher launcher,
	    const box<3>& execution_range, buffer_access_allocation_map access_allocations,
	    buffer_access_allocation_map reduction_allocations CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(
	        , const task_type oob_task_type, const task_id oob_task_id, std::string oob_task_name))
	    : acceptor_base(iid, priority), m_device_id(did), m_launcher(std::move(launcher)), m_execution_range(execution_range),
	      m_access_allocations(std::move(access_allocations)), m_reduction_allocations(std::move(reduction_allocations))                                      //
	      CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, m_oob_task_type(oob_task_type), m_oob_task_id(oob_task_id), m_oob_task_name(std::move(oob_task_name))) //
	{}

	device_id get_device_id() const { return m_device_id; }
	const device_kernel_launcher& get_launcher() const { return m_launcher; }
	const box<3>& get_execution_range() const { return m_execution_range; }
	const buffer_access_allocation_map& get_access_allocations() const { return m_access_allocations; }
	const buffer_access_allocation_map& get_reduction_allocations() const { return m_reduction_allocations; }

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	task_type get_oob_task_type() const { return m_oob_task_type; }
	task_id get_oob_task_id() const { return m_oob_task_id; }
	const std::string& get_oob_task_name() const { return m_oob_task_name; }
#endif

  private:
	device_id m_device_id;
	device_kernel_launcher m_launcher;
	box<3> m_execution_range;
	buffer_access_allocation_map m_access_allocations;
	buffer_access_allocation_map m_reduction_allocations;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	task_type m_oob_task_type;
	task_id m_oob_task_id;
	std::string m_oob_task_name;
#endif
};

/// Launches a host task in a thread pool. Bound accessors are hydrated through a buffer_access_allocation_map.
class host_task_instruction final : public matchbox::implement_acceptor<instruction, host_task_instruction> {
  public:
	host_task_instruction(const instruction_id iid, const int priority, host_task_launcher launcher, const box<3>& execution_range,
	    const range<3>& global_range, buffer_access_allocation_map access_allocations,
	    const collective_group_id cgid CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(
	        , const task_type oob_task_type, const task_id oob_task_id, std::string oob_task_name))
	    : acceptor_base(iid, priority), m_launcher(std::move(launcher)), m_global_range(global_range), m_execution_range(execution_range),
	      m_access_allocations(std::move(access_allocations)), m_cgid(cgid)                                                                                   //
	      CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, m_oob_task_type(oob_task_type), m_oob_task_id(oob_task_id), m_oob_task_name(std::move(oob_task_name))) //
	{}

	const range<3>& get_global_range() const { return m_global_range; }
	const host_task_launcher& get_launcher() const { return m_launcher; }
	collective_group_id get_collective_group_id() const { return m_cgid; }
	const box<3>& get_execution_range() const { return m_execution_range; }
	const buffer_access_allocation_map& get_access_allocations() const { return m_access_allocations; }

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	task_type get_oob_task_type() const { return m_oob_task_type; }
	task_id get_oob_task_id() const { return m_oob_task_id; }
	const std::string& get_oob_task_name() const { return m_oob_task_name; }
#endif

  private:
	host_task_launcher m_launcher;
	range<3> m_global_range;
	box<3> m_execution_range;
	buffer_access_allocation_map m_access_allocations;
	collective_group_id m_cgid;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	task_type m_oob_task_type;
	task_id m_oob_task_id;
	std::string m_oob_task_name;
#endif
};

/// (MPI_) sends a subrange of an allocation to a single remote node. The send must have be announced by transmitting a pilot_message first.
class send_instruction final : public matchbox::implement_acceptor<instruction, send_instruction> {
  public:
	explicit send_instruction(const instruction_id iid, const int priority, const node_id to_nid, const message_id msgid, const allocation_id source_aid,
	    const range<3>& source_alloc_range, const id<3>& offset_in_alloc, const range<3>& send_range, const size_t elem_size)
	    : acceptor_base(iid, priority), m_to_nid(to_nid), m_message_id(msgid), m_source_aid(source_aid), m_source_range(source_alloc_range),
	      m_offset_in_source(offset_in_alloc), m_send_range(send_range), m_elem_size(elem_size) {}

	node_id get_dest_node_id() const { return m_to_nid; }
	message_id get_message_id() const { return m_message_id; }
	allocation_id get_source_allocation_id() const { return m_source_aid; }
	const range<3>& get_source_allocation_range() const { return m_source_range; }
	const id<3>& get_offset_in_source_allocation() const { return m_offset_in_source; }
	const range<3>& get_send_range() const { return m_send_range; }
	size_t get_element_size() const { return m_elem_size; }

  private:
	node_id m_to_nid;
	message_id m_message_id;
	allocation_id m_source_aid;
	range<3> m_source_range;
	id<3> m_offset_in_source;
	range<3> m_send_range;
	size_t m_elem_size;
};

/// Common implementation mixin for receive_instruction and split_receive_instruction.
class receive_instruction_impl {
  public:
	explicit receive_instruction_impl(
	    const transfer_id& trid, region<3> request, const allocation_id dest_allocation, const box<3>& allocated_box, const size_t elem_size)
	    : m_trid(trid), m_request(std::move(request)), m_dest_aid(dest_allocation), m_allocated_box(allocated_box), m_elem_size(elem_size) {}

	const transfer_id& get_transfer_id() const { return m_trid; }
	const region<3>& get_requested_region() const { return m_request; }
	allocation_id get_dest_allocation_id() const { return m_dest_aid; }
	const box<3>& get_allocated_box() const { return m_allocated_box; }
	size_t get_element_size() const { return m_elem_size; }

  private:
	transfer_id m_trid;
	region<3> m_request;
	allocation_id m_dest_aid;
	box<3> m_allocated_box;
	size_t m_elem_size;
};

/// Requests the receive of a single buffer region. The receive can be fulfilled by one or more non-overlapping incoming transfers, but the receive is only
/// complete once all of its constituent parts have arrived.
class receive_instruction final : public matchbox::implement_acceptor<instruction, receive_instruction>, public receive_instruction_impl {
  public:
	explicit receive_instruction(const instruction_id iid, const int priority, const transfer_id& trid, region<3> request, const allocation_id dest_allocation,
	    const box<3>& allocated_box, const size_t elem_size)
	    : acceptor_base(iid, priority), receive_instruction_impl(trid, std::move(request), dest_allocation, allocated_box, elem_size) {}
};

/// Informs the receive arbiter about the bounding box allocation for a series of incoming transfers. The boxes of remote send_instructions do not necessarily
/// coincide with await_receive_instructions - sends can fulfil subsets or supersets of receives, so the executor needs to be able to handle all send-patterns
/// that cover the region of the original await_push command. To make this happen, the instruction_graph_generator allocates the bounding box of each
/// 2/4/8-connected component of the await_push region and passes it on to the receive_arbiter through a split_receive_instruction.
class split_receive_instruction final : public matchbox::implement_acceptor<instruction, split_receive_instruction>, public receive_instruction_impl {
  public:
	explicit split_receive_instruction(const instruction_id iid, const int priority, const transfer_id& trid, region<3> request,
	    const allocation_id dest_allocation, const box<3>& allocated_box, const size_t elem_size)
	    : acceptor_base(iid, priority), receive_instruction_impl(trid, std::move(request), dest_allocation, allocated_box, elem_size) {}
};

/// Waits on the receive arbiter to complete part of the receive.
/// TODO for RDMA receives where different subranges of the await-push are required by different devices, this instruction could pass the device (staging)
/// buffer allocation that receive_arbiter would use in the "happy path" where there is a 1-to-1 correspondence between sends and receives.
class await_receive_instruction final : public matchbox::implement_acceptor<instruction, await_receive_instruction> {
  public:
	explicit await_receive_instruction(const instruction_id iid, const int priority, const transfer_id& trid, region<3> recv_region)
	    : acceptor_base(iid, priority), m_trid(trid), m_recv_region(std::move(recv_region)) {}

	transfer_id get_transfer_id() const { return m_trid; }
	const region<3>& get_received_region() const { return m_recv_region; }

  private:
	transfer_id m_trid;
	region<3> m_recv_region;
};

/// A special type of receive instruction used for global reductions: Instructs the receive arbiter to wait for incoming transfers of the same region from every
/// peer node and place the chunks side-by-side in a contiguous allocation. The offset in the output allocation are equal to the sender node id.
class gather_receive_instruction final : public matchbox::implement_acceptor<instruction, gather_receive_instruction> {
  public:
	explicit gather_receive_instruction(
	    const instruction_id iid, const int priority, const transfer_id& trid, const allocation_id dest_aid, const size_t node_chunk_size)
	    : acceptor_base(iid, priority), m_trid(trid), m_dest_aid(dest_aid), m_node_chunk_size(node_chunk_size) {}

	transfer_id get_transfer_id() const { return m_trid; }
	allocation_id get_dest_allocation_id() const { return m_dest_aid; }
	size_t get_node_chunk_size() const { return m_node_chunk_size; }

  private:
	transfer_id m_trid;
	allocation_id m_dest_aid;
	size_t m_node_chunk_size;
};

/// Fills an allocation with the identity value of a reduction. Used as a predecessor to gather_receive_instruction to ensure that peer nodes that do not
/// contribute a partial reduction result leave the identity value in their gather slot.
class fill_identity_instruction final : public matchbox::implement_acceptor<instruction, fill_identity_instruction> {
  public:
	explicit fill_identity_instruction(
	    const instruction_id iid, const int priority, const reduction_id rid, const allocation_id allocation_id, const size_t num_values)
	    : acceptor_base(iid, priority), m_rid(rid), m_aid(allocation_id), m_num_values(num_values) {}

	reduction_id get_reduction_id() const { return m_rid; }
	allocation_id get_allocation_id() const { return m_aid; }
	size_t get_num_values() const { return m_num_values; }

  private:
	reduction_id m_rid;
	allocation_id m_aid;
	size_t m_num_values;
};

/// Performs an out-of-memory reduction by reading from a gather allocation and writing to a single (buffer) allocation.
class reduce_instruction final : public matchbox::implement_acceptor<instruction, reduce_instruction> {
  public:
	explicit reduce_instruction(const instruction_id iid, const int priority, const reduction_id rid, const allocation_id source_allocation_id,
	    const size_t num_source_values, const allocation_id dest_allocation_id)
	    : acceptor_base(iid, priority), m_rid(rid), m_source_aid(source_allocation_id), m_num_source_values(num_source_values), m_dest_aid(dest_allocation_id) {
	}

	reduction_id get_reduction_id() const { return m_rid; }
	allocation_id get_source_allocation_id() const { return m_source_aid; }
	size_t get_num_source_values() const { return m_num_source_values; }
	allocation_id get_dest_allocation_id() const { return m_dest_aid; }

  private:
	reduction_id m_rid;
	allocation_id m_source_aid;
	size_t m_num_source_values;
	allocation_id m_dest_aid;
};

/// Fulfills a fence promise. Issued directly after a copy_instruction to user memory in case of a buffer fence.
class fence_instruction final : public matchbox::implement_acceptor<instruction, fence_instruction> {
  public:
	explicit fence_instruction(const instruction_id iid, const int priority, fence_promise* promise) : acceptor_base(iid, priority), m_promise(promise) {}

	fence_promise* get_promise() const { return m_promise; };

  private:
	fence_promise* m_promise;
};

/// Host object instances are owned by the instruction executor, so once the last reference to a host_object goes out of scope in userland, this instruction
/// deallocates that host object.
class destroy_host_object_instruction final : public matchbox::implement_acceptor<instruction, destroy_host_object_instruction> {
  public:
	explicit destroy_host_object_instruction(const instruction_id iid, const int priority, const host_object_id hoid)
	    : acceptor_base(iid, priority), m_hoid(hoid) {}

	host_object_id get_host_object_id() const { return m_hoid; }

  private:
	host_object_id m_hoid;
};

/// The executor will maintain runtime state about entities that the instruction graph generator references through ids. When the entity goes out of scope or is
/// otherwise not needed for further instructions, we can delete that state from the executor. We attach the list of these ids to horizon and epoch commands
/// because they conveniently depend on the entire previous execution front and are thus scheduled to run after the last instruction using each entity.
struct instruction_garbage {
	std::vector<reduction_id> reductions;
	std::vector<allocation_id> user_allocations;
};

/// Instruction-graph equivalent of a horizon task or command.
class horizon_instruction final : public matchbox::implement_acceptor<instruction, horizon_instruction> {
  public:
	explicit horizon_instruction(const instruction_id iid, const int priority, const task_id horizon_tid, instruction_garbage garbage)
	    : acceptor_base(iid, priority), m_horizon_tid(horizon_tid), m_garbage(std::move(garbage)) {}

	task_id get_horizon_task_id() const { return m_horizon_tid; }
	const instruction_garbage& get_garbage() const { return m_garbage; }

  private:
	task_id m_horizon_tid;
	instruction_garbage m_garbage;
};

/// Instruction-graph equivalent of an epoch task or command.
class epoch_instruction final : public matchbox::implement_acceptor<instruction, epoch_instruction> {
  public:
	explicit epoch_instruction(const instruction_id iid, const int priority, const task_id epoch_tid, const epoch_action action, instruction_garbage garbage)
	    : acceptor_base(iid, priority), m_epoch_tid(epoch_tid), m_epoch_action(action), m_garbage(std::move(garbage)) {}

	task_id get_epoch_task_id() const { return m_epoch_tid; }
	epoch_action get_epoch_action() const { return m_epoch_action; }
	const instruction_garbage& get_garbage() const { return m_garbage; }

  private:
	task_id m_epoch_tid;
	epoch_action m_epoch_action;
	instruction_garbage m_garbage;
};

/// The instruction graph (IDAG) provides a static, parallel schedule of operations executed on a single Celerity node. It manages allocation and transfer
/// operations between host- and all device memories installed in the node and issues kernel launches, inter-node data transfers and reductions. Unlike the
/// higher-level task and command graphs which track data dependencies in terms of buffers, it operates on the lower level of allocations, which (among other
/// uses) can back sub-regions of the (virtual) global buffer.
///
/// The `instruction_graph` struct keeps ownership of all instructions that have not yet been pruned by epoch or horizon application.
class instruction_graph {
  public:
	// Call this before pushing horizon or epoch instruction in order to be able to call erase_before_epoch on the same task id later.
	void begin_epoch(const task_id tid) {
		assert(m_epochs.empty() || (m_epochs.back().epoch_tid < tid && !m_epochs.back().instructions.empty()));
		m_epochs.push_back({tid, {}});
	}

	// Add an instruction to the current epoch. Its instruction id must be higher than any instruction inserted into the graph before.
	void push_instruction(std::unique_ptr<instruction> instr) {
		assert(!m_epochs.empty());
		auto& instructions = m_epochs.back().instructions;
		assert(instructions.empty() || instruction_id_less{}(instructions.back().get(), instr.get()));
		instructions.push_back(std::move(instr));
	}

	// Free all instructions that were pushed before begin_epoch(tid) was called.
	void prune_before_epoch(const task_id tid) {
		const auto first_retained =
		    std::partition_point(m_epochs.begin(), m_epochs.end(), [=](const instruction_epoch& epoch) { return epoch.epoch_tid < tid; });
		assert(first_retained != m_epochs.end() && first_retained->epoch_tid == tid);
		m_epochs.erase(m_epochs.begin(), first_retained);
	}

	// The total number of instructions currently owned and not yet pruned, across all epochs.
	size_t get_live_instruction_count() const {
		return std::accumulate(
		    m_epochs.begin(), m_epochs.end(), size_t{0}, [](const size_t acc, const instruction_epoch& epoch) { return acc + epoch.instructions.size(); });
	}

  private:
	struct instruction_epoch {
		task_id epoch_tid;
		std::vector<std::unique_ptr<instruction>> instructions; // instruction pointers are stable, so it is safe to hand them to another thread
	};

	std::vector<instruction_epoch> m_epochs;
};

} // namespace celerity::detail
