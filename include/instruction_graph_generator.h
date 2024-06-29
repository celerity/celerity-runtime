#pragma once

#include "grid.h"
#include "ranges.h"
#include "types.h"

#include <limits>


namespace celerity::detail {

class abstract_command;
class instruction;
class instruction_graph;
class instruction_recorder;
struct outbound_pilot;
struct system_info;
class task_manager;

} // namespace celerity::detail

namespace celerity::detail::instruction_graph_generator_detail {

/// MPI limits the extent of send/receive operations and strides (if any) to INT_MAX elements in each dimension. Since we do not require this value anywhere
/// else we define it in instruction_graph_generator_detail, even though it does not logically originate here.
constexpr size_t communicator_max_coordinate = static_cast<size_t>(std::numeric_limits<int>::max());

class generator_impl;

} // namespace celerity::detail::instruction_graph_generator_detail

namespace celerity::detail {

/// Tracks the node-local state of buffers and host objects, receives commands at node granularity and emits instructions to allocate memory, establish
/// coherence between devices and distribute work among the devices on the local system.
class instruction_graph_generator {
  public:
	/// Implement this as the owner of instruction_graph_generator to receive callbacks on generated instructions and pilot messages.
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		/// Called whenever new instructions have been generated and inserted into the instruction graph, and / or new pilot messages have been generated that
		/// must be transmitted to peer nodes before they can accept data transmitted through `send_instruction`s originating from the local node.
		///
		/// The vector of instructions is in topological order of dependencies, and so is the concatenation of all vectors that are passed through this
		/// function. Topological order here means that sequential execution in that order would fulfill all internal dependencies. The instruction graph
		/// generator guarantees that instruction pointers are stable and the pointed-to instructions are both immutable and safe to read from other threads.
		///
		/// This is exposed as a single function on vectors to minimize lock contention in a threaded delegate implementations.
		virtual void flush(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) = 0;
	};

	struct policy_set {
		/// Reported when the user requests a `hint::oversubscribe`, but oversubscription is unsafe because the task has side effects, participates in a
		/// collective-group operation, performs a reduction (current limitation of our implementation) or its iteration space simply cannot be split.
		error_policy unsafe_oversubscription_error = error_policy::panic;

		/// Reported when a task attempts to read data that has neither been await-pushed nor generated on the local node. This error is usually caught on a
		/// higher level by the task and command graph generator.
		error_policy uninitialized_read_error = error_policy::panic;

		/// Reported when two or more chunks of a device kernel or host task attempt to write the same buffer elements. instruction_graph_generator will produce
		/// an executable graph even when this error is being ignored, but will cause race conditions between instructions on the executor level.
		error_policy overlapping_write_error = error_policy::panic;
	};

	/// Instruction graph generation requires information about the target system. `num_nodes` and `local_nid` affect the generation of communication
	/// instructions and reductions, and `system` is used to determine work assignment, memory allocation and data migration between memories.
	///
	/// Generated instructions are inserted into (and subsequently owned by) the provided `idag`, and if `dlg` is provided, it is notified about any newly
	/// created instructions and outbound pilots.
	///
	/// If and only if a `recorder` is present, the generator will collect debug information about each generated instruction and pass it to the recorder. Set
	/// this to `nullptr` in production code in order to avoid a performance penalty.
	///
	/// Specify a non-default `policy` to influence what user-errors are detected at runtime and how they are reported. The default is is to throw exceptions
	/// which catch errors early in tests, but users of this class will want to ease these settings. Any policy set to a value other than
	/// `error_policy::ignore` will have a performance penalty.
	explicit instruction_graph_generator(const task_manager& tm, size_t num_nodes, node_id local_nid, const system_info& system, instruction_graph& idag,
	    delegate* dlg = nullptr, instruction_recorder* recorder = nullptr, const policy_set& policy = default_policy_set());

	instruction_graph_generator(const instruction_graph_generator&) = delete;
	instruction_graph_generator(instruction_graph_generator&&) = default;
	instruction_graph_generator& operator=(const instruction_graph_generator&) = delete;
	instruction_graph_generator& operator=(instruction_graph_generator&&) = default;

	~instruction_graph_generator();

	/// Begin tracking local data distribution and dependencies on the buffer with id `bid`. Allocations are made lazily on first access.
	///
	/// Passing `user_allocation_id != null_allocation_id` means that the buffer is considered coherent in user memory and data will be lazily copied from that
	/// allocation when read from host tasks or device kernels.
	void notify_buffer_created(buffer_id bid, const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_allocation_id);

	/// Changing an existing buffer's debug name causes all future instructions to refer to that buffer by the new name (if a recorder is present).
	void notify_buffer_debug_name_changed(buffer_id bid, const std::string& name);

	/// End tracking buffer with the id `bid`. Emits `free_instructions` for all current allocations of that buffer.
	void notify_buffer_destroyed(buffer_id bid);

	/// Begin tracking dependencies on the host object with id `hoid`. If `owns_instance` is true, a `destroy_host_object_instruction` will be emitted when
	/// `destroy_host_object` is subsequently called.
	void notify_host_object_created(host_object_id hoid, bool owns_instance);

	/// End tracking the host object with id `hoid`. Emits `destroy_host_object_instruction` if `create_host_object` was called with `owns_instance == true`.
	void notify_host_object_destroyed(host_object_id hoid);

	/// Compiles a command-graph node into a set of instructions, which are inserted into the shared instruction graph, and updates tracking structures.
	void compile(const abstract_command& cmd);

  private:
	/// Default-constructs a `policy_set` - this must be a function because we can't use the implicit default constructor of `policy_set`, which has member
	/// initializers, within its surrounding class (Clang diagnostic).
	constexpr static policy_set default_policy_set() { return {}; }

	std::unique_ptr<instruction_graph_generator_detail::generator_impl> m_impl;
};

} // namespace celerity::detail


// forward declaration for tests

namespace celerity::detail::instruction_graph_generator_detail {

template <int Dims>
bool boxes_edge_connected(const box<Dims>& box1, const box<Dims>& box2);

template <int Dims>
box_vector<Dims> connected_subregion_bounding_boxes(const region<Dims>& region);

box_vector<3> split_into_communicator_compatible_boxes(const range<3>& buffer_range, const box<3>& send_box);

template <int Dims>
void symmetrically_split_overlapping_regions(std::vector<region<Dims>>& regions);

} // namespace celerity::detail::instruction_graph_generator_detail
