#include "types.h"

#include <memory>
#include <optional>

namespace celerity::detail::out_of_order_engine_detail {
struct engine_impl;
}

namespace celerity::detail {

class instruction;
struct system_info;

/// State machine controlling when and in what manner instructions are assigned to execution resources in adherence to the dependency graph.
///
/// Based on their type, instructions can either be assigned to begin executing immediately as soon as all their predecessors are complete, or they can be
/// enqueued eagerly in one of the backend's thread queues or in-order device queues to hide launch latencies where possible.
class out_of_order_engine {
  public:
	/// Identifies a category of execution resources that may be distinguished further by a device- or lane id.
	enum class target {
		/// Execution can begin immediately, no queueing takes place in the backend and no lane is assigned. Used for low-overhead instructions that do not
		/// profit from additional concurrency such as horizons, as well as for instructions where asynchronicity is managed outside the backend (p2p transfers
		/// through communicator and receive_arbiter).
		immediate,

		/// The instruction shall be inserted to the backend's (singular) host allocation queue. No lane is assigned. Since at least CUDA serializes the slow
		/// alloc / free operations anyway to update page tables globally, the added concurrency from multiple thread queues would not increase throughput. The
		/// separation between alloc_queue and host_queues further enforces a host round-trip between every alloc_instruction and its first successor, which we
		/// use to inform the executor of the newly allocated pointer for the purpose of accessor hydration.
		alloc_queue,

		/// The instruction shall be submitted to a backend thread queue identified by the lane id. Used for host tasks and host-to-host copies.
		host_queue,

		/// The instruction shall be submitted to a backend in-order device queue identified by the lane id. Used for device kernels and host-to-device /
		/// device-to-device / device-to-host copies.
		device_queue,
	};

	/// A lane identifies a thread queue for target::host_task and an in-order device queue with target::device_queue.
	using lane_id = size_t;

	/// Directions on how a single (ready) instruction is to be dispatched by the executor.
	struct assignment {
		const detail::instruction* instruction = nullptr;
		out_of_order_engine::target target = out_of_order_engine::target::immediate;
		std::optional<device_id> device; ///< Identifies the device to submit to (if target == device_queue) or to allocate on (if target == alloc_queue).
		std::optional<lane_id> lane; ///< Identifies the thread queue (target == host_queue) or the in-order queue for the given device (target == alloc_queue).

		assignment(const detail::instruction* instruction, const out_of_order_engine::target target, const std::optional<device_id> device = std::nullopt,
		    const std::optional<lane_id> lane = std::nullopt)
		    : instruction(instruction), target(target), device(device), lane(lane) {}
	};

	/// Constructor requires a `system_info` to enumerate devices and perform `memory_id -> device_id` mapping.
	explicit out_of_order_engine(const system_info& system);

	out_of_order_engine(const out_of_order_engine&) = delete;
	out_of_order_engine(out_of_order_engine&&) = default;
	out_of_order_engine& operator=(const out_of_order_engine&) = delete;
	out_of_order_engine& operator=(out_of_order_engine&&) = default;

	~out_of_order_engine();

	/// True when all submitted instructions have completed.
	bool is_idle() const;

	/// Returns the number of instructions currently awaiting normal or eager assignment (diagnostic use only).
	size_t get_assignment_queue_length() const;

	/// Begin tracking an instruction so that it is eventually returned through `assign_one`. Must be called in topological order of dependencies, i.e. no
	/// instruction must be submitted before one of its predecessors in the graph.
	void submit(const instruction* instr);

	/// Produce an assignment for one instruction for which either all predecessors have completed, or for which all incomplete predecessors can be implicitly
	/// fulfilled by submitting to the same backend queue. If multiple instructions are eligible, assign the one with the highest priority.
	[[nodiscard]] std::optional<assignment> assign_one();

	/// Call once an instruction that was previously returned from `assign_one` has completed synchronously or asynchronously. For simplicity it is permitted
	/// to mark assigned instructions as completed in any order, even if that would violate internal dependency order.
	void complete_assigned(const instruction_id iid);

  private:
	std::unique_ptr<out_of_order_engine_detail::engine_impl> m_impl;
};

} // namespace celerity::detail
