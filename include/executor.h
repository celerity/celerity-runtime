#pragma once

#include "types.h"

#include <memory>

namespace celerity::detail {

struct host_object_instance;
class instruction;
struct outbound_pilot;
class reducer;

/// An executor processes receives and processes an instruction stream to process in a background thread.
class executor {
  public:
	/// Implement this as the owner of an executor to receive callbacks on completed horizons and epochs.
	class delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		/// Called from the executor thread as soon as a horizon_instruction has finished executing.
		virtual void horizon_reached(task_id tid) = 0;

		/// Called from the executor thread as soon as an epoch_instruction has finished executing.
		virtual void epoch_reached(task_id tid) = 0;
	};

	executor() = default;
	executor(const executor&) = delete;
	executor(executor&&) = delete;
	executor& operator=(const executor&) = delete;
	executor& operator=(executor&&) = delete;

	/// Waits until an epoch with `epoch_action::shutdown` has executed and the executor thread has exited.
	virtual ~executor() = default;

	/// Informs the executor about the runtime address of an allocation on user_memory_id. Must be called before submitting any instruction referring to the
	/// allocation id in question. User allocations are later removed from executor tracking as they appear in an instruction_garbage list attached to a horizon
	/// or epoch instruction.
	virtual void track_user_allocation(allocation_id aid, void* ptr) = 0;

	/// Transfer ownership of a host object instance to the executor. The executor will later destroy this instance when executing a matching
	/// destroy_host_object_instruction.
	virtual void track_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) = 0;

	/// Informs the executor about the runtime behavior of a reduction. Will be used by any fill_identity_instruction and reduce_instruction later submitted on
	/// the same reduction_id. Reducer instances are removed from executor tracking and destroyed when they later appear in an instruction_garbage list attached
	/// to a horizon or epoch instruction.
	virtual void track_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) = 0;

	/// Submits a list of instructions to execute once their dependencies have been fulfilled, and a list of outbound pilots to be transmitted to their
	/// recipients as soon as possible. Instructions must be in topological order of dependencies, as must be the concatenation of all vectors passed to
	/// subsequent invocations of this function.
	virtual void submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) = 0;
};

} // namespace celerity::detail
