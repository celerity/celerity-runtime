#pragma once

#include "double_buffered_queue.h"
#include "executor.h"

#include <memory>
#include <optional>
#include <thread>
#include <variant>

namespace celerity::detail::live_executor_detail {

struct instruction_pilot_batch {
	std::vector<const instruction*> instructions;
	std::vector<outbound_pilot> pilots;
};
struct user_allocation_transfer {
	allocation_id aid;
	void* ptr = nullptr;
};
struct host_object_transfer {
	host_object_id hoid = 0;
	std::unique_ptr<host_object_instance> instance;
};
struct reducer_transfer {
	reduction_id rid = 0;
	std::unique_ptr<reducer> reduction;
};
using submission = std::variant<instruction_pilot_batch, user_allocation_transfer, host_object_transfer, reducer_transfer>;

} // namespace celerity::detail::live_executor_detail

namespace celerity::detail {

class communicator;
struct system_info;
class backend;

/// Executor implementation for a normal (non-dry) run of a Celerity application. Internal instruction dependencies are resolved by means of an
/// out_of_order_engine and receive_arbiter, and the resulting operations dispatched to a `backend` and `communicator` implementation.
class live_executor final : public executor {
  public:
	struct policy_set {
		std::optional<std::chrono::milliseconds> progress_warning_timeout = std::chrono::seconds(20);
	};

	/// Operations are dispatched to `backend` and `root_comm` or one of its clones.
	/// `dlg` (optional) receives notifications about reached horizons and epochs from the executor thread.
	explicit live_executor(
	    std::unique_ptr<backend> backend, std::unique_ptr<communicator> root_comm, delegate* dlg, const policy_set& policy = default_policy_set());

	live_executor(const live_executor&) = delete;
	live_executor(live_executor&&) = delete;
	live_executor& operator=(const live_executor&) = delete;
	live_executor& operator=(live_executor&&) = delete;

	~live_executor() override;

	void track_user_allocation(allocation_id aid, void* ptr) override;
	void track_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) override;
	void track_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) override;

	void submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) override;

  private:
	friend struct executor_testspy;

	std::unique_ptr<communicator> m_root_comm; // created and destroyed outside of executor thread
	double_buffered_queue<live_executor_detail::submission> m_submission_queue;
	std::thread m_thread;

	void thread_main(std::unique_ptr<backend> backend, delegate* dlg, const policy_set& policy);

	/// Default-constructs a `policy_set` - this must be a function because we can't use the implicit default constructor of `policy_set`, which has member
	/// initializers, within its surrounding class (Clang diagnostic).
	constexpr static policy_set default_policy_set() { return {}; }
};

} // namespace celerity::detail
