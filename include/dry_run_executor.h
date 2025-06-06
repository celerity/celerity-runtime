#pragma once

#include "double_buffered_queue.h"
#include "executor.h"
#include "types.h"

#include <memory>
#include <thread>
#include <utility>
#include <variant>
#include <vector>


namespace celerity::detail {

/// Executor implementation selected when Celerity performs a dry run, that is, graph generation for debugging purposes without actually allocating memory,
/// launching kernels or issuing data transfers.
///
/// `dry_run_executor` still executes horizon-, epoch- and fence instructions to guarantee forward progress in the user application.
class dry_run_executor final : public executor {
  public:
	/// `dlg` (optional) receives notifications about reached horizons and epochs from the executor thread.
	explicit dry_run_executor(executor::delegate* dlg);

	dry_run_executor(const dry_run_executor&) = delete;
	dry_run_executor(dry_run_executor&&) = delete;
	dry_run_executor& operator=(const dry_run_executor&) = delete;
	dry_run_executor& operator=(dry_run_executor&&) = delete;

	~dry_run_executor() override;

	void track_user_allocation(allocation_id aid, void* ptr) override;
	void track_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) override;
	void track_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) override;

	void submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) override;

	void notify_scheduler_idle(bool is_idle) override;

	std::chrono::nanoseconds get_starvation_time() const override;
	std::chrono::nanoseconds get_active_time() const override;

  private:
	using host_object_transfer = std::pair<host_object_id, std::unique_ptr<host_object_instance>>;
	using submission = std::variant<std::vector<const instruction*>, host_object_transfer>;

	double_buffered_queue<submission> m_submission_queue;
	std::thread m_thread;

	void thread_main(executor::delegate* dlg);
};

} // namespace celerity::detail
