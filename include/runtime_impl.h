#pragma once

#include "runtime.h"

#include "affinity.h"
#include "cgf.h"
#include "device.h"
#include "executor.h"
#include "host_object.h"
#include "instruction_graph_generator.h"
#include "reduction.h"
#include "scheduler.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

#include <atomic>
#include <string>


namespace celerity::detail {

class config;

class runtime_impl final : public runtime, private task_manager::delegate, private scheduler::delegate, private executor::delegate {
  public:
	runtime_impl(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector);

	runtime_impl(const runtime_impl&) = delete;
	runtime_impl(runtime_impl&&) = delete;
	runtime_impl& operator=(const runtime_impl&) = delete;
	runtime_impl& operator=(runtime_impl&&) = delete;

	~runtime_impl() override;

	task_id submit(raw_command_group&& cg) override;

	task_id fence(buffer_access access, std::unique_ptr<task_promise> fence_promise) override;

	task_id fence(host_object_effect effect, std::unique_ptr<task_promise> fence_promise) override;

	task_id sync(detail::epoch_action action) override;

	void create_queue() override;

	void destroy_queue() override;

	allocation_id create_user_allocation(void* ptr) override;

	buffer_id create_buffer(const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid) override;

	void set_buffer_debug_name(buffer_id bid, const std::string& debug_name) override;

	void destroy_buffer(buffer_id bid) override;

	host_object_id create_host_object(std::unique_ptr<host_object_instance> instance /* optional */) override;

	void destroy_host_object(host_object_id hoid) override;

	reduction_id create_reduction(std::unique_ptr<reducer> reducer) override;

	bool is_dry_run() const override;

	void set_scheduler_lookahead(experimental::lookahead lookahead) override;

	void flush_scheduler() override;

  private:
	friend struct runtime_testspy;

	// `runtime` is not thread safe except for its delegate implementations, so we store the id of the thread where it was instantiated (the application
	// thread) in order to throw if the user attempts to issue a runtime operation from any other thread. One case where this may happen unintentionally
	// is capturing a buffer into a host-task by value, where this capture is the last reference to the buffer: The runtime would attempt to destroy itself
	// from a thread that it also needs to await, which would at least cause a deadlock. This variable is immutable, so reading it from a different thread
	// for the purpose of the check is safe.
	std::thread::id m_application_thread;

	std::unique_ptr<config> m_cfg;
	size_t m_num_nodes = 0;
	node_id m_local_nid = 0;
	size_t m_num_local_devices = 0;

	// track all instances of celerity::queue, celerity::buffer and celerity::host_object to sanity-check runtime destruction
	size_t m_num_live_queues = 0;
	std::unordered_set<buffer_id> m_live_buffers;
	std::unordered_set<host_object_id> m_live_host_objects;

	buffer_id m_next_buffer_id = 0;
	raw_allocation_id m_next_user_allocation_id = 1;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = no_reduction_id + 1;

	task_graph m_tdag;
	std::unique_ptr<task_manager> m_task_mngr;
	std::unique_ptr<scheduler> m_schdlr;
	std::unique_ptr<executor> m_exec;

	std::optional<task_id> m_latest_horizon_reached; // only accessed by executor thread
	std::atomic<size_t> m_latest_epoch_reached;      // task_id, but cast to size_t to work with std::atomic
	task_id m_last_epoch_pruned_before = 0;

	std::unique_ptr<detail::task_recorder> m_task_recorder;               // accessed by task manager (application thread)
	std::unique_ptr<detail::command_recorder> m_command_recorder;         // accessed only by scheduler thread (until shutdown)
	std::unique_ptr<detail::instruction_recorder> m_instruction_recorder; // accessed only by scheduler thread (until shutdown)

	std::unique_ptr<detail::thread_pinning::thread_pinner> m_thread_pinner; // thread safe, manages lifetime of thread pinning machinery

	/// Panic when not called from m_application_thread (see that variable for more info on the matter). Since there are thread-safe and non thread-safe
	/// member functions, we call this check at the beginning of all the non-safe ones.
	void require_call_from_application_thread() const;

	void maybe_prune_task_graph();

	// task_manager::delegate
	void task_created(const task* tsk) override;

	// scheduler::delegate
	void flush(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilot) override;

	// executor::delegate
	void horizon_reached(task_id horizon_tid) override;
	void epoch_reached(task_id epoch_tid) override;

	/// True when no buffers, host objects or queues are live that keep the runtime alive.
	bool is_unreferenced() const;
};

} // namespace celerity::detail
