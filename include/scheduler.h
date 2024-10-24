#pragma once

#include "command_graph_generator.h"
#include "instruction_graph_generator.h"
#include "ranges.h"
#include "types.h"


namespace celerity::detail::scheduler_detail {
struct scheduler_impl;
}

namespace celerity::detail {

class command_recorder;
class instruction_recorder;
class task;

// Abstract base class to allow different threading implementation in tests
class scheduler {
  private:
	friend struct scheduler_testspy;
	friend class test_benchmark_scheduler;

  public:
	using delegate = instruction_graph_generator::delegate;

	struct policy_set {
		detail::command_graph_generator::policy_set command_graph_generator;
		detail::instruction_graph_generator::policy_set instruction_graph_generator;
	};

	scheduler(size_t num_nodes, node_id local_node_id, const system_info& system_info, const task_manager& tm, delegate* delegate, command_recorder* crec,
	    instruction_recorder* irec, const policy_set& policy = {});

	scheduler(const scheduler&) = delete;
	scheduler(scheduler&&) = default;
	scheduler& operator=(const scheduler&) = delete;
	scheduler& operator=(scheduler&&) = default;

	~scheduler();

	/**
	 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
	 */
	void notify_task_created(const task* const tsk);

	void notify_buffer_created(
	    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_allocation_id);

	void notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name);

	void notify_buffer_destroyed(const buffer_id bid);

	void notify_host_object_created(const host_object_id hoid, const bool owns_instance);

	void notify_host_object_destroyed(const host_object_id hoid);

	void notify_epoch_reached(const task_id tid);

	void set_lookahead(const experimental::lookahead lookahead);

	void flush_commands();

  private:
	struct test_start_idle_tag {};

	std::unique_ptr<scheduler_detail::scheduler_impl> m_impl;

	scheduler(test_start_idle_tag, size_t num_nodes, node_id local_node_id, const system_info& system_info, const task_manager& tm, delegate* delegate,
	    command_recorder* crec, instruction_recorder* irec, const policy_set& policy = {});
	void test_invoke_thread_main();
	void test_inspect(std::function<void()> inspector);
	size_t test_get_live_command_count();
	size_t test_get_live_instruction_count();
};

} // namespace celerity::detail
