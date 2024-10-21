#pragma once

#include "command_graph.h"
#include "command_graph_generator.h"
#include "instruction_graph_generator.h"
#include "ranges.h"
#include "types.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>


namespace celerity::detail::scheduler_detail {

/// executed inside scheduler thread, making it safe to access scheduler members
using test_inspector = std::function<void(const command_graph&, const instruction_graph&)>;

struct scheduler_impl;

} // namespace celerity::detail::scheduler_detail

namespace celerity::detail {

class command_recorder;
class instruction_recorder;
class task;

// Abstract base class to allow different threading implementation in tests
class scheduler {
  private:
	friend struct scheduler_testspy;

  public:
	using delegate = instruction_graph_generator::delegate;

	struct policy_set {
		detail::command_graph_generator::policy_set command_graph_generator;
		detail::instruction_graph_generator::policy_set instruction_graph_generator;
	};

	scheduler(size_t num_nodes, node_id local_node_id, const system_info& system_info, scheduler::delegate* delegate, command_recorder* crec,
	    instruction_recorder* irec, const policy_set& policy = {});

	scheduler(const scheduler&) = delete;
	scheduler(scheduler&&) = default;
	scheduler& operator=(const scheduler&) = delete;
	scheduler& operator=(scheduler&&) = default;

	~scheduler();

	/**
	 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
	 */
	void notify_task_created(const task* tsk);

	void notify_buffer_created(buffer_id bid, const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_allocation_id);

	void notify_buffer_debug_name_changed(buffer_id bid, const std::string& name);

	void notify_buffer_destroyed(buffer_id bid);

	void notify_host_object_created(host_object_id hoid, bool owns_instance);

	void notify_host_object_destroyed(host_object_id hoid);

	void notify_epoch_reached(task_id tid);

	void set_lookahead(experimental::lookahead lookahead);

	void flush_commands();

  private:
	struct test_threadless_tag {};

	std::unique_ptr<scheduler_detail::scheduler_impl> m_impl;

	// used in scheduler_testspy
	scheduler(test_threadless_tag, size_t num_nodes, node_id local_node_id, const system_info& system_info, scheduler::delegate* delegate,
	    command_recorder* crec, instruction_recorder* irec, const policy_set& policy = {});
	void test_scheduling_loop();
	void test_inspect(scheduler_detail::test_inspector inspector);
};

} // namespace celerity::detail
