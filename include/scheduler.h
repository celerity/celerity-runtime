#pragma once

#include "command_graph_generator.h"
#include "instruction_graph_generator.h"
#include "ranges.h"
#include "types.h"

#include <cstddef>
#include <memory>
#include <string>


namespace celerity::detail::scheduler_detail {

struct scheduler_impl;

} // namespace celerity::detail::scheduler_detail

namespace celerity::detail {

class command_recorder;
class instruction_recorder;
class loop_template;
class task;

// Abstract base class to allow different threading implementation in tests
class scheduler {
  private:
	friend struct scheduler_testspy;

  public:
	class delegate : public instruction_graph_generator::delegate {
	  protected:
		delegate() = default;
		delegate(const delegate&) = default;
		delegate(delegate&&) = default;
		delegate& operator=(const delegate&) = default;
		delegate& operator=(delegate&&) = default;
		~delegate() = default; // do not allow destruction through base pointer

	  public:
		virtual void on_scheduler_idle() = 0;
		virtual void on_scheduler_busy() = 0;
	};

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

	void enable_loop_template(loop_template& templ);

	void complete_loop_iteration();

	void finalize_loop_template(loop_template& templ); // NOCOMMIT Take unique_ptr instead

	void leak_memory();

  private:
	scheduler() = default; // used by scheduler_testspy

	std::unique_ptr<scheduler_detail::scheduler_impl> m_impl;
};

} // namespace celerity::detail
