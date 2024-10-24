#pragma once
#include <thread>
#include <variant>

#include "command_graph_generator.h"
#include "double_buffered_queue.h"
#include "instruction_graph_generator.h"
#include "ranges.h"
#include "types.h"


namespace celerity::detail {

class command_graph;
class command_recorder;
class instruction;
class instruction_graph;
class instruction_recorder;
struct outbound_pilot;
class task;

// Abstract base class to allow different threading implementation in tests
class scheduler {
  protected:
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
	scheduler(scheduler&&) = delete;
	scheduler& operator=(const scheduler&) = delete;
	scheduler& operator=(scheduler&&) = delete;

	~scheduler();

	/**
	 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
	 */
	void notify_task_created(const task* const tsk) { m_task_queue.push(event_task_available{tsk}); }

	void notify_buffer_created(
	    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_allocation_id) {
		m_task_queue.push(event_buffer_created{bid, range, elem_size, elem_align, user_allocation_id});
	}

	void notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name) { m_task_queue.push(event_buffer_debug_name_changed{bid, name}); }

	void notify_buffer_destroyed(const buffer_id bid) { m_task_queue.push(event_buffer_destroyed{bid}); }

	void notify_host_object_created(const host_object_id hoid, const bool owns_instance) { m_task_queue.push(event_host_object_created{hoid, owns_instance}); }

	void notify_host_object_destroyed(const host_object_id hoid) { m_task_queue.push(event_host_object_destroyed{hoid}); }

	void notify_epoch_reached(const task_id tid) { m_task_queue.push(event_epoch_reached{tid}); }

	void set_lookahead(const experimental::lookahead lookahead) { m_task_queue.push(event_set_lookahead{lookahead}); }

  private:
	struct event_task_available {
		const task* tsk;
	};
	struct event_command_available {
		const abstract_command* cmd;
	};
	struct event_buffer_created {
		buffer_id bid;
		celerity::range<3> range;
		size_t elem_size;
		size_t elem_align;
		allocation_id user_allocation_id;
	};
	struct event_buffer_debug_name_changed {
		buffer_id bid;
		std::string debug_name;
	};
	struct event_buffer_destroyed {
		buffer_id bid;
	};
	struct event_host_object_created {
		host_object_id hoid;
		bool owns_instance;
	};
	struct event_host_object_destroyed {
		host_object_id hoid;
	};
	struct event_epoch_reached {
		task_id tid;
	};
	struct event_set_lookahead {
		experimental::lookahead lookahead;
	};
	struct event_test_inspect {        // only used by scheduler_testspy
		std::function<void()> inspect; // executed inside scheduler thread, making it safe to access scheduler members
	};

	struct task_queue {
		using event = std::variant<event_task_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed,
		    event_host_object_created, event_host_object_destroyed, event_epoch_reached, event_set_lookahead, event_test_inspect>;

		double_buffered_queue<event> global_queue;
		std::deque<event> local_queue;

		bool empty() const { return !global_queue.nonempty() && local_queue.empty(); }
		void push(event&& evt);
		event pop();
	};

	struct command_queue {
		using event = std::variant<event_command_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed,
		    event_host_object_created, event_host_object_destroyed, event_set_lookahead>;

		std::deque<event> queue;
		int num_queued_fences_and_epochs = 0;
		int num_queued_horizons = 0;

		bool empty() const { return queue.empty(); }
		bool next_is_command() const { return !queue.empty() && std::holds_alternative<event_command_available>(queue.front()); }
		void push(event&& evt);
		event pop();
	};

	struct start_idle_tag {
	} inline static constexpr start_idle;

	std::unique_ptr<command_graph> m_cdag;
	command_recorder* m_crec;
	std::unique_ptr<command_graph_generator> m_cggen;
	experimental::lookahead m_lookahead = experimental::lookahead::automatic;
	std::unique_ptr<instruction_graph> m_idag;
	instruction_recorder* m_irec;
	std::unique_ptr<instruction_graph_generator> m_iggen;

	task_queue m_task_queue;
	command_queue m_command_queue;

	std::optional<task_id> m_shutdown_epoch_created = std::nullopt;
	bool m_shutdown_epoch_reached = false;

	std::thread m_thread;

	scheduler(start_idle_tag, size_t num_nodes, node_id local_node_id, const system_info& system_info, const task_manager& tm, delegate* delegate,
	    command_recorder* crec, instruction_recorder* irec, const policy_set& policy = {});

	std::vector<const abstract_command*> build_task(const task& tsk);
	void compile_command(const abstract_command& cmd);

	void process_task_queue_event(const task_queue::event& evt);
	void process_command_queue_event(const command_queue::event& evt);
	bool should_dequeue_more_command_events() const;

	void test_inspect(std::function<void()> inspector) { m_task_queue.push(event_test_inspect{std::move(inspector)}); }
	size_t test_get_live_command_count() { return m_cdag->command_count(); }
	size_t test_get_live_instruction_count() { return m_idag->get_live_instruction_count(); }

	void thread_main();
};

} // namespace celerity::detail
