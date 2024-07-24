#pragma once

#include <thread>
#include <variant>

#include "distributed_graph_generator.h"
#include "double_buffered_queue.h"
#include "instruction_graph_generator.h"
#include "ranges.h"
#include "types.h"


namespace celerity {
namespace detail {

	class command_graph;
	class command_recorder;
	class instruction;
	class instruction_graph;
	class instruction_recorder;
	struct outbound_pilot;
	class task;

	// Abstract base class to allow different threading implementation in tests
	class abstract_scheduler {
	  protected:
		friend struct scheduler_testspy;

	  public:
		using delegate = instruction_graph_generator::delegate;

		struct policy_set {
			detail::distributed_graph_generator::policy_set command_graph_generator;
			detail::instruction_graph_generator::policy_set instruction_graph_generator;
		};

		abstract_scheduler(size_t num_nodes, node_id local_node_id, const system_info& system_info, const task_manager& tm, delegate* delegate,
		    command_recorder* crec, instruction_recorder* irec, const policy_set& policy = {});

		abstract_scheduler(const abstract_scheduler&) = delete;
		abstract_scheduler(abstract_scheduler&&) = delete;
		abstract_scheduler& operator=(const abstract_scheduler&) = delete;
		abstract_scheduler& operator=(abstract_scheduler&&) = delete;

		virtual ~abstract_scheduler();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(const task* const tsk) { notify(event_task_available{tsk}); }

		void notify_buffer_created(
		    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_allocation_id) {
			notify(event_buffer_created{bid, range, elem_size, elem_align, user_allocation_id});
		}

		void notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name) { notify(event_buffer_debug_name_changed{bid, name}); }

		void notify_buffer_destroyed(const buffer_id bid) { notify(event_buffer_destroyed{bid}); }

		void notify_host_object_created(const host_object_id hoid, const bool owns_instance) { notify(event_host_object_created{hoid, owns_instance}); }

		void notify_host_object_destroyed(const host_object_id hoid) { notify(event_host_object_destroyed{hoid}); }

		void notify_epoch_reached(const task_id tid) { notify(event_epoch_reached{tid}); }

	  protected:
		/**
		 * This is called by the worker thread.
		 */
		void schedule();

	  private:
		struct event_task_available {
			const task* tsk;
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
		struct event_test_inspect {        // only used by scheduler_testspy
			std::function<void()> inspect; // executed inside scheduler thread, making it safe to access scheduler members
		};
		using event = std::variant<event_task_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed,
		    event_host_object_created, event_host_object_destroyed, event_epoch_reached, event_test_inspect>;

		std::unique_ptr<command_graph> m_cdag;
		command_recorder* m_crec;
		std::unique_ptr<distributed_graph_generator> m_dggen;
		std::unique_ptr<instruction_graph> m_idag;
		instruction_recorder* m_irec;
		std::unique_ptr<instruction_graph_generator> m_iggen;

		double_buffered_queue<event> m_event_queue;

		void notify(event&& evt);
	};

	class scheduler final : public abstract_scheduler {
		friend struct scheduler_testspy;

	  public:
		scheduler(size_t num_nodes, node_id local_node_id, const system_info& system, const task_manager& tm, delegate* delegate, command_recorder* crec,
		    instruction_recorder* irec, const policy_set& policy = {});

		scheduler(const scheduler&) = delete;
		scheduler(scheduler&&) = delete;
		scheduler& operator=(const scheduler&) = delete;
		scheduler& operator=(scheduler&&) = delete;

		~scheduler() override;

	  private:
		std::thread m_thread;

		void thread_main();
	};

} // namespace detail
} // namespace celerity
