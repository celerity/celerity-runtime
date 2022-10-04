#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

#include "ranges.h"
#include "types.h"

namespace celerity {
namespace detail {

	class distributed_graph_generator;
	class executor;
	class task;

	// Abstract base class to allow different threading implementation in tests
	class abstract_scheduler {
	  public:
		abstract_scheduler(bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, executor& exec, size_t num_nodes);

		virtual ~abstract_scheduler() = default;

		virtual void startup() = 0;

		virtual void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(const task* const tsk) { notify(event_task_available{tsk}); }

		void notify_buffer_registered(const buffer_id bid, const celerity::range<3>& range) { notify(event_buffer_registered{bid, range}); }

	  protected:
		/**
		 * This is called by the worker thread.
		 */
		void schedule();

	  private:
		struct event_shutdown {};
		struct event_task_available {
			const task* tsk;
		};
		struct event_buffer_registered {
			buffer_id bid;
			celerity::range<3> range;
		};
		using event = std::variant<event_shutdown, event_task_available, event_buffer_registered>;

		bool m_is_dry_run;
		std::unique_ptr<distributed_graph_generator> m_dggen;
		executor& m_exec;

		std::queue<event> m_available_events;
		std::queue<event> m_in_flight_events;

		mutable std::mutex m_events_mutex;
		std::condition_variable m_events_cv;

		const size_t m_num_nodes;

		void notify(const event& evt);
	};

	class scheduler final : public abstract_scheduler {
		friend struct scheduler_testspy;

	  public:
		using abstract_scheduler::abstract_scheduler;

		void startup() override;

		void shutdown() override;

	  private:
		std::thread m_worker_thread;
	};

} // namespace detail
} // namespace celerity
