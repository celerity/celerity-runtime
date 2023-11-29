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
		abstract_scheduler(const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, executor& exec);

		virtual ~abstract_scheduler() = default;

		virtual void startup() = 0;

		virtual void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(const task* const tsk) { notify(event_task_available{tsk}); }

		void notify_buffer_registered(const buffer_id bid, const int dims, const range<3>& range, bool host_initialized) {
			notify(event_buffer_registered{bid, dims, range, host_initialized});
		}

	  protected:
		/**
		 * This is called by the worker thread.
		 */
		void schedule();

		// Constructor for tests that does not require an executor
		abstract_scheduler(const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen)
		    : m_is_dry_run(is_dry_run), m_dggen(std::move(dggen)), m_exec(nullptr) {}

	  private:
		struct event_shutdown {};
		struct event_task_available {
			const task* tsk;
		};
		struct event_buffer_registered {
			buffer_id bid;
			int dims;
			celerity::range<3> range;
			bool host_initialized;
		};
		using event = std::variant<event_shutdown, event_task_available, event_buffer_registered>;

		bool m_is_dry_run;
		std::unique_ptr<distributed_graph_generator> m_dggen;
		executor* m_exec; // Pointer instead of reference so we can omit for tests / benchmarks

		std::queue<event> m_available_events;
		std::queue<event> m_in_flight_events;

		mutable std::mutex m_events_mutex;
		std::condition_variable m_events_cv;

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
