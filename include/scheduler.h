#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <variant>

#include "distributed_graph_generator.h"
#include "ranges.h"
#include "types.h"

namespace celerity {
namespace detail {

	class command_graph;
	class command_recorder;
	class legacy_executor;
	class task;

	// Abstract base class to allow different threading implementation in tests
	class abstract_scheduler {
	  public:
		abstract_scheduler(const bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, legacy_executor& exec);

		virtual ~abstract_scheduler() = default;

		virtual void startup() = 0;

		virtual void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(const task* const tsk) { notify(event_task_available{tsk}); }

		void notify_buffer_created(const buffer_id bid, const range<3>& range, bool host_initialized) {
			notify(event_buffer_created{bid, range, host_initialized});
		}

		void notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name) { notify(event_buffer_debug_name_changed{bid, name}); }

		void notify_buffer_destroyed(const buffer_id bid) { notify(event_buffer_destroyed{bid}); }

		void notify_host_object_created(const host_object_id hoid) { notify(event_host_object_created{hoid}); }

		void notify_host_object_destroyed(const host_object_id hoid) { notify(event_host_object_destroyed{hoid}); }

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
		struct event_buffer_created {
			buffer_id bid;
			celerity::range<3> range;
			bool host_initialized;
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
		};
		struct event_host_object_destroyed {
			host_object_id hoid;
		};
		using event = std::variant<event_shutdown, event_task_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed,
		    event_host_object_created, event_host_object_destroyed>;

		bool m_is_dry_run;
		std::unique_ptr<distributed_graph_generator> m_dggen;
		legacy_executor* m_exec; // Pointer instead of reference so we can omit for tests / benchmarks

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
