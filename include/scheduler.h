#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "types.h"

namespace celerity {
namespace detail {

	class graph_generator;
	class graph_serializer;

	enum class scheduler_event_type { TASK_AVAILABLE, SHUTDOWN };

	struct scheduler_event {
		scheduler_event_type type;
		size_t data;
	};

	// Abstract base class to allow different threading implementation in tests
	class abstract_scheduler {
	  public:
		abstract_scheduler(graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes);

		virtual ~abstract_scheduler() = default;

		virtual void startup() = 0;

		virtual void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(task_id tid) { notify(scheduler_event_type::TASK_AVAILABLE, tid); }

	  protected:
		/**
		 * This is called by the worker thread.
		 */
		void schedule();

	  private:
		graph_generator& ggen;
		graph_serializer& gsrlzr;

		std::queue<scheduler_event> events;
		mutable std::mutex events_mutex;
		std::condition_variable events_cv;

		const size_t num_nodes;

		void notify(scheduler_event_type type, size_t data);
	};

	class scheduler final : public abstract_scheduler {
		friend struct scheduler_testspy;

	  public:
		using abstract_scheduler::abstract_scheduler;

		void startup() override;

		void shutdown() override;

	  private:
		std::thread worker_thread;
	};

} // namespace detail
} // namespace celerity
