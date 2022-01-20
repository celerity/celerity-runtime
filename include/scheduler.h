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

	class scheduler {
	  public:
		scheduler(graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes);

		void startup();

		void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created(task_id tid) { notify(scheduler_event_type::TASK_AVAILABLE, tid); }

		/**
		 * @brief Returns true if the scheduler is idle (has no events to process).
		 */
		bool is_idle() const noexcept;

	  private:
		graph_generator& ggen;
		graph_serializer& gsrlzr;

		std::queue<scheduler_event> events;
		mutable std::mutex events_mutex;
		std::condition_variable events_cv;

		const size_t num_nodes;

		std::thread worker_thread;

		/**
		 * This is called by the worker thread.
		 */
		void schedule();

		void notify(scheduler_event_type type, size_t data);
	};

} // namespace detail
} // namespace celerity
