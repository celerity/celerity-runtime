#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace celerity {
namespace detail {

	class graph_generator;

	enum class scheduler_event_type { TASK_AVAILABLE, SHUTDOWN };

	class scheduler {
	  public:
		scheduler(std::shared_ptr<graph_generator> ggen);

		void startup();

		void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created() { notify(scheduler_event_type::TASK_AVAILABLE); }

		/**
		 * @brief Returns true if the scheduler is idle (has no events to process).
		 */
		bool is_idle() const noexcept;

	  private:
		std::shared_ptr<graph_generator> ggen;

		std::queue<scheduler_event_type> events;
		mutable std::mutex events_mutex;
		std::condition_variable events_cv;

		std::thread worker_thread;

		/**
		 * This is called by the worker thread.
		 */
		void schedule();

		void notify(scheduler_event_type type);
	};

} // namespace detail
} // namespace celerity
