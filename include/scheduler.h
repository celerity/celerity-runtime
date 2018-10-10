#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace celerity {
namespace detail {

	class graph_generator;

	class scheduler {
	  public:
		scheduler(std::shared_ptr<graph_generator> ggen, size_t num_nodes);

		void startup();

		void shutdown();

		/**
		 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
		 */
		void notify_task_created();

	  private:
		std::shared_ptr<graph_generator> ggen;
		size_t num_nodes;

		size_t unscheduled_tasks = 0;
		std::atomic<bool> should_shutdown = {false};
		std::mutex tasks_available_mutex;
		std::condition_variable tasks_available_cv;
		std::thread schd_thrd;

		/**
		 * This is called by the worker thread.
		 */
		void schedule();
	};

} // namespace detail
} // namespace celerity
