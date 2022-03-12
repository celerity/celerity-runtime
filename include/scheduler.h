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

	class background_loop {
	  public:
		virtual void loop() = 0;

	  protected:
		// non-virtual dtor cannot be used to destroy through a base class pointer
		~background_loop() = default;
	};

	class background_thread {
		friend struct background_thread_testspy;

	  public:
		static const std::string default_debug_name;

		background_thread();
		~background_thread();

		void start(background_loop& lo, const std::string& debug_name = default_debug_name);
		void wait();

	  private:
		static background_loop* const loop_empty;
		static background_loop* const loop_shutdown;

		std::mutex loop_mutex;
		background_loop* loop = loop_empty;
		std::condition_variable loop_changed;
		std::thread thread{&background_thread::main, this};

		void main();

		void wait(std::unique_lock<std::mutex>& lk);
	};

	enum class scheduler_event_type { TASK_AVAILABLE, SHUTDOWN };

	struct scheduler_event {
		scheduler_event_type type;
		size_t data;
	};

	class scheduler final : private background_loop {
		friend struct scheduler_testspy;

	  public:
		scheduler(background_thread& thrd, graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes);

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
		background_thread& thrd;
		graph_generator& ggen;
		graph_serializer& gsrlzr;

		std::queue<scheduler_event> events;
		mutable std::mutex events_mutex;
		std::condition_variable events_cv;

		const size_t num_nodes;

		/**
		 * This is called by the background_thread.
		 */
		void loop() override;

		void notify(scheduler_event_type type, size_t data);
	};

} // namespace detail
} // namespace celerity
