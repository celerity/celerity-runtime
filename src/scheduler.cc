#include "scheduler.h"

#include "graph_generator.h"
#include "graph_serializer.h"
#include "named_threads.h"
#include "transformers/naive_split.h"

namespace celerity {
namespace detail {

	const std::string background_thread::default_debug_name = "cy-thread";
	background_loop* const background_thread::loop_empty = nullptr;
	background_loop* const background_thread::loop_shutdown = loop_empty + 1;

	background_thread::background_thread() { set_thread_name(thread.native_handle(), default_debug_name); }

	background_thread::~background_thread() {
		{
			std::unique_lock lk{loop_mutex};
			wait(lk);
			loop = loop_shutdown;
			loop_changed.notify_one();
		}
		thread.join();
	}

	void background_thread::start(background_loop& lo, const std::string& debug_name) {
		std::unique_lock lk{loop_mutex};
		wait(lk);
		loop = &lo;
		loop_changed.notify_all();
		set_thread_name(thread.native_handle(), debug_name);
	}

	void background_thread::wait() {
		std::unique_lock lk{loop_mutex};
		wait(lk);
	}

	void background_thread::main() {
		std::unique_lock lk{loop_mutex};
		for(;;) {
			loop_changed.wait(lk, [this] { return loop != nullptr; });
			if(loop == loop_shutdown) break;
			loop->loop();
			loop = nullptr;
			loop_changed.notify_all();
		}
	}

	void background_thread::wait(std::unique_lock<std::mutex>& lk) {
		loop_changed.wait(lk, [this] {
			assert(loop != loop_shutdown);
			return loop == loop_empty;
		});
		set_thread_name(thread.native_handle(), default_debug_name);
	}

	scheduler::scheduler(background_thread& thrd, graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes)
	    : thrd(thrd), ggen(ggen), gsrlzr(gsrlzr), num_nodes(num_nodes) {}

	void scheduler::startup() { thrd.start(*this, "cy-scheduler"); }

	void scheduler::shutdown() {
		notify(scheduler_event_type::SHUTDOWN, 0);
		thrd.wait();
	}

	bool scheduler::is_idle() const noexcept {
		std::lock_guard<std::mutex> lock(events_mutex);
		return events.empty();
	}

	void scheduler::loop() {
		std::unique_lock<std::mutex> lk(events_mutex);

		while(true) {
			// TODO: We currently operate in lockstep with the main thread. This is less than ideal.
			events_cv.wait(lk, [this] { return !events.empty(); });
			const auto event = events.front();
			events.pop();

			const task_id tid = event.data;
			if(event.type == scheduler_event_type::TASK_AVAILABLE) {
				naive_split_transformer naive_split(num_nodes, num_nodes);
				ggen.build_task(tid, {&naive_split});
				gsrlzr.flush(tid);
			} else if(event.type == scheduler_event_type::SHUTDOWN) {
				assert(events.empty());
				return;
			}
		}
	}

	void scheduler::notify(scheduler_event_type type, size_t data) {
		{
			std::lock_guard<std::mutex> lk(events_mutex);
			events.push({type, data});
		}
		events_cv.notify_one();
	}

} // namespace detail
} // namespace celerity
