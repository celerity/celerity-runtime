#include "scheduler.h"

#include "graph_generator.h"

namespace celerity {
namespace detail {

	scheduler::scheduler(std::shared_ptr<graph_generator> ggen) : ggen(ggen) {}

	void scheduler::startup() { worker_thread = std::thread(&scheduler::schedule, this); }

	void scheduler::shutdown() {
		notify(scheduler_event_type::SHUTDOWN);
		if(worker_thread.joinable()) { worker_thread.join(); }
	}

	bool scheduler::is_idle() const noexcept {
		std::lock_guard<std::mutex> lock(events_mutex);
		return events.empty();
	}

	void scheduler::schedule() {
		std::unique_lock<std::mutex> lk(events_mutex);

		while(true) {
			// TODO: We currently operate in lockstep with the main thread. This is less than ideal.
			events_cv.wait(lk, [this] { return !events.empty(); });
			const auto event = events.front();
			events.pop();

			if(event == scheduler_event_type::SHUTDOWN) {
				assert(events.empty());
				return;
			}

			if(event == scheduler_event_type::TASK_AVAILABLE) {
				const auto tid = ggen->get_unbuilt_task();
				assert(tid != boost::none);
				ggen->build_task(*tid);
				ggen->flush(*tid);
			}
		}
	}

	void scheduler::notify(scheduler_event_type type) {
		{
			std::lock_guard<std::mutex> lk(events_mutex);
			events.push(type);
		}
		events_cv.notify_one();
	}

} // namespace detail
} // namespace celerity
