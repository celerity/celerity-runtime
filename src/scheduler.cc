#include "scheduler.h"

#include "graph_generator.h"
#include "graph_serializer.h"
#include "transformers/naive_split.h"

namespace celerity {
namespace detail {

	scheduler::scheduler(graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes) : ggen(ggen), gsrlzr(gsrlzr), num_nodes(num_nodes) {}

	void scheduler::startup() { worker_thread = std::thread(&scheduler::schedule, this); }

	void scheduler::shutdown() {
		notify(scheduler_event_type::SHUTDOWN, 0);
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
