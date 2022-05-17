#include "scheduler.h"

#include "graph_generator.h"
#include "graph_serializer.h"
#include "named_threads.h"
#include "transformers/naive_split.h"

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes)
	    : ggen(ggen), gsrlzr(gsrlzr), num_nodes(num_nodes) {}

	void abstract_scheduler::shutdown() { notify(scheduler_event_type::SHUTDOWN, 0); }

	void abstract_scheduler::schedule() {
		std::queue<scheduler_event> in_flight_events;
		while(true) {
			{
				std::unique_lock lk(events_mutex);
				events_cv.wait(lk, [this] { return !available_events.empty(); });
				std::swap(available_events, in_flight_events);
			}

			while(!in_flight_events.empty()) {
				const auto event = std::move(in_flight_events.front()); // NOLINT(performance-move-const-arg)
				in_flight_events.pop();

				if(event.type == scheduler_event_type::SHUTDOWN) {
					assert(in_flight_events.empty());
					return;
				}

				assert(event.type == scheduler_event_type::TASK_AVAILABLE);

				const task_id tid = event.data;
				assert(tsk != nullptr);
				naive_split_transformer naive_split(num_nodes, num_nodes);
				ggen.build_task(tid, {&naive_split});
				gsrlzr.flush(tid);
			}
		}
	}

	void abstract_scheduler::notify(scheduler_event_type type, size_t data) {
		{
			std::lock_guard lk(events_mutex);
			available_events.push({type, data});
		}
		events_cv.notify_one();
	}

	void scheduler::startup() {
		worker_thread = std::thread(&scheduler::schedule, this);
		set_thread_name(worker_thread.native_handle(), "cy-scheduler");
	}

	void scheduler::shutdown() {
		abstract_scheduler::shutdown();
		if(worker_thread.joinable()) { worker_thread.join(); }
	}

} // namespace detail
} // namespace celerity
