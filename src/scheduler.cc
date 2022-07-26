#include "scheduler.h"

#include "graph_generator.h"
#include "graph_serializer.h"
#include "named_threads.h"
#include "transformers/naive_split.h"

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(graph_generator& ggen, graph_serializer& gsrlzr, size_t num_nodes)
	    : m_ggen(ggen), m_gsrlzr(gsrlzr), m_num_nodes(num_nodes) {}

	void abstract_scheduler::shutdown() { notify(scheduler_event_type::shutdown, 0); }

	void abstract_scheduler::schedule() {
		std::queue<scheduler_event> in_flight_events;
		while(true) {
			{
				std::unique_lock lk(m_events_mutex);
				m_events_cv.wait(lk, [this] { return !m_available_events.empty(); });
				std::swap(m_available_events, in_flight_events);
			}

			while(!in_flight_events.empty()) {
				const auto event = std::move(in_flight_events.front()); // NOLINT(performance-move-const-arg)
				in_flight_events.pop();

				if(event.type == scheduler_event_type::shutdown) {
					assert(in_flight_events.empty());
					return;
				}

				assert(event.type == scheduler_event_type::task_available);

				const task* tsk = event.tsk;
				assert(tsk != nullptr);
				naive_split_transformer naive_split(m_num_nodes, m_num_nodes);
				m_ggen.build_task(*tsk, {&naive_split});
				m_gsrlzr.flush(tsk->get_id());
			}
		}
	}

	void abstract_scheduler::notify(scheduler_event_type type, const task* tsk) {
		{
			std::lock_guard lk(m_events_mutex);
			m_available_events.push({type, tsk});
		}
		m_events_cv.notify_one();
	}

	void scheduler::startup() {
		m_worker_thread = std::thread(&scheduler::schedule, this);
		set_thread_name(m_worker_thread.native_handle(), "cy-scheduler");
	}

	void scheduler::shutdown() {
		abstract_scheduler::shutdown();
		if(m_worker_thread.joinable()) { m_worker_thread.join(); }
	}

} // namespace detail
} // namespace celerity
