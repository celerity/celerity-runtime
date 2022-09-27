#include "scheduler.h"

#include "graph_generator.h"
#include "graph_serializer.h"
#include "named_threads.h"
#include "transformers/naive_split.h"
#include "utils.h"

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(std::unique_ptr<graph_generator> ggen, std::unique_ptr<graph_serializer> gser, size_t num_nodes)
	    : m_ggen(std::move(ggen)), m_gser(std::move(gser)), m_num_nodes(num_nodes) {
		assert(m_ggen != nullptr);
		assert(m_gser != nullptr);
	}

	void abstract_scheduler::shutdown() { notify(event_shutdown{}); }

	void abstract_scheduler::schedule() {
		std::queue<event> in_flight_events;
		bool shutdown = false;
		while(!shutdown) {
			{
				std::unique_lock lk(m_events_mutex);
				m_events_cv.wait(lk, [this] { return !m_available_events.empty(); });
				std::swap(m_available_events, in_flight_events);
			}

			while(!in_flight_events.empty()) {
				const auto event = std::move(in_flight_events.front()); // NOLINT(performance-move-const-arg)
				in_flight_events.pop();

				utils::match(
				    event,
				    [this](const event_task_available& e) {
					    assert(e.tsk != nullptr);
					    naive_split_transformer naive_split(m_num_nodes, m_num_nodes);
					    m_ggen->build_task(*e.tsk, {&naive_split});
					    m_gser->flush(e.tsk->get_id());
				    },
				    [this](const event_buffer_registered& e) { //
					    m_ggen->add_buffer(e.bid, e.range);
				    },
				    [&](const event_shutdown&) {
					    assert(in_flight_events.empty());
					    shutdown = true;
				    });
			}
		}
	}

	void abstract_scheduler::notify(const event& evt) {
		{
			const std::lock_guard lk(m_events_mutex);
			m_available_events.push(evt);
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
