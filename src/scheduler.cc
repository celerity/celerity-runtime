#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "executor.h"
#include "frame.h"
#include "graph_serializer.h"
#include "named_threads.h"
#include "transformers/naive_split.h"
#include "utils.h"

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, executor& exec, size_t num_nodes)
	    : m_is_dry_run(is_dry_run), m_dggen(std::move(dggen)), m_exec(exec), m_num_nodes(num_nodes) {
		assert(m_dggen != nullptr);
	}

	void abstract_scheduler::shutdown() { notify(event_shutdown{}); }

	void abstract_scheduler::schedule() {
		// NOCOMMIT Get rid of this, no need to serialize commands
		graph_serializer serializer(m_dggen->NOCOMMIT_get_cdag(), [this](node_id, command_pkg pkg) {
			if(m_is_dry_run && pkg.get_command_type() != command_type::epoch) { return; }
			m_exec.enqueue(std::move(pkg));
		});

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
				    [this, &serializer](const event_task_available& e) {
					    assert(e.tsk != nullptr);
					    naive_split_transformer naive_split(m_num_nodes, m_num_nodes);
					    const auto cmds = m_dggen->build_task(*e.tsk);
					    serializer.flush(cmds);
				    },
				    [this](const event_buffer_registered& e) { //
					    m_dggen->add_buffer(e.bid, e.range, e.dims);
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
			std::lock_guard lk(m_events_mutex);
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
