#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "frame.h"
#include "graph_serializer.h"
#include "legacy_executor.h"
#include "named_threads.h"

#include <matchbox.hh>

namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(bool is_dry_run, std::unique_ptr<distributed_graph_generator> dggen, legacy_executor& exec)
	    : m_is_dry_run(is_dry_run), m_dggen(std::move(dggen)), m_exec(&exec) {
		assert(m_dggen != nullptr);
	}

	void abstract_scheduler::shutdown() { notify(event_shutdown{}); }

	void abstract_scheduler::schedule() {
		graph_serializer serializer([this](command_pkg&& pkg) {
			if(m_is_dry_run && pkg.get_command_type() != command_type::epoch && pkg.get_command_type() != command_type::horizon
			    && pkg.get_command_type() != command_type::fence) {
				// in dry runs, skip everything except epochs, horizons and fences
				return;
			}
			if(m_is_dry_run && pkg.get_command_type() == command_type::fence) {
				CELERITY_WARN("Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set. "
				              "The result of this operation will not match the expected output of an actual run.");
			}
			// Executor may not be set during tests / benchmarks
			if(m_exec != nullptr) { m_exec->enqueue(std::move(pkg)); }
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

				matchbox::match(
				    event,
				    [&](const event_task_available& e) {
					    assert(!shutdown);
					    assert(e.tsk != nullptr);
					    const auto cmds = m_dggen->build_task(*e.tsk);
					    serializer.flush(cmds);
				    },
				    [&](const event_buffer_created& e) {
					    assert(!shutdown);
					    m_dggen->notify_buffer_created(e.bid, e.range, e.host_initialized);
				    },
				    [&](const event_buffer_debug_name_changed& e) {
					    assert(!shutdown);
					    m_dggen->notify_buffer_debug_name_changed(e.bid, e.debug_name);
				    },
				    [&](const event_buffer_destroyed& e) {
					    assert(!shutdown);
					    m_dggen->notify_buffer_destroyed(e.bid);
				    },
				    [&](const event_host_object_created& e) {
					    assert(!shutdown);
					    m_dggen->notify_host_object_created(e.hoid);
				    },
				    [&](const event_host_object_destroyed& e) {
					    assert(!shutdown);
					    m_dggen->notify_host_object_destroyed(e.hoid);
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
