#include "scheduler.h"

#include "distributed_graph_generator.h"
#include "instruction_graph.h"
#include "instruction_graph_generator.h"
#include "named_threads.h"
#include "recorders.h"
#include "task.h"
#include "tracy.h"


namespace celerity {
namespace detail {

	abstract_scheduler::abstract_scheduler(const size_t num_nodes, const node_id local_node_id, instruction_graph_generator::system_info system_info,
	    const task_manager& tm, delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
	    : m_cdag(std::make_unique<command_graph>()), m_crec(crec),
	      m_dggen(std::make_unique<distributed_graph_generator>(num_nodes, local_node_id, *m_cdag, tm, crec, policy.command_graph_generator)),
	      m_idag(std::make_unique<instruction_graph>()), m_irec(irec), //
	      m_iggen(std::make_unique<instruction_graph_generator>(
	          tm, num_nodes, local_node_id, std::move(system_info), *m_idag, delegate, irec, policy.instruction_graph_generator)) {}

	abstract_scheduler::~abstract_scheduler() = default;

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

				matchbox::match(
				    event,
				    [&](const event_task_available& e) {
					    assert(!shutdown);
					    assert(e.tsk != nullptr);
					    const auto commands = m_dggen->build_task(*e.tsk);

					    for(const auto cmd : sort_topologically(commands)) {
						    m_iggen->compile(*cmd);

						    if(e.tsk->get_type() == task_type::epoch && e.tsk->get_epoch_action() == epoch_action::shutdown) {
							    shutdown = true;
							    // m_iggen.delegate must be considered dangling as soon as the instructions for the shutdown epoch have been emitted
						    }
					    }
				    },
				    [&](const event_buffer_created& e) {
					    assert(!shutdown);
					    m_dggen->notify_buffer_created(e.bid, e.range, e.user_allocation_id != null_allocation_id);
					    m_iggen->notify_buffer_created(e.bid, e.range, e.elem_size, e.elem_align, e.user_allocation_id);
				    },
				    [&](const event_buffer_debug_name_changed& e) {
					    assert(!shutdown);
					    m_dggen->notify_buffer_debug_name_changed(e.bid, e.debug_name);
					    m_iggen->notify_buffer_debug_name_changed(e.bid, e.debug_name);
				    },
				    [&](const event_buffer_destroyed& e) {
					    assert(!shutdown);
					    m_dggen->notify_buffer_destroyed(e.bid);
					    m_iggen->notify_buffer_destroyed(e.bid);
				    },
				    [&](const event_host_object_created& e) {
					    assert(!shutdown);
					    m_dggen->notify_host_object_created(e.hoid);
					    m_iggen->notify_host_object_created(e.hoid, e.owns_instance);
				    },
				    [&](const event_host_object_destroyed& e) {
					    assert(!shutdown);
					    m_dggen->notify_host_object_destroyed(e.hoid);
					    m_iggen->notify_host_object_destroyed(e.hoid);
				    },
				    [&](const event_epoch_reached& e) { //
					    // The dggen automatically prunes the CDAG on generation, which is safe because it's not used across threads.
					    // We might want to refactor this to match the IDAG behavior in the future.
					    m_idag->prune_before_epoch(e.tid);
				    },
				    [&](const test_event_signal_idle& e) {
					    // No thread must submit more events until signal has been awaited and all test inspections of the scheduler have taken place.
					    // This check only catches some violations of that synchronization requirement; the test application must ensure that no thread
					    // interacts with the scheduler until all inspections have completed.
					    assert(in_flight_events.empty());

					    *e.idle = true;
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

	scheduler::scheduler(const size_t num_nodes, const node_id local_node_id, instruction_graph_generator::system_info system_info, const task_manager& tm,
	    delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
	    : abstract_scheduler(num_nodes, local_node_id, std::move(system_info), tm, delegate, crec, irec, policy), m_thread(&scheduler::thread_main, this) {
		set_thread_name(m_thread.native_handle(), "cy-scheduler");
	}

	scheduler::~scheduler() {
		// schedule() will exit as soon as it has processed the shutdown epoch
		m_thread.join();
	}

	void scheduler::thread_main() {
		CELERITY_DETAIL_TRACY_SET_CURRENT_THREAD_NAME("cy-scheduler")
		try {
			schedule();
		} catch(const std::exception& e) {
			CELERITY_CRITICAL("[scheduler] {}", e.what());
			std::abort();
		}
	}

} // namespace detail
} // namespace celerity
