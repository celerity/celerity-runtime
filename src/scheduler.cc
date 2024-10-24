#include "scheduler.h"

#include "command_graph_generator.h"
#include "double_buffered_queue.h"
#include "instruction_graph_generator.h"
#include "log.h"
#include "named_threads.h"
#include "recorders.h"
#include "tracy.h"

#include <deque>
#include <functional>
#include <thread>

#include <matchbox.hh>


namespace celerity::detail::scheduler_detail {

struct event_task_available {
	const task* tsk;
};
struct event_command_available {
	const abstract_command* cmd;
};
struct event_buffer_created {
	buffer_id bid;
	celerity::range<3> range;
	size_t elem_size;
	size_t elem_align;
	allocation_id user_allocation_id;
};
struct event_buffer_debug_name_changed {
	buffer_id bid;
	std::string debug_name;
};
struct event_buffer_destroyed {
	buffer_id bid;
};
struct event_host_object_created {
	host_object_id hoid;
	bool owns_instance;
};
struct event_host_object_destroyed {
	host_object_id hoid;
};
struct event_epoch_reached {
	task_id tid;
};
struct event_set_lookahead {
	experimental::lookahead lookahead;
};
struct event_flush_commands {};
struct test_event_inspect {
	std::function<void()> inspect; // executed inside scheduler thread, making it safe to access scheduler members
};

struct task_queue {
	using event = std::variant<event_task_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed, event_host_object_created,
	    event_host_object_destroyed, event_epoch_reached, event_set_lookahead, event_flush_commands, test_event_inspect>;

	double_buffered_queue<event> global_queue;
	std::deque<event> local_queue;

	bool empty() const { return !global_queue.nonempty() && local_queue.empty(); }

	void push(event&& evt) { global_queue.push(std::move(evt)); }

	event pop() {
		if(local_queue.empty()) {
			// We can frequently suspend / resume the scheduler thread without adding latency as long as the executor remains busy
			global_queue.wait_while_empty();
			const auto& batch = global_queue.pop_all();
			local_queue.insert(local_queue.end(), batch.begin(), batch.end());
			assert(!local_queue.empty());
		}
		auto evt = std::move(local_queue.front());
		local_queue.pop_front();
		return evt;
	}
};

struct command_queue {
	using event = std::variant<event_command_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed,
	    event_host_object_created, event_host_object_destroyed, event_flush_commands, event_set_lookahead>;

	std::deque<event> queue;
	int num_queued_flushes = 0;
	int num_queued_horizons = 0;

	bool empty() const { return queue.empty(); }

	bool next_is_command() const { return !queue.empty() && std::holds_alternative<event_command_available>(queue.front()); }

	void push(event&& evt) {
		const auto avail = std::get_if<event_command_available>(&evt);
		if(std::holds_alternative<event_flush_commands>(evt)
		    || (avail != nullptr && (avail->cmd->get_type() == command_type::fence || avail->cmd->get_type() == command_type::epoch))) {
			num_queued_flushes += 1;
		} else if(avail != nullptr && avail->cmd->get_type() == command_type::horizon) {
			num_queued_horizons += 1;
		}
		queue.push_back(std::move(evt));
	}

	event pop() {
		assert(!queue.empty());
		auto evt = std::move(queue.front());
		queue.pop_front();

		const auto avail = std::get_if<event_command_available>(&evt);
		if(std::holds_alternative<event_flush_commands>(evt)
		    || (avail != nullptr && (avail->cmd->get_type() == command_type::fence || avail->cmd->get_type() == command_type::epoch))) {
			assert(num_queued_flushes > 0);
			num_queued_flushes -= 1;
		} else if(avail != nullptr && avail->cmd->get_type() == command_type::horizon) {
			assert(num_queued_horizons > 0);
			num_queued_horizons -= 1;
		}

		return evt;
	}
};

struct scheduler_impl {
	command_graph cdag;
	command_recorder* crec;
	command_graph_generator cggen;
	instruction_graph idag;
	instruction_recorder* irec;
	instruction_graph_generator iggen;

	experimental::lookahead lookahead = experimental::lookahead::automatic;

	task_queue task_queue;
	command_queue command_queue;

	std::optional<task_id> shutdown_epoch_created = std::nullopt;
	bool shutdown_epoch_reached = false;

	std::thread thread;

	scheduler_impl(const bool test_start_idle, const size_t num_nodes, const node_id local_node_id, const system_info& system, const task_manager& tm,
	    scheduler::delegate* const dlg, command_recorder* const crec, instruction_recorder* const irec, const scheduler::policy_set& policy)
	    : cdag(), crec(crec), cggen(num_nodes, local_node_id, cdag, tm, crec, policy.command_graph_generator), idag(), irec(irec),
	      iggen(tm, num_nodes, local_node_id, system, idag, dlg, irec, policy.instruction_graph_generator) {
		if(!test_start_idle) {
			thread = std::thread(&scheduler_impl::thread_main, this);
			set_thread_name(thread.native_handle(), "cy-scheduler");
		}
	}

	// immovable: self-referential via `thread`
	scheduler_impl(const scheduler_impl&) = delete;
	scheduler_impl(scheduler_impl&&) = delete;
	scheduler_impl& operator=(const scheduler_impl&) = delete;
	scheduler_impl& operator=(scheduler_impl&&) = delete;

	~scheduler_impl() {
		// schedule() will exit as soon as it has processed the shutdown epoch
		if(thread.joinable()) { thread.join(); }
	}

	std::vector<const abstract_command*> build_task(const task& tsk);
	void compile_command(const abstract_command& cmd);

	void process_task_queue_event(const task_queue::event& evt);
	bool should_dequeue_more_command_events() const;
	void process_command_queue_event(const command_queue::event& evt);

	void thread_main();
};

std::vector<const abstract_command*> scheduler_impl::build_task(const task& tsk) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::build_task", WebMaroon, "T{} build", tsk.get_id());
	CELERITY_DETAIL_TRACY_ZONE_TEXT(utils::make_task_debug_label(tsk.get_type(), tsk.get_id(), tsk.get_debug_name()));
	return cggen.build_task(tsk);
}

void scheduler_impl::compile_command(const abstract_command& cmd) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::compile_command", MidnightBlue, "C{} compile", cmd.get_cid());
	CELERITY_DETAIL_TRACY_ZONE_TEXT("{}", cmd.get_type());
	iggen.compile(cmd);
}

// TODO split CDAG / IDAG schedulers to simplify queueing?
// - IDAG scheduler will always need a branch "perform now or later?" so we can compile the first instruction ASAP
// - We must not opportunistically anticipate() things later in the queue, otherwise we get non-deterministic IDAGs

void scheduler_impl::process_task_queue_event(const task_queue::event& evt) {
	matchbox::match(
	    evt,
	    [&](const event_task_available& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    assert(e.tsk != nullptr);
		    auto& tsk = *e.tsk;

		    auto commands = build_task(tsk);

		    for(const auto cmd : commands) {
			    // If there are multiple commands, the shutdown epoch must come last. m_iggen.delegate must be considered dangling after receiving
			    // the corresponding instruction, as runtime will begin destroying the executor after it has observed the epoch to be reached.
			    assert(!shutdown_epoch_created);
			    if(tsk.get_type() == task_type::epoch && tsk.get_epoch_action() == epoch_action::shutdown) { shutdown_epoch_created = tsk.get_id(); }

			    command_queue.push(event_command_available{cmd});
		    }
	    },
	    [&](const event_buffer_created& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_created", DarkGreen, "B{} create", e.bid);
		    cggen.notify_buffer_created(e.bid, e.range, e.user_allocation_id != null_allocation_id);
		    command_queue.push(e);
	    },
	    [&](const event_buffer_debug_name_changed& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_name_changed", DarkGreen, "B{} set name", e.bid);
		    cggen.notify_buffer_debug_name_changed(e.bid, e.debug_name);
		    command_queue.push(e);
	    },
	    [&](const event_buffer_destroyed& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_destroyed", DarkGreen, "B{} destroy", e.bid);
		    cggen.notify_buffer_destroyed(e.bid);
		    command_queue.push(e);
	    },
	    [&](const event_host_object_created& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_created", DarkGreen, "H{} create", e.hoid);
		    cggen.notify_host_object_created(e.hoid);
		    command_queue.push(e);
	    },
	    [&](const event_host_object_destroyed& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_destroyed", DarkGreen, "H{} destroy", e.hoid);
		    cggen.notify_host_object_destroyed(e.hoid);
		    command_queue.push(e);
	    },
	    [&](const event_epoch_reached& e) { //
		    assert(!shutdown_epoch_reached);
		    {
			    // The cggen automatically prunes the CDAG on generation, which is safe because commands are not shared across threads.
			    // We might want to refactor this to match the IDAG behavior in the future.
			    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::prune_idag", Gray);
			    idag.prune_before_epoch(e.tid);
		    }

		    // The scheduler will receive the shutdown-epoch completion event via the runtime even if executor destruction has already begun.
		    if(shutdown_epoch_created && e.tid == *shutdown_epoch_created) { shutdown_epoch_reached = true; }
	    },
	    [&](const event_set_lookahead& e) { //
		    command_queue.push(e);
	    },
	    [&](const event_flush_commands& e) { command_queue.push(e); },
	    [&](const test_event_inspect& e) { //
		    e.inspect();
	    });
}

bool scheduler_impl::should_dequeue_more_command_events() const {
	if(command_queue.empty()) return false;
	// always flush set_lookahead() and metadata changes if they are at the front, otherwise set_lookahead(none) after queue creation does not work
	// TODO evaluate if `q.submit(); q.set_lookahead(none); q.submit()` not flushing immediately is surprising or expected.
	if(!command_queue.next_is_command()) return true;
	if(command_queue.num_queued_flushes > 0) return true;
	switch(lookahead) {
	case experimental::lookahead::none: return true;
	case experimental::lookahead::automatic: return command_queue.num_queued_horizons > 1;
	case experimental::lookahead::infinite: return false;
	}
}

void scheduler_impl::process_command_queue_event(const command_queue::event& evt) {
	matchbox::match(
	    evt, //
	    [&](const event_command_available& e) { compile_command(*e.cmd); },
	    [&](const event_buffer_created& e) { iggen.notify_buffer_created(e.bid, e.range, e.elem_size, e.elem_align, e.user_allocation_id); },
	    [&](const event_buffer_debug_name_changed& e) { iggen.notify_buffer_debug_name_changed(e.bid, e.debug_name); },
	    [&](const event_buffer_destroyed& e) { iggen.notify_buffer_destroyed(e.bid); },
	    [&](const event_host_object_created& e) { iggen.notify_host_object_created(e.hoid, e.owns_instance); },
	    [&](const event_host_object_destroyed& e) { iggen.notify_host_object_destroyed(e.hoid); },
	    [&](const event_set_lookahead& e) { lookahead = e.lookahead; }, //
	    [&](const event_flush_commands& /* e */) {});
}

void scheduler_impl::thread_main() {
	CELERITY_DETAIL_TRACY_SET_THREAD_NAME_AND_ORDER("cy-scheduler", tracy_detail::thread_order::scheduler)
	try {
		while(!shutdown_epoch_reached) {
			process_task_queue_event(task_queue.pop());
			while(should_dequeue_more_command_events()) {
				process_command_queue_event(command_queue.pop());
			}
		}
		assert(task_queue.empty());
		assert(command_queue.empty());
	}
	// LCOV_EXCL_START
	catch(const std::exception& e) {
		CELERITY_CRITICAL("[scheduler] {}", e.what());
		std::abort();
	}
	// LCOV_EXCL_STOP
}

} // namespace celerity::detail::scheduler_detail

using namespace celerity::detail::scheduler_detail;

namespace celerity::detail {

scheduler::scheduler(const size_t num_nodes, const node_id local_node_id, const system_info& system, const task_manager& tm, delegate* const delegate,
    command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
    : m_impl(std::make_unique<scheduler_impl>(false /* test_start_idle */, num_nodes, local_node_id, system, tm, delegate, crec, irec, policy)) {}

scheduler::scheduler(const test_start_idle_tag /* test_start_idle */, const size_t num_nodes, const node_id local_node_id, const system_info& system,
    const task_manager& tm, delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec, const policy_set& policy)
    : m_impl(std::make_unique<scheduler_impl>(true /* test_start_idle */, num_nodes, local_node_id, system, tm, delegate, crec, irec, policy)) {}

scheduler::~scheduler() = default;

/**
 * @brief Notifies the scheduler that a new task has been created and is ready for scheduling.
 */
void scheduler::notify_task_created(const task* const tsk) { m_impl->task_queue.push(event_task_available{tsk}); }

void scheduler::notify_buffer_created(
    const buffer_id bid, const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_allocation_id) {
	m_impl->task_queue.push(event_buffer_created{bid, range, elem_size, elem_align, user_allocation_id});
}

void scheduler::notify_buffer_debug_name_changed(const buffer_id bid, const std::string& name) {
	m_impl->task_queue.push(event_buffer_debug_name_changed{bid, name});
}

void scheduler::notify_buffer_destroyed(const buffer_id bid) { m_impl->task_queue.push(event_buffer_destroyed{bid}); }

void scheduler::notify_host_object_created(const host_object_id hoid, const bool owns_instance) {
	m_impl->task_queue.push(event_host_object_created{hoid, owns_instance});
}

void scheduler::notify_host_object_destroyed(const host_object_id hoid) { m_impl->task_queue.push(event_host_object_destroyed{hoid}); }

void scheduler::notify_epoch_reached(const task_id tid) { m_impl->task_queue.push(event_epoch_reached{tid}); }

void scheduler::set_lookahead(const experimental::lookahead lookahead) { m_impl->task_queue.push(event_set_lookahead{lookahead}); }

void scheduler::flush_commands() { m_impl->task_queue.push(event_flush_commands{}); }

void scheduler::test_invoke_thread_main() { m_impl->thread_main(); }
void scheduler::test_inspect(std::function<void()> inspector) { m_impl->task_queue.push(test_event_inspect{std::move(inspector)}); }
size_t scheduler::test_get_live_command_count() { return m_impl->cdag.command_count(); }
size_t scheduler::test_get_live_instruction_count() { return m_impl->idag.get_live_instruction_count(); }

} // namespace celerity::detail
