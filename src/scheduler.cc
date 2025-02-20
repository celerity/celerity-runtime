#include "scheduler.h"

#include "command_graph.h"
#include "command_graph_generator.h"
#include "double_buffered_queue.h"
#include "instruction_graph_generator.h"
#include "loop_template.h"
#include "named_threads.h"
#include "print_utils.h"
#include "print_utils_internal.h"
#include "ranges.h"
#include "recorders.h"
#include "testspy/scheduler_testspy.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>

#include <matchbox.hh>


namespace celerity::detail::scheduler_detail {

struct event_task_available {
	const task* tsk;
};
struct event_command_available {
	const command* cmd;
	std::optional<instruction_graph_generator::scheduling_hint> hint;
	loop_template* templ;

	// For instruction recording
	task_id part_of_tid;
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
struct event_enable_loop_template {
	loop_template* templ;
};
struct event_complete_loop_iteration {
	loop_template* templ; // NOCOMMIT Only set/used for IDAG - weird
};
struct event_finalize_loop_template {
	loop_template* templ;
};
struct event_leak_memory {};

/// An event passed from task_manager or runtime through the public scheduler interface.
using task_event = std::variant<event_task_available, event_buffer_created, event_buffer_debug_name_changed, event_buffer_destroyed, event_host_object_created,
    event_host_object_destroyed, event_epoch_reached, event_set_lookahead, event_flush_commands, event_enable_loop_template, event_complete_loop_iteration,
    event_finalize_loop_template, event_leak_memory, scheduler_testspy::event_inspect>;

class task_queue {
  public:
	void push(task_event&& evt) { m_global_queue.push(std::move(evt)); }

	task_event wait_and_pop() {
		if(m_local_queue.empty()) {
			// We can frequently suspend / resume the scheduler thread without adding latency as long as the executor remains busy
			m_global_queue.wait_while_empty();
			const auto& batch = m_global_queue.pop_all();
			m_local_queue.insert(m_local_queue.end(), batch.begin(), batch.end());
			assert(!m_local_queue.empty());
		}
		auto evt = std::move(m_local_queue.front());
		m_local_queue.pop_front();
		return evt;
	}

	bool empty() const { return m_local_queue.empty() && m_global_queue.empty(); }

  private:
	double_buffered_queue<task_event> m_global_queue;
	std::deque<task_event> m_local_queue; // "triple buffer" here because double_buffered_queue only gives us a temporary reference to its read-end
};

/// An event originating from command_graph_generator, or forwarded from the task_queue because it requires in-order processing with commands.
using command_event = std::variant<event_command_available, event_buffer_debug_name_changed, event_buffer_destroyed, event_host_object_destroyed,
    event_flush_commands, event_set_lookahead, event_complete_loop_iteration, event_finalize_loop_template, event_leak_memory>;

class command_queue {
  public:
	bool should_dequeue(const experimental::lookahead lookahead) const {
		if(m_queue.empty()) return false;
		if(lookahead == experimental::lookahead::none) return true;                        // unconditionally dequeue, and do not inspect scheduling hints
		if(m_num_flushes_in_queue > 0) return true;                                        // force-dequeue until all flushing events are processed
		if(!std::holds_alternative<event_command_available>(m_queue.front())) return true; // only commands carry a hint and are thus worth delaying
		const auto& avail = std::get<event_command_available>(m_queue.front());
		assert(avail.hint.has_value()); // only nullopt when lookahead == none, which we checked above
		if(avail.hint == instruction_graph_generator::scheduling_hint::is_self_contained) return true; // don't delay scheduling of self-contained commands
		if(lookahead == experimental::lookahead::infinite) return false;
		assert(lookahead == experimental::lookahead::automatic);
		return m_num_horizons_since_last_mergeable_cmd >= 2; // heuristic: passing two horizons suggests we have arrived at an "allocation steady state"
	}

	void push(command_event&& evt) {
		if(is_flush(evt)) m_num_flushes_in_queue += 1;
		if(const auto avail = std::get_if<event_command_available>(&evt)) {
			if(utils::isa<horizon_command>(avail->cmd)) { m_num_horizons_since_last_mergeable_cmd += 1; }
			if(avail->hint == instruction_graph_generator::scheduling_hint::could_merge_with_future_commands) { m_num_horizons_since_last_mergeable_cmd = 0; }
		}
		m_queue.push_back(std::move(evt));
	}

	command_event pop() {
		assert(!m_queue.empty());
		auto evt = std::move(m_queue.front());
		m_queue.pop_front();
		if(is_flush(evt)) { m_num_flushes_in_queue -= 1; }
		return evt;
	}

	bool empty() const { return m_queue.empty(); }

  private:
	std::deque<command_event> m_queue;
	int m_num_flushes_in_queue = 0;
	int m_num_horizons_since_last_mergeable_cmd = 0;

	static bool is_flush(const command_event& evt) {
		if(std::holds_alternative<event_flush_commands>(evt)) return true;
		// Flushing on all changes to the lookahead setting avoids complicated decisions on when to "anticipate" commands from incoming tasks
		if(std::holds_alternative<event_set_lookahead>(evt)) return true;
		if(const auto avail = std::get_if<event_command_available>(&evt)) {
			return utils::isa<fence_command>(avail->cmd) || utils::isa<epoch_command>(avail->cmd);
		}
		return false;
	}
};

struct scheduler_impl {
	scheduler::delegate* dlg;
	command_graph cdag;
	command_recorder* crec;
	command_graph_generator cggen;
	instruction_graph idag;
	instruction_recorder* irec;
	instruction_graph_generator iggen;

	experimental::lookahead lookahead = experimental::lookahead::automatic;

	loop_template* active_loop_template = nullptr; // NOCOMMIT Only for commands though - naming?! Or don't store this, but pass it in with each task?

	class task_queue task_queue;
	class command_queue command_queue;

	std::optional<task_id> shutdown_epoch_created = std::nullopt;
	bool shutdown_epoch_reached = false;

	task_id highest_seen_tid = 0; ///< Used for recording task boundaries

	std::thread thread;

	scheduler_impl(bool start_thread, size_t num_nodes, node_id local_node_id, const system_info& system, scheduler::delegate* dlg, command_recorder* crec,
	    instruction_recorder* irec, const scheduler::policy_set& policy);

	// immovable: self-referential via `thread`
	scheduler_impl(const scheduler_impl&) = delete;
	scheduler_impl(scheduler_impl&&) = delete;
	scheduler_impl& operator=(const scheduler_impl&) = delete;
	scheduler_impl& operator=(scheduler_impl&&) = delete;

	~scheduler_impl();

	void process_task_queue_event(const task_event& evt);
	void process_command_queue_event(const command_event& evt);

	void scheduling_loop();
	void thread_main();
};

scheduler_impl::scheduler_impl(const bool start_thread, const size_t num_nodes, const node_id local_node_id, const system_info& system,
    scheduler::delegate* const dlg, command_recorder* const crec, instruction_recorder* const irec, const scheduler::policy_set& policy)
    : dlg(dlg), cdag(), crec(crec), cggen(num_nodes, local_node_id, cdag, crec, policy.command_graph_generator), idag(), irec(irec),
      iggen(num_nodes, local_node_id, system, idag, dlg, irec, policy.instruction_graph_generator) {
	if(start_thread) { thread = std::thread(&scheduler_impl::thread_main, this); }
}

scheduler_impl::~scheduler_impl() {
	// schedule() will exit as soon as it has processed the shutdown epoch
	if(thread.joinable()) { thread.join(); }
}

void scheduler_impl::process_task_queue_event(const task_event& evt) {
	matchbox::match(
	    evt,
	    [&](const event_task_available& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    assert(e.tsk != nullptr);
		    auto& tsk = *e.tsk;

		    const auto commands = [&] {
			    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::build_task", scheduler_build_task, "T{} build", tsk.get_id());
			    CELERITY_DETAIL_TRACY_ZONE_TEXT(utils::make_task_debug_label(tsk.get_type(), tsk.get_id(), tsk.get_debug_name()));
			    return cggen.build_task(tsk, active_loop_template);
		    }(); // IIFE

		    for(const auto cmd : commands) {
			    // If there are multiple commands, the shutdown epoch must come last. m_iggen.delegate must be considered dangling after receiving
			    // the corresponding instruction, as runtime will begin destroying the executor after it has observed the epoch to be reached.
			    assert(!shutdown_epoch_created);
			    if(tsk.get_type() == task_type::epoch && tsk.get_epoch_action() == epoch_action::shutdown) { shutdown_epoch_created = tsk.get_id(); }

			    std::optional<instruction_graph_generator::scheduling_hint> hint;
			    // NOCOMMIT TODO: No need to anticipate when we're currently in an active loop template
			    if(lookahead != experimental::lookahead::none) { hint = iggen.anticipate(*cmd); }
			    command_queue.push(event_command_available{.cmd = cmd, .hint = hint, .templ = active_loop_template, .part_of_tid = tsk.get_id()});
		    }
	    },
	    [&](const event_buffer_created& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_created", scheduler_buffer_created, "B{} create", e.bid);
		    cggen.notify_buffer_created(e.bid, e.range, e.user_allocation_id != null_allocation_id);
		    // Buffer creation must be applied immediately (and out-of-order when necessary) so that instruction_graph_generator::anticipate() does not operate
		    // on unknown buffers. This is fine as buffer creation never has dependencies on other commands and we do not re-use buffer ids.
		    iggen.notify_buffer_created(e.bid, e.range, e.elem_size, e.elem_align, e.user_allocation_id);
	    },
	    [&](const event_buffer_debug_name_changed& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_name_changed", scheduler_buffer_name_changed, "B{} set name", e.bid);
		    cggen.notify_buffer_debug_name_changed(e.bid, e.debug_name);
		    // buffer-name changes are enqueued in-order to ensure that instruction records have the buffer names as they existed at task creation time.
		    command_queue.push(e);
	    },
	    [&](const event_buffer_destroyed& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_destroyed", scheduler_buffer_destroyed, "B{} destroy", e.bid);
		    cggen.notify_buffer_destroyed(e.bid);
		    // host-object destruction must happen in-order, otherwise iggen would need to compile commands on already-deleted buffers.
		    command_queue.push(e);
	    },
	    [&](const event_host_object_created& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_created", scheduler_host_object_created, "H{} create", e.hoid);
		    cggen.notify_host_object_created(e.hoid);
		    // instruction_graph_generator::anticipate() does not examine host objects (unlike it does with buffers), but it doesn't hurt to create them early
		    // either since we don't re-use host object ids.
		    iggen.notify_host_object_created(e.hoid, e.owns_instance);
	    },
	    [&](const event_host_object_destroyed& e) {
		    assert(!shutdown_epoch_created && !shutdown_epoch_reached);
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_destroyed", scheduler_host_object_destroyed, "H{} destroy", e.hoid);
		    cggen.notify_host_object_destroyed(e.hoid);
		    // host-object destruction must happen in-order, otherwise iggen would need to compile commands on already-deleted host objects.
		    command_queue.push(e);
	    },
	    [&](const event_epoch_reached& e) { //
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED("scheduler::prune", scheduler_prune);
		    cdag.erase_before_epoch(e.tid);
		    idag.erase_before_epoch(e.tid);

		    // The scheduler will receive the shutdown-epoch completion event via the runtime even if executor destruction has already begun.
		    assert(!shutdown_epoch_reached);
		    if(shutdown_epoch_created && e.tid == *shutdown_epoch_created) { shutdown_epoch_reached = true; }
	    },
	    [&](const event_set_lookahead& e) { //
		    command_queue.push(e);
	    },
	    [&](const event_flush_commands& e) { //
		    command_queue.push(e);
	    },
	    [&](const event_enable_loop_template& e) { active_loop_template = e.templ; },
	    [&](const event_complete_loop_iteration& e) {
		    assert(active_loop_template != nullptr);
		    active_loop_template->cdag.complete_iteration();
		    command_queue.push(event_complete_loop_iteration{.templ = active_loop_template});
	    },
	    [&](const event_finalize_loop_template& e) {
		    cggen.finalize_loop_template(*e.templ);
		    active_loop_template = nullptr;
		    command_queue.push(e);
	    },
	    [&](const event_leak_memory& e) { //
		    command_queue.push(e);
	    },
	    [&](const scheduler_testspy::event_inspect& e) { //
		    e.inspector({.cdag = &cdag, .idag = &idag, .lookahead = lookahead});
	    });
}

void scheduler_impl::process_command_queue_event(const command_event& evt) {
	matchbox::match(
	    evt, //
	    [&](const event_command_available& e) {
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::compile_command", scheduler_compile_command, "C{} compile", e.cmd->get_id());
		    CELERITY_DETAIL_TRACY_ZONE_TEXT("{}", print_command_type(*e.cmd));
		    if(irec != nullptr && (e.part_of_tid == 0 || e.part_of_tid > highest_seen_tid)) {
			    irec->record_task_boundary(e.part_of_tid);
			    highest_seen_tid = e.part_of_tid;
		    }
		    iggen.compile(*e.cmd, e.templ);
	    },
	    [&](const event_buffer_debug_name_changed& e) {
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_name_changed", scheduler_buffer_name_changed, "B{} set name", e.bid);
		    iggen.notify_buffer_debug_name_changed(e.bid, e.debug_name);
	    },
	    [&](const event_buffer_destroyed& e) {
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::buffer_destroyed", scheduler_buffer_destroyed, "B{} destroy", e.bid);
		    iggen.notify_buffer_destroyed(e.bid);
	    },
	    [&](const event_host_object_destroyed& e) {
		    CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("scheduler::host_object_destroyed", scheduler_host_object_destroyed, "H{} destroy", e.hoid);
		    iggen.notify_host_object_destroyed(e.hoid);
	    },
	    [&](const event_set_lookahead& e) {
		    // setting the lookahead must happen in the command queue, not task queue, to make sure all previous commands are flushed first
		    this->lookahead = e.lookahead;
	    },
	    [&](const event_flush_commands&) {
		    // no-op, but must still reside in command_queue to ensure a correct num_flushes_in_queue count
	    },
	    [&](const event_complete_loop_iteration& e) {
		    assert(e.templ != nullptr);
		    e.templ->idag.complete_iteration();
	    },
	    [&](const event_finalize_loop_template& e) { iggen.finalize_loop_template(*e.templ); },
	    [&](const event_leak_memory&) { //
		    iggen.leak_memory();
	    });
}

void scheduler_impl::scheduling_loop() {
	bool is_idle = true;
	while(!shutdown_epoch_reached) {
		if(dlg != nullptr && !is_idle && command_queue.empty() && task_queue.empty()) {
			dlg->on_scheduler_idle();
			is_idle = true;
		}
		const auto tsk_evt = task_queue.wait_and_pop();
		if(dlg != nullptr && is_idle) {
			dlg->on_scheduler_busy();
			is_idle = false;
		}
		process_task_queue_event(tsk_evt);
		while(command_queue.should_dequeue(lookahead)) {
			process_command_queue_event(command_queue.pop());
		}
	}
	assert(task_queue.empty());
	assert(command_queue.empty());
}

void scheduler_impl::thread_main() {
	name_and_pin_and_order_this_thread(named_threads::thread_type::scheduler);
	scheduling_loop();
}

} // namespace celerity::detail::scheduler_detail

using namespace celerity::detail::scheduler_detail;

namespace celerity::detail {

scheduler::scheduler(const size_t num_nodes, const node_id local_node_id, const system_info& system, delegate* const delegate, command_recorder* const crec,
    instruction_recorder* const irec, const policy_set& policy)
    : m_impl(std::make_unique<scheduler_impl>(true /* start_thread */, num_nodes, local_node_id, system, delegate, crec, irec, policy)) {}

scheduler::~scheduler() = default;

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

void scheduler::enable_loop_template(loop_template& templ) { m_impl->task_queue.push(event_enable_loop_template{&templ}); }

void scheduler::complete_loop_iteration() { m_impl->task_queue.push(event_complete_loop_iteration{}); }

void scheduler::finalize_loop_template(loop_template& templ) { m_impl->task_queue.push(event_finalize_loop_template{&templ}); }

void scheduler::leak_memory() { m_impl->task_queue.push(event_leak_memory{}); }

} // namespace celerity::detail


#define CELERITY_DETAIL_TAIL_INCLUDE
#include "testspy/scheduler_testspy.inl"
