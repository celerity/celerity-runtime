#pragma once

#include "scheduler.h"

#include <functional>
#include <future>
#include <type_traits>


namespace celerity::detail {

struct scheduler_testspy {
	struct scheduler_state {
		const command_graph* cdag = nullptr;
		const instruction_graph* idag = nullptr;
		experimental::lookahead lookahead = experimental::lookahead::automatic;
	};

	struct event_inspect {
		/// executed inside scheduler thread, making it safe to access scheduler members
		std::function<void(const scheduler_state&)> inspector;
	};

	static scheduler make_threadless_scheduler(size_t num_nodes, node_id local_node_id, const system_info& system_info, scheduler::delegate* delegate,
	    command_recorder* crec, instruction_recorder* irec, const scheduler::policy_set& policy = {});

	static void run_scheduling_loop(scheduler& schdlr);

	static void begin_inspect_thread(scheduler& schdlr, event_inspect inspector);

	template <typename F>
	static auto inspect_thread(scheduler& schdlr, F&& f) {
		using return_t = std::invoke_result_t<F, const scheduler_state&>;
		std::promise<return_t> channel;
		begin_inspect_thread(schdlr, event_inspect{[&](const scheduler_state& state) {
			if constexpr(std::is_void_v<return_t>) {
				f(state), channel.set_value();
			} else {
				channel.set_value(f(state));
			}
		}});
		return channel.get_future().get();
	}
};

} // namespace celerity::detail
