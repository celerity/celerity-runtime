#include "graph_serializer.h"

#include <cassert>

#include "command.h"
#include "command_graph.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	void graph_serializer::flush(task_id tid) { flush(cdag.task_commands(tid)); }

	bool is_virtual_dependency(const abstract_command* const cmd) {
		// The initial epoch command is not flushed, so including it in dependencies is not useful
		// TODO we might want to generate and flush init tasks explicitly to avoid this kind of special casing
		const auto ecmd = dynamic_cast<const epoch_command*>(cmd);
		return ecmd && ecmd->get_tid() == task_manager::initial_epoch_task;
	}

	void graph_serializer::flush(const std::vector<task_command*>& cmds) {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		task_id check_tid = task_id(-1);
#endif

		std::vector<std::pair<task_command*, std::vector<command_id>>> cmds_and_deps;
		cmds_and_deps.reserve(cmds.size());
		for(auto cmd : cmds) {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			// Verify that all commands belong to the same task
			assert(check_tid == task_id(-1) || check_tid == cmd->get_tid());
			check_tid = cmd->get_tid();
#endif

			cmds_and_deps.emplace_back();
			auto& cad = *cmds_and_deps.rbegin();
			cad.first = cmd;

			// Iterate over first level of dependencies.
			// These might either be commands from other tasks that have been flushed previously or generated data transfer / reduction commands.
			for(auto d : cmd->get_dependencies()) {
				if(!is_virtual_dependency(d.node)) { cad.second.push_back(d.node->get_cid()); }

				// Sanity check: All dependencies must be on the same node.
				assert(d.node->get_nid() == cmd->get_nid());

				if(auto* tcmd = dynamic_cast<task_command*>(d.node)) {
					// Task command dependencies must be from a different task and have already been flushed.
					assert(tcmd->get_tid() != cmd->get_tid());
					assert(tcmd->is_flushed());
					continue;
				}

				// Flush dependency right away
				if(!d.node->is_flushed()) flush_dependency(d.node);
			}
		}

		// Finally, flush all the task commands.
		for(auto& cad : cmds_and_deps) {
			serialize_and_flush(cad.first, cad.second);
		}
	}

	void graph_serializer::flush_dependency(abstract_command* dep) const {
		// Special casing for AWAIT_PUSH commands: Also flush the corresponding PUSH.
		// This is necessary as we would otherwise not reach it when starting from task commands alone
		// (unless there exists an anti-dependency, which is not true in most cases).
		if(isa<await_push_command>(dep)) {
			const auto pcmd = static_cast<await_push_command*>(dep)->get_source();
			if(!pcmd->is_flushed()) flush_dependency(pcmd);
		}

		std::vector<command_id> dep_deps;
		// Iterate over second level of dependencies. These will usually be flushed already. One notable exception are reduction dependencies, which generate
		// a tree of push_await_commands and reduction_commands as a dependency.
		// TODO: We could probably do some pruning here (e.g. omit tasks we know are already finished)
		for(auto dd : dep->get_dependencies()) {
			if(!dd.node->is_flushed()) {
				assert(isa<reduction_command>(dep) && isa<await_push_command>(dd.node));
				flush_dependency(dd.node);
			}
			if(!is_virtual_dependency(dd.node)) { dep_deps.push_back(dd.node->get_cid()); }
		}
		serialize_and_flush(dep, dep_deps);
	}

	void graph_serializer::serialize_and_flush(abstract_command* cmd, const std::vector<command_id>& dependencies) const {
		assert(!cmd->is_flushed() && "Command has already been flushed.");

		command_pkg pkg;
		pkg.cid = cmd->get_cid();
		if(const auto* ecmd = dynamic_cast<epoch_command*>(cmd)) {
			pkg.data = epoch_data{ecmd->get_tid(), ecmd->get_epoch_action()};
		} else if(const auto* xcmd = dynamic_cast<execution_command*>(cmd)) {
			pkg.data = execution_data{xcmd->get_tid(), xcmd->get_execution_range(), xcmd->is_reduction_initializer()};
		} else if(const auto* pcmd = dynamic_cast<push_command*>(cmd)) {
			pkg.data = push_data{pcmd->get_bid(), pcmd->get_rid(), pcmd->get_target(), pcmd->get_range()};
		} else if(const auto* apcmd = dynamic_cast<await_push_command*>(cmd)) {
			auto* source = apcmd->get_source();
			pkg.data = await_push_data{source->get_bid(), source->get_rid(), source->get_nid(), source->get_cid(), source->get_range()};
		} else if(const auto* rcmd = dynamic_cast<reduction_command*>(cmd)) {
			pkg.data = reduction_data{rcmd->get_rid()};
		} else if(const auto* hcmd = dynamic_cast<horizon_command*>(cmd)) {
			pkg.data = horizon_data{hcmd->get_tid()};
		} else {
			assert(false && "Unknown command");
		}

		flush_cb(cmd->get_nid(), pkg, dependencies);
		cmd->mark_as_flushed();
	}

} // namespace detail
} // namespace celerity
