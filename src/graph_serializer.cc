#include "graph_serializer.h"

#include <cassert>

#include "command.h"
#include "command_graph.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	bool is_virtual_dependency(const abstract_command* const cmd) {
		// The initial epoch command is not flushed, so including it in dependencies is not useful
		// TODO we might want to generate and flush init tasks explicitly to avoid this kind of special casing
		const auto ecmd = dynamic_cast<const epoch_command*>(cmd);
		return ecmd && ecmd->get_tid() == task_manager::initial_epoch_task;
	}

	void graph_serializer::flush(const std::unordered_set<abstract_command*>& cmds) {
		[[maybe_unused]] task_id check_tid = task_id(-1);

		// Separate push commands from task commands. We flush pushes first to avoid deadlocking the executor.
		// This is always safe as no other unflushed command within a single task can precede a push.
		std::vector<abstract_command*> push_cmds;
		push_cmds.reserve(cmds.size() / 2);
		std::vector<abstract_command*> task_cmds;
		task_cmds.reserve(cmds.size() / 2); // Somewhat overzealous, we are likely to have more push commands

		for(const auto& cmd : cmds) {
			if(isa<push_command>(cmd)) {
				push_cmds.push_back(cmd);
			} else if(isa<task_command>(cmd)) {
				task_cmds.push_back(cmd);
			}
		}

		// Flush a command and all of its unflushed predecessors, recursively. Usually this will only require one level of recursion.
		// One notable exception are reductions, which generate a tree of await push commands and reduction commands as successors.
		[[maybe_unused]] size_t flush_count = 0;
		const auto flush_recursive = [this, &check_tid, &flush_count](abstract_command* cmd, auto recurse) -> void {
			(void)check_tid;
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			if(isa<task_command>(cmd)) {
				// Verify that all commands belong to the same task
				assert(check_tid == task_id(-1) || check_tid == static_cast<task_command*>(cmd)->get_tid());
				check_tid = static_cast<task_command*>(cmd)->get_tid();
			}
#endif
			std::vector<command_id> deps;
			for(auto dep : cmd->get_dependencies()) {
				if(!dep.node->is_flushed()) { recurse(dep.node, recurse); }
				if(!is_virtual_dependency(dep.node)) { deps.push_back(dep.node->get_cid()); }
			}
			serialize_and_flush(cmd, std::move(deps));
			flush_count++;
		};

		for(const auto& cmds_ptr : {&push_cmds, &task_cmds}) {
			for(auto& cmd : *cmds_ptr) {
				flush_recursive(cmd, flush_recursive);
			}
		}

		assert(flush_count == cmds.size());
	}

	void graph_serializer::serialize_and_flush(abstract_command* cmd, std::vector<command_id>&& dependencies) const {
		assert(!cmd->is_flushed() && "Command has already been flushed.");

		command_pkg pkg;
		pkg.dependencies = std::move(dependencies);

		pkg.cid = cmd->get_cid();
		if(const auto* ecmd = dynamic_cast<epoch_command*>(cmd)) {
			pkg.data = epoch_data{ecmd->get_tid(), ecmd->get_epoch_action()};
		} else if(const auto* xcmd = dynamic_cast<execution_command*>(cmd)) {
			pkg.data = execution_data{xcmd->get_tid(), xcmd->get_execution_range(), xcmd->is_reduction_initializer()};
		} else if(const auto* pcmd = dynamic_cast<push_command*>(cmd)) {
			pkg.data = push_data{pcmd->get_bid(), pcmd->get_reduction_id(), pcmd->get_target(), pcmd->get_transfer_id(), pcmd->get_range()};
		} else if(const auto* apcmd = dynamic_cast<await_push_command*>(cmd)) {
			pkg.data = await_push_data{apcmd->get_bid(), apcmd->get_reduction_id(), apcmd->get_transfer_id(), apcmd->get_region()};
		} else if(const auto* rcmd = dynamic_cast<reduction_command*>(cmd)) {
			pkg.data = reduction_data{rcmd->get_reduction_info().rid};
		} else if(const auto* hcmd = dynamic_cast<horizon_command*>(cmd)) {
			pkg.data = horizon_data{hcmd->get_tid()};
		} else if(const auto* fcmd = dynamic_cast<fence_command*>(cmd)) {
			pkg.data = fence_data{fcmd->get_tid()};
		} else {
			assert(false && "Unknown command");
		}

		m_flush_cb(std::move(pkg));
		cmd->mark_as_flushed();
	}

} // namespace detail
} // namespace celerity
