#include "graph_serializer.h"

#include <cassert>

#include "command.h"
#include "command_graph.h"
#include "logger.h"

namespace celerity {
namespace detail {

	void graph_serializer::flush(task_id tid) {
		const auto cmd_range = cdag.task_commands(tid);
		const std::vector<task_command*> cmds(cmd_range.begin(), cmd_range.end());
		flush(cmds);
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
			// These might either be data transfer commands, or task commands from other tasks.
			for(auto d : cmd->get_dependencies()) {
				if(isa<nop_command>(d.node)) continue;
				cad.second.push_back(d.node->get_cid());

				// Sanity check: All dependencies must be on the same node.
				assert(d.node->get_nid() == cmd->get_nid());

				if(isa<task_command>(d.node)) {
					// Task command dependencies must be from a different task and have already been flushed.
					assert(static_cast<task_command*>(d.node)->get_tid() != cmd->get_tid());
					assert(static_cast<task_command*>(d.node)->is_flushed());
					continue;
				}

				// Flush dependency right away
				if(!d.node->is_flushed()) flush_dependency(d.node);

				// Special casing for AWAIT_PUSH commands: Also flush the corresponding PUSH.
				// This is necessary as we would otherwise not reach it when starting from task commands alone
				// (unless there exists an anti-dependency, which is not true in most cases).
				if(isa<await_push_command>(d.node)) {
					const auto pcmd = static_cast<await_push_command*>(d.node)->get_source();
					if(!pcmd->is_flushed()) flush_dependency(pcmd);
				}
			}
		}

		// Finally, flush all the task commands.
		for(auto& cad : cmds_and_deps) {
			serialize_and_flush(cad.first, cad.second);
		}
	}


	void graph_serializer::flush_horizons() {
		auto& horizon_cmds = cdag.get_active_horizons();
		for(auto& horizon_cmd : horizon_cmds) {
			flush_dependency(horizon_cmd);
		}
		horizon_cmds.clear();
	}

	void graph_serializer::flush_dependency(abstract_command* dep) const {
		std::vector<command_id> dep_deps;
		// Iterate over second level of dependencies. All of these must already have been flushed.
		// TODO: We could probably do some pruning here (e.g. omit tasks we know are already finished)
		for(auto dd : dep->get_dependencies()) {
			assert(dd.node->is_flushed());
			dep_deps.push_back(dd.node->get_cid());
		}
		serialize_and_flush(dep, dep_deps);
	}

	void graph_serializer::serialize_and_flush(abstract_command* cmd, const std::vector<command_id>& dependencies) const {
		assert(!cmd->is_flushed() && "Command has already been flushed.");

		if(isa<nop_command>(cmd)) return;
		command_pkg pkg{cmd->get_cid(), command_type::NOP, command_data{nop_data{}}};

		if(const auto* tcmd = dynamic_cast<task_command*>(cmd)) {
			pkg.cmd = command_type::TASK;
			pkg.data = task_data{tcmd->get_tid(), tcmd->get_execution_range()};
		} else if(const auto* pcmd = dynamic_cast<push_command*>(cmd)) {
			pkg.cmd = command_type::PUSH;
			pkg.data = push_data{pcmd->get_bid(), pcmd->get_target(), pcmd->get_range()};
		} else if(const auto* apcmd = dynamic_cast<await_push_command*>(cmd)) {
			pkg.cmd = command_type::AWAIT_PUSH;
			pkg.data = await_push_data{
			    apcmd->get_source()->get_bid(), apcmd->get_source()->get_nid(), apcmd->get_source()->get_cid(), apcmd->get_source()->get_range()};
		} else if(dynamic_cast<horizon_command*>(cmd)) {
			pkg.cmd = command_type::HORIZON;
		} else {
			assert(false && "Unknown command");
		}

		flush_cb(cmd->get_nid(), pkg, dependencies);
		cmd->mark_as_flushed();
	}

} // namespace detail
} // namespace celerity
