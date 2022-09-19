#include "graph_serializer.h"

#include <cassert>

#include "command.h"
#include "command_graph.h"
#include "task_manager.h"

namespace celerity::detail {

graph_serializer::graph_serializer(const size_t num_nodes, command_graph& cdag, flush_callback flush_cb)
    : m_num_nodes(num_nodes), m_cdag(cdag), m_flush_cb(std::move(flush_cb)) {}

void graph_serializer::flush(const task_id tid) {
	node_command_map pending_node_cmds(m_num_nodes);
	for(const auto tcmd : m_cdag.task_commands(tid)) {
		assert(tcmd->get_tid() == tid);
		collect_task_command(tcmd, pending_node_cmds);
	}
	for(node_id nid = 0; nid < m_num_nodes; ++nid) {
		if(!pending_node_cmds[nid].empty()) { m_flush_cb(nid, serialize(pending_node_cmds[nid])); }
	}
}

void graph_serializer::collect_task_command(task_command* const cmd, node_command_map& pending_node_cmds) const {
	cmd->mark_as_flushed();

	const auto nid = cmd->get_nid();
	assert(nid <= m_num_nodes);

	// Iterate over first level of dependencies.
	// These might either be commands from other tasks that have been flushed previously or generated data transfer / reduction commands.
	for(const auto& edge : cmd->get_dependencies()) {
		const auto dep = edge.node;

		// Sanity check: All dependencies must be on the same node.
		assert(dep->get_nid() == nid);

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		// Task command dependencies must be from a different task and have already been flushed.
		if(auto* tdep = dynamic_cast<task_command*>(dep)) {
			assert(tdep->get_tid() != cmd->get_tid());
			assert(tdep->is_flushed());
		}
#endif

		if(!dep->is_flushed()) collect_dependency(dep, pending_node_cmds);
	}

	pending_node_cmds[nid].emplace_back(cmd);
}

void graph_serializer::collect_dependency(abstract_command* const cmd, node_command_map& pending_node_cmds) const {
	cmd->mark_as_flushed();

	const auto nid = cmd->get_nid();
	assert(nid <= m_num_nodes);

	// Special casing for await_push commands: Also flush the corresponding push.
	// This is necessary as we would otherwise not reach it when starting from task commands alone
	// (unless there exists an anti-dependency, which is not true in most cases).
	if(isa<await_push_command>(cmd)) {
		const auto pcmd = static_cast<await_push_command*>(cmd)->get_source();
		if(!pcmd->is_flushed()) collect_dependency(pcmd, pending_node_cmds);
	}

	// Iterate over second level of dependencies. These will usually be flushed already. One notable exception are reduction dependencies, which generate
	// a tree of push_await_commands and reduction_commands as a dependency.
	// TODO: We could probably do some pruning here (e.g. omit tasks we know are already finished)
	for(const auto& edge : cmd->get_dependencies()) {
		const auto dep = edge.node;

		// Sanity check: All dependencies must be on the same node.
		assert(dep->get_nid() == nid);

		if(!dep->is_flushed()) {
			assert(isa<reduction_command>(cmd) || isa<await_push_command>(dep));
			collect_dependency(dep, pending_node_cmds);
		}
	}

	pending_node_cmds[nid].emplace_back(cmd);
}

frame_vector<command_frame> graph_serializer::serialize(const command_vector& cmds) const {
	frame_vector_layout<command_frame> layout;
	for(const auto cmd : cmds) {
		const auto num_deps = std::distance(cmd->get_dependencies().begin(), cmd->get_dependencies().end());
		layout.reserve_back(from_payload_count, num_deps);
	}

	frame_vector_builder<command_frame> builder(layout);
	for(const auto cmd : cmds) {
		const auto num_deps = std::distance(cmd->get_dependencies().begin(), cmd->get_dependencies().end());
		auto& frame = builder.emplace_back(from_payload_count, num_deps);

		frame.pkg.cid = cmd->get_cid();
		if(const auto* ecmd = dynamic_cast<const epoch_command*>(cmd)) {
			frame.pkg.data = epoch_data{ecmd->get_tid(), ecmd->get_epoch_action()};
		} else if(const auto* xcmd = dynamic_cast<const execution_command*>(cmd)) {
			frame.pkg.data = execution_data{xcmd->get_tid(), xcmd->get_execution_range(), xcmd->is_reduction_initializer()};
		} else if(const auto* pcmd = dynamic_cast<const push_command*>(cmd)) {
			frame.pkg.data = push_data{pcmd->get_bid(), pcmd->get_rid(), pcmd->get_target(), pcmd->get_range()};
		} else if(const auto* apcmd = dynamic_cast<const await_push_command*>(cmd)) {
			auto* source = apcmd->get_source();
			frame.pkg.data = await_push_data{source->get_bid(), source->get_rid(), source->get_nid(), source->get_cid(), source->get_range()};
		} else if(const auto* rcmd = dynamic_cast<const reduction_command*>(cmd)) {
			frame.pkg.data = reduction_data{rcmd->get_reduction_info().rid};
		} else if(const auto* hcmd = dynamic_cast<const horizon_command*>(cmd)) {
			frame.pkg.data = horizon_data{hcmd->get_tid()};
		} else {
			assert(false && "Unknown command");
		}

		frame.num_dependencies = num_deps;
		std::transform(cmd->get_dependencies().begin(), cmd->get_dependencies().end(), frame.dependencies,
		    [](const abstract_command::dependency& edge) { return edge.node->get_cid(); });
	}

	return std::move(builder).into_vector();
}

} // namespace celerity::detail
