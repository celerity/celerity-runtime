#include "recorders.h"
#include "task.h"

namespace celerity::detail {

// Tasks

access_list build_access_list(const buffer_access_map& bam, int dims, const range<3>& global_size, const subrange<3>& exec_range) {
	access_list ret;
	for(const auto bid : bam.get_accessed_buffers()) {
		for(const auto mode : bam.get_access_modes(bid)) {
			const auto req = bam.get_mode_requirements(bid, mode, dims, exec_range, global_size);
			ret.push_back({bid, mode, req});
		}
	}
	return ret;
}

access_list build_access_list(const task& tsk) {
	return build_access_list(
	    tsk.get_buffer_access_map(), tsk.get_dimensions(), tsk.get_global_size(), subrange<3>{tsk.get_global_offset(), tsk.get_global_size()});
}

reduction_list build_reduction_list(const task& tsk) {
	reduction_list ret;
	for(const auto& reduction : tsk.get_reductions()) {
		ret.push_back({reduction.rid, reduction.bid, reduction.init_from_buffer});
	}
	return ret;
}

task_dependency_list build_task_dependency_list(const task& tsk) {
	task_dependency_list ret;
	for(const auto& dep : tsk.get_dependencies()) {
		ret.push_back({dep.node->get_id(), dep.kind, dep.origin});
	}
	return ret;
}

task_record::task_record(const task& from)
    : tid(from.get_id()), debug_name(utils::simplify_task_name(from.get_debug_name())), cgid(from.get_collective_group_id()), type(from.get_type()),
      geometry(from.get_geometry()), reductions(build_reduction_list(from)), accesses(build_access_list(from)), side_effect_map(from.get_side_effect_map()),
      dependencies(build_task_dependency_list(from)) {}

// Commands

command_type get_command_type(const abstract_command& cmd) {
	if(utils::isa<epoch_command>(&cmd)) return command_type::epoch;
	if(utils::isa<horizon_command>(&cmd)) return command_type::horizon;
	if(utils::isa<execution_command>(&cmd)) return command_type::execution;
	if(utils::isa<push_command>(&cmd)) return command_type::push;
	if(utils::isa<await_push_command>(&cmd)) return command_type::await_push;
	if(utils::isa<reduction_command>(&cmd)) return command_type::reduction;
	if(utils::isa<fence_command>(&cmd)) return command_type::fence;
	CELERITY_CRITICAL("Unexpected command type");
	std::terminate();
}

std::optional<epoch_action> get_epoch_action(const abstract_command& cmd) {
	const auto* epoch_cmd = dynamic_cast<const epoch_command*>(&cmd);
	return epoch_cmd != nullptr ? epoch_cmd->get_epoch_action() : std::optional<epoch_action>{};
}

std::optional<subrange<3>> get_execution_range(const abstract_command& cmd) {
	const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd);
	return execution_cmd != nullptr ? execution_cmd->get_execution_range() : std::optional<subrange<3>>{};
}

std::optional<reduction_id> get_reduction_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_reduction_id();
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_reduction_id();
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().rid;
	return {};
}

std::optional<buffer_id> get_buffer_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_bid();
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_bid();
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().bid;
	return {};
}

std::optional<node_id> get_target(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_target();
	return {};
}

std::optional<region<3>> get_await_region(const abstract_command& cmd) {
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_region();
	return {};
}

std::optional<subrange<3>> get_push_range(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_range();
	return {};
}

std::optional<transfer_id> get_transfer_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id();
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id();
	return {};
}

std::optional<task_id> get_task_id(const abstract_command& cmd) {
	if(const auto* task_cmd = dynamic_cast<const task_command*>(&cmd)) return task_cmd->get_tid();
	return {};
}

bool get_is_reduction_initializer(const abstract_command& cmd) {
	if(const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd)) return execution_cmd->is_reduction_initializer();
	return false;
}

access_list build_cmd_access_list(const abstract_command& cmd, const task* tsk) {
	if(tsk != nullptr) {
		const auto execution_range = get_execution_range(cmd).value_or(subrange<3>{tsk->get_global_offset(), tsk->get_global_size()});
		return build_access_list(tsk->get_buffer_access_map(), tsk->get_dimensions(), tsk->get_global_size(), execution_range);
	}
	return {};
}

command_dependency_list build_command_dependency_list(const abstract_command& cmd) {
	command_dependency_list ret;
	for(const auto& dep : cmd.get_dependencies()) {
		ret.push_back({dep.node->get_cid(), dep.kind, dep.origin});
	}
	return ret;
}

command_record::command_record(const abstract_command& cmd, const task* tsk)
    : cid(cmd.get_cid()), type(get_command_type(cmd)), epoch_action(get_epoch_action(cmd)), execution_range(get_execution_range(cmd)),
      reduction_id(get_reduction_id(cmd)), buffer_id(get_buffer_id(cmd)), target(get_target(cmd)), await_region(get_await_region(cmd)),
      push_range(get_push_range(cmd)), transfer_id(get_transfer_id(cmd)), task_id(get_task_id(cmd)),
      is_reduction_initializer(get_is_reduction_initializer(cmd)), accesses(build_cmd_access_list(cmd, tsk)), dependencies(build_command_dependency_list(cmd)) {
}

} // namespace celerity::detail
