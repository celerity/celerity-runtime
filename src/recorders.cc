#include "recorders.h"
#include "command.h"
#include "task_manager.h"

namespace celerity::detail {

// Naming

std::string get_buffer_name(const buffer_id bid, const buffer_name_map& accessed_buffer_names) {
	if(const auto it = accessed_buffer_names.find(bid); it != accessed_buffer_names.end()) { return it->second; }
	return {};
}

// Tasks

access_list build_access_list(const task& tsk, const buffer_name_map& accessed_buffer_names, const std::optional<subrange<3>> execution_range = {}) {
	access_list ret;
	const auto exec_range = execution_range.value_or(subrange<3>{tsk.get_global_offset(), tsk.get_global_size()});
	const auto& bam = tsk.get_buffer_access_map();
	for(const auto bid : bam.get_accessed_buffers()) {
		for(const auto mode : bam.get_access_modes(bid)) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), exec_range, tsk.get_global_size());
			ret.push_back({bid, get_buffer_name(bid, accessed_buffer_names), mode, req});
		}
	}
	return ret;
}

reduction_list build_reduction_list(const task& tsk, const buffer_name_map& accessed_buffer_names) {
	reduction_list ret;
	for(const auto& reduction : tsk.get_reductions()) {
		ret.push_back({reduction.rid, reduction.bid, get_buffer_name(reduction.bid, accessed_buffer_names), reduction.init_from_buffer});
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

task_record::task_record(const task& tsk, const buffer_name_map& accessed_buffer_names)
    : tid(tsk.get_id()), debug_name(tsk.get_debug_name()), cgid(tsk.get_collective_group_id()), type(tsk.get_type()), geometry(tsk.get_geometry()),
      reductions(build_reduction_list(tsk, accessed_buffer_names)), accesses(build_access_list(tsk, accessed_buffer_names)),
      side_effect_map(tsk.get_side_effect_map()), dependencies(build_task_dependency_list(tsk)) {}

// Commands

command_type get_command_type(const abstract_command& cmd) {
	if(utils::isa<epoch_command>(&cmd)) return command_type::epoch;
	if(utils::isa<horizon_command>(&cmd)) return command_type::horizon;
	if(utils::isa<execution_command>(&cmd)) return command_type::execution;
	if(utils::isa<push_command>(&cmd)) return command_type::push;
	if(utils::isa<await_push_command>(&cmd)) return command_type::await_push;
	if(utils::isa<reduction_command>(&cmd)) return command_type::reduction;
	if(utils::isa<fence_command>(&cmd)) return command_type::fence;
	utils::panic("Unexpected command type");
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
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id().rid;
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id().rid;
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().rid;
	return {};
}

std::optional<buffer_id> get_buffer_id(const abstract_command& cmd) {
	if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id().bid;
	if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id().bid;
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().bid;
	return {};
}

std::string get_cmd_buffer_name(const std::optional<buffer_id>& bid, const buffer_name_map& accessed_buffer_names) {
	if(bid.has_value()) return get_buffer_name(*bid, accessed_buffer_names);
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

access_list build_cmd_access_list(const abstract_command& cmd, const task& tsk, const buffer_name_map& accessed_buffer_names) {
	const auto execution_range_a = get_execution_range(cmd);
	const auto execution_range_b = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
	const auto execution_range = execution_range_a.value_or(execution_range_b);
	return build_access_list(tsk, accessed_buffer_names, execution_range);
}

command_dependency_list build_command_dependency_list(const abstract_command& cmd) {
	command_dependency_list ret;
	for(const auto& dep : cmd.get_dependencies()) {
		ret.push_back({dep.node->get_cid(), dep.kind, dep.origin});
	}
	return ret;
}

std::string get_task_name(const task& tsk) { return tsk.get_debug_name(); } // TODO remove?

command_record::command_record(const abstract_command& cmd, const task& tsk, const buffer_name_map& accessed_buffer_names)
    : cid(cmd.get_cid()), type(get_command_type(cmd)), epoch_action(get_epoch_action(cmd)), execution_range(get_execution_range(cmd)),
      reduction_id(get_reduction_id(cmd)), buffer_id(get_buffer_id(cmd)), buffer_name(get_cmd_buffer_name(buffer_id, accessed_buffer_names)),
      target(get_target(cmd)), await_region(get_await_region(cmd)), push_range(get_push_range(cmd)), transfer_id(get_transfer_id(cmd)),
      task_id(get_task_id(cmd)), task_geometry(tsk.get_geometry()), is_reduction_initializer(get_is_reduction_initializer(cmd)),
      accesses(build_cmd_access_list(cmd, tsk, accessed_buffer_names)), reductions(build_reduction_list(tsk, accessed_buffer_names)),
      side_effects(tsk.get_side_effect_map()), dependencies(build_command_dependency_list(cmd)), task_name(get_task_name(tsk)), task_type(tsk.get_type()),
      collective_group_id(tsk.get_collective_group_id()) {}

} // namespace celerity::detail
