#include "recorders.h"
#include "buffer_manager.h"
#include "task_manager.h"

#include <regex>

namespace celerity::detail {

// Naming

std::string get_buffer_name(const buffer_id bid, const buffer_manager* buff_man) {
	return buff_man != nullptr ? buff_man->get_buffer_info(bid).debug_name : "";
}

// Tasks

access_list build_access_list(const task& tsk, const buffer_manager* buff_man, const std::optional<subrange<3>> execution_range = {}) {
	access_list ret;
	const auto exec_range = execution_range.value_or(subrange<3>{tsk.get_global_offset(), tsk.get_global_size()});
	const auto& bam = tsk.get_buffer_access_map();
	for(const auto bid : bam.get_accessed_buffers()) {
		for(const auto mode : bam.get_access_modes(bid)) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), exec_range, tsk.get_global_size());
			ret.push_back({bid, get_buffer_name(bid, buff_man), mode, req});
		}
	}
	return ret;
}

reduction_list build_reduction_list(const task& tsk, const buffer_manager* buff_man) {
	reduction_list ret;
	for(const auto& reduction : tsk.get_reductions()) {
		ret.push_back({reduction.rid, reduction.bid, get_buffer_name(reduction.bid, buff_man), reduction.init_from_buffer});
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

task_record::task_record(const task& from, const buffer_manager* buff_mngr)
    : tid(from.get_id()), debug_name(utils::simplify_task_name(from.get_debug_name())), cgid(from.get_collective_group_id()), type(from.get_type()),
      geometry(from.get_geometry()), reductions(build_reduction_list(from, buff_mngr)), accesses(build_access_list(from, buff_mngr)),
      side_effect_map(from.get_side_effect_map()), dependencies(build_task_dependency_list(from)) {}

void task_recorder::record_task(const task& tsk) { //
	m_recorded_tasks.emplace_back(tsk, m_buff_mngr);
}

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

std::string get_cmd_buffer_name(const std::optional<buffer_id>& bid, const buffer_manager* buff_mngr) {
	if(buff_mngr == nullptr || !bid.has_value()) return "";
	return get_buffer_name(bid.value(), buff_mngr);
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

const task* get_task_for(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* task_cmd = dynamic_cast<const task_command*>(&cmd)) {
		if(task_mngr != nullptr) {
			assert(task_mngr->has_task(task_cmd->get_tid()));
			return task_mngr->get_task(task_cmd->get_tid());
		}
	}
	return nullptr;
}

std::optional<task_geometry> get_task_geometry(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return tsk->get_geometry();
	return {};
}

bool get_is_reduction_initializer(const abstract_command& cmd) {
	if(const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd)) return execution_cmd->is_reduction_initializer();
	return false;
}

access_list build_cmd_access_list(const abstract_command& cmd, const task_manager* task_mngr, const buffer_manager* buff_man) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) {
		const auto execution_range = get_execution_range(cmd).value_or(subrange<3>{tsk->get_global_offset(), tsk->get_global_size()});
		return build_access_list(*tsk, buff_man, execution_range);
	}
	return {};
}

reduction_list build_cmd_reduction_list(const abstract_command& cmd, const task_manager* task_mngr, const buffer_manager* buff_man) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return build_reduction_list(*tsk, buff_man);
	return {};
}

side_effect_map get_side_effects(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return tsk->get_side_effect_map();
	return {};
}

command_dependency_list build_command_dependency_list(const abstract_command& cmd) {
	command_dependency_list ret;
	for(const auto& dep : cmd.get_dependencies()) {
		ret.push_back({dep.node->get_cid(), dep.kind, dep.origin});
	}
	return ret;
}

std::string get_task_name(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return utils::simplify_task_name(tsk->get_debug_name());
	return {};
}

std::optional<task_type> get_task_type(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return tsk->get_type();
	return {};
}

std::optional<collective_group_id> get_collective_group_id(const abstract_command& cmd, const task_manager* task_mngr) {
	if(const auto* tsk = get_task_for(cmd, task_mngr)) return tsk->get_collective_group_id();
	return {};
}

command_record::command_record(const abstract_command& cmd, const task_manager* task_mngr, const buffer_manager* buff_mngr)
    : cid(cmd.get_cid()), type(get_command_type(cmd)), epoch_action(get_epoch_action(cmd)), execution_range(get_execution_range(cmd)),
      reduction_id(get_reduction_id(cmd)), buffer_id(get_buffer_id(cmd)), buffer_name(get_cmd_buffer_name(buffer_id, buff_mngr)), target(get_target(cmd)),
      await_region(get_await_region(cmd)), push_range(get_push_range(cmd)), transfer_id(get_transfer_id(cmd)), task_id(get_task_id(cmd)),
      task_geometry(get_task_geometry(cmd, task_mngr)), is_reduction_initializer(get_is_reduction_initializer(cmd)),
      accesses(build_cmd_access_list(cmd, task_mngr, buff_mngr)), reductions(build_cmd_reduction_list(cmd, task_mngr, buff_mngr)),
      side_effects(get_side_effects(cmd, task_mngr)), dependencies(build_command_dependency_list(cmd)), task_name(get_task_name(cmd, task_mngr)),
      task_type(get_task_type(cmd, task_mngr)), collective_group_id(get_collective_group_id(cmd, task_mngr)) {}

void command_recorder::record_command(const abstract_command& com) { //
	m_recorded_commands.emplace_back(com, m_task_mngr, m_buff_mngr);
}

} // namespace celerity::detail
