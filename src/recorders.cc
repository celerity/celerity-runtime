#include "recorders.h"
#include "command.h"

namespace celerity::detail {

// Tasks

access_list build_access_list(const task& tsk, const buffer_name_map& get_buffer_debug_name, const std::optional<subrange<3>> execution_range = {}) {
	access_list ret;
	const auto exec_range = execution_range.value_or(subrange<3>{tsk.get_global_offset(), tsk.get_global_size()});
	const auto& bam = tsk.get_buffer_access_map();
	for(const auto bid : bam.get_accessed_buffers()) {
		for(const auto mode : bam.get_access_modes(bid)) {
			const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), exec_range, tsk.get_global_size());
			ret.push_back({bid, get_buffer_debug_name(bid), mode, req});
		}
	}
	return ret;
}

reduction_list build_reduction_list(const task& tsk, const buffer_name_map& get_buffer_debug_name) {
	reduction_list ret;
	for(const auto& reduction : tsk.get_reductions()) {
		ret.push_back({reduction.rid, reduction.bid, get_buffer_debug_name(reduction.bid), reduction.init_from_buffer});
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

task_record::task_record(const task& tsk, const buffer_name_map& get_buffer_debug_name)
    : tid(tsk.get_id()), debug_name(tsk.get_debug_name()), cgid(tsk.get_collective_group_id()), type(tsk.get_type()), geometry(tsk.get_geometry()),
      reductions(build_reduction_list(tsk, get_buffer_debug_name)), accesses(build_access_list(tsk, get_buffer_debug_name)),
      side_effect_map(tsk.get_side_effect_map()), dependencies(build_task_dependency_list(tsk)) {}

// Commands

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

std::string get_cmd_buffer_name(const std::optional<buffer_id>& bid, const buffer_name_map& get_buffer_debug_name) {
	if(bid.has_value()) return get_buffer_debug_name(*bid);
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

bool get_has_local_contribution(const abstract_command& cmd) {
	if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->has_local_contribution();
	return false;
}

std::vector<reduction_id> get_completed_reductions(const abstract_command& cmd) {
	return matchbox::match(
	    cmd,                                                                         //
	    [](const horizon_command& hcmd) { return hcmd.get_completed_reductions(); }, //
	    [](const epoch_command& ecmd) { return ecmd.get_completed_reductions(); },   //
	    [](const auto&) { return std::vector<reduction_id>{}; });
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

command_record::command_record(const abstract_command& cmd, const task& tsk, const buffer_name_map& get_buffer_debug_name)
    : cid(cmd.get_cid()), type(cmd.get_type()), epoch_action(get_epoch_action(cmd)), execution_range(get_execution_range(cmd)),
      reduction_id(get_reduction_id(cmd)), buffer_id(get_buffer_id(cmd)), buffer_name(get_cmd_buffer_name(buffer_id, get_buffer_debug_name)),
      target(get_target(cmd)), await_region(get_await_region(cmd)), push_range(get_push_range(cmd)), transfer_id(get_transfer_id(cmd)),
      task_id(get_task_id(cmd)), task_geometry(tsk.get_geometry()), is_reduction_initializer(get_is_reduction_initializer(cmd)),
      has_local_contribution(get_has_local_contribution(cmd)), accesses(build_cmd_access_list(cmd, tsk, get_buffer_debug_name)),
      reductions(build_reduction_list(tsk, get_buffer_debug_name)), side_effects(tsk.get_side_effect_map()), dependencies(build_command_dependency_list(cmd)),
      task_name(get_task_name(tsk)), task_type(tsk.get_type()), collective_group_id(tsk.get_collective_group_id()),
      completed_reductions(get_completed_reductions(cmd)) {}

// Instructions

instruction_record::instruction_record(const instruction& instr) : id(instr.get_id()), priority(instr.get_priority()) {}

clone_collective_group_instruction_record::clone_collective_group_instruction_record(const clone_collective_group_instruction& ccginstr)
    : acceptor_base(ccginstr), original_collective_group_id(ccginstr.get_original_collective_group_id()),
      new_collective_group_id(ccginstr.get_new_collective_group_id()) {}

alloc_instruction_record::alloc_instruction_record(
    const alloc_instruction& ainstr, const alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation, std::optional<size_t> num_chunks)
    : acceptor_base(ainstr), allocation_id(ainstr.get_allocation_id()), size_bytes(ainstr.get_size_bytes()), alignment_bytes(ainstr.get_alignment_bytes()),
      origin(origin), buffer_allocation(std::move(buffer_allocation)), num_chunks(num_chunks) {}

free_instruction_record::free_instruction_record(const free_instruction& finstr, const size_t size, std::optional<buffer_allocation_record> buffer_allocation)
    : acceptor_base(finstr), allocation_id(finstr.get_allocation_id()), size(size), buffer_allocation(std::move(buffer_allocation)) {}

copy_instruction_record::copy_instruction_record(
    const copy_instruction& cinstr, const copy_origin origin, const detail::buffer_id buffer_id, std::string buffer_name)
    : acceptor_base(cinstr), source_allocation_id(cinstr.get_source_allocation_id()), dest_allocation_id(cinstr.get_dest_allocation_id()),
      source_layout(cinstr.get_source_layout()), dest_layout(cinstr.get_dest_layout()), copy_region(cinstr.get_copy_region()),
      element_size(cinstr.get_element_size()), origin(origin), buffer_id(buffer_id), buffer_name(std::move(buffer_name)) {}

device_kernel_instruction_record::device_kernel_instruction_record(const device_kernel_instruction& dkinstr, const task_id cg_tid,
    const command_id execution_cid, const std::string& debug_name, const std::vector<buffer_memory_record>& buffer_memory_allocation_map,
    const std::vector<buffer_reduction_record>& buffer_memory_reduction_map)
    : acceptor_base(dkinstr), device_id(dkinstr.get_device_id()), execution_range(dkinstr.get_execution_range()),
      estimated_global_memory_traffic_bytes(dkinstr.get_estimated_global_memory_traffic_bytes()), command_group_task_id(cg_tid),
      execution_command_id(execution_cid), debug_name(debug_name) //
{
	assert(dkinstr.get_access_allocations().size() == buffer_memory_allocation_map.size());
	access_map.reserve(dkinstr.get_access_allocations().size());
	for(size_t i = 0; i < dkinstr.get_access_allocations().size(); ++i) {
		access_map.emplace_back(dkinstr.get_access_allocations()[i], buffer_memory_allocation_map[i]);
	}

	assert(dkinstr.get_reduction_allocations().size() == buffer_memory_reduction_map.size());
	reduction_map.reserve(dkinstr.get_reduction_allocations().size());
	for(size_t i = 0; i < dkinstr.get_reduction_allocations().size(); ++i) {
		reduction_map.emplace_back(dkinstr.get_reduction_allocations()[i], buffer_memory_reduction_map[i]);
	}
}

host_task_instruction_record::host_task_instruction_record(const host_task_instruction& htinstr, const task_id cg_tid, const command_id execution_cid,
    const std::string& debug_name, const std::vector<buffer_memory_record>& buffer_memory_allocation_map)
    : acceptor_base(htinstr), collective_group_id(htinstr.get_collective_group_id()), execution_range(htinstr.get_execution_range()),
      command_group_task_id(cg_tid), execution_command_id(execution_cid), debug_name(debug_name) //
{
	assert(htinstr.get_access_allocations().size() == buffer_memory_allocation_map.size());
	access_map.reserve(htinstr.get_access_allocations().size());
	for(size_t i = 0; i < htinstr.get_access_allocations().size(); ++i) {
		access_map.emplace_back(htinstr.get_access_allocations()[i], buffer_memory_allocation_map[i]);
	}
}

send_instruction_record::send_instruction_record(const send_instruction& sinstr, const command_id push_cid, const detail::transfer_id& trid,
    std::string buffer_name, const celerity::id<3>& offset_in_buffer)
    : acceptor_base(sinstr), dest_node_id(sinstr.get_dest_node_id()), message_id(sinstr.get_message_id()),
      source_allocation_id(sinstr.get_source_allocation_id()), source_allocation_range(sinstr.get_source_allocation_range()),
      offset_in_source_allocation(sinstr.get_offset_in_source_allocation()), send_range(sinstr.get_send_range()), element_size(sinstr.get_element_size()),
      push_cid(push_cid), transfer_id(trid), buffer_name(std::move(buffer_name)), offset_in_buffer(offset_in_buffer) {}

receive_instruction_record_impl::receive_instruction_record_impl(const receive_instruction_impl& rinstr, std::string buffer_name)
    : transfer_id(rinstr.get_transfer_id()), buffer_name(std::move(buffer_name)), requested_region(rinstr.get_requested_region()),
      dest_allocation_id(rinstr.get_dest_allocation_id()), allocated_box(rinstr.get_allocated_box()), element_size(rinstr.get_element_size()) {}

receive_instruction_record::receive_instruction_record(const receive_instruction& rinstr, std::string buffer_name)
    : acceptor_base(rinstr), receive_instruction_record_impl(rinstr, std::move(buffer_name)) {}

split_receive_instruction_record::split_receive_instruction_record(const split_receive_instruction& srinstr, std::string buffer_name)
    : acceptor_base(srinstr), receive_instruction_record_impl(srinstr, std::move(buffer_name)) {}

await_receive_instruction_record::await_receive_instruction_record(const await_receive_instruction& arinstr, std::string buffer_name)
    : acceptor_base(arinstr), transfer_id(arinstr.get_transfer_id()), buffer_name(std::move(buffer_name)), received_region(arinstr.get_received_region()) {}

gather_receive_instruction_record::gather_receive_instruction_record(
    const gather_receive_instruction& grinstr, std::string buffer_name, const box<3>& gather_box, size_t num_nodes)
    : acceptor_base(grinstr), transfer_id(grinstr.get_transfer_id()), buffer_name(std::move(buffer_name)), allocation_id(grinstr.get_dest_allocation_id()),
      node_chunk_size(grinstr.get_node_chunk_size()), gather_box(gather_box), num_nodes(num_nodes) {}

fill_identity_instruction_record::fill_identity_instruction_record(const fill_identity_instruction& fiinstr)
    : acceptor_base(fiinstr), reduction_id(fiinstr.get_reduction_id()), allocation_id(fiinstr.get_allocation_id()), num_values(fiinstr.get_num_values()) {}

reduce_instruction_record::reduce_instruction_record(const reduce_instruction& rinstr, const std::optional<detail::command_id> reduction_cid,
    const detail::buffer_id bid, std::string buffer_name, const detail::box<3>& box, const reduction_scope scope)
    : acceptor_base(rinstr), reduction_id(rinstr.get_reduction_id()), source_allocation_id(rinstr.get_source_allocation_id()),
      num_source_values(rinstr.get_num_source_values()), dest_allocation_id(rinstr.get_dest_allocation_id()), reduction_command_id(reduction_cid),
      buffer_id(bid), buffer_name(std::move(buffer_name)), box(box), scope(scope) {}

fence_instruction_record::fence_instruction_record(
    const fence_instruction& finstr, const task_id tid, const command_id cid, const buffer_id bid, std::string buffer_name, const box<3>& box)
    : acceptor_base(finstr), tid(tid), cid(cid), variant(buffer_variant{bid, std::move(buffer_name), box}) {}

fence_instruction_record::fence_instruction_record(const fence_instruction& finstr, const task_id tid, const command_id cid, const host_object_id hoid)
    : acceptor_base(finstr), tid(tid), cid(cid), variant(host_object_variant{hoid}) {}

destroy_host_object_instruction_record::destroy_host_object_instruction_record(const destroy_host_object_instruction& dhoinstr)
    : acceptor_base(dhoinstr), host_object_id(dhoinstr.get_host_object_id()) {}

horizon_instruction_record::horizon_instruction_record(const horizon_instruction& hinstr, const command_id horizon_cid)
    : acceptor_base(hinstr), horizon_task_id(hinstr.get_horizon_task_id()), horizon_command_id(horizon_cid), garbage(hinstr.get_garbage()) {}

epoch_instruction_record::epoch_instruction_record(const epoch_instruction& einstr, const command_id epoch_cid)
    : acceptor_base(einstr), epoch_task_id(einstr.get_epoch_task_id()), epoch_command_id(epoch_cid), epoch_action(einstr.get_epoch_action()),
      garbage(einstr.get_garbage()) {}

void instruction_recorder::record_await_push_command_id(const transfer_id& trid, const command_id cid) {
	assert(m_await_push_cids.count(trid) == 0 || m_await_push_cids.at(trid) == cid);
	m_await_push_cids.emplace(trid, cid);
}

command_id instruction_recorder::get_await_push_command_id(const transfer_id& trid) const { return m_await_push_cids.at(trid); }

} // namespace celerity::detail
