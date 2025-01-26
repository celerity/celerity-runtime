#include "recorders.h"

#include "command_graph.h"
#include "grid.h"
#include "instruction_graph.h"
#include "print_utils_internal.h"
#include "ranges.h"
#include "system_info.h"
#include "task.h"
#include "types.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <graphs.hpp>
#include <mpi.h>

#include <fmt/ranges.h> // NOCOMMIT Needed?


namespace celerity::detail {

// Tasks

access_list build_access_list(const task& tsk, const buffer_name_map& get_buffer_debug_name, const std::optional<subrange<3>> execution_range = {}) {
	access_list ret;
	const auto& bam = tsk.get_buffer_access_map();
	for(size_t i = 0; i < bam.get_num_accesses(); ++i) {
		const auto [bid, mode] = bam.get_nth_access(i);
		const auto req = bam.get_requirements_for_nth_access(i, execution_range);
		ret.push_back({bid, get_buffer_debug_name(bid), mode, req});
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

std::optional<execution_spec> get_execution_spec(const command& cmd) {
	if(const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd); execution_cmd != nullptr) { return execution_cmd->get_execution_spec(); }
	return std::nullopt;
}

access_list build_cmd_access_list(const command& cmd, const task& tsk, const buffer_name_map& accessed_buffer_names) {
	const auto exec_spec = get_execution_spec(cmd);
	if(!exec_spec.has_value()) {
		return build_access_list(tsk, accessed_buffer_names, subrange<3>{get_global_offset(tsk.get_geometry()), get_global_size(tsk.get_geometry())});
	}
	return matchbox::match(
	    *exec_spec, //
	    [&](const subrange<3>& sr) { return build_access_list(tsk, accessed_buffer_names, sr); },
	    [&](const std::vector<device_execution_range>& exec_ranges) {
		    // NOCOMMIT TODO: Add test for this
		    access_list ret = build_access_list(tsk, accessed_buffer_names, exec_ranges.at(0).range);
		    for(size_t i = 1; i < exec_ranges.size(); ++i) {
			    const auto& er = exec_ranges[i];
			    const auto al = build_access_list(tsk, accessed_buffer_names, er.range);
			    for(size_t j = 0; j < al.size(); ++j) {
				    assert(ret[j].buffer_name == al[j].buffer_name);
				    assert(ret[j].bid == al[j].bid);
				    assert(ret[j].mode == al[j].mode);
				    ret[j].req = region_union(ret[j].req, al[j].req);
			    }
		    }
		    return ret;
	    });
}

command_record::command_record(const command& cmd) : id(cmd.get_id()) {}

push_command_record::push_command_record(const push_command& pcmd, std::string buffer_name)
    : acceptor_base(pcmd), trid(pcmd.get_transfer_id()), target_regions(pcmd.get_target_regions()), buffer_name(std::move(buffer_name)) {
	// Sort regions by node id for easier testing and consistent output in graph printing
	std::sort(target_regions.begin(), target_regions.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
}

await_push_command_record::await_push_command_record(const await_push_command& apcmd, std::string buffer_name)
    : acceptor_base(apcmd), trid(apcmd.get_transfer_id()), await_region(apcmd.get_region()), buffer_name(std::move(buffer_name)) {}

reduction_command_record::reduction_command_record(const reduction_command& rcmd, std::string buffer_name)
    : acceptor_base(rcmd), rid(rcmd.get_reduction_info().rid), bid(rcmd.get_reduction_info().bid), buffer_name(std::move(buffer_name)),
      init_from_buffer(rcmd.get_reduction_info().init_from_buffer), has_local_contribution(rcmd.has_local_contribution()) {}

task_command_record::task_command_record(const task& tsk)
    : tid(tsk.get_id()), type(tsk.get_type()), debug_name(tsk.get_debug_name()), cgid(tsk.get_collective_group_id()) {}

epoch_command_record::epoch_command_record(const epoch_command& ecmd, const task& tsk)
    : acceptor_base(ecmd), task_command_record(tsk), action(ecmd.get_epoch_action()), completed_reductions(ecmd.get_completed_reductions()) {}

horizon_command_record::horizon_command_record(const horizon_command& hcmd, const task& tsk)
    : acceptor_base(hcmd), task_command_record(tsk), completed_reductions(hcmd.get_completed_reductions()) {}

execution_command_record::execution_command_record(const execution_command& ecmd, const task& tsk, const buffer_name_map& get_buffer_debug_name)
    : acceptor_base(ecmd), task_command_record(tsk), exec_spec(ecmd.get_execution_spec()), is_reduction_initializer(ecmd.is_reduction_initializer()),
      accesses(build_cmd_access_list(ecmd, tsk, get_buffer_debug_name)), side_effects(tsk.get_side_effect_map()),
      reductions(build_reduction_list(tsk, get_buffer_debug_name)) {}

fence_command_record::fence_command_record(const fence_command& fcmd, const task& tsk, const buffer_name_map& get_buffer_debug_name)
    : acceptor_base(fcmd), task_command_record(tsk), accesses(build_cmd_access_list(fcmd, tsk, get_buffer_debug_name)),
      side_effects(tsk.get_side_effect_map()) {}

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

device_kernel_instruction_record::device_kernel_instruction_record(const device_kernel_instruction& dkinstr, const task_id cg_tid,
    const command_id execution_cid, const std::string& debug_name, const device_kernel_instruction_record& other)
    : acceptor_base(dkinstr), device_id(dkinstr.get_device_id()), execution_range(dkinstr.get_execution_range()), access_map(other.access_map),
      reduction_map(other.reduction_map), estimated_global_memory_traffic_bytes(dkinstr.get_estimated_global_memory_traffic_bytes()),
      command_group_task_id(cg_tid), execution_command_id(execution_cid), debug_name(debug_name) {}

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

host_task_instruction_record::host_task_instruction_record(const host_task_instruction& htinstr, const task_id cg_tid, const command_id execution_cid,
    const std::string& debug_name, const host_task_instruction_record& other)
    : acceptor_base(htinstr), collective_group_id(htinstr.get_collective_group_id()), execution_range(htinstr.get_execution_range()),
      access_map(other.access_map), command_group_task_id(cg_tid), execution_command_id(execution_cid), debug_name(debug_name) //
{}

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

///////////// ========================= TODO: Move into separate TU

// TODO: Optionally include all types of instructions?
const std::vector<std::pair<std::type_index, std::string_view>> instruction_types = {
    {typeid(alloc_instruction_record), "alloc_instruction"},
    {typeid(free_instruction_record), "free_instruction"},
    {typeid(copy_instruction_record), "copy_instruction"},
    {typeid(device_kernel_instruction_record), "device_kernel_instruction"},
    {typeid(host_task_instruction_record), "host_task_instruction"},
    {typeid(send_instruction_record), "send_instruction"},
    {typeid(receive_instruction_record), "receive_instruction"},
    {typeid(split_receive_instruction_record), "split_receive_instruction"},
    {typeid(await_receive_instruction_record), "await_receive_instruction"},
};

// TODO: Naming "report"?
// TODO: Move raw data into separate struct? => instruction_package and performance_report
// TODO: Transpose - have one type map lookup outside
struct instruction_performance_package {
	using duration = instruction_performance_recorder::duration;

	std::string name;
	bool is_global = false; // TODO: Enum "scope"?

	// Raw data
	std::unordered_map<std::type_index, std::vector<size_t>> times; // need to store as size_t for histogram

	// Derived
	std::unordered_map<std::type_index, size_t> count;
	std::unordered_map<std::type_index, duration> sum_durations;
	std::unordered_map<std::type_index, duration> min_durations;
	std::unordered_map<std::type_index, duration> mean_durations;
	std::unordered_map<std::type_index, duration> median_durations;
	std::unordered_map<std::type_index, duration> max_durations;

	/// Total number of bytes transferred using send instructions
	uint64_t bytes_sent = 0;

	uint64_t bytes_copied = 0;
	uint64_t bytes_copied_u2h = 0;
	uint64_t bytes_copied_h2h = 0;
	uint64_t bytes_copied_h2d = 0;
	uint64_t bytes_copied_d2h = 0;
	uint64_t bytes_copied_d2d_self = 0;
	uint64_t bytes_copied_d2d_peer = 0;

	uint64_t bytes_allocated = 0;
	uint64_t bytes_allocated_host = 0;
	uint64_t bytes_allocated_host_buffer = 0;
	uint64_t bytes_allocated_host_staging = 0;
	uint64_t bytes_allocated_device = 0;
	uint64_t bytes_allocated_device_buffer = 0;
	uint64_t bytes_allocated_device_staging = 0;
};


// TODO: [x] Aggregate across all nodes
// TODO: [x] Group by task name
// TODO:    [ ] Compute task duration (first to last instruction)
// TODO:    [ ] For tasks with more than 1 instance: Compute total span (first instruction of first task to last instruction of last task)
// TODO:       [ ] Then: Compute %-overlap
//             => Actually we can't do that with current setup. We only store elapsed time, not start/end timestamps.
//             => It would be nice if we could compute %-device occupied. But I think we would need graph-based start/end markers for tasks to do that..?
// TODO: [x] Write total number of bytes transferred / copied
// TODO: [ ] Write warning when kernel profiling is disabled (and durations are short?)
// TODO: [ ] Compute high/low mean?
// TODO: [ ] Compute stddev? Does that make sense for non-normal-distributed samples?
// TODO: [ ] Skip histogram if it cannot be displayed with available width => Couldn't we just increase bucket size?
// TODO: [x] Add subsection for copies: D2D, H2D, D2H, U2H
// TODO: [ ] Receives - average number of involved peers - can we compute from pilots? (problem: we only have SENT pilots, need to exchange)

// Q: How do we handle grouping etc when we have multiple nodes?
//    The problem is we still need the raw data to compute median (unless we do a distributed quick select or something)
void print_report(const instruction_performance_package& pkg, const node_id local_nid) {
	using aligned_as_sub_second = as_sub_second<right_padded_time_units>;
	using aligned_as_decimal_size = as_decimal_size<single_digit_right_padded_byte_size_units>;

	fmt::print("\n{} ({}):\n\n", pkg.name, pkg.is_global ? "global" : fmt::format("node {}", local_nid));
	fmt::print("{:<30} {:<12} {:>9} {:>9} {:>9} {:>9}\n", "Type", "Count", "Min", "Mean", "Median", "Max");
	for(const auto& [type, name] : instruction_types) {
		fmt::print("{:<30} {:<12} {:>9.1f} {:>9.1f} {:>9.1f} {:>9.1f}\n", name, fmt::group_digits(pkg.count.at(type)),
		    aligned_as_sub_second(pkg.min_durations.at(type)), aligned_as_sub_second(pkg.mean_durations.at(type)),
		    aligned_as_sub_second(pkg.median_durations.at(type)), aligned_as_sub_second(pkg.max_durations.at(type)));
	}

	fmt::print("\n");
	fmt::print("{:<40} {:>9.1f}\n", "Data transferred between nodes:", aligned_as_decimal_size(pkg.bytes_sent));

	fmt::print("\n");
	fmt::print("{:<40} {:>9.1f}\n", "Data copied:", aligned_as_decimal_size(pkg.bytes_copied));
	fmt::print("{:<40} {:>9.1f}\n", "  Device to device (self):", aligned_as_decimal_size(pkg.bytes_copied_d2d_self));
	fmt::print("{:<40} {:>9.1f}\n", "  Device to device (peer):", aligned_as_decimal_size(pkg.bytes_copied_d2d_peer));
	fmt::print("{:<40} {:>9.1f}\n", "  Device to host:", aligned_as_decimal_size(pkg.bytes_copied_d2h));
	fmt::print("{:<40} {:>9.1f}\n", "  Host to host:", aligned_as_decimal_size(pkg.bytes_copied_h2h));
	fmt::print("{:<40} {:>9.1f}\n", "  Host to device:", aligned_as_decimal_size(pkg.bytes_copied_h2d));
	fmt::print("{:<40} {:>9.1f}\n", "  User to host:", aligned_as_decimal_size(pkg.bytes_copied_u2h));

	// TODO: Can we compute a high-water mark? We wouldn't need to walk the graph, just know in which order instructions are executed...
	fmt::print("\n");
	fmt::print("{:<40} {:>9.1f}\n", "Memory allocated:", aligned_as_decimal_size(pkg.bytes_allocated));
	fmt::print("{:<40} {:>9.1f}\n", "  Device:", aligned_as_decimal_size(pkg.bytes_allocated_device));
	fmt::print("{:<40} {:>9.1f}\n", "    Buffer:", aligned_as_decimal_size(pkg.bytes_allocated_device_buffer));
	fmt::print("{:<40} {:>9.1f}\n", "    Staging:", aligned_as_decimal_size(pkg.bytes_allocated_device_staging));
	fmt::print("{:<40} {:>9.1f}\n", "  Host:", aligned_as_decimal_size(pkg.bytes_allocated_host));
	fmt::print("{:<40} {:>9.1f}\n", "    Buffer:", aligned_as_decimal_size(pkg.bytes_allocated_host_buffer));
	fmt::print("{:<40} {:>9.1f}\n", "    Staging:", aligned_as_decimal_size(pkg.bytes_allocated_host_staging));

	// NOCOMMIT TODO: Do manual histogram?
	// fmt::print("\nHistogram for device_kernel_instruction:\n");
	// graphs::histogram(80, 160, 20, 400, 0, 0, pkg.times.at(typeid(device_kernel_instruction_record)));
}

template <typename T>
T compute_median(const size_t num_nodes, const node_id local_nid, std::vector<T>& times) {
	// NOCOMMIT DEBUG: Compute brute-force on rank 0
	T brute_force_median = {};
	{
		const int local_size_bytes = times.size() * sizeof(T);
		std::vector<int> per_node_size_bytes(num_nodes, 0);
		per_node_size_bytes[local_nid] = local_size_bytes;
		MPI_Gather(local_nid == 0 ? MPI_IN_PLACE : &per_node_size_bytes[local_nid], 1, MPI_INT, per_node_size_bytes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

		std::vector<int> displs(num_nodes, 0);
		for(size_t i = 1; i < num_nodes; ++i) {
			displs[i] = displs[i - 1] + per_node_size_bytes[i - 1];
		}
		// fmt::print("[{}] Counts per node: {}\n", local_nid, fmt::join(per_node_size_bytes, ", "));
		// fmt::print("[{}] Displacements per node: {}\n", local_nid, fmt::join(displs, ", "));
		std::vector<T> all_times((displs.back() + per_node_size_bytes.back()) / sizeof(T));
		MPI_Gatherv(times.data(), local_size_bytes, MPI_BYTE, all_times.data(), per_node_size_bytes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

		if(local_nid == 0) {
			std::sort(all_times.begin(), all_times.end());
			// fmt::print("Brute-force median: {}\n", all_times[all_times.size() / 2]);
			brute_force_median = all_times[all_times.size() / 2];
		}
	}


	using iterator = std::vector<T>::iterator;

	iterator begin = times.begin();
	iterator pivot = times.begin();
	iterator end = times.end();

	std::vector<int> counts(num_nodes, 0);

	const auto update_counts = [&]() {
		counts[local_nid] = std::distance(begin, end);
		MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
	};

	std::mt19937 gen(num_nodes);
	std::uniform_int_distribution<> random_node(0, num_nodes - 1);
	std::random_device rd;

	// TODO: Use hybrid pivot-of-pivot / proper random approach
	T previous_pivot = {};
	const auto select_pivot = [&]() {
		T pivot = {};
		// It's important that we pick the pivot from a random node, otherwise we can get stuck.
		const size_t start_idx = random_node(gen);
		size_t i = start_idx;
		while(true) {
			if(counts[i] != 0) {
				if(i == local_nid) {
					std::mt19937 gen(rd());
					std::uniform_int_distribution<> dist(0, counts[i] - 1);
					auto offset = dist(gen);
					pivot = *(begin + offset);
					if(pivot == previous_pivot) {
						for(size_t j = 0; j < counts[i]; ++j) {
							offset = (offset + 1) % counts[i];
							pivot = *(begin + offset);
							if(pivot != previous_pivot) {
								fmt::print("Node {} local duplicate pivot avoidance successful: Advanced by {} steps to find {} instead of {}\n", local_nid,
								    j + 1, pivot, previous_pivot);
								break;
							}
						}
					}
				}
				MPI_Bcast(&pivot, sizeof(T), MPI_BYTE, i, MPI_COMM_WORLD);
				if(pivot != previous_pivot) {
					break;
				} else if(local_nid == 0) {
					fmt::print("Global duplicate pivot avoidance: Node {} returned pivot {} which is equal to previous. Trying again\n", i, pivot);
				}
			}
			i = (i + 1) % num_nodes;
			if(i == start_idx) {
				// NOCOMMIT TODO: Msg is wrong for duplicate pivot avoidance
				utils::panic("All nodes have zero elements - cannot compute median");
			}
		}
		previous_pivot = pivot;
		return pivot;
	};

	update_counts();
	size_t k = std::round(std::accumulate(counts.begin(), counts.end(), 0) / 2.f);

	std::string debug_trace = "";
	for(size_t s = 0; true; ++s) {
		if(s > 0) update_counts();
		const auto pivot_value = select_pivot();
		// fmt::print("{} selected pivot value {}. k = {}\n", local_nid, pivot_value, k);

		pivot = std::partition(begin, end, [&](const auto& x) { return x < pivot_value; });
		const size_t equal_to_pivot = std::count(pivot, end, pivot_value);
		const size_t count_right = std::distance(pivot, end);
		// fmt::print("Node {} left: {}, right: {}, equal to pivot: {}\n", local_nid, fmt::join(begin, pivot, ","), fmt::join(pivot, end, ","), equal_to_pivot);

		uint64_t sum_right = 0;
		MPI_Allreduce(&count_right, &sum_right, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
		uint64_t sum_equal_to_pivot = 0;
		MPI_Allreduce(&equal_to_pivot, &sum_equal_to_pivot, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

		// TODO: Properly document this. Think of "in which partition is the median"? (or k-th element)
		if(sum_right - sum_equal_to_pivot + 1 > k) {
			begin = pivot;
		} else if(sum_right < k) {
			end = pivot;
			k -= sum_right;
		} else /* sum_right == k */ { // TODO: What is the actual condition now?
			// if(local_nid == 0) fmt::print("Found median after {} iterations: {}\n", s, pivot_value);
			if(local_nid == 0 && pivot_value != brute_force_median) { fmt::print("ERROR: Median does not match brute force result!!\n"); }
			return pivot_value;
		}

		const auto global_count = std::accumulate(counts.begin(), counts.end(), 0);
		auto debug_info = fmt::format("Node {} @ step {}: pivot={}, k={}, left: {}, right: {}, global right: {}, global equal to pivot: {}, global count: {}\n",
		    local_nid, s, pivot_value, k, fmt::join(begin, pivot, ","), fmt::join(pivot, end, ","), sum_right, sum_equal_to_pivot, global_count);
		if(local_nid == 0) {
			debug_trace += debug_info;
		} else {
			debug_trace = debug_info;
		}

		// TODO: What is a good cutoff?
		if(s > 30) { utils::panic("Aborting distributed median computation after {} iterations - something went wrong. {}", s, debug_trace); }
	}
}

instruction_performance_package create_global_report(instruction_performance_package& local_pkg, const size_t num_nodes, const node_id local_nid) {
	using duration = instruction_performance_recorder::duration;

	instruction_performance_package global_report{
	    .name = local_pkg.name,
	    .is_global = true,
	}; // NOCOMMIT TODO: Get rid of Wmissing-field-initializers?

	for(auto& [type, name] : instruction_types) {
		uint64_t sum_durations = local_pkg.sum_durations[type].count();
		MPI_Reduce(local_nid == 0 ? MPI_IN_PLACE : &sum_durations, &sum_durations, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
		uint64_t count = local_pkg.times[type].size();
		// Allreduce: We need to know on all nodes whether to compute median or not
		MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
		uint64_t min = local_pkg.min_durations[type].count();
		MPI_Reduce(local_nid == 0 ? MPI_IN_PLACE : &min, &min, 1, MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
		uint64_t max = local_pkg.max_durations[type].count();
		MPI_Reduce(local_nid == 0 ? MPI_IN_PLACE : &max, &max, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);

		const uint64_t mean = count > 0 ? sum_durations / count : 0;
		// fmt::print("Computing median for {} which has {} elements\n", name, count);
		const auto median = count > 0 ? compute_median(num_nodes, local_nid, local_pkg.times[type]) : 0;

		global_report.count[type] = count;
		global_report.sum_durations[type] = duration(sum_durations);
		global_report.min_durations[type] = duration(min);
		global_report.mean_durations[type] = duration(mean);
		global_report.median_durations[type] = std::chrono::microseconds(median); // TODO: Using microseconds here is ugly
		global_report.max_durations[type] = duration(max);
	}

	const auto compute_sum = [local_nid](const uint64_t value) {
		uint64_t sum = value;
		MPI_Reduce(local_nid == 0 ? MPI_IN_PLACE : &sum, &sum, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
		return sum;
	};

	global_report.bytes_sent = compute_sum(local_pkg.bytes_sent);
	global_report.bytes_copied = compute_sum(local_pkg.bytes_copied);
	global_report.bytes_copied_u2h = compute_sum(local_pkg.bytes_copied_u2h);
	global_report.bytes_copied_h2h = compute_sum(local_pkg.bytes_copied_h2h);
	global_report.bytes_copied_h2d = compute_sum(local_pkg.bytes_copied_h2d);
	global_report.bytes_copied_d2h = compute_sum(local_pkg.bytes_copied_d2h);
	global_report.bytes_copied_d2d_self = compute_sum(local_pkg.bytes_copied_d2d_self);
	global_report.bytes_copied_d2d_peer = compute_sum(local_pkg.bytes_copied_d2d_peer);

	global_report.bytes_allocated = compute_sum(local_pkg.bytes_allocated);
	global_report.bytes_allocated_host = compute_sum(local_pkg.bytes_allocated_host);
	global_report.bytes_allocated_host_buffer = compute_sum(local_pkg.bytes_allocated_host_buffer);
	global_report.bytes_allocated_host_staging = compute_sum(local_pkg.bytes_allocated_host_staging);
	global_report.bytes_allocated_device = compute_sum(local_pkg.bytes_allocated_device);
	global_report.bytes_allocated_device_buffer = compute_sum(local_pkg.bytes_allocated_device_buffer);
	global_report.bytes_allocated_device_staging = compute_sum(local_pkg.bytes_allocated_device_staging);

	return global_report;
}

template <typename InstructionPointerT>
instruction_performance_package create_local_report(const std::vector<InstructionPointerT>& instructions,
    const dense_map<instruction_id, instruction_performance_recorder::duration>& execution_times, std::string name) //
{
	using duration = instruction_performance_recorder::duration;

	instruction_performance_package report{.name = std::move(name)};

	const auto record_generic_data = [&](const auto& instr) {
		report.times[typeid(instr)].push_back(std::chrono::duration_cast<std::chrono::microseconds>(execution_times[instr.id]).count());
		report.sum_durations[typeid(instr)] += execution_times[instr.id];
	};

	for(const auto& instr : instructions) {
		matchbox::match(
		    *instr,                                                 //
		    [&](const auto& instr) { record_generic_data(instr); }, //
		    [&](const alloc_instruction_record& ainstr) {
			    record_generic_data(ainstr);
			    report.bytes_allocated += ainstr.size_bytes;
			    if(ainstr.allocation_id.get_memory_id() == host_memory_id) {
				    report.bytes_allocated_host += ainstr.size_bytes;
				    if(ainstr.origin == alloc_instruction_record::alloc_origin::buffer) {
					    report.bytes_allocated_host_buffer += ainstr.size_bytes;
				    } else {
					    report.bytes_allocated_host_staging += ainstr.size_bytes;
				    }
			    } else if(ainstr.allocation_id.get_memory_id() >= first_device_memory_id) {
				    report.bytes_allocated_device += ainstr.size_bytes;
				    if(ainstr.origin == alloc_instruction_record::alloc_origin::buffer) {
					    report.bytes_allocated_device_buffer += ainstr.size_bytes;
				    } else {
					    report.bytes_allocated_device_staging += ainstr.size_bytes;
				    }
			    }
		    },
		    [&](const copy_instruction_record& cinstr) {
			    record_generic_data(cinstr);
			    const auto copy_bytes = cinstr.copy_region.get_area() * cinstr.element_size;
			    const auto from_mid = cinstr.source_allocation_id.get_memory_id();
			    const auto to_mid = cinstr.dest_allocation_id.get_memory_id();

			    report.bytes_copied += copy_bytes;
			    if(from_mid == user_memory_id) {
				    if(to_mid == host_memory_id) {
					    report.bytes_copied_u2h += copy_bytes;
				    } else {
					    // We dont' support that
					    fmt::print("Unexpected user-to-user or user-to-device copy\n");
				    }
			    } else if(from_mid == host_memory_id) {
				    if(to_mid == host_memory_id) {
					    report.bytes_copied_h2h += copy_bytes;
				    } else if(to_mid >= first_device_memory_id) {
					    report.bytes_copied_h2d += copy_bytes;
				    } else {
					    // We don't support that
					    fmt::print("Unexpected host-to-user copy\n");
				    }
			    } else if(from_mid >= first_device_memory_id) {
				    if(to_mid == host_memory_id) {
					    report.bytes_copied_d2h += copy_bytes;
				    } else if(to_mid == from_mid) {
					    report.bytes_copied_d2d_self += copy_bytes;
				    } else if(to_mid >= first_device_memory_id) {
					    report.bytes_copied_d2d_peer += copy_bytes;
				    } else {
					    // We don't support that
					    fmt::print("Unexpected device-to-user copy\n");
				    }
			    }
		    },
		    [&](const send_instruction_record& sinstr) {
			    record_generic_data(sinstr);
			    report.bytes_sent += sinstr.send_range.size() * sinstr.element_size;
		    });
	}

	for(auto& [type, name] : instruction_types) {
		auto& times = report.times[type]; // allow default-insert
		report.count[type] = times.size();
		std::sort(times.begin(), times.end());
		report.min_durations[type] = times.size() > 0 ? std::chrono::microseconds(times.front()) : duration{0};
		report.mean_durations[type] = times.size() > 0 ? duration{report.sum_durations[type] / times.size()} : duration{0};
		// TODO: Do we compute high or low median in distributed variant? Do the same here
		report.median_durations[type] = times.size() > 0 ? std::chrono::microseconds(times[times.size() / 2]) : duration{0};
		report.max_durations[type] = times.size() > 0 ? std::chrono::microseconds(times.back()) : duration{0};
	}

	return report;
}

void instruction_performance_recorder::print_summary(const instruction_recorder& irec, const task_recorder& trec) const {
	const auto before = std::chrono::steady_clock::now();

	std::unordered_map<std::string, std::vector<task_id>> named_tasks;
	for(const auto& t : trec.get_graph_nodes()) {
		if(t->debug_name != "") { named_tasks[t->debug_name].push_back(t->tid); }
	}
	std::vector<std::string> task_names_in_program_order;
	for(const auto& [name, _] : named_tasks) {
		task_names_in_program_order.push_back(name);
	}
	std::sort(task_names_in_program_order.begin(), task_names_in_program_order.end(),
	    [&](const auto& a, const auto& b) { return named_tasks[a].front() < named_tasks[b].front(); });

	for(const auto& name : task_names_in_program_order) {
		const auto& tids = named_tasks[name];

		// TODO: Only print tasks that make up e.g. 10% of overall tasks?
		std::vector<instruction_record*> instrs;
		const auto& task_boundaries = irec.get_task_boundaries();
		const auto& all_instrs = irec.get_graph_nodes();
		auto task_it = tids.begin();
		auto instr_it = all_instrs.begin() + task_boundaries[*task_it];
		auto until_it = all_instrs.begin() + task_boundaries[*task_it + 1]; // +1 always safe because of shutdown epoch
		while(task_it != tids.end()) {
			while(instr_it != until_it) {
				instrs.push_back(instr_it->get());
				++instr_it;
			}
			++task_it;
			if(task_it != tids.end()) { until_it = all_instrs.begin() + task_boundaries[*task_it + 1]; }
		}

		auto local_report =
		    create_local_report(instrs, m_execution_times, fmt::format("Task '{}' ({} instance{})", name, tids.size(), tids.size() > 1 ? "s" : ""));
		const auto global_report = create_global_report(local_report, m_num_nodes, m_local_nid);
		if(m_local_nid == 0) { print_report(global_report, m_local_nid); }
	}

	auto all_instrs_local = create_local_report(irec.get_graph_nodes(), m_execution_times, "All instructions");
	// print_report(local_report, m_local_nid);

	// TODO: Compute distributed median BEFORE sorting locally for better average performance (quickselect)
	//  	 => Why is sorting actually bad for quickselect..?
	const auto all_instrs_global = create_global_report(all_instrs_local, m_num_nodes, m_local_nid);
	if(m_local_nid == 0) {
		print_report(all_instrs_global, m_local_nid);
		const auto after = std::chrono::steady_clock::now();
		fmt::print("\nReport generation took {:.1f}\n", as_sub_second(after - before));
	}
}

} // namespace celerity::detail
