#pragma once

#include "command.h"
#include "instruction_graph.h"
#include "pilot.h"
#include "task.h"

#include <functional>
#include <variant>
#include <vector>


namespace celerity::detail {

class task_manager;

// General recording

struct access_record {
	const buffer_id bid;
	const std::string buffer_name;
	const access_mode mode;
	const region<3> req;
};
using access_list = std::vector<access_record>;
using buffer_name_map = std::function<std::string(buffer_id)>;

struct reduction_record {
	const reduction_id rid;
	const buffer_id bid;
	const std::string buffer_name;
	const bool init_from_buffer;
};
using reduction_list = std::vector<reduction_record>;

template <typename IdType>
struct dependency_record {
	const IdType node;
	const dependency_kind kind;
	const dependency_origin origin;
};

// Task recording

using task_dependency_list = std::vector<dependency_record<task_id>>;

struct task_record {
	task_record(const task& tsk, const buffer_name_map& get_buffer_debug_name);

	task_id tid;
	std::string debug_name;
	collective_group_id cgid;
	task_type type;
	task_geometry geometry;
	reduction_list reductions;
	access_list accesses;
	detail::side_effect_map side_effect_map;
	task_dependency_list dependencies;
};

class task_recorder {
  public:
	void record(task_record&& record) { m_recorded_tasks.push_back(std::move(record)); }

	const std::vector<task_record>& get_tasks() const { return m_recorded_tasks; }

	const task_record& get_task(const task_id tid) const {
		const auto it = std::find_if(m_recorded_tasks.begin(), m_recorded_tasks.end(), [tid](const task_record& rec) { return rec.tid == tid; });
		assert(it != m_recorded_tasks.end());
		return *it;
	}

  private:
	std::vector<task_record> m_recorded_tasks;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_record {
	command_id cid;
	command_type type;

	std::optional<detail::epoch_action> epoch_action;
	std::optional<subrange<3>> execution_range;
	std::optional<detail::reduction_id> reduction_id;
	std::optional<detail::buffer_id> buffer_id;
	std::string buffer_name;
	std::optional<node_id> target;
	std::optional<region<3>> await_region;
	std::optional<subrange<3>> push_range;
	std::optional<detail::transfer_id> transfer_id;
	std::optional<detail::task_id> task_id;
	std::optional<detail::task_geometry> task_geometry;
	bool is_reduction_initializer;
	bool has_local_contribution;
	std::optional<access_list> accesses;
	std::optional<reduction_list> reductions;
	std::optional<side_effect_map> side_effects;
	command_dependency_list dependencies;
	std::string task_name;
	std::optional<detail::task_type> task_type;
	std::optional<detail::collective_group_id> collective_group_id;
	std::vector<detail::reduction_id> completed_reductions;

	command_record(const abstract_command& cmd, const task& tsk, const buffer_name_map& get_buffer_debug_name);
};

class command_recorder {
  public:
	void record(command_record&& record) { m_recorded_commands.push_back(std::move(record)); }

	const std::vector<detail::command_record>& get_commands() const { return m_recorded_commands; }

	const command_record& get_command(const command_id cid) const {
		const auto it = std::find_if(m_recorded_commands.begin(), m_recorded_commands.end(), [cid](const command_record& rec) { return rec.cid == cid; });
		assert(it != m_recorded_commands.end());
		return *it;
	}

  private:
	std::vector<detail::command_record> m_recorded_commands;
};

// Instruction recording

struct buffer_allocation_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	detail::box<3> box;

	friend bool operator==(const buffer_allocation_record& lhs, const buffer_allocation_record& rhs) {
		return lhs.buffer_id == rhs.buffer_id && lhs.buffer_name == rhs.buffer_name && lhs.box == rhs.box;
	}
	friend bool operator!=(const buffer_allocation_record& lhs, const buffer_allocation_record& rhs) { return !(lhs == rhs); }
};

enum class instruction_dependency_origin {
	allocation_lifetime,  ///< Dependency between an alloc / free instruction and the first / last access front on that allocation
	write_to_allocation,  ///< An anti- or output dependency on data present in an allocation
	read_from_allocation, ///< True dataflow dependency on data present in an allocation
	side_effect, ///< Dependency between two host tasks that affect the same host object, or between such a host task and `destroy_host_object_instruction`
	collective_group_order, ///< Serializing dependency between two host tasks that participate in the same `collective_group`
	last_epoch,             ///< Fall-back dependency to the effective epoch for instructions that have no other dependency
	execution_front,        ///< Dependency from a new epoch- or horizon instruction to the previous execution front
	split_receive,          ///< Ordering dependency between a `split_receive_instruction` and its `await_receive_instruction`s
};

struct instruction_dependency_record {
	instruction_id predecessor;
	instruction_id successor;
	instruction_dependency_origin origin;

	instruction_dependency_record(const instruction_id predecessor, const instruction_id successor, const instruction_dependency_origin origin)
	    : predecessor(predecessor), successor(successor), origin(origin) {}
};

/// IDAG base record type for `detail::instruction`.
struct instruction_record
    : matchbox::acceptor<struct clone_collective_group_instruction_record, struct alloc_instruction_record, struct free_instruction_record,
          struct copy_instruction_record, struct device_kernel_instruction_record, struct host_task_instruction_record, struct send_instruction_record,
          struct receive_instruction_record, struct split_receive_instruction_record, struct await_receive_instruction_record,
          struct gather_receive_instruction_record, struct fill_identity_instruction_record, struct reduce_instruction_record, struct fence_instruction_record,
          struct destroy_host_object_instruction_record, struct horizon_instruction_record,
          struct epoch_instruction_record> //
{
	instruction_id id;
	int priority;

	explicit instruction_record(const instruction& instr);
};

/// IDAG record type for `clone_collective_group_instruction`.
struct clone_collective_group_instruction_record : matchbox::implement_acceptor<instruction_record, clone_collective_group_instruction_record> {
	collective_group_id original_collective_group_id;
	collective_group_id new_collective_group_id;

	explicit clone_collective_group_instruction_record(const clone_collective_group_instruction& ccginstr);
};

/// IDAG record type for `alloc_instruction`.
struct alloc_instruction_record : matchbox::implement_acceptor<instruction_record, alloc_instruction_record> {
	enum class alloc_origin {
		buffer,
		gather,
	};

	detail::allocation_id allocation_id;
	size_t size_bytes;
	size_t alignment_bytes;
	alloc_origin origin;
	std::optional<buffer_allocation_record> buffer_allocation;
	std::optional<size_t> num_chunks;

	alloc_instruction_record(
	    const alloc_instruction& ainstr, alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation, std::optional<size_t> num_chunks);
};

/// IDAG record type for `free_instruction`.
struct free_instruction_record : matchbox::implement_acceptor<instruction_record, free_instruction_record> {
	detail::allocation_id allocation_id;
	size_t size;
	std::optional<buffer_allocation_record> buffer_allocation;

	free_instruction_record(const free_instruction& finstr, size_t size, std::optional<buffer_allocation_record> buffer_allocation);
};

/// IDAG record type for `copy_instruction`.
struct copy_instruction_record : matchbox::implement_acceptor<instruction_record, copy_instruction_record> {
	enum class copy_origin {
		resize,
		coherence,
		gather,
		fence,
	};

	allocation_with_offset source_allocation;
	allocation_with_offset dest_allocation;
	box<3> source_box;
	box<3> dest_box;
	region<3> copy_region;
	size_t element_size;
	copy_origin origin;
	detail::buffer_id buffer_id;
	std::string buffer_name;

	copy_instruction_record(const copy_instruction& cinstr, copy_origin origin, detail::buffer_id buffer_id, std::string buffer_name);
};

/// IDAG debug info for device-kernel / host-task access to a single allocation (not part of the actual graph).
struct buffer_memory_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
};

/// IDAG debug info for a device-kernel access to a reduction output buffer (not part of the actual graph).
struct buffer_reduction_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	detail::reduction_id reduction_id;
};

/// IDAG combined record for a device-kernel / host-task buffer access via a single allocation.
struct buffer_access_allocation_record : buffer_access_allocation, buffer_memory_record {
	buffer_access_allocation_record(const buffer_access_allocation& aa, buffer_memory_record mr)
	    : buffer_access_allocation(aa), buffer_memory_record(std::move(mr)) {}
};

/// IDAG combined record for a device-kernel access to a reduction-output buffer allocation.
struct buffer_reduction_allocation_record : buffer_access_allocation, buffer_reduction_record {
	buffer_reduction_allocation_record(const buffer_access_allocation& aa, buffer_reduction_record mrr)
	    : buffer_access_allocation(aa), buffer_reduction_record(std::move(mrr)) {}
};

/// IDAG record type for a `device_kernel_instruction`.
struct device_kernel_instruction_record : matchbox::implement_acceptor<instruction_record, device_kernel_instruction_record> {
	detail::device_id device_id;
	box<3> execution_range;
	std::vector<buffer_access_allocation_record> access_map;
	std::vector<buffer_reduction_allocation_record> reduction_map;
	size_t estimated_global_memory_traffic_bytes;
	task_id command_group_task_id;
	command_id execution_command_id;
	std::string debug_name;

	device_kernel_instruction_record(const device_kernel_instruction& dkinstr, task_id cg_tid, command_id execution_cid, const std::string& debug_name,
	    const std::vector<buffer_memory_record>& buffer_memory_allocation_map, const std::vector<buffer_reduction_record>& buffer_memory_reduction_map);
};

/// IDAG record type for a `host_task_instruction`.
struct host_task_instruction_record : matchbox::implement_acceptor<instruction_record, host_task_instruction_record> {
	detail::collective_group_id collective_group_id;
	box<3> execution_range;
	std::vector<buffer_access_allocation_record> access_map;
	task_id command_group_task_id;
	command_id execution_command_id;
	std::string debug_name;

	host_task_instruction_record(const host_task_instruction& htinstr, task_id cg_tid, command_id execution_cid, const std::string& debug_name,
	    const std::vector<buffer_memory_record>& buffer_memory_allocation_map);
};

/// IDAG record type for a `send_instruction`.
struct send_instruction_record : matchbox::implement_acceptor<instruction_record, send_instruction_record> {
	node_id dest_node_id;
	detail::message_id message_id;
	allocation_id source_allocation_id;
	range<3> source_allocation_range;
	celerity::id<3> offset_in_source_allocation;
	range<3> send_range;
	size_t element_size;
	command_id push_cid;
	detail::transfer_id transfer_id;
	std::string buffer_name;
	celerity::id<3> offset_in_buffer;

	send_instruction_record(
	    const send_instruction& sinstr, command_id push_cid, const detail::transfer_id& trid, std::string buffer_name, const celerity::id<3>& offset_in_buffer);
};

/// Base implementation for IDAG record types of `receive_instruction` and `split_receive_instruction`.
struct receive_instruction_record_impl {
	detail::transfer_id transfer_id;
	std::string buffer_name;
	region<3> requested_region;
	allocation_id dest_allocation_id;
	box<3> allocated_box;
	size_t element_size;

	receive_instruction_record_impl(const receive_instruction_impl& rinstr, std::string buffer_name);
};

/// IDAG record type for a `receive_instruction`.
struct receive_instruction_record : matchbox::implement_acceptor<instruction_record, receive_instruction_record>, receive_instruction_record_impl {
	receive_instruction_record(const receive_instruction& rinstr, std::string buffer_name);
};

/// IDAG record type for a `split_receive_instruction`.
struct split_receive_instruction_record : matchbox::implement_acceptor<instruction_record, split_receive_instruction_record>, receive_instruction_record_impl {
	split_receive_instruction_record(const split_receive_instruction& srinstr, std::string buffer_name);
};

/// IDAG record type for a `await_receive_instruction`.
struct await_receive_instruction_record : matchbox::implement_acceptor<instruction_record, await_receive_instruction_record> {
	detail::transfer_id transfer_id;
	std::string buffer_name;
	region<3> received_region;

	await_receive_instruction_record(const await_receive_instruction& arinstr, std::string buffer_name);
};

/// IDAG record type for a `gather_receive_instruction`.
struct gather_receive_instruction_record : matchbox::implement_acceptor<instruction_record, gather_receive_instruction_record> {
	detail::transfer_id transfer_id;
	std::string buffer_name;
	detail::allocation_id allocation_id;
	size_t node_chunk_size;
	box<3> gather_box;
	size_t num_nodes;

	gather_receive_instruction_record(const gather_receive_instruction& grinstr, std::string buffer_name, const box<3>& gather_box, size_t num_nodes);
};

/// IDAG record type for a `fill_identity_instruction`.
struct fill_identity_instruction_record : matchbox::implement_acceptor<instruction_record, fill_identity_instruction_record> {
	detail::reduction_id reduction_id;
	detail::allocation_id allocation_id;
	size_t num_values;

	fill_identity_instruction_record(const fill_identity_instruction& fiinstr);
};

/// IDAG record type for a `reduce_instruction`.
struct reduce_instruction_record : matchbox::implement_acceptor<instruction_record, reduce_instruction_record> {
	enum class reduction_scope {
		global,
		local,
	};

	detail::reduction_id reduction_id;
	allocation_id source_allocation_id;
	size_t num_source_values;
	allocation_id dest_allocation_id;
	std::optional<command_id> reduction_command_id;
	detail::buffer_id buffer_id;
	std::string buffer_name;
	detail::box<3> box;
	reduction_scope scope;

	reduce_instruction_record(const reduce_instruction& rinstr, std::optional<detail::command_id> reduction_cid, detail::buffer_id bid, std::string buffer_name,
	    const detail::box<3>& box, reduction_scope scope);
};

/// IDAG record type for a `fence_instruction`.
struct fence_instruction_record : matchbox::implement_acceptor<instruction_record, fence_instruction_record> {
	struct buffer_variant {
		buffer_id bid;
		std::string name;
		detail::box<3> box;
	};
	struct host_object_variant {
		host_object_id hoid;
	};

	task_id tid;
	command_id cid;
	std::variant<buffer_variant, host_object_variant> variant;

	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, buffer_id bid, std::string buffer_name, const box<3>& box);
	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, host_object_id hoid);
};

/// IDAG record type for a `destroy_host_object_instruction`.
struct destroy_host_object_instruction_record : matchbox::implement_acceptor<instruction_record, destroy_host_object_instruction_record> {
	detail::host_object_id host_object_id;

	explicit destroy_host_object_instruction_record(const destroy_host_object_instruction& dhoinstr);
};

/// IDAG record type for a `horizon_instruction`.
struct horizon_instruction_record : matchbox::implement_acceptor<instruction_record, horizon_instruction_record> {
	task_id horizon_task_id;
	command_id horizon_command_id;
	instruction_garbage garbage;

	horizon_instruction_record(const horizon_instruction& hinstr, command_id horizon_cid);
};

/// IDAG record type for a `epoch_instruction`.
struct epoch_instruction_record : matchbox::implement_acceptor<instruction_record, epoch_instruction_record> {
	task_id epoch_task_id;
	command_id epoch_command_id;
	detail::epoch_action epoch_action;
	instruction_garbage garbage;

	epoch_instruction_record(const epoch_instruction& einstr, command_id epoch_cid);
};

/// Records instructions and outbound pilots on instruction-graph generation.
class instruction_recorder {
  public:
	void record_await_push_command_id(const transfer_id& trid, const command_id cid);

	void record_instruction(std::unique_ptr<instruction_record> record) { m_recorded_instructions.push_back(std::move(record)); }

	void record_outbound_pilot(const outbound_pilot& pilot) { m_recorded_pilots.push_back(pilot); }

	void record_dependency(const instruction_dependency_record& dependency) { m_recorded_dependencies.push_back(dependency); }

	const std::vector<std::unique_ptr<instruction_record>>& get_instructions() const { return m_recorded_instructions; }

	const std::vector<instruction_dependency_record>& get_dependencies() const { return m_recorded_dependencies; }

	const instruction_record& get_instruction(const instruction_id iid) const {
		const auto it = std::find_if(
		    m_recorded_instructions.begin(), m_recorded_instructions.end(), [=](const std::unique_ptr<instruction_record>& instr) { return instr->id == iid; });
		assert(it != m_recorded_instructions.end());
		return **it;
	}

	const std::vector<outbound_pilot>& get_outbound_pilots() const { return m_recorded_pilots; }

	command_id get_await_push_command_id(const transfer_id& trid) const;

  private:
	std::vector<std::unique_ptr<instruction_record>> m_recorded_instructions;
	std::vector<instruction_dependency_record> m_recorded_dependencies;
	std::vector<outbound_pilot> m_recorded_pilots;
	std::unordered_map<transfer_id, command_id> m_await_push_cids;
};

} // namespace celerity::detail
