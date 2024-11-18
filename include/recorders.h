#pragma once

#include "command_graph.h"
#include "dense_map.h"
#include "grid.h"
#include "instruction_graph.h"
#include "intrusive_graph.h"
#include "nd_memory.h"
#include "pilot.h"
#include "ranges.h"
#include "task.h"
#include "types.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <matchbox.hh>


namespace celerity::detail {

class task_manager;

// General recording

struct access_record {
	buffer_id bid;
	std::string buffer_name;
	access_mode mode;
	region<3> req;

	bool operator==(const access_record&) const = default;
};
using access_list = std::vector<access_record>;
using buffer_name_map = std::function<std::string(buffer_id)>;

struct reduction_record {
	const reduction_id rid;
	const buffer_id bid;
	const std::string buffer_name;
	const bool init_from_buffer;

	bool operator==(const reduction_record&) const = default;
};
using reduction_list = std::vector<reduction_record>;

template <typename IdType>
struct dependency_record {
	const IdType node;
	const dependency_kind kind;
	const dependency_origin origin;

	bool operator==(const dependency_record&) const = default;
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

	bool operator==(const task_record&) const = default;
};

class task_recorder {
  public:
	void record(std::unique_ptr<task_record> record) { m_recorded_tasks.push_back(std::move(record)); }

	const std::vector<std::unique_ptr<task_record>>& get_graph_nodes() const { return m_recorded_tasks; }

	void filter_by_task_id(const task_id tid, const size_t before, const size_t after) {
		auto it = std::lower_bound(
		    m_recorded_tasks.begin(), m_recorded_tasks.end(), tid, [](const std::unique_ptr<task_record>& rec, const task_id tid) { return rec->tid < tid; });
		if(it == m_recorded_tasks.end()) {
			fprintf(stderr, "Failed to filter by task id: Task with id %zu not found\n", tid.value);
			return;
		}
		std::unordered_set<task_id> tasks_to_keep;
		tasks_to_keep.insert((*it)->tid);

		const auto get_task = [&](const task_id tid) {
			auto it = std::lower_bound(m_recorded_tasks.begin(), m_recorded_tasks.end(), tid,
			    [](const std::unique_ptr<task_record>& rec, const task_id tid) { return rec->tid < tid; });
			assert(it != m_recorded_tasks.end());
			return **it;
		};

		std::unordered_set<std::pair<task_id, size_t>, utils::pair_hash> tasks_to_visit;
		tasks_to_visit.insert({tid, 0});
		while(!tasks_to_visit.empty()) {
			const auto [current_tid, distance] = *tasks_to_visit.begin();
			tasks_to_visit.erase(tasks_to_visit.begin());
			const auto& current_task = get_task(current_tid);
			tasks_to_keep.insert(current_tid);
			if(distance < before) {
				for(const auto& dep : current_task.dependencies) {
					tasks_to_visit.insert({dep.node, distance + 1});
				}
			}
		}
		// TODO: Implement downward dependencies (we don't store these in records atm)

		std::vector<std::unique_ptr<task_record>> records_to_keep;
		for(auto& tsk : m_recorded_tasks) {
			if(tasks_to_keep.contains(tsk->tid)) { records_to_keep.push_back(std::move(tsk)); }
		}
		m_recorded_tasks = std::move(records_to_keep);
	}

  private:
	std::vector<std::unique_ptr<task_record>> m_recorded_tasks;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_dependency_record {
	command_id predecessor;
	command_id successor;
	dependency_kind kind;
	dependency_origin origin;

	command_dependency_record(const command_id predecessor, const command_id successor, const dependency_kind kind, const dependency_origin origin)
	    : predecessor(predecessor), successor(successor), kind(kind), origin(origin) {}

	bool operator==(const command_dependency_record& other) const = default;
};

struct command_record : matchbox::acceptor<struct push_command_record, struct await_push_command_record, struct reduction_command_record,
                            struct epoch_command_record, struct horizon_command_record, struct execution_command_record, struct fence_command_record> {
	command_id id;

	// Set by command recorder if a loop template is currently active
	// NOCOMMIT TODO: Also for TDAG ?
	bool is_cloned = false;

	explicit command_record(const command& cmd);

	bool operator==(const command_record& other) const { return id == other.id; };
};

struct push_command_record : matchbox::implement_acceptor<command_record, push_command_record> {
	transfer_id trid;
	std::vector<std::pair<node_id, region<3>>> target_regions;
	std::string buffer_name;

	explicit push_command_record(const push_command& pcmd, std::string buffer_name);

	bool operator==(const push_command_record&) const = default;
};

struct await_push_command_record : matchbox::implement_acceptor<command_record, await_push_command_record> {
	transfer_id trid;
	region<3> await_region;
	std::string buffer_name;

	explicit await_push_command_record(const await_push_command& apcmd, std::string buffer_name);

	bool operator==(const await_push_command_record&) const = default;
};

struct reduction_command_record : matchbox::implement_acceptor<command_record, reduction_command_record> {
	reduction_id rid;
	buffer_id bid;
	std::string buffer_name;
	bool init_from_buffer;
	bool has_local_contribution;

	explicit reduction_command_record(const reduction_command& rcmd, std::string buffer_name);

	bool operator==(const reduction_command_record&) const = default;
};

/// Base class for task command records
struct task_command_record {
	task_id tid;
	task_type type;
	std::string debug_name;
	collective_group_id cgid;

	explicit task_command_record(const task& tsk);

	bool operator==(const task_command_record&) const = default;
};

struct epoch_command_record : matchbox::implement_acceptor<command_record, epoch_command_record>, task_command_record {
	epoch_action action;
	std::vector<reduction_id> completed_reductions;

	explicit epoch_command_record(const epoch_command& ecmd, const task& tsk);

	bool operator==(const epoch_command_record&) const = default;
};

struct horizon_command_record : matchbox::implement_acceptor<command_record, horizon_command_record>, task_command_record {
	std::vector<reduction_id> completed_reductions;

	explicit horizon_command_record(const horizon_command& hcmd, const task& tsk);

	bool operator==(const horizon_command_record&) const = default;
};

struct execution_command_record : matchbox::implement_acceptor<command_record, execution_command_record>, task_command_record {
	execution_spec exec_spec;
	bool is_reduction_initializer;
	access_list accesses;
	side_effect_map side_effects;
	reduction_list reductions;

	explicit execution_command_record(const execution_command& ecmd, const task& tsk, const buffer_name_map& get_buffer_debug_name);

	// NOCOMMIT TODO: Add tests that task/command/instruction records are equality comparable! (and for each class, so we don't accidentally just inherit base
	// class comparison)
	bool operator==(const execution_command_record&) const = default;
};

struct fence_command_record : matchbox::implement_acceptor<command_record, fence_command_record>, task_command_record {
	explicit fence_command_record(const fence_command& fcmd, const task& tsk, const buffer_name_map& get_buffer_debug_name);

	access_list accesses;
	side_effect_map side_effects;

	bool operator==(const fence_command_record&) const = default;
};

class command_recorder {
  public:
	// NOCOMMIT TODO: Naming - this designates the point where we begin INSTANTIATING a template. This is different from the *GGENs, where we begin a template
	// w/ priming.
	void begin_loop_template() {
		assert(!m_loop_template_active);
		m_loop_template_active = true;
	}

	void end_loop_template() {
		assert(m_loop_template_active);
		m_loop_template_active = false;
	}

	void record_command(std::unique_ptr<command_record> record) {
		if(m_loop_template_active) { record->is_cloned = true; }
		m_recorded_commands.push_back(std::move(record));
	}

	void record_dependency(const command_dependency_record& dependency) { m_recorded_dependencies.push_back(dependency); }

	const std::vector<std::unique_ptr<command_record>>& get_graph_nodes() const { return m_recorded_commands; }

	const std::vector<command_dependency_record>& get_dependencies() const { return m_recorded_dependencies; }

	void filter_by_task_id(const task_id tid, const size_t before, const size_t after) {
		// TODO: Would be nice if we could use binary search here as well - need to skip all non-task commands somehow though
		auto it = std::find_if(m_recorded_commands.begin(), m_recorded_commands.end(), [tid](const std::unique_ptr<command_record>& rec) {
			if(utils::isa<task_command_record>(rec.get())) { return dynamic_cast<const task_command_record*>(rec.get())->tid == tid; }
			return false;
		});
		if(it == m_recorded_commands.end()) {
			fprintf(stderr, "Failed to filter by task id: Execution command for task %zu not found\n", tid.value);
			return;
		}

		std::unordered_set<command_id> commands_to_keep;
		enum class visit_direction { forward, backward };
		const auto visit_all = [&](const visit_direction dir) {
			// TODO: We may end up visiting some nodes multiple times. In same cases we have to, b/c we may have found a shorter path.
			std::unordered_set<std::pair<command_id, size_t>, utils::pair_hash> to_visit;
			to_visit.insert({(*it)->id, 0});
			while(!to_visit.empty()) {
				const auto [id, distance] = *to_visit.begin();
				to_visit.erase(to_visit.begin());
				commands_to_keep.insert(id);
				if(dir == visit_direction::backward && distance < before) {
					// FIXME: Oof
					for(const auto& dep : m_recorded_dependencies) {
						if(dep.successor == id) { to_visit.insert({dep.predecessor, distance + 1}); }
					}
				}
				if(dir == visit_direction::forward && distance < after) {
					// FIXME: Oof
					for(const auto& dep : m_recorded_dependencies) {
						if(dep.predecessor == id) { to_visit.insert({dep.successor, distance + 1}); }
					}
				}
			}
		};
		visit_all(visit_direction::forward);
		visit_all(visit_direction::backward);

		std::vector<std::unique_ptr<command_record>> records_to_keep;
		for(auto& cmd : m_recorded_commands) {
			if(commands_to_keep.contains(cmd->id)) { records_to_keep.push_back(std::move(cmd)); }
		}
		std::vector<command_dependency_record> dependencies_to_keep;
		for(const auto& dep : m_recorded_dependencies) {
			if(commands_to_keep.contains(dep.predecessor) && commands_to_keep.contains(dep.successor)) { dependencies_to_keep.push_back(dep); }
		}

		m_recorded_commands = std::move(records_to_keep);
		m_recorded_dependencies = std::move(dependencies_to_keep);
	}

  private:
	std::vector<std::unique_ptr<command_record>> m_recorded_commands;
	std::vector<command_dependency_record> m_recorded_dependencies;
	bool m_loop_template_active = false;
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

	// Set by instruction recorder if a loop template is currently active
	bool is_cloned = false;

	explicit instruction_record(const instruction& instr);

	bool operator==(const instruction_record& other) const { return id == other.id && priority == other.priority; }
};

/// IDAG record type for `clone_collective_group_instruction`.
struct clone_collective_group_instruction_record : matchbox::implement_acceptor<instruction_record, clone_collective_group_instruction_record> {
	collective_group_id original_collective_group_id;
	collective_group_id new_collective_group_id;

	explicit clone_collective_group_instruction_record(const clone_collective_group_instruction& ccginstr);

	bool operator==(const clone_collective_group_instruction_record&) const = default;
};

/// IDAG record type for `alloc_instruction`.
struct alloc_instruction_record : matchbox::implement_acceptor<instruction_record, alloc_instruction_record> {
	enum class alloc_origin {
		buffer,
		gather,
		staging,
	};

	detail::allocation_id allocation_id;
	size_t size_bytes;
	size_t alignment_bytes;
	alloc_origin origin;
	std::optional<buffer_allocation_record> buffer_allocation;
	std::optional<size_t> num_chunks;

	alloc_instruction_record(
	    const alloc_instruction& ainstr, alloc_origin origin, std::optional<buffer_allocation_record> buffer_allocation, std::optional<size_t> num_chunks);

	bool operator==(const alloc_instruction_record&) const = default;
};

/// IDAG record type for `free_instruction`.
struct free_instruction_record : matchbox::implement_acceptor<instruction_record, free_instruction_record> {
	detail::allocation_id allocation_id;
	size_t size;
	std::optional<buffer_allocation_record> buffer_allocation;

	free_instruction_record(const free_instruction& finstr, size_t size, std::optional<buffer_allocation_record> buffer_allocation);

	bool operator==(const free_instruction_record&) const = default;
};

/// IDAG record type for `copy_instruction`.
struct copy_instruction_record : matchbox::implement_acceptor<instruction_record, copy_instruction_record> {
	enum class copy_origin {
		resize,
		coherence,
		gather,
		fence,
		staging,
		linearizing,
		delinearizing,
	};

	allocation_id source_allocation_id;
	allocation_id dest_allocation_id;
	region_layout source_layout;
	region_layout dest_layout;
	region<3> copy_region;
	size_t element_size;
	copy_origin origin;
	detail::buffer_id buffer_id;
	std::string buffer_name;

	copy_instruction_record(const copy_instruction& cinstr, copy_origin origin, detail::buffer_id buffer_id, std::string buffer_name);

	bool operator==(const copy_instruction_record&) const = default;
};

/// IDAG debug info for device-kernel / host-task access to a single allocation (not part of the actual graph).
struct buffer_memory_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	region<3> accessed_region_in_buffer;

	bool operator==(const buffer_memory_record&) const = default;
};

/// IDAG debug info for a device-kernel access to a reduction output buffer (not part of the actual graph).
struct buffer_reduction_record {
	detail::buffer_id buffer_id;
	std::string buffer_name;
	detail::reduction_id reduction_id;

	bool operator==(const buffer_reduction_record&) const = default;
};

/// IDAG combined record for a device-kernel / host-task buffer access via a single allocation.
struct buffer_access_allocation_record : buffer_access_allocation, buffer_memory_record {
	buffer_access_allocation_record(const buffer_access_allocation& aa, buffer_memory_record mr)
	    : buffer_access_allocation(aa), buffer_memory_record(std::move(mr)) {}

	bool operator==(const buffer_access_allocation_record&) const = default;
};

/// IDAG combined record for a device-kernel access to a reduction-output buffer allocation.
struct buffer_reduction_allocation_record : buffer_access_allocation, buffer_reduction_record {
	buffer_reduction_allocation_record(const buffer_access_allocation& aa, buffer_reduction_record mrr)
	    : buffer_access_allocation(aa), buffer_reduction_record(std::move(mrr)) {}

	bool operator==(const buffer_reduction_allocation_record&) const = default;
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

	// Cloning constructor
	device_kernel_instruction_record(const device_kernel_instruction& dkinstr, task_id cg_tid, command_id execution_cid, const std::string& debug_name,
	    const device_kernel_instruction_record& other);

	bool operator==(const device_kernel_instruction_record&) const = default;
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

	// Cloning constructor
	host_task_instruction_record(const host_task_instruction& htinstr, task_id cg_tid, command_id execution_cid, const std::string& debug_name,
	    const host_task_instruction_record& other);

	bool operator==(const host_task_instruction_record&) const = default;
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

	bool operator==(const send_instruction_record&) const = default;
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

	bool operator==(const receive_instruction_record_impl&) const = default;
};

/// IDAG record type for a `receive_instruction`.
struct receive_instruction_record : matchbox::implement_acceptor<instruction_record, receive_instruction_record>, receive_instruction_record_impl {
	receive_instruction_record(const receive_instruction& rinstr, std::string buffer_name);

	bool operator==(const receive_instruction_record&) const = default;
};

/// IDAG record type for a `split_receive_instruction`.
struct split_receive_instruction_record : matchbox::implement_acceptor<instruction_record, split_receive_instruction_record>, receive_instruction_record_impl {
	split_receive_instruction_record(const split_receive_instruction& srinstr, std::string buffer_name);

	bool operator==(const split_receive_instruction_record&) const = default;
};

/// IDAG record type for a `await_receive_instruction`.
struct await_receive_instruction_record : matchbox::implement_acceptor<instruction_record, await_receive_instruction_record> {
	detail::transfer_id transfer_id;
	std::string buffer_name;
	region<3> received_region;

	await_receive_instruction_record(const await_receive_instruction& arinstr, std::string buffer_name);

	bool operator==(const await_receive_instruction_record&) const = default;
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

	bool operator==(const gather_receive_instruction_record&) const = default;
};

/// IDAG record type for a `fill_identity_instruction`.
struct fill_identity_instruction_record : matchbox::implement_acceptor<instruction_record, fill_identity_instruction_record> {
	detail::reduction_id reduction_id;
	detail::allocation_id allocation_id;
	size_t num_values;

	fill_identity_instruction_record(const fill_identity_instruction& fiinstr);

	bool operator==(const fill_identity_instruction_record&) const = default;
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

	bool operator==(const reduce_instruction_record&) const = default;
};

/// IDAG record type for a `fence_instruction`.
struct fence_instruction_record : matchbox::implement_acceptor<instruction_record, fence_instruction_record> {
	struct buffer_variant {
		buffer_id bid;
		std::string name;
		detail::box<3> box;

		bool operator==(const buffer_variant&) const = default;
	};
	struct host_object_variant {
		host_object_id hoid;

		bool operator==(const host_object_variant&) const = default;
	};

	task_id tid;
	command_id cid;
	std::variant<buffer_variant, host_object_variant> variant;

	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, buffer_id bid, std::string buffer_name, const box<3>& box);
	fence_instruction_record(const fence_instruction& finstr, task_id tid, command_id cid, host_object_id hoid);

	bool operator==(const fence_instruction_record&) const = default;
};

/// IDAG record type for a `destroy_host_object_instruction`.
struct destroy_host_object_instruction_record : matchbox::implement_acceptor<instruction_record, destroy_host_object_instruction_record> {
	detail::host_object_id host_object_id;

	explicit destroy_host_object_instruction_record(const destroy_host_object_instruction& dhoinstr);

	bool operator==(const destroy_host_object_instruction_record&) const = default;
};

/// IDAG record type for a `horizon_instruction`.
struct horizon_instruction_record : matchbox::implement_acceptor<instruction_record, horizon_instruction_record> {
	task_id horizon_task_id;
	command_id horizon_command_id;
	instruction_garbage garbage;

	horizon_instruction_record(const horizon_instruction& hinstr, command_id horizon_cid);

	bool operator==(const horizon_instruction_record&) const = default;
};

/// IDAG record type for a `epoch_instruction`.
struct epoch_instruction_record : matchbox::implement_acceptor<instruction_record, epoch_instruction_record> {
	task_id epoch_task_id;
	command_id epoch_command_id;
	detail::epoch_action epoch_action;
	instruction_garbage garbage;

	epoch_instruction_record(const epoch_instruction& einstr, command_id epoch_cid);

	bool operator==(const epoch_instruction_record&) const = default;
};

/// Records instructions and outbound pilots on instruction-graph generation.
class instruction_recorder {
  public:
	// NOCOMMIT TODO: Naming - this designates the point where we begin INSTANTIATING a template. This is different from the *GGENs, where we begin a template
	// w/ priming.
	void begin_loop_template() {
		assert(!m_loop_template_active);
		m_loop_template_active = true;
	}

	void end_loop_template() {
		assert(m_loop_template_active);
		m_loop_template_active = false;
	}

	void record_task_boundary(const task_id tid) {
		assert((tid <= 1 || m_task_boundaries[tid - 1] != 0) && "Missing previous task boundary");
		m_task_boundaries.insert(tid, m_highest_recorded_instruction_id + 1);
	}

	void record_await_push_command_id(const transfer_id& trid, const command_id cid);

	void record_instruction(std::unique_ptr<instruction_record> record) {
		if(m_loop_template_active) { record->is_cloned = true; }
		m_highest_recorded_instruction_id++;
		assert(record->id == m_highest_recorded_instruction_id);
		m_recorded_instructions.push_back(std::move(record));
	}

	void record_outbound_pilot(const outbound_pilot& pilot) { m_recorded_pilots.push_back(pilot); }

	void record_dependency(const instruction_dependency_record& dependency) { m_recorded_dependencies.push_back(dependency); }

	const dense_map<task_id, instruction_id>& get_task_boundaries() const { return m_task_boundaries; }

	const std::vector<std::unique_ptr<instruction_record>>& get_graph_nodes() const { return m_recorded_instructions; }

	const std::vector<instruction_dependency_record>& get_dependencies() const { return m_recorded_dependencies; }

	const std::vector<outbound_pilot>& get_outbound_pilots() const { return m_recorded_pilots; }

	command_id get_await_push_command_id(const transfer_id& trid) const;

	void filter_by_task_id(const task_id tid, const size_t before, const size_t after) {
		std::unordered_set<instruction_id> instructions_to_keep;

		for(const auto& inst : m_recorded_instructions) {
			matchbox::match(
			    *inst, //
			    [&](const device_kernel_instruction_record& dkinstr) {
				    if(dkinstr.command_group_task_id == tid) { instructions_to_keep.insert(dkinstr.id); }
			    },
			    [&](const host_task_instruction_record& hinstr) {
				    if(hinstr.command_group_task_id == tid) { instructions_to_keep.insert(hinstr.id); }
			    },
			    [&](const fence_instruction_record& finstr) {
				    if(finstr.tid == tid) { instructions_to_keep.insert(finstr.id); }
			    },
			    [&](const horizon_instruction_record& hinstr) {
				    if(hinstr.horizon_task_id == tid) { instructions_to_keep.insert(hinstr.id); }
			    },
			    [&](const epoch_instruction_record& einstr) {
				    if(einstr.epoch_task_id == tid) { instructions_to_keep.insert(einstr.id); }
			    },
			    [&](const auto&) {
				    // nop
			    });
		}
		if(instructions_to_keep.empty()) {
			fprintf(stderr, "Failed to filter by task id: No instructions for task %zu found\n", tid.value);
			return;
		}

		enum class visit_direction { forward, backward };
		const auto visit_all = [&](const visit_direction dir) {
			// TODO: We may end up visiting some nodes multiple times. In same cases we have to, b/c we may have found a shorter path.
			std::unordered_set<std::pair<instruction_id, size_t>, utils::pair_hash> to_visit;
			for(const auto id : instructions_to_keep) {
				to_visit.insert({id, 0});
			}
			while(!to_visit.empty()) {
				const auto [id, distance] = *to_visit.begin();
				to_visit.erase(to_visit.begin());
				instructions_to_keep.insert(id);
				if(dir == visit_direction::backward && distance < before) {
					// FIXME: Oof
					for(const auto& dep : m_recorded_dependencies) {
						if(dep.successor == id) { to_visit.insert({dep.predecessor, distance + 1}); }
					}
				}
				if(dir == visit_direction::forward && distance < after) {
					// FIXME: Oof
					for(const auto& dep : m_recorded_dependencies) {
						if(dep.predecessor == id) { to_visit.insert({dep.successor, distance + 1}); }
					}
				}
			}
		};
		visit_all(visit_direction::forward);
		visit_all(visit_direction::backward);

		std::vector<std::unique_ptr<instruction_record>> records_to_keep;
		std::unordered_set<message_id> message_ids_to_keep;
		for(auto& instr : m_recorded_instructions) {
			if(instructions_to_keep.contains(instr->id)) {
				if(auto sinstr = dynamic_cast<send_instruction_record*>(instr.get())) { message_ids_to_keep.insert(sinstr->message_id); }
				records_to_keep.push_back(std::move(instr));
			}
		}
		std::vector<instruction_dependency_record> dependencies_to_keep;
		for(const auto& dep : m_recorded_dependencies) {
			if(instructions_to_keep.contains(dep.predecessor) && instructions_to_keep.contains(dep.successor)) { dependencies_to_keep.push_back(dep); }
		}
		std::vector<outbound_pilot> pilots_to_keep;
		for(const auto& pilot : m_recorded_pilots) {
			if(message_ids_to_keep.contains(pilot.message.id)) { pilots_to_keep.push_back(pilot); }
		}

		m_recorded_instructions = std::move(records_to_keep);
		m_recorded_dependencies = std::move(dependencies_to_keep);
		m_recorded_pilots = std::move(pilots_to_keep);
	}

  private:
	bool m_loop_template_active = false;
	std::vector<std::unique_ptr<instruction_record>> m_recorded_instructions;
	instruction_id m_highest_recorded_instruction_id = std::numeric_limits<instruction_id::value_type>::max(); // overflows on first increment
	std::vector<instruction_dependency_record> m_recorded_dependencies;
	std::vector<outbound_pilot> m_recorded_pilots;
	std::unordered_map<transfer_id, command_id> m_await_push_cids;
	dense_map<task_id, instruction_id> m_task_boundaries; ///< The first instruction id of each task
};

class instruction_performance_recorder {
  public:
	using duration = std::chrono::steady_clock::duration;

	instruction_performance_recorder(const size_t num_nodes, const node_id local_nid) : m_num_nodes(num_nodes), m_local_nid(local_nid) {
		// Make some initial space
		m_execution_times.resize(1000);
	}

	void record_execution_time(const instruction_id iid, const duration time) {
		if(iid >= m_execution_times.size()) { m_execution_times.resize(m_execution_times.size() * 2); }
		m_execution_times[iid] = time;
	}

	duration get_execution_time(const instruction_id iid) const { return m_execution_times[iid]; }

	// TODO: Move out of recorder
	void print_summary(const instruction_recorder& irec, const task_recorder& trec) const;

  private:
	size_t m_num_nodes;
	node_id m_local_nid;
	dense_map<instruction_id, duration> m_execution_times;
};

} // namespace celerity::detail
