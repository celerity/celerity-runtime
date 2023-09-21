#pragma once

#include "command.h"
#include "task.h"

namespace celerity::detail {

class task_manager;

// General recording

struct access_record {
	buffer_id bid;
	access_mode mode;
	region<3> req;
};
using access_list = std::vector<access_record>;

struct reduction_record {
	reduction_id rid;
	buffer_id bid;
	bool init_from_buffer;
};
using reduction_list = std::vector<reduction_record>;

template <typename IdType>
struct dependency_record {
	IdType node;
	dependency_kind kind;
	dependency_origin origin;
};

// Buffer recording

struct buffer_record {
	buffer_id bid;
	std::string debug_name;
};

class buffer_recorder {
  public:
	using buffer_records = std::vector<buffer_record>;

	void create_buffer(const buffer_id bid) { m_recorded_buffers.push_back(buffer_record{bid, {}}); }

	void set_buffer_debug_name(const buffer_id bid, std::string name) {
		const auto it = std::find_if(m_recorded_buffers.begin(), m_recorded_buffers.end(), [bid](const buffer_record& rec) { return rec.bid == bid; });
		assert(it != m_recorded_buffers.end());
		it->debug_name = std::move(name);
	}
	const buffer_records& get_buffers() const { return m_recorded_buffers; }

	const buffer_record& get_buffer(const buffer_id bid) const {
		const auto it = std::find_if(m_recorded_buffers.begin(), m_recorded_buffers.end(), [bid](const buffer_record& rec) { return rec.bid == bid; });
		assert(it != m_recorded_buffers.end());
		return *it;
	}

  private:
	buffer_records m_recorded_buffers;
};

// Task recording

using task_dependency_list = std::vector<dependency_record<task_id>>;

struct task_record {
	explicit task_record(const task& tsk);

	task_id tid;
	std::string debug_name;
	collective_group_id cgid;
	task_type type;
	task_geometry geometry;
	reduction_list reductions;
	access_list accesses;
	side_effect_map side_effect_map;
	task_dependency_list dependencies;
};

class task_recorder {
  public:
	using task_records = std::vector<task_record>;

	void record_task(const task& tsk) { m_recorded_tasks.push_back(task_record(tsk)); }

	const task_records& get_tasks() const { return m_recorded_tasks; }

	const task_record& get_task(const task_id tid) const {
		const auto it = std::find_if(m_recorded_tasks.begin(), m_recorded_tasks.end(), [tid](const task_record& rec) { return rec.tid == tid; });
		assert(it != m_recorded_tasks.end());
		return *it;
	}

  private:
	task_records m_recorded_tasks;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_record {
	command_id cid;
	command_type type;

	std::optional<epoch_action> epoch_action;
	std::optional<subrange<3>> execution_range;
	std::optional<reduction_id> reduction_id;
	std::optional<buffer_id> buffer_id;
	std::optional<node_id> target;
	std::optional<region<3>> await_region;
	std::optional<subrange<3>> push_range;
	std::optional<transfer_id> transfer_id;
	std::optional<task_id> task_id;
	bool is_reduction_initializer;
	std::optional<access_list> accesses;
	command_dependency_list dependencies;

	explicit command_record(const abstract_command& cmd, const task* tsk);
};

class command_recorder {
  public:
	using command_records = std::vector<command_record>;

	void record_command(const abstract_command& cmd, const task* tsk) { m_recorded_commands.push_back(command_record(cmd, tsk)); }

	const command_records& get_commands() const { return m_recorded_commands; }

	const command_record& get_command(const command_id cid) const {
		const auto it = std::find_if(m_recorded_commands.begin(), m_recorded_commands.end(), [cid](const command_record& rec) { return rec.cid == cid; });
		assert(it != m_recorded_commands.end());
		return *it;
	}

  private:
	command_records m_recorded_commands;
};

} // namespace celerity::detail
