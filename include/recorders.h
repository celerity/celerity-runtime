#pragma once

#include "command.h"
#include "task.h"

#include <functional>


namespace celerity::detail {

class buffer_manager;
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
	std::optional<access_list> accesses;
	std::optional<reduction_list> reductions;
	std::optional<side_effect_map> side_effects;
	command_dependency_list dependencies;
	std::string task_name;
	std::optional<detail::task_type> task_type;
	std::optional<detail::collective_group_id> collective_group_id;

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

} // namespace celerity::detail
