#pragma once

#include "command.h"
#include "task.h"

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
	task_record(const task& from, const buffer_manager* buff_mngr);

	const task_id tid;
	const std::string debug_name;
	const collective_group_id cgid;
	const task_type type;
	const task_geometry geometry;
	const reduction_list reductions;
	const access_list accesses;
	const side_effect_map side_effect_map;
	const task_dependency_list dependencies;
};

class task_recorder {
  public:
	using task_record = std::vector<task_record>;

	task_recorder(const buffer_manager* buff_mngr = nullptr) : m_buff_mngr(buff_mngr) {}

	void record_task(const task& tsk);

	const task_record& get_tasks() const { return m_recorded_tasks; }

  private:
	task_record m_recorded_tasks;
	const buffer_manager* m_buff_mngr;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_record {
	const command_id cid;
	const command_type type;

	const std::optional<epoch_action> epoch_action;
	const std::optional<subrange<3>> execution_range;
	const std::optional<reduction_id> reduction_id;
	const std::optional<buffer_id> buffer_id;
	const std::string buffer_name;
	const std::optional<node_id> target;
	const std::optional<region<3>> await_region;
	const std::optional<subrange<3>> push_range;
	const std::optional<transfer_id> transfer_id;
	const std::optional<task_id> task_id;
	const std::optional<task_geometry> task_geometry;
	const bool is_reduction_initializer;
	const std::optional<access_list> accesses;
	const std::optional<reduction_list> reductions;
	const std::optional<side_effect_map> side_effects;
	const command_dependency_list dependencies;
	const std::string task_name;
	const std::optional<task_type> task_type;
	const std::optional<collective_group_id> collective_group_id;

	command_record(const abstract_command& cmd, const task_manager* task_mngr, const buffer_manager* buff_mngr);
};

class command_recorder {
  public:
	using command_record = std::vector<command_record>;

	command_recorder(const task_manager* task_mngr, const buffer_manager* buff_mngr = nullptr) : m_task_mngr(task_mngr), m_buff_mngr(buff_mngr) {}

	void record_command(const abstract_command& com);

	const command_record& get_commands() const { return m_recorded_commands; }

  private:
	command_record m_recorded_commands;
	const task_manager* m_task_mngr;
	const buffer_manager* m_buff_mngr;
};

} // namespace celerity::detail
