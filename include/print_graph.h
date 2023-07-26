#pragma once

#include <memory>
#include <string>

#include "command.h"
#include "intrusive_graph.h"
#include "task.h"
#include "task_ring_buffer.h"

namespace celerity {
namespace detail {

	class buffer_manager;
	class command_graph;
	class task_manager;

	// General recording

	struct access_record {
		const buffer_id bid;
		const std::string buffer_name;
		const access_mode mode;
		const GridRegion<3> req;
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
		IdType node;
		dependency_kind kind;
		dependency_origin origin;
	};

	// Task recording

	using task_dependency_list = std::vector<dependency_record<task_id>>;

	struct task_printing_information {
		task_printing_information(const task& from, const buffer_manager* buff_man);

		const task_id m_tid;
		const std::string m_debug_name;
		const collective_group_id m_cgid;
		const task_type m_type;
		const task_geometry m_geometry;
		const reduction_list m_reductions;
		const access_list m_accesses;
		const side_effect_map m_side_effect_map;
		const task_dependency_list m_dependencies;
	};

	class task_recorder {
	  public:
		using task_record = std::vector<task_printing_information>;

		task_recorder(const buffer_manager* buff_man = nullptr) : m_buff_man(buff_man) {}

		void record_task(const task& tsk);

		const task_record& get_tasks() const { return m_recorded_tasks; }

	  private:
		task_record m_recorded_tasks;
		const buffer_manager* m_buff_man;
	};

	const std::optional<task_recorder> no_task_recorder = {};

	// Command recording

	using command_dependency_list = std::vector<dependency_record<command_id>>;

	struct command_printing_information {
		const command_id m_cid;
		const command_type m_type;

		const std::optional<epoch_action> m_epoch_action;
		const std::optional<subrange<3>> m_execution_range;
		const std::optional<reduction_id> m_reduction_id;
		const std::optional<buffer_id> m_buffer_id;
		const std::string m_buffer_name;
		const std::optional<node_id> m_target;
		const std::optional<GridRegion<3>> m_await_region;
		const std::optional<subrange<3>> m_push_range;
		const std::optional<transfer_id> m_transfer_id;
		const std::optional<task_id> m_task_id;
		const std::optional<task_geometry> m_task_geometry;
		const bool m_is_reduction_initializer;
		const std::optional<access_list> m_accesses;
		const std::optional<reduction_list> m_reductions;
		const std::optional<side_effect_map> m_side_effects;
		const command_dependency_list m_dependencies;
		const std::string m_task_name;
		const std::optional<task_type> m_task_type;
		const std::optional<collective_group_id> m_collective_group_id;

		command_printing_information(const abstract_command& cmd, const task_manager* task_man, const buffer_manager* buff_man);
	};

	class command_recorder {
	  public:
		using command_record = std::vector<command_printing_information>;

		command_recorder(const task_manager* t_man, const buffer_manager* buff_man = nullptr) : m_task_man(t_man), m_buff_man(buff_man) {}

		void record_command(const abstract_command& com);

		const command_record& get_commands() const { return m_recorded_commands; }

	  private:
		command_record m_recorded_commands;
		const task_manager* m_task_man;
		const buffer_manager* m_buff_man;
	};

	const std::optional<command_recorder> no_command_recorder = {};

	// Printing interface

	std::string print_task_graph(const task_recorder& recorder);
	std::string print_command_graph(const node_id local_nid, const command_recorder& recorder);
	std::string combine_command_graphs(const std::vector<std::string>& graphs);

} // namespace detail
} // namespace celerity
