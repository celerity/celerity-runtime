#pragma once

#include <memory>
#include <string>

#include "intrusive_graph.h"
#include "task.h"
#include "task_ring_buffer.h"

namespace celerity {
namespace detail {

	class buffer_manager;
	class command_graph;
	class task_manager;

	struct task_printing_information {
		struct access_record {
			const buffer_id bid;
			const access_mode mode;
			const GridRegion<3> req;
		};
		using access_list = std::vector<access_record>;

		task_printing_information(const task& from);

		const task_id m_tid;
		const std::string m_debug_name;
		const collective_group_id m_cgid;
		const task_type m_type;
		const task_geometry m_geometry;
		const reduction_set m_reductions;
		const access_list m_accesses;
		const side_effect_map m_side_effect_map;
		const gch::small_vector<intrusive_graph_node<task>::dependency> m_dependencies;
	};


	class task_recorder {
	  public:
		using task_record = std::vector<task_printing_information>;

		void record_task(const task& from);

		const task_record& get_tasks() const { return m_recorded_tasks; }

	  private:
		task_record m_recorded_tasks;
	};

	void set_recording(bool enabled);
	void record_buffer_name(const buffer_id bid, const std::string& name);

	std::string print_task_graph(const task_recorder& recorder);
	std::string print_command_graph(const node_id local_nid, const command_graph& cdag, const task_manager& tm, const buffer_manager* bm);
	std::string combine_command_graphs(const std::vector<std::string>& graphs);

} // namespace detail
} // namespace celerity
