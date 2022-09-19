#pragma once

#include <unordered_map>

#include "ranges.h"
#include "region_map.h"
#include "types.h"

namespace celerity::detail {

class task;
class task_manager;
class command_graph;
class abstract_command;

class distributed_graph_generator {
  public:
	distributed_graph_generator(const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm)
	    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_cdag(cdag), m_task_mngr(tm) {}

	void add_buffer(const buffer_id bid, const range<3>& range);

	void build_task(const task& tsk);

	command_graph& NOCOMMIT_get_cdag() { return m_cdag; }

  private:
	void reduce_execution_front_to(abstract_command* const new_front);

	void generate_epoch_command(const task& tsk);

  private:
	size_t m_num_nodes;
	node_id m_local_nid;
	command_graph& m_cdag;
	const task_manager& m_task_mngr;
	std::unordered_map<buffer_id, region_map<command_id>> m_buffer_last_writer; // NOCOMMIT Naming
	// This is an optimization; we could also walk the task graph to figure this out each time we don't have newest data locally.
	std::unordered_map<buffer_id, region_map<task_id>> m_buffer_last_writer_task; // NOCOMMIT Naming
};
} // namespace celerity::detail