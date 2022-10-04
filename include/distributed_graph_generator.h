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
	// write_command_state is basically a command id with one bit of additional information:
	// Whether the data written by this command is globally still the newest version ("fresh")
	// or whether it has been superseded by a command on another node ("stale").
	// => Now it's two bits: Also store whether data is replicated or not.
	class write_command_state {
		constexpr static int64_t mask = 1ull << 63;
		constexpr static int64_t mask2 = 1ull << 62;
		static_assert(sizeof(mask) == sizeof(command_id));

	  public:
		constexpr write_command_state() = default;
		/* explicit(false) */ constexpr write_command_state(command_id cid) : m_cid(cid) {}
		constexpr write_command_state(command_id cid, bool is_replicated) : m_cid(cid) {
			if(is_replicated) { m_cid |= mask2; }
		}

		bool is_fresh() const { return !(m_cid & mask); }

		bool is_replicated() const { return m_cid & mask2; }

		void mark_as_stale() { m_cid |= mask; }

		operator command_id() const { return m_cid & ~mask & ~mask2; }

		bool operator==(const write_command_state& other) const { return m_cid == other.m_cid; }

	  private:
		command_id m_cid;
	};

	inline static write_command_state no_command = write_command_state(static_cast<command_id>(-1));

	struct buffer_state {
		region_map<write_command_state> local_last_writer;
		// This is an optimization; we could also walk the task graph to figure this out each time we don't have newest data locally.
		region_map<task_id> global_last_writer;
	};

  public:
	distributed_graph_generator(const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm)
	    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_cdag(cdag), m_task_mngr(tm), m_per_node_transaction_ids(num_nodes, 0) {}

	void add_buffer(const buffer_id bid, const range<3>& range);

	void build_task(const task& tsk);

	command_graph& NOCOMMIT_get_cdag() { return m_cdag; }

  private:
	void generate_anti_dependencies(
	    task_id tid, buffer_id bid, const region_map<write_command_state>& last_writers_map, const GridRegion<3>& write_req, abstract_command* write_cmd);

	void reduce_execution_front_to(abstract_command* const new_front);

	void generate_epoch_command(const task& tsk);

  private:
	using buffer_read_map = std::unordered_map<buffer_id, GridRegion<3>>;

	size_t m_num_nodes;
	node_id m_local_nid;
	command_graph& m_cdag;
	const task_manager& m_task_mngr;
	std::unordered_map<buffer_id, buffer_state> m_buffer_states;
	std::vector<transaction_id> m_per_node_transaction_ids;

	// For proper handling of anti-dependencies we also have to store for each command which buffer regions it reads.
	// We do this because we cannot reconstruct the requirements from a command within the graph alone (e.g. for compute commands).
	// While we could apply range mappers again etc., that is a bit wasteful. This is basically an optimization.
	std::unordered_map<command_id, buffer_read_map> m_command_buffer_reads;
};

} // namespace celerity::detail