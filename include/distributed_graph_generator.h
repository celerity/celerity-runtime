#pragma once

#include <bitset>
#include <unordered_map>

#include "ranges.h"
#include "types.h"

#define USE_COOL_REGION_MAP 1
#if USE_COOL_REGION_MAP
#include "cool_region_map.h"
template <typename T>
using region_map_t = celerity::detail::my_cool_region_map_wrapper<T>;
#else
#include "region_map.h"
template <typename T>
using region_map_t = celerity::detail::region_map<T>;
#endif

namespace celerity::detail {

class task;
class task_manager;
class command_graph;
class abstract_command;

// TODO: Make compile-time configurable
constexpr size_t max_num_nodes = 256;
using node_bitset = std::bitset<max_num_nodes>;

class distributed_graph_generator {
	friend struct distributed_graph_generator_testspy;

  public: // NOCOMMIT need write_command_state public for std::hash specialization
	// write_command_state is basically a command id with one bit of additional information:
	// Whether the data written by this command is globally still the newest version ("fresh")
	// or whether it has been superseded by a command on another node ("stale").
	// => Now it's two bits: Also store whether data is replicated or not (replicated from somewhere else, i.e., we are not the owner)
	// 		=> TODO: Rename to is_owned or something? Or does that conceptually overlap with freshness?
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

		bool operator!=(const write_command_state& other) const { return !(*this == other); }

	  private:
		command_id m_cid;
	};

	inline static write_command_state no_command = write_command_state(static_cast<command_id>(-1));

	struct buffer_state {
		// NOCOMMIT Just a hack for cool region map. get rid of
		buffer_state(region_map_t<write_command_state> lw, region_map_t<std::bitset<max_num_nodes>> rr)
		    : local_last_writer(std::move(lw)), replicated_regions(std::move(rr)) {}

		region_map_t<write_command_state> local_last_writer;
		region_map_t<node_bitset> replicated_regions;
	};

  public:
	distributed_graph_generator(const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm);

	void add_buffer(const buffer_id bid, const range<3>& range, int dims);

	void build_task(const task& tsk);

	command_graph& NOCOMMIT_get_cdag() { return m_cdag; }

  private:
	void generate_execution_commands(const task& tsk);

	void generate_anti_dependencies(
	    task_id tid, buffer_id bid, const region_map_t<write_command_state>& last_writers_map, const GridRegion<3>& write_req, abstract_command* write_cmd);

	void set_epoch_for_new_commands(const abstract_command* const epoch_or_horizon);

	void reduce_execution_front_to(abstract_command* const new_front);

	void generate_epoch_command(const task& tsk);

	void generate_horizon_command(const task& tsk);

	void generate_epoch_dependencies(abstract_command* cmd);

	void prune_commands_before(const command_id epoch);

  private:
	using buffer_read_map = std::unordered_map<buffer_id, GridRegion<3>>;

	size_t m_num_nodes;
	node_id m_local_nid;
	command_graph& m_cdag;
	const task_manager& m_task_mngr;
	std::unordered_map<buffer_id, buffer_state> m_buffer_states;
	command_id m_epoch_for_new_commands = 0;
	command_id m_epoch_last_pruned_before = 0;
	command_id m_current_horizon = no_command;

	// For proper handling of anti-dependencies we also have to store for each command which buffer regions it reads.
	// We do this because we cannot reconstruct the requirements from a command within the graph alone (e.g. for compute commands).
	// While we could apply range mappers again etc., that is a bit wasteful. This is basically an optimization.
	std::unordered_map<command_id, buffer_read_map> m_command_buffer_reads;
};

} // namespace celerity::detail

// NOCOMMIT Just for cool_region_map::get_num_regions - remove again?
namespace std {
template <>
struct hash<celerity::detail::distributed_graph_generator::write_command_state> {
	size_t operator()(const celerity::detail::distributed_graph_generator::write_command_state& wcs) const {
		return std::hash<size_t>{}(static_cast<celerity::detail::command_id>(wcs));
	}
};
} // namespace std