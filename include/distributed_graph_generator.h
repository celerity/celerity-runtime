#pragma once

#include <bitset>
#include <unordered_map>
#include <variant>

#include "command_graph.h"
#include "ranges.h"
#include "region_map.h"
#include "types.h"

namespace celerity::detail {

class task;
class task_manager;
class abstract_command;
class task_recorder;
class command_recorder;

// TODO: Make compile-time configurable
constexpr size_t max_num_nodes = 256;
using node_bitset = std::bitset<max_num_nodes>;

/**
 * write_command_state is a command_id with two bits of additional information:
 *   - Whether the data written by this command is globally still the newest version ("fresh" or "stale")
 *   - Whether this data has been replicated from somewhere else, i.e., we are not the original producer
 */
class write_command_state {
	constexpr static int64_t fresh_bit = 1ll << 63;
	constexpr static int64_t replicated_bit = 1ll << 62;
	static_assert(sizeof(fresh_bit) == sizeof(command_id));

  public:
	constexpr write_command_state() = default;

	/* explicit(false) */ constexpr write_command_state(command_id cid) : m_cid(cid) {}

	constexpr write_command_state(command_id cid, bool is_replicated) : m_cid(cid) {
		if(is_replicated) { m_cid |= replicated_bit; }
	}

	bool is_fresh() const { return (m_cid & fresh_bit) == 0u; }

	bool is_replicated() const { return (m_cid & replicated_bit) != 0u; }

	void mark_as_stale() { m_cid |= fresh_bit; }

	operator command_id() const { return m_cid & ~fresh_bit & ~replicated_bit; }

	friend bool operator==(const write_command_state& lhs, const write_command_state& rhs) { return lhs.m_cid == rhs.m_cid; }

	friend bool operator!=(const write_command_state& lhs, const write_command_state& rhs) { return !(lhs == rhs); }

  private:
	command_id m_cid = 0;
};

class distributed_graph_generator {
	friend struct distributed_graph_generator_testspy;

	inline static const write_command_state no_command = write_command_state(static_cast<command_id>(-1));

	struct buffer_state {
		buffer_state(region_map<write_command_state> lw, region_map<std::bitset<max_num_nodes>> rr)
		    : local_last_writer(std::move(lw)), replicated_regions(std::move(rr)), pending_reduction(std::nullopt) {}

		region_map<write_command_state> local_last_writer;
		region_map<node_bitset> replicated_regions;

		// When a buffer is used as the output of a reduction, we do not insert reduction_commands right away,
		// but mark it as having a pending reduction. The final reduction will then be generated when the buffer
		// is used in a subsequent read requirement. This avoids generating unnecessary reduction commands.
		std::optional<reduction_info> pending_reduction;
	};

  public:
	distributed_graph_generator(
	    const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm, detail::command_recorder* recorder);

	void add_buffer(const buffer_id bid, const int dims, const range<3>& range);

	std::unordered_set<abstract_command*> build_task(const task& tsk);

	command_graph& get_command_graph() { return m_cdag; }

  private:
	// Wrapper around command_graph::create that adds commands to current batch set.
	template <typename T, typename... Args>
	T* create_command(Args&&... args) {
		auto* const cmd = m_cdag.create<T>(std::forward<Args>(args)...);
		m_current_cmd_batch.insert(cmd);
		return cmd;
	}

	/**
	 * Generates command(s) that need to be processed by every node in the system,
	 * because they may require data transfers.
	 */
	void generate_distributed_commands(const task& tsk);

	void generate_anti_dependencies(
	    task_id tid, buffer_id bid, const region_map<write_command_state>& last_writers_map, const region<3>& write_req, abstract_command* write_cmd);

	void process_task_side_effect_requirements(const task& tsk);

	void set_epoch_for_new_commands(const abstract_command* const epoch_or_horizon);

	void reduce_execution_front_to(abstract_command* const new_front);

	void generate_epoch_command(const task& tsk);

	void generate_horizon_command(const task& tsk);

	void generate_epoch_dependencies(abstract_command* cmd);

	void prune_commands_before(const command_id epoch);

  private:
	using buffer_read_map = std::unordered_map<buffer_id, region<3>>;
	using side_effect_map = std::unordered_map<host_object_id, command_id>;

	size_t m_num_nodes;
	node_id m_local_nid;
	command_graph& m_cdag;
	const task_manager& m_task_mngr;
	std::unordered_map<buffer_id, buffer_state> m_buffer_states;
	command_id m_epoch_for_new_commands = 0;
	command_id m_epoch_last_pruned_before = 0;
	command_id m_current_horizon = no_command;

	// Batch of commands currently being generated. Returned (and thereby emptied) by build_task().
	std::unordered_set<abstract_command*> m_current_cmd_batch;

	// For proper handling of anti-dependencies we also have to store for each command which buffer regions it reads.
	// We do this because we cannot reconstruct the requirements from a command within the graph alone (e.g. for compute commands).
	// While we could apply range mappers again etc., that is a bit wasteful. This is basically an optimization.
	std::unordered_map<command_id, buffer_read_map> m_command_buffer_reads;

	// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
	// they are executed in the same order on every node.
	std::unordered_map<collective_group_id, command_id> m_last_collective_commands;

	// Side effects on the same host object create true dependencies between task commands, so we track the last effect per host object.
	side_effect_map m_host_object_last_effects;

	// Generated commands will be recorded to this recorder if it is set
	detail::command_recorder* m_recorder = nullptr;
};

} // namespace celerity::detail

namespace std {
template <>
struct hash<celerity::detail::write_command_state> {
	size_t operator()(const celerity::detail::write_command_state& wcs) const { return std::hash<size_t>{}(static_cast<celerity::detail::command_id>(wcs)); }
};
} // namespace std