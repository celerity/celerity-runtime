#pragma once

#include "command_graph.h"
#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "recorders.h"
#include "reduction.h"
#include "region_map.h"
#include "types.h"
#include "utils.h"

#include <bitset>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>


namespace celerity::detail {

class task;
class command;
class task_recorder;
class command_recorder;

// TODO: Make compile-time configurable
constexpr size_t max_num_nodes = 256;
using node_bitset = std::bitset<max_num_nodes>;

/**
 * write_command_state is a command pointer with two bits of additional information:
 *   - Whether the data written by this command is globally still the newest version ("fresh" or "stale")
 *   - Whether this data has been replicated from somewhere else, i.e., we are not the original producer
 */
class write_command_state {
	friend struct std::hash<celerity::detail::write_command_state>;

	static_assert(alignof(command) > 0b11); // so we have 2 spare bits to encode the masks below
	constexpr static uintptr_t stale_bit = 0b01;
	constexpr static uintptr_t replicated_bit = 0b10;

  public:
	constexpr write_command_state() = default;

	explicit(false) write_command_state(command* const cmd) : m_bits(reinterpret_cast<uintptr_t>(cmd)) {}

	write_command_state(command* const cmd, bool is_replicated) : m_bits(reinterpret_cast<uintptr_t>(cmd)) {
		if(is_replicated) { m_bits |= replicated_bit; }
	}

	command* get_command() const { return reinterpret_cast<command*>(m_bits & ~stale_bit & ~replicated_bit); }

	bool is_fresh() const { return (m_bits & stale_bit) == 0u; }

	bool is_replicated() const { return (m_bits & replicated_bit) != 0u; }

	void mark_as_stale() { m_bits |= stale_bit; }

	operator command*() const { return get_command(); }

	friend bool operator==(const write_command_state& lhs, const write_command_state& rhs) { return lhs.m_bits == rhs.m_bits; }
	friend bool operator==(const write_command_state& lhs, const std::nullptr_t /* rhs */) { return lhs.get_command() == nullptr; }
	friend bool operator==(const std::nullptr_t /* lhs */, const write_command_state& rhs) { return rhs.get_command() == nullptr; }

  private:
	uintptr_t m_bits = 0;
};

class command_graph_generator {
	friend struct command_graph_generator_testspy;

	inline static const write_command_state no_command = {};

	struct buffer_state {
		explicit buffer_state(const range<3>& range, const write_command_state initial_wcs, const node_bitset& replicated_on_nodes)
		    : local_last_writer(range, initial_wcs), replicated_regions(range, replicated_on_nodes) {}

		region<3> initialized_region; // for detecting uninitialized reads (only if policies.uninitialized_read != error_policy::ignore)
		region_map<write_command_state> local_last_writer;
		region_map<node_bitset> replicated_regions;

		// When a buffer is used as the output of a reduction, we do not insert reduction_commands right away,
		// but mark it as having a pending reduction. The final reduction will then be generated when the buffer
		// is used in a subsequent read requirement. This avoids generating unnecessary reduction commands.
		std::optional<reduction_info> pending_reduction;

		std::string debug_name;
	};

	struct host_object_state {
		explicit host_object_state(command* const initial_writer) : last_side_effect(initial_writer) {}

		// Side effects on the same host object create true dependencies between task commands, so we track the last effect per host object.
		command* last_side_effect;
	};

	struct collective_group_state {
		explicit collective_group_state(command* const initial_command) : last_collective_command(initial_command) {}

		// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
		// they are executed in the same order on every node.
		command* last_collective_command;
	};

  public:
	struct policy_set {
		error_policy uninitialized_read_error = error_policy::panic;
		error_policy overlapping_write_error = error_policy::panic;
	};

	command_graph_generator(const size_t num_nodes, const node_id local_nid, command_graph& cdag, detail::command_recorder* recorder,
	    const policy_set& policy = default_policy_set());

	void notify_buffer_created(buffer_id bid, const range<3>& range, bool host_initialized);

	void notify_buffer_debug_name_changed(buffer_id bid, const std::string& debug_name);

	void notify_buffer_destroyed(buffer_id bid);

	void notify_host_object_created(host_object_id hoid);

	void notify_host_object_destroyed(host_object_id hoid);

	/// Generates the set of commands required to execute the given task.
	/// This includes resolving local data dependencies, generating await push commands for local reads of remote data,
	/// as well as push commands for remote reads of local data.
	/// Commands are returned in topologically sorted order, i.e., sequential execution would satisfy all internal dependencies.
	std::vector<const command*> build_task(const task& tsk);

	/// Only for testing: Instead of (at most) a single chunk per node, generate `multiplier` chunks per node for each task.
	/// This is to ensure that CDAG generation logic works for arbitrary numbers of chunks, even though we don't provide
	/// a way for users to specify more than one chunk per node... yet.
	void test_set_chunk_multiplier(const size_t multiplier) {
		assert(multiplier > 0);
		m_test_chunk_multiplier = multiplier;
	}

  private:
	using batch = std::vector<const command*>;

	/// True if a recorder is present and create_command() will call the `record_with` lambda passed as its last parameter.
	bool is_recording() const { return m_recorder != nullptr; }

	/// Maps command DAG types to their record type.
	template <typename Command>
	using record_type_for_t = utils::type_switch_t<Command, push_command(push_command_record), await_push_command(await_push_command_record),
	    reduction_command(reduction_command_record), epoch_command(epoch_command_record), horizon_command(horizon_command_record),
	    execution_command(execution_command_record), fence_command(fence_command_record)>;

	template <std::derived_from<command> Command, typename... CtorParamsAndRecordWithFn, size_t... CtorParamIndices, size_t RecordWithFnIndex>
	Command* create_command_internal(batch& current_batch, std::tuple<CtorParamsAndRecordWithFn...>&& ctor_params_and_record_with,
	    std::index_sequence<CtorParamIndices...> /* ctor_param_indices */, std::index_sequence<RecordWithFnIndex> /* record_with_fn_index */) //
	{
		const auto cid = m_next_cid++;
		auto unique_cmd = std::make_unique<Command>(cid, std::move(std::get<CtorParamIndices>(ctor_params_and_record_with))...);
		const auto cmd = m_cdag->retain_in_current_epoch(std::move(unique_cmd));
		m_execution_front.insert(cmd);
		current_batch.push_back(cmd);

		if(is_recording()) {
			const auto& record_with = std::get<RecordWithFnIndex>(ctor_params_and_record_with);
			[[maybe_unused]] bool recorded = false;
			record_with([&](auto&&... debug_info) {
				m_recorder->record_command(
				    std::make_unique<record_type_for_t<Command>>(std::as_const(*cmd), std::forward<decltype(debug_info)>(debug_info)...));
				recorded = true;
			});
			assert(recorded && "record_debug_info() not called within recording function");
		}
		return cmd;
	}

	/// Wrapper around command_graph::create that adds commands to the current batch.
	/// Records the command if a recorder is present.
	///
	/// Invoke as
	/// ```
	/// create<command-type>(command-ctor-params,
	///     [&](const auto record_debug_info) { return record_debug_info(command-record-additional-ctor-params)})
	/// ```
	template <std::derived_from<command> Command, typename... CtorParamsAndRecordWithFn>
	Command* create_command(batch& current_batch, CtorParamsAndRecordWithFn&&... args) {
		constexpr auto n_args = sizeof...(CtorParamsAndRecordWithFn);
		static_assert(n_args > 0);
		return create_command_internal<Command>(current_batch, std::forward_as_tuple(std::forward<CtorParamsAndRecordWithFn>(args)...),
		    std::make_index_sequence<n_args - 1>{}, std::index_sequence<n_args - 1>{});
	}

	/// Adds a new dependency between two commands and records it if recording is enabled.
	void add_dependency(command* const from, command* const to, dependency_kind kind, dependency_origin origin) {
		assert(to != from);
		from->add_dependency({to, kind, origin});
		m_execution_front.erase(to);

		// Sanity check: For non-dataflow dependencies the commands can only be of specific types
		if(origin == dependency_origin::execution_front) { assert(utils::isa<epoch_command>(from) || utils::isa<horizon_command>(from)); }
		if(origin == dependency_origin::collective_group_serialization) {
			assert(utils::isa<execution_command>(from));
			// The original execution command may have been subsumed by a horizon / epoch
			assert(utils::isa<execution_command>(to) || utils::isa<epoch_command>(to) || utils::isa<horizon_command>(to));
		}
		if(origin == dependency_origin::last_epoch) { assert(utils::isa<epoch_command>(to) || utils::isa<horizon_command>(to)); }

		// Sanity check for unit tests, where we may have multiple CDAGS
		// TODO assert(m_commands.at(from->get_id()).get() == depender);
		// TODO assert(m_commands.at(to->get_id()).get() == dependee);
		if(is_recording()) { m_recorder->record_dependency(command_dependency_record(to->get_id(), from->get_id(), kind, origin)); }
	}

	struct assigned_chunk {
		node_id executed_on = -1;
		chunk<3> chnk;
	};

	struct buffer_requirements {
		buffer_id bid = -1;
		region<3> consumed;
		region<3> produced;
	};

	using buffer_requirements_list = std::vector<buffer_requirements>;

	struct assigned_chunks_with_requirements {
		using with_requirements = std::pair<assigned_chunk, buffer_requirements_list>;

		// We process both local (to be executed on this node) and remote (to be execute on other nodes) chunks.
		// The latter are required to determine whether we currently own data that needs to be pushed to other nodes.
		std::vector<with_requirements> local_chunks;
		std::vector<with_requirements> remote_chunks;
	};

	std::vector<assigned_chunk> split_task_and_assign_chunks(const task& tsk) const;

	buffer_requirements_list get_buffer_requirements_for_mapped_access(const task& tsk, const subrange<3>& sr) const;

	assigned_chunks_with_requirements compute_per_chunk_requirements(const task& tsk, const std::vector<assigned_chunk>& chunks) const;

	/// Resolve requirements on buffers with pending reductions.
	/// For local chunks, create a reduction command and a single await_push command that receives the partial reduction results from all other nodes.
	/// For remote chunks, always create a push command, regardless of whether we own a partial reduction result or not.
	/// This is required because remote nodes do not know how many partial reduction results there are.
	void resolve_pending_reductions(batch& current_batch, const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements);

	/// For all remote chunks, find read requirements intersecting with owned buffer regions and generate push commands for those regions.
	void generate_pushes(batch& current_batch, const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements);

	/// For all local chunks, find read requirements on remote data.
	/// Generate a single await push command for each buffer that awaits the entire required region.
	/// This will then be fulfilled by one or more incoming pushes.
	void generate_await_pushes(batch& current_batch, const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements);

	/// Determine which local data is fresh or stale by comparing global (task-level) and local writes.
	void update_local_buffer_fresh_regions(const task& tsk, const std::unordered_map<buffer_id, region<3>>& per_buffer_local_writes);

	/**
	 * Generates command(s) that need to be processed by every node in the system,
	 * because they may require data transfers.
	 */
	void generate_distributed_commands(batch& current_batch, const task& tsk);

	void generate_anti_dependencies(
	    const task& tsk, buffer_id bid, const region_map<write_command_state>& last_writers_map, const region<3>& write_req, command* write_cmd);

	void set_epoch_for_new_commands(command* const epoch_or_horizon);

	void reduce_execution_front_to(command* const new_front);

	void generate_epoch_command(batch& current_batch, const task& tsk);

	void generate_horizon_command(batch& current_batch, const task& tsk);

	void generate_epoch_dependencies(command* cmd);

	void report_overlapping_writes(const task& tsk, const box_vector<3>& local_chunks) const;

  private:
	using buffer_read_map = std::unordered_map<buffer_id, region<3>>;

	// default-constructs a policy_set - this must be a function because we can't use the implicit default constructor of policy_set, which has member
	// initializers, within its surrounding class (Clang)
	constexpr static policy_set default_policy_set() { return {}; }

	// In the master/worker model, we used to try and find the node best suited for initializing multiple
	// reductions that do not initialize_to_identity based on current data distribution.
	// This is more difficult in a distributed setting, so for now we just hard code it to node 0.
	// TODO: Revisit this at some point.
	constexpr static node_id reduction_initializer_nid = 0;

	std::string print_buffer_debug_label(buffer_id bid) const;

	size_t m_num_nodes;
	node_id m_local_nid;
	policy_set m_policy;

	command_id m_next_cid = 0;
	command_graph* m_cdag;
	command_set m_execution_front;

	std::unordered_map<buffer_id, buffer_state> m_buffers;
	std::unordered_map<host_object_id, host_object_state> m_host_objects;
	std::unordered_map<collective_group_id, collective_group_state> m_collective_groups;

	command* m_epoch_for_new_commands = nullptr;
	command_id m_epoch_last_pruned_before = 0;
	command* m_current_horizon = nullptr;

	size_t m_test_chunk_multiplier = 1;

	// List of reductions that have either completed globally or whose result has been discarded. This list will be appended to the next horizon to eventually
	// inform the instruction executor that it can safely garbage-collect runtime info on the reduction operation.
	std::vector<reduction_id> m_completed_reductions;

	// For proper handling of anti-dependencies we also have to store for each command which buffer regions it reads.
	// We do this because we cannot reconstruct the requirements from a command within the graph alone (e.g. for compute commands).
	// While we could apply range mappers again etc., that is a bit wasteful. This is basically an optimization.
	std::unordered_map<command_id, buffer_read_map> m_command_buffer_reads;

	// Generated commands will be recorded to this recorder if it is set
	detail::command_recorder* m_recorder = nullptr;
};

} // namespace celerity::detail

namespace std {
template <>
struct hash<celerity::detail::write_command_state> {
	size_t operator()(const celerity::detail::write_command_state& wcs) const { return std::hash<uintptr_t>{}(wcs.m_bits); }
};

} // namespace std
