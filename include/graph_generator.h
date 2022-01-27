#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <vector>

#include "region_map.h"
#include "task.h"
#include "types.h"


namespace celerity {
namespace detail {

	class task_manager;
	class reduction_manager;
	class graph_transformer;
	class command_graph;
	class abstract_command;
	class horizon_command;

	// TODO: Move to utility header..?
	// Implementation from Boost.ContainerHash, licensed under the Boost Software License, Version 1.0.
	inline void hash_combine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

	struct pair_hash {
		template <typename U, typename V>
		std::size_t operator()(const std::pair<U, V>& p) const {
			std::size_t seed = 0;
			hash_combine(seed, std::hash<U>{}(p.first));
			hash_combine(seed, std::hash<V>{}(p.second));
			return seed;
		}
	};

	class graph_generator {
		friend struct graph_generator_testspy;

		// Common case: the buffer is scattered and/or replicated among nodes.
		struct distributed_state {
			region_map<std::vector<node_id>> region_sources;
		};

		// When a buffer is used as the output of a reduction, we do not insert reduction_commands right away, but mark it as pending_reduction. The final
		// reduction will then be generated when the buffer is used in a subsequent read requirement. This avoids generating unnecessary reduction commands
		// or multi-hop transfers.
		struct pending_reduction_state {
			reduction_id reduction;
			std::vector<node_id> operand_sources;
		};

		// Currently only unit-size reductions are supported, so no buffer can be partially in both a distributed and a pending_reduction state. This means
		// we can avoid any region_map shenanigans for the pending_reduction case.
		using buffer_state = std::variant<distributed_state, pending_reduction_state>;

		using buffer_state_map = std::unordered_map<buffer_id, buffer_state>;
		using buffer_read_map = std::unordered_map<buffer_id, GridRegion<3>>;
		using buffer_writer_map = std::unordered_map<buffer_id, region_map<std::optional<command_id>>>;
		using side_effect_map = std::unordered_map<host_object_id, command_id>;

		struct per_node_data {
			// The epoch command is used as the last writer for host-initialized buffers.
			// To ensure correct ordering, all commands that have no other true-dependencies depend on this command.
			// This is useful so we can correctly generate anti-dependencies onto commands that read host-initialized buffers.
			command_id current_epoch_cid;
			// We store for each node which command last wrote to a buffer region. This includes both newly generated data (from a execution command),
			// as well as already existing data that was pushed in from another node. This is used for determining anti-dependencies.
			buffer_writer_map buffer_last_writer;
			// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
			// they are executed in the same order on every node.
			std::unordered_map<collective_group_id, command_id> last_collective_commands;
			// Side effects on the same host object create true dependencies between task commands, so we track the last effect per host object on each node.
			side_effect_map host_object_last_effects;
		};

	  public:
		/**
		 * @param num_nodes Number of CELERITY nodes, including the master node.
		 * @param tm
		 * @param cdag The command graph this generator should operate on.
		 */
		graph_generator(size_t num_nodes, task_manager& tm, reduction_manager& rm, command_graph& cdag);

		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range);

		// Build the commands for a single task
		void build_task(task_id tid, const std::vector<graph_transformer*>& transformers);

	  private:
		task_manager& task_mngr;
		reduction_manager& reduction_mngr;
		const size_t num_nodes;
		command_graph& cdag;

		// The most recent horizon command per node.
		std::vector<horizon_command*> current_horizon_cmds;
		// The id for the next cleanup horizon (commands with ids lower than the cleanup horizon will be deleted next)
		detail::command_id cleanup_horizon_id = 0;

		// NOTE: We have several data structures that keep track of the "global state" of the distributed program, across all tasks and nodes.
		// While it might seem that this is problematic when the ordering of tasks can be chosen freely (by the scheduler),
		// as long as all dependencies within the task graph are respected, this is in fact fine.

		// This is a data structure which keeps track of where (= on which node) valid regions of a buffer can be found.
		// A valid region is any region that has not been written to on another node.
		// This is updated with every task that is processed.
		buffer_state_map buffer_states;

		// For proper handling of anti-dependencies we also have to store for each command which buffer regions it reads.
		// We do this because we cannot reconstruct the requirements from a command within the graph alone (e.g. for compute commands).
		// While we could apply range mappers again etc., that is a bit wasteful. This is basically an optimization.
		std::unordered_map<command_id, buffer_read_map> command_buffer_reads;

		std::unordered_map<node_id, per_node_data> node_data;

		// This mutex mainly serves to protect per-buffer data structures, as new buffers might be added at any time.
		std::mutex buffer_mutex;

		void generate_anti_dependencies(task_id tid, buffer_id bid, const region_map<std::optional<command_id>>& last_writers_map,
		    const GridRegion<3>& write_req, abstract_command* write_cmd);

		void process_task_data_requirements(task_id tid);

		void process_task_side_effect_requirements(task_id tid);

		void generate_horizon(task_id tid);
	};

} // namespace detail
} // namespace celerity
