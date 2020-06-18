#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "region_map.h"
#include "task.h"
#include "types.h"


namespace celerity {
namespace detail {

	class task_manager;
	class graph_transformer;
	class command_graph;
	class abstract_command;
	class horizon_command;

	class graph_generator {
		friend struct graph_generator_testspy;

		using buffer_state_map = std::unordered_map<buffer_id, region_map<std::vector<node_id>>>;
		using buffer_read_map = std::unordered_map<buffer_id, GridRegion<3>>;
		using buffer_writer_map = std::unordered_map<buffer_id, region_map<std::optional<command_id>>>;

		struct per_node_data {
			// For each node we generate a separate INIT command which acts as a proxy last writer for host-initialized buffers.
			command_id init_cid;
			// We store for each node which command last wrote to a buffer region. This includes both newly generated data (from a execution command),
			// as well as already existing data that was pushed in from another node. This is used for determining anti-dependencies.
			buffer_writer_map buffer_last_writer;
		};

	  public:
		/**
		 * @param num_nodes Number of CELERITY nodes, including the master node.
		 * @param tm
		 * @param cdag The command graph this generator should operate on.
		 */
		graph_generator(size_t num_nodes, task_manager& tm, command_graph& cdag);

		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range);

		// Build the commands for a single task
		void build_task(task_id tid, const std::vector<graph_transformer*>& transformers);

		void set_horizon_step_size(unsigned step_size) { horizon_step_size = step_size; }

	  private:
		task_manager& task_mngr;
		const size_t num_nodes;
		command_graph& cdag;

		// Number of cpath steps which should occur before a new horizon is inserted.
		unsigned horizon_step_size = 4;
		// Keeps track of the "position" of the previous horizon to allow inserting new horizons with a controlled frequency.
		unsigned prev_horizon_cpath_max = 0;
		// The most recent horizon command per node.
		std::vector<horizon_command*> prev_horizon_cmds;
		// The id for the next cleanup horizon (after which we can delete commands)
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

		// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
		// they are executed in the same order on every node.
		std::unordered_map<std::pair<node_id, collective_group_id>, command_id, boost::hash<std::pair<node_id, collective_group_id>>> last_collective_commands;

		// This mutex mainly serves to protect per-buffer data structures, as new buffers might be added at any time.
		std::mutex buffer_mutex;

		void generate_anti_dependencies(task_id tid, buffer_id bid, const region_map<std::optional<command_id>>& last_writers_map,
		    const GridRegion<3>& write_req, abstract_command* write_cmd);

		void process_task_data_requirements(task_id tid);

		bool should_generate_horizon() const;
		void generate_horizon();
	};

} // namespace detail
} // namespace celerity
