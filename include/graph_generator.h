#pragma once

#include <memory>
#include <vector>

#include "buffer_state.h"
#include "graph.h"
#include "task.h"
#include "types.h"

#include "transformers/naive_split.h"

namespace celerity {

class distr_queue;

namespace detail {

	using buffer_state_map = std::unordered_map<buffer_id, std::shared_ptr<buffer_state>>;

	class graph_builder;

	std::pair<cdag_vertex, cdag_vertex> create_task_commands(const task_dag& task_graph, command_dag& command_graph, graph_builder& gb, task_id tid);

	class graph_generator {
	  public:
		/**
		 * @param num_nodes Number of CELERITY nodes, including the master node.
		 */
		graph_generator(size_t num_nodes);

		void set_queue(distr_queue* queue);

		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range);

		void register_transformer(std::shared_ptr<graph_transformer> gt);

		// Build the commands for a single task
		void build_task();

		bool has_unbuilt_tasks() const;

		// FIXME: Currently used for debug graph printing and distributing commands - both of which should be handled internally
		const command_dag& get_command_graph() const { return command_graph; }

	  private:
		distr_queue* queue = nullptr;
		const size_t num_nodes;
		command_dag command_graph;

		// This is a data structure which encodes where (= on which node) valid
		// regions of a buffer can be found after a task has been completed.
		// A valid region is any region that has not been written to on another node.
		std::unordered_map<task_id, buffer_state_map> task_buffer_states;

		// We maintain an additional empty buffer state map for initializing root tasks.
		buffer_state_map empty_buffer_states;

		std::vector<std::shared_ptr<graph_transformer>> transformers;

		void process_task_data_requirements(task_id tid);
	};

} // namespace detail
} // namespace celerity
