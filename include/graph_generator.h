#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <boost/optional.hpp>

#include "buffer_state.h"
#include "graph.h"
#include "task.h"
#include "types.h"

#include "transformers/naive_split.h"

namespace celerity {

class logger;

namespace detail {

	class task_manager;
	class graph_builder;

	std::pair<cdag_vertex, cdag_vertex> create_task_commands(const task_dag& task_graph, command_dag& command_graph, graph_builder& gb, task_id tid);

	class graph_generator {
		using buffer_state_map = std::unordered_map<buffer_id, std::shared_ptr<buffer_state>>;
		using flush_callback = std::function<void(node_id, command_pkg)>;

	  public:
		/**
		 * @param num_nodes Number of CELERITY nodes, including the master node.
		 * @param tm
		 * @param flush_cb Callback invoked for each command that is being flushed
		 */
		graph_generator(size_t num_nodes, task_manager& tm, flush_callback flush_cb);

		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range);

		void register_transformer(std::shared_ptr<graph_transformer> gt);

		// Build the commands for a single task
		void build_task(task_id tid);

		boost::optional<task_id> get_unbuilt_task() const;

		void flush(task_id tid) const;

		void print_graph(std::shared_ptr<logger>& graph_logger);

	  private:
		task_manager& task_mngr;
		const size_t num_nodes;
		command_dag command_graph;
		flush_callback flush_cb;

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
