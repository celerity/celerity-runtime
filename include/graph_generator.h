#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <boost/optional.hpp>

#include "graph.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

#include "transformers/naive_split.h"

namespace celerity {
namespace detail {

	// Data structure used to map valid buffer regions to the node(s) that currently hold that region, as well as the commands that produced the region.
	struct valid_buffer_source {
		node_id nid;
		command_id cid;
	};

	inline bool operator==(const valid_buffer_source& vbs1, const valid_buffer_source& vbs2) { return vbs1.nid == vbs2.nid && vbs1.cid == vbs2.cid; }

} // namespace detail
} // namespace celerity

namespace std {

template <>
struct hash<celerity::detail::valid_buffer_source> {
	size_t operator()(const celerity::detail::valid_buffer_source& vbs) const noexcept {
		const auto h1 = std::hash<celerity::node_id>{}(vbs.nid);
		const auto h2 = std::hash<celerity::command_id>{}(vbs.cid);
		return h1 ^ (h2 << 1);
	}
};

} // namespace std

namespace celerity {

class logger;

namespace detail {

	class task_manager;
	class graph_builder;

	std::pair<cdag_vertex, cdag_vertex> create_task_commands(const task_dag& task_graph, command_dag& command_graph, graph_builder& gb, task_id tid);

	class graph_generator {
		using buffer_state_map = std::unordered_map<buffer_id, region_map<std::unordered_set<valid_buffer_source>>>;
		using buffer_read_map = std::unordered_map<buffer_id, std::vector<std::pair<command_id, GridRegion<3>>>>;
		using buffer_writer_map = std::unordered_map<buffer_id, region_map<boost::optional<command_id>>>;
		using flush_callback = std::function<void(node_id, command_pkg, const std::vector<command_id>&)>;

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

		void print_graph(logger& graph_logger);

	  private:
		task_manager& task_mngr;
		const size_t num_nodes;
		command_dag command_graph;
		flush_callback flush_cb;

		// NOTE: We have several data structures that keep track of the "global state" of the distributed program, across all tasks and nodes.
		// While it might seem that this is problematic when the ordering of tasks can be chosen freely (by the scheduler),
		// as long as all dependencies within the task graph are respected, this is in fact fine.

		// This is a data structure which keeps track of where (= on which node) valid regions of a buffer can be found.
		// A valid region is any region that has not been written to on another node.
		// This is updated with every task that is processed.
		buffer_state_map buffer_states;

		// For proper handling of anti-dependencies we also have to store for each task which buffer regions are read by which commands.
		// TODO: Look into freeing this for tasks that won't be needed anymore
		std::unordered_map<task_id, buffer_read_map> task_buffer_reads;

		// We also store for each node which command last wrote to a buffer region. This includes both newly generated data (from a execution command),
		// as well as already existing data that was pushed in from another node. This is used for determining anti-dependencies.
		std::unordered_map<node_id, buffer_writer_map> node_buffer_last_writer;

		std::vector<std::shared_ptr<graph_transformer>> transformers;

		void generate_anti_dependencies(task_id tid, buffer_id bid, const region_map<boost::optional<command_id>>& last_writers_map,
		    const GridRegion<3>& write_req, command_id write_cid, graph_builder& gb);
		void process_task_data_requirements(task_id tid);
	};

} // namespace detail
} // namespace celerity
