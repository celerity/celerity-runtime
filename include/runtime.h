#pragma once

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <mpi.h>

#include "buffer_state.h"
#include "buffer_storage.h"
#include "buffer_transfer_manager.h"
#include "distr_queue.h"
#include "graph.h"
#include "graph_utils.h"
#include "logger.h"
#include "types.h"
#include "worker_job.h"

namespace celerity {

using chunk_id = size_t;

// FIXME: Untangle these data structures somehow. MSVC already warns about long names (C4503).
using chunk_buffer_requirements_map = std::unordered_map<chunk_id, std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<3>>>>;
using chunk_buffer_source_map = std::unordered_map<chunk_id, std::unordered_map<buffer_id, std::vector<std::pair<GridBox<3>, std::unordered_set<node_id>>>>>;
using buffer_writers_map = std::unordered_map<buffer_id, std::unordered_map<node_id, std::vector<GridRegion<3>>>>;
using buffer_state_map = std::unordered_map<buffer_id, std::unique_ptr<detail::buffer_state>>;

class runtime {
  public:
	static void init(int* argc, char** argv[]);
	static runtime& get_instance();

	~runtime();

	void TEST_do_work();
	void register_queue(distr_queue* queue);
	distr_queue& get_queue();

	template <typename DataT, int Dims>
	buffer_id register_buffer(cl::sycl::range<Dims> size, cl::sycl::buffer<DataT, Dims>& buf) {
		const buffer_id bid = buffer_count++;
		valid_buffer_regions[bid] = std::make_unique<detail::buffer_state>(cl::sycl::range<3>(size), num_nodes);
		buffer_ptrs[bid] = std::make_unique<detail::buffer_storage<DataT, Dims>>(buf);
		return bid;
	}

	void unregister_buffer(buffer_id bid) {
		buffer_ptrs.erase(bid);
		valid_buffer_regions.erase(bid);
	}

	detail::raw_data_read_handle get_buffer_data(buffer_id bid, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) {
		assert(buffer_ptrs.at(bid) != nullptr);
		return buffer_ptrs[bid]->get_data(offset, range);
	}

	void set_buffer_data(buffer_id bid, const detail::raw_data_range& dr) {
		assert(buffer_ptrs.at(bid) != nullptr);
		buffer_ptrs[bid]->set_data(dr);
	}

	std::shared_ptr<logger> get_logger() const { return default_logger; }

  private:
	static std::unique_ptr<runtime> instance;
	std::shared_ptr<logger> default_logger;
	std::shared_ptr<logger> graph_logger;

	distr_queue* queue = nullptr;
	size_t num_nodes;
	bool is_master;

	size_t buffer_count = 0;
	std::unordered_map<buffer_id, std::unique_ptr<detail::buffer_storage_base>> buffer_ptrs;

	// This is a data structure which encodes where (= on which node) valid
	// regions of a buffer can be found. A valid region is any region that has not
	// been written to on another node.
	// NOTE: This represents the buffer regions after all commands in the current
	// command graph have been completed.
	buffer_state_map valid_buffer_regions;

	command_id next_cmd_id = 0;
	command_dag command_graph;
	std::unordered_map<task_id, graph_utils::task_vertices> cmd_dag_task_vertices;

	std::unique_ptr<buffer_transfer_manager> btm;
	job_set jobs;

	runtime(int* argc, char** argv[]);
	runtime(const runtime&) = delete;
	runtime(runtime&&) = delete;

	void build_command_graph();

	void process_task_data_requirements(task_id tid, const std::unordered_map<chunk_id, node_id>& chunk_nodes,
	    const chunk_buffer_requirements_map& chunk_requirements, const chunk_buffer_source_map& chunk_buffer_sources,
	    const std::vector<vertex>& chunk_command_vertices, buffer_writers_map& buffer_writers);

	/**
	 * @brief Sends commands to their designated nodes
	 *
	 * The command graph is traversed in a breadth-first fashion, starting at the root tasks, i.e. tasks without any dependencies.
	 * @param master_command_queue Queue of commands to be executed by the master node
	 */
	void distribute_commands(std::queue<command_pkg>& master_command_queue);

	friend class master_access_job;
	void execute_master_access_task(task_id tid) const;

	void handle_command_pkg(const command_pkg& pkg);

	size_t num_jobs = 0;

	template <typename Job, typename... Args>
	void create_job(const command_pkg& pkg, Args&&... args) {
		auto logger = default_logger->create_context({{"task", std::to_string(pkg.tid)}, {"job", std::to_string(num_jobs)}});
		auto job = std::make_shared<Job>(pkg, logger, std::forward<Args>(args)...);
		job->initialize(*queue, jobs);
		jobs.insert(job);
		num_jobs++;
	}
};

} // namespace celerity
