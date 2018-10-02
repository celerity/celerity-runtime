#pragma once

#include <memory>
#include <queue>
#include <unordered_map>

#include <mpi.h>

#include "buffer_transfer_manager.h"
#include "distr_queue.h"
#include "graph_generator.h"
#include "logger.h"
#include "types.h"
#include "worker_job.h"

namespace celerity {

namespace detail {
	class buffer_storage_base;
}

class runtime {
  public:
	static void init(int* argc, char** argv[]);
	static runtime& get_instance();

	~runtime();

	void TEST_do_work();
	void register_queue(distr_queue* queue);
	distr_queue& get_queue();

	buffer_id register_buffer(cl::sycl::range<3> range, std::shared_ptr<detail::buffer_storage_base> buf_storage);

	/**
	 * Currently this is being called by the distr_queue on shutdown (dtor).
	 * We have to make sure all SYCl objects are free'd before the queue is destroyed.
	 * TODO: Once we get rid of TEST_do_work we'll need an alternative solution that blocks the distr_queue dtor until we're done.
	 */
	void free_buffers();

	/**
	 * This is currently a no-op. We don't know whether it is safe to free a buffer.
	 * TODO: We could mark when a buffer is no longer needed in the task graph, and free the memory accordingly.
	 */
	void unregister_buffer(buffer_id bid) {}

	std::shared_ptr<detail::raw_data_read_handle> get_buffer_data(buffer_id bid, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) {
		assert(buffer_ptrs.at(bid) != nullptr);
		return buffer_ptrs[bid]->get_data(offset, range);
	}

	void set_buffer_data(buffer_id bid, const detail::raw_data_handle& dh) {
		assert(buffer_ptrs.at(bid) != nullptr);
		buffer_ptrs[bid]->set_data(dh);
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
	std::unordered_map<buffer_id, std::shared_ptr<detail::buffer_storage_base>> buffer_ptrs;

	std::unique_ptr<celerity::detail::graph_generator> ggen;

	std::unique_ptr<buffer_transfer_manager> btm;
	job_set jobs;

	runtime(int* argc, char** argv[]);
	runtime(const runtime&) = delete;
	runtime(runtime&&) = delete;

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

#ifdef CELERITY_TEST
	// ------------------------------------------ TESTING UTILS ------------------------------------------
	// We have to jump through some hoops to be able to re-initialize the runtime for unit testing.
	// MPI does not like being initialized more than once per process, so we have to skip that part for
	// re-initialization.
	// ---------------------------------------------------------------------------------------------------

  public:
	/**
	 * Initializes the runtime singleton without running the MPI lifecycle more than once per process.
	 */
	static void init_for_testing() {
		if(instance == nullptr) {
			init(nullptr, nullptr);
			return;
		}
		test_skip_mpi_lifecycle = true;
		instance.reset();
		init(nullptr, nullptr);
		test_skip_mpi_lifecycle = false;
	}
#endif
  private:
	static bool test_skip_mpi_lifecycle;
};

} // namespace celerity
