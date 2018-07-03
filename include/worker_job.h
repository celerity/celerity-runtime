#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <limits>
#include <unordered_set>
#include <utility>

#include "buffer_transfer_manager.h"
#include "command.h"
#include "logger.h"

namespace celerity {

class distr_queue;

class worker_job;
using job_set = std::unordered_set<std::shared_ptr<worker_job>>;

enum class job_type : int { COMPUTE, PUSH, AWAIT_PUSH, MASTER_ACCESS };
constexpr const char* job_type_string[] = {"COMPUTE", "PUSH", "AWAIT_PUSH", "MASTER_ACCESS"};

class worker_job {
  public:
	worker_job(command_pkg pkg, std::shared_ptr<logger> job_logger) : pkg(pkg), job_logger(job_logger) {}
	worker_job(const worker_job&) = delete;
	worker_job(worker_job&&) = delete;

	virtual ~worker_job() = default;

	void initialize(const distr_queue& queue, const job_set& jobs) { dependencies = find_dependencies(queue, jobs); }

	void update();

	bool is_done() const { return done; }
	// TODO: This should return a job_type, not a command
	command get_type() const { return pkg.cmd; }
	task_id get_task_id() const { return pkg.tid; }

	// FIXME Remove this, required for send_job workaround (see below)
	const command_pkg& WORKAROUND_get_pkg() const { return pkg; }

  private:
	command_pkg pkg;
	std::shared_ptr<logger> job_logger;
	bool done = false;
	bool running = false;
	job_set dependencies;

	// Benchmarking
	static constexpr size_t BENCH_MOVING_AVG_SAMPLES = 1000;
	std::array<std::chrono::microseconds::rep, BENCH_MOVING_AVG_SAMPLES> bench_samples;
	std::chrono::high_resolution_clock bench_clock;
	size_t bench_sample_count = 0;
	// Benchmarking results
	double bench_avg = 0.0;
	std::chrono::microseconds::rep bench_min = std::numeric_limits<std::chrono::microseconds::rep>::max();
	std::chrono::microseconds::rep bench_max = std::numeric_limits<std::chrono::microseconds::rep>::min();

	virtual job_set find_dependencies(const distr_queue& queue, const job_set& jobs) { return job_set(); }
	virtual bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) = 0;

	/**
	 * Returns the job description in the form of the job type, as well as a string describing the parameters.
	 * Used for logging.
	 */
	virtual std::pair<job_type, std::string> get_description(const command_pkg& pkg) = 0;
};

/**
 * Informs the data_transfer_manager about the awaited push, then waits until the transfer has been received and completed.
 */
class await_push_job : public worker_job {
  public:
	await_push_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm) : worker_job(pkg, job_logger), btm(btm) {
		assert(pkg.cmd == command::AWAIT_PUSH);
	}

  private:
	buffer_transfer_manager& btm;
	std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

	/**
	 * This is a workaround for a race-condition occuring between consecutive master access tasks.
	 *
	 * Scenario: If a master access task is used to e.g. write results somewhere outside the scope of CELERITY, and that task is repeated multiple times, we
	 * have to make sure that the buffer contents read within each task are in the correct state at that particular point in time. This is problematic however,
	 * if worker nodes continue to push new results to the master node: Consecutive AWAIT PUSH jobs will happily accept the new data and use it to update the
	 * buffer, even if previous master access tasks have not yet excuted.
	 *
	 * NOTE: This needs a bit more investigation, however it may very well be that this type of race condition can also occur for compute tasks.
	 * FIXME: Get rid of this hack, solve at command graph level
	 */
	job_set find_dependencies(const distr_queue& queue, const job_set& jobs) override {
		job_set dependencies;
		for(auto& job : jobs) {
			if(job->get_type() == command::MASTER_ACCESS && job->get_task_id() < get_task_id()) { dependencies.insert(job); }
		}
		return dependencies;
	}

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;
};

/**
 * Pushes buffer data to a node.
 * Waits on any compute (and master-access) jobs on which the pushing task has dependencies on (on a task level).
 * TODO: Optimization opportunity: Check whether compute jobs actually write to that buffer and range and if not, send immediately
 */
class push_job : public worker_job {
  public:
	push_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm) : worker_job(pkg, job_logger), btm(btm) {
		assert(pkg.cmd == command::PUSH);
	}

  private:
	buffer_transfer_manager& btm;
	std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

	job_set find_dependencies(const distr_queue& queue, const job_set& jobs) override;

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;
};

/**
 * Executes a kernel on this node.
 * Waits for any await-pushes within the same task and any computes on which this task has dependencies on.
 *
 * TODO: Optimization opportunity: If we don't have any outstanding await-pushes, submitting the kernel to SYCL right away may be faster,
 * as it can already start copying buffers to the device (i.e. let SYCL do the scheduling).
 */
class compute_job : public worker_job {
  public:
	compute_job(command_pkg pkg, std::shared_ptr<logger> job_logger, distr_queue& queue) : worker_job(pkg, job_logger), queue(queue) {
		assert(pkg.cmd == command::COMPUTE);
	}

  private:
	distr_queue& queue;
	cl::sycl::event event;
	bool submitted = false;

	job_set find_dependencies(const distr_queue& queue, const job_set& jobs) override;

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;
};

/**
 * Runs a master access functor.
 * Waits for any pushes within the same task, as well as other master accesses (and computes) on which it has a task-level dependency.
 */
class master_access_job : public worker_job {
  public:
	master_access_job(command_pkg pkg, std::shared_ptr<logger> job_logger) : worker_job(pkg, job_logger) { assert(pkg.cmd == command::MASTER_ACCESS); }

  private:
	job_set find_dependencies(const distr_queue& queue, const job_set& jobs) override;

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;
};


} // namespace celerity
