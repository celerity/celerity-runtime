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

enum class job_type : int { COMPUTE, PULL, AWAIT_PULL, SEND, MASTER_ACCESS };
constexpr const char* job_type_string[] = {"COMPUTE", "PULL", "AWAIT_PULL", "SEND", "MASTER_ACCESS"};

class worker_job {
  public:
	worker_job(command_pkg pkg, std::shared_ptr<logger> job_logger) : pkg(pkg), job_logger(job_logger) {}
	worker_job(const worker_job&) = delete;
	worker_job(worker_job&&) = delete;

	virtual ~worker_job() = default;

	void initialize(const distr_queue& queue, const job_set& jobs) { dependencies = find_dependencies(queue, jobs); }

	void update();

	bool is_done() const { return done; }
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
 * Sends a pull request to another node using the buffer_transfer_manager, then waits for the data to be ready.
 * This mainly acts as a dependency for compute jobs.
 */
class pull_job : public worker_job {
  public:
	pull_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm) : worker_job(pkg, job_logger), btm(btm) {
		assert(pkg.cmd == command::PULL);
	}

  private:
	buffer_transfer_manager& btm;
	std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

	job_set find_dependencies(const distr_queue& queue, const job_set& jobs) override {
		// TODO: A pull job doesn't really have any dependencies, other than the fact that the corresponding buffer subrange shouldn't be written
		// to by this node. That shouldn't happen anyway, but we could assert it.
		return job_set();
	}

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;
};

/**
 * Informs the data_transfer_manager about the awaited pull, then waits until it has been received and completed.
 * TODO: This should ensure that future compute jobs don't overwrite the buffer in question, by creating a copy of the data (only if needed).
 */
class await_pull_job : public worker_job {
  public:
	await_pull_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm) : worker_job(pkg, job_logger), btm(btm) {
		assert(pkg.cmd == command::AWAIT_PULL);
	}

  private:
	buffer_transfer_manager& btm;
	std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;
};

/**
 * Sends buffer data to a node in response to an incoming pull request.
 * Waits on any compute jobs on which the pulling task has dependencies on (on a task level).
 *
 * TODO: Optimization opportunity: Check whether compute jobs actually write to that buffer and range and if not, send immediately
 */
class send_job : public worker_job {
  public:
	send_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm, node_id recipient)
	    : worker_job(pkg, job_logger), btm(btm), recipient(recipient) {
		assert(pkg.cmd == command::PULL);
	}

  private:
	node_id recipient;
	buffer_transfer_manager& btm;
	std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

	job_set find_dependencies(const distr_queue& queue, const job_set& jobs) override;

	bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
	std::pair<job_type, std::string> get_description(const command_pkg& pkg) override;

	// FIXME: WORKAROUND - Remove this at some point
  private:
	const distr_queue* queue = nullptr;
	const job_set* jobs = nullptr;
	bool safe_to_send = false;
	std::shared_ptr<const await_pull_job> corresponding_await_pull = nullptr;
	job_set additional_dependencies;

	bool WORKAROUND_avoid_race_condition(std::shared_ptr<logger> logger);
};

/**
 * Executes a kernel on this node.
 * Waits for any pulls within the same task and any computes on which this task has dependencies on.
 *
 * TODO: Optimization opportunity: If we don't have any outstanding pulls, submitting the kernel to SYCL right away may be faster,
 * as it can already start copying buffers to the device.
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
 * Waits for any pulls within the same task.
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
