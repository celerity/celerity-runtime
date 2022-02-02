#pragma once

#include <chrono>
#include <thread>

#include "buffer_manager.h"
#include "buffer_transfer_manager.h"
#include "worker_job.h"

namespace celerity {

namespace detail {

	class host_queue;
	class device_queue;
	class task_manager;

	class duration_metric {
	  public:
		void resume();
		void pause();
		bool is_running() const { return running; }
		std::chrono::microseconds get() const { return duration; }

	  private:
		bool running = false;
		std::chrono::steady_clock clock;
		std::chrono::time_point<std::chrono::steady_clock> current_start;
		std::chrono::microseconds duration = {};
	};

	struct executor_metrics {
		// How much time occurs before the first job is started
		duration_metric initial_idle;
		// How much is spent not executing any device compute job
		duration_metric device_idle;
		// How much time is spent without any jobs (excluding initial idle)
		duration_metric starvation;
	};

	class executor {
		friend struct executor_testspy;

	  public:
		// TODO: Try to decouple this more.
		executor(
		    node_id local_nid, host_queue& h_queue, device_queue& d_queue, task_manager& tm, buffer_manager& buffer_mngr, reduction_manager& reduction_mngr);

		void startup();

		/**
		 * @brief Waits until all commands have been processed, and the SHUTDOWN command has been received.
		 */
		void shutdown();

	  private:
		node_id local_nid;
		host_queue& h_queue;
		device_queue& d_queue;
		task_manager& task_mngr;
		// FIXME: We currently need this for buffer locking in some jobs, which is a bit of a band-aid fix. Get rid of this at some point.
		buffer_manager& buffer_mngr;
		reduction_manager& reduction_mngr;
		std::unique_ptr<buffer_transfer_manager> btm;
		std::thread exec_thrd;
		size_t running_device_compute_jobs = 0;

		// Jobs are identified by the command id they're processing

		struct job_handle {
			std::unique_ptr<worker_job> job;
			command_type cmd;
			std::vector<command_id> dependents;
			size_t unsatisfied_dependencies;
		};

		std::unordered_map<command_id, job_handle> jobs;

		executor_metrics metrics;
		bool first_command_received = false;

		template <typename Job, typename... Args>
		void create_job(const command_pkg& pkg, const std::vector<command_id>& dependencies, Args&&... args) {
			jobs[pkg.cid] = {std::make_unique<Job>(pkg, std::forward<Args>(args)...), pkg.cmd, {}, 0};

			// If job doesn't exist we assume it has already completed.
			// This is true as long as we're respecting task-graph (anti-)dependencies when processing tasks.
			for(const command_id& d : dependencies) {
				const auto it = jobs.find(d);
				if(it != jobs.end()) {
					it->second.dependents.push_back(pkg.cid);
					jobs[pkg.cid].unsatisfied_dependencies++;
				}
			}
		}

		void run();
		bool handle_command(const command_pkg& pkg, const std::vector<command_id>& dependencies);

		void update_metrics();
	};

} // namespace detail
} // namespace celerity
