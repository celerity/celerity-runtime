#pragma once

#include <thread>

#include "logger.h"
#include "worker_job.h"

namespace celerity {

class distr_queue;

namespace detail {

	class task_manager;

	class executor {
	  public:
		// TODO: Try to decouple this more.
		executor(distr_queue& queue, task_manager& tm, buffer_transfer_manager& btm, std::shared_ptr<logger> execution_logger);

		void startup();

		/**
		 * @brief Waits until all commands have been processed, and the SHUTDOWN command has been received.
		 */
		void shutdown();

	  private:
		distr_queue& queue;
		task_manager& task_mngr;
		buffer_transfer_manager& btm;
		std::shared_ptr<logger> execution_logger;
		std::thread exec_thrd;

		// Jobs are identified by the command id they're processing

		struct job_handle {
			std::unique_ptr<worker_job> job;
			std::vector<command_id> dependants;
			size_t unsatisfied_dependencies;
		};

		std::unordered_map<command_id, job_handle> jobs;

		template <typename Job, typename... Args>
		void create_job(const command_pkg& pkg, const std::vector<command_id>& dependencies, Args&&... args) {
			auto logger = execution_logger->create_context({{"task", std::to_string(pkg.tid)}, {"job", std::to_string(pkg.cid)}});
			jobs[pkg.cid] = {std::make_unique<Job>(pkg, logger, std::forward<Args>(args)...), {}, 0};

			// If job doesn't exist we assume it has already completed.
			// This is true as long as we're respecting task-graph (anti-)dependencies when processing tasks.
			for(const command_id& d : dependencies) {
				const auto it = jobs.find(d);
				if(it != jobs.end()) {
					it->second.dependants.push_back(pkg.cid);
					jobs[pkg.cid].unsatisfied_dependencies++;
				}
			}
		}

		void run();
		void handle_command(const command_pkg& pkg, const std::vector<command_id>& dependencies);
	};

} // namespace detail
} // namespace celerity
