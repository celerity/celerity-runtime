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
		job_set jobs;

		size_t num_jobs = 0;

		template <typename Job, typename... Args>
		void create_job(const command_pkg& pkg, Args&&... args) {
			auto logger = execution_logger->create_context({{"task", std::to_string(pkg.tid)}, {"job", std::to_string(num_jobs)}});
			auto job = std::make_shared<Job>(pkg, logger, std::forward<Args>(args)...);
			job->initialize(task_mngr, jobs);
			jobs.insert(job);
			num_jobs++;
		}

		void run();
		void handle_command_pkg(const command_pkg& pkg);
	};

} // namespace detail
} // namespace celerity
