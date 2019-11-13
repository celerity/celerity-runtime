#pragma once

#include <chrono>
#include <thread>

#include "buffer_transfer_manager.h"
#include "logger.h"
#include "worker_job.h"

namespace celerity {

namespace detail {

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
		// How much is spent not executing any compute tasks
		duration_metric compute_idle;
		// How much time is spent without any jobs (excluding initial idle)
		duration_metric starvation;
	};

	class executor {
	  public:
		// TODO: Try to decouple this more.
		executor(device_queue& queue, task_manager& tm, std::shared_ptr<logger> execution_logger);

		void startup();

		/**
		 * @brief Waits until all commands have been processed, and the SHUTDOWN command has been received.
		 */
		void shutdown();

		/**
		 * @brief Get the id of the highest executed global sync operation.
		 */
		uint64_t get_highest_executed_sync_id() const noexcept;

	  private:
		device_queue& queue;
		task_manager& task_mngr;
		std::unique_ptr<buffer_transfer_manager> btm;
		std::shared_ptr<logger> execution_logger;
		std::thread exec_thrd;
		std::unordered_map<command_type, size_t> job_count_by_cmd;
		std::atomic<uint64_t> highest_executed_sync_id = {0};

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
			auto logger = execution_logger->create_context({{"job", std::to_string(pkg.cid)}});
			if(pkg.cmd == command_type::COMPUTE || pkg.cmd == command_type::MASTER_ACCESS) {
				logger = logger->create_context({{"task",
				    std::to_string(pkg.cmd == command_type::COMPUTE ? boost::get<compute_data>(pkg.data).tid : boost::get<master_access_data>(pkg.data).tid)}});
			}
			jobs[pkg.cid] = {std::make_unique<Job>(pkg, logger, std::forward<Args>(args)...), pkg.cmd, {}, 0};

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
		void handle_command(const command_pkg& pkg, const std::vector<command_id>& dependencies);

		void update_metrics();
	};

} // namespace detail
} // namespace celerity
