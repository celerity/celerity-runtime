#pragma once

#include <chrono>
#include <thread>

#include "buffer_transfer_manager.h"
#include "worker_job.h"

namespace celerity {

namespace detail {

	class local_devices;
	class task_manager;
	class buffer_manager;

	class duration_metric {
	  public:
		void resume();
		void pause();
		bool is_running() const { return m_running; }
		std::chrono::microseconds get() const { return m_duration; }

	  private:
		bool m_running = false;
		std::chrono::steady_clock m_clock;
		std::chrono::time_point<std::chrono::steady_clock> m_current_start;
		std::chrono::microseconds m_duration = {};
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
		executor(node_id local_nid, local_devices& devices, task_manager& tm, buffer_manager& buffer_mngr, reduction_manager& reduction_mngr);

		void startup();

		void enqueue(unique_frame_ptr<command_frame> frame) {
			std::scoped_lock lk(m_command_queue_mutex);
			m_command_queue.push(std::move(frame));
		}

		/**
		 * @brief Waits until all commands have been processed, and the SHUTDOWN command has been received.
		 */
		void shutdown();

	  private:
		node_id m_local_nid;
		local_devices& m_local_devices;
		std::vector<size_t> m_active_compute_jobs_by_device;
		task_manager& m_task_mngr;
		// FIXME: We currently need this for buffer locking in some jobs, which is a bit of a band-aid fix. Get rid of this at some point.
		buffer_manager& m_buffer_mngr;
		reduction_manager& m_reduction_mngr;
		std::unique_ptr<buffer_transfer_manager> m_btm;
		std::thread m_exec_thrd;
		size_t m_running_device_compute_jobs = 0;

		std::mutex m_command_queue_mutex;
		std::queue<unique_frame_ptr<command_frame>> m_command_queue;

		// Jobs are identified by the command id they're processing

		struct job_handle {
			std::unique_ptr<worker_job> job;
			command_type cmd;
			std::vector<command_id> dependents;
			size_t unsatisfied_dependencies;
		};

		std::unordered_map<command_id, job_handle> m_jobs;

		executor_metrics m_metrics;
		bool m_first_command_received = false;

		template <typename Job, typename... Args>
		void create_job(const command_frame& frame, Args&&... args) {
			const auto& pkg = frame.pkg;
			m_jobs[pkg.cid] = {std::make_unique<Job>(pkg, std::forward<Args>(args)...), pkg.get_command_type(), {}, 0};

			// If job doesn't exist we assume it has already completed.
			// This is true as long as we're respecting task-graph (anti-)dependencies when processing tasks.
			for(const auto dcid : frame.iter_dependencies()) {
				if(const auto it = m_jobs.find(dcid); it != m_jobs.end()) {
					it->second.dependents.push_back(pkg.cid);
					m_jobs[pkg.cid].unsatisfied_dependencies++;
				}
			}
		}

		void run();
		bool handle_command(const command_frame& frame);

		void update_metrics();
	};

} // namespace detail
} // namespace celerity
