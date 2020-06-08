#include "executor.h"

#include <queue>

#include "distr_queue.h"
#include "mpi_support.h"

// TODO: Get rid of this. (This could potentialy even cause deadlocks on large clusters)
constexpr size_t MAX_CONCURRENT_JOBS = 20;

namespace celerity {
namespace detail {
	void duration_metric::resume() {
		assert(!running);
		current_start = clock.now();
		running = true;
	}

	void duration_metric::pause() {
		assert(running);
		duration += std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - current_start);
		running = false;
	}

	executor::executor(device_queue& queue, task_manager& tm, std::shared_ptr<logger> execution_logger)
	    : queue(queue), task_mngr(tm), execution_logger(execution_logger) {
		btm = std::make_unique<buffer_transfer_manager>(execution_logger);
		metrics.initial_idle.resume();
	}

	void executor::startup() { exec_thrd = std::thread(&executor::run, this); }

	void executor::shutdown() {
		if(exec_thrd.joinable()) { exec_thrd.join(); }

		execution_logger->trace(logger_map{{"initialIdleTime", std::to_string(metrics.initial_idle.get().count())}});
		execution_logger->trace(logger_map{{"computeIdleTime", std::to_string(metrics.compute_idle.get().count())}});
		execution_logger->trace(logger_map{{"starvationTime", std::to_string(metrics.starvation.get().count())}});
	}

	uint64_t executor::get_highest_executed_sync_id() const noexcept { return highest_executed_sync_id; }

	void executor::run() {
		bool done = false;
		constexpr uint64_t NOT_SYNCING = std::numeric_limits<uint64_t>::max();
		uint64_t syncing_on_id = NOT_SYNCING;

		struct command_info {
			command_pkg pkg;
			std::vector<command_id> dependencies;
		};
		std::queue<command_info> command_queue;

		while(!done || !jobs.empty()) {
			if(syncing_on_id != NOT_SYNCING && jobs.empty()) {
				MPI_Barrier(MPI_COMM_WORLD);
				highest_executed_sync_id = syncing_on_id;
				syncing_on_id = NOT_SYNCING;
			}

			// We poll transfers from here (in the same thread, interleaved with job updates),
			// as it allows us to omit any sort of locking when interacting with the BTM through jobs.
			// This actually makes quite a big difference, especially for lots of small transfers.
			// The BTM uses non-blocking MPI routines internally, making this a relatively cheap operation.
			btm->poll();

			std::vector<command_id> ready_jobs;
			for(auto it = jobs.begin(); it != jobs.end();) {
				auto& job_handle = it->second;

				if(job_handle.unsatisfied_dependencies > 0) {
					++it;
					continue;
				}

				if(!job_handle.job->is_running()) {
					if(std::find(ready_jobs.cbegin(), ready_jobs.cend(), it->first) == ready_jobs.cend()) { ready_jobs.push_back(it->first); }
					++it;
					continue;
				}

				if(!job_handle.job->is_done()) {
					job_handle.job->update();
					++it;
				} else {
					for(const auto& d : job_handle.dependents) {
						assert(jobs.count(d) == 1);
						jobs[d].unsatisfied_dependencies--;
						if(jobs[d].unsatisfied_dependencies == 0) { ready_jobs.push_back(d); }
					}
					job_count_by_cmd[job_handle.cmd]--;
					it = jobs.erase(it);
				}
			}

			// Process newly available jobs
			if(!ready_jobs.empty()) {
				// Make sure to start any PUSH jobs before other jobs, as on some platforms copying data from a compute device while
				// also reading it from within a kernel is not supported. To avoid stalling other nodes, we thus perform the PUSH first.
				std::sort(ready_jobs.begin(), ready_jobs.end(),
				    [this](command_id a, command_id b) { return jobs[a].cmd == command_type::PUSH && jobs[b].cmd != command_type::PUSH; });
				for(command_id cid : ready_jobs) {
					jobs[cid].job->start();
					jobs[cid].job->update();
					job_count_by_cmd[jobs[cid].cmd]++;
				}
			}

			MPI_Status status;
			int flag;
			MPI_Message msg;
			MPI_Improbe(MPI_ANY_SOURCE, mpi_support::TAG_CMD, MPI_COMM_WORLD, &flag, &msg, &status);
			if(flag == 1) {
				// Commands should be small enough to block here (TODO: Re-evaluate this now that we also transfer dependencies)
				command_queue.emplace<command_info>({});
				auto& pkg = command_queue.back().pkg;
				auto& dependencies = command_queue.back().dependencies;
				int count;
				MPI_Get_count(&status, MPI_CHAR, &count);
				const size_t deps_size = count - sizeof(command_pkg);
				dependencies.resize(deps_size / sizeof(command_id));
				const auto data_type = mpi_support::build_single_use_composite_type({{sizeof(command_pkg), &pkg}, {deps_size, dependencies.data()}});
				MPI_Mrecv(MPI_BOTTOM, 1, *data_type, &msg, &status);

				if(!first_command_received) {
					metrics.initial_idle.pause();
					metrics.compute_idle.resume();
					first_command_received = true;
				}
			}

			if(syncing_on_id == NOT_SYNCING && jobs.size() < MAX_CONCURRENT_JOBS && !command_queue.empty()) {
				const auto info = command_queue.front();
				command_queue.pop();
				if(info.pkg.cmd == command_type::SHUTDOWN) {
					assert(command_queue.empty());
					done = true;
				} else if(info.pkg.cmd == command_type::SYNC) {
					syncing_on_id = std::get<sync_data>(info.pkg.data).sync_id;
				} else {
					handle_command(info.pkg, info.dependencies);
				}
			}

			if(first_command_received) { update_metrics(); }
		}

#ifndef NDEBUG
		for(const auto it : job_count_by_cmd) {
			assert(it.second == 0);
		}
#endif
	}

	void executor::handle_command(const command_pkg& pkg, const std::vector<command_id>& dependencies) {
		switch(pkg.cmd) {
		case command_type::HORIZON: create_job<horizon_job>(pkg, dependencies); break;
		case command_type::PUSH: create_job<push_job>(pkg, dependencies, *btm); break;
		case command_type::AWAIT_PUSH: create_job<await_push_job>(pkg, dependencies, *btm); break;
		case command_type::COMPUTE: create_job<compute_job>(pkg, dependencies, queue, task_mngr); break;
		case command_type::MASTER_ACCESS: create_job<master_access_job>(pkg, dependencies, task_mngr); break;
		default: {
			assert(false && "Unexpected command");
		}
		}
	}

	void executor::update_metrics() {
		if(job_count_by_cmd[command_type::COMPUTE] == 0) {
			if(!metrics.compute_idle.is_running()) { metrics.compute_idle.resume(); }
		} else {
			if(metrics.compute_idle.is_running()) { metrics.compute_idle.pause(); }
		}
		if(jobs.empty()) {
			if(!metrics.starvation.is_running()) { metrics.starvation.resume(); }
		} else {
			if(metrics.starvation.is_running()) { metrics.starvation.pause(); }
		}
	}
} // namespace detail
} // namespace celerity
