#include "executor.h"

#include <queue>

#include "distr_queue.h"
#include "mpi_support.h"

// TODO: Get rid of this. (This could potentialy even cause deadlocks on large clusters)
constexpr size_t MAX_CONCURRENT_JOBS = 20;

namespace celerity {
namespace detail {
	executor::executor(distr_queue& queue, task_manager& tm, buffer_transfer_manager& btm, std::shared_ptr<logger> execution_logger)
	    : queue(queue), task_mngr(tm), btm(btm), execution_logger(execution_logger) {}

	void executor::startup() { exec_thrd = std::thread(&executor::run, this); }

	void executor::shutdown() {
		if(exec_thrd.joinable()) { exec_thrd.join(); }
	}

	void executor::run() {
		bool done = false;

		struct command_info {
			command_pkg pkg;
			std::vector<command_id> dependencies;
		};
		std::queue<command_info> command_queue;

		while(!done || !jobs.empty()) {
			btm.poll();

			for(auto it = jobs.begin(); it != jobs.end();) {
				auto& job_handle = it->second;

				if(job_handle.unsatisfied_dependencies > 0) {
					++it;
					continue;
				}

				job_handle.job->update();
				if(job_handle.job->is_done()) {
					for(const auto& d : job_handle.dependants) {
						assert(jobs.count(d) == 1);
						jobs[d].unsatisfied_dependencies--;
					}
					it = jobs.erase(it);
				} else {
					++it;
				}
			}

			MPI_Status status;
			int flag;
			MPI_Message msg;
			MPI_Improbe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &flag, &msg, &status);
			if(flag == 1) {
				// Commands should be small enough to block here (TODO: Re-evaluate this now that also transfer dependencies)
				command_queue.emplace<command_info>({});
				auto& pkg = command_queue.back().pkg;
				auto& dependencies = command_queue.back().dependencies;
				int count;
				MPI_Get_count(&status, MPI_CHAR, &count);
				const size_t deps_size = count - sizeof(command_pkg);
				dependencies.resize(deps_size / sizeof(command_id));
				const auto data_type = mpi_support::build_single_use_composite_type({{sizeof(command_pkg), &pkg}, {deps_size, dependencies.data()}});
				MPI_Mrecv(MPI_BOTTOM, 1, *data_type, &msg, &status);
			}

			if(jobs.size() < MAX_CONCURRENT_JOBS && !command_queue.empty()) {
				const auto info = command_queue.front();
				command_queue.pop();
				if(info.pkg.cmd == command::SHUTDOWN) {
					assert(command_queue.empty());
					done = true;
				} else {
					handle_command(info.pkg, info.dependencies);
				}
			}
		}
	}

	void executor::handle_command(const command_pkg& pkg, const std::vector<command_id>& dependencies) {
		switch(pkg.cmd) {
		case command::PUSH: create_job<push_job>(pkg, dependencies, btm); break;
		case command::AWAIT_PUSH: create_job<await_push_job>(pkg, dependencies, btm); break;
		case command::COMPUTE: create_job<compute_job>(pkg, dependencies, queue, task_mngr); break;
		case command::MASTER_ACCESS: create_job<master_access_job>(pkg, dependencies, task_mngr); break;
		default: { assert(false && "Unexpected command"); }
		}
	}
} // namespace detail
} // namespace celerity
