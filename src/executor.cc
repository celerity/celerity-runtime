#include "executor.h"

#include <queue>

#include "distr_queue.h"

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
		std::queue<command_pkg> command_queue;

		while(!done || !jobs.empty()) {
			btm.poll();

			for(auto it = jobs.begin(); it != jobs.end();) {
				auto job = *it;
				job->update();
				if(job->is_done()) {
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
				// Commands should be small enough to block here
				command_pkg pkg;
				MPI_Mrecv(&pkg, sizeof(command_pkg), MPI_BYTE, &msg, &status);
				command_queue.push(pkg);
			}

			if(jobs.size() < MAX_CONCURRENT_JOBS && !command_queue.empty()) {
				const auto pkg = command_queue.front();
				command_queue.pop();
				if(pkg.cmd == command::SHUTDOWN) {
					assert(command_queue.empty());
					done = true;
				} else {
					handle_command_pkg(pkg);
				}
			}
		}
	}

	void executor::handle_command_pkg(const command_pkg& pkg) {
		switch(pkg.cmd) {
		case command::PUSH: create_job<push_job>(pkg, btm); break;
		case command::AWAIT_PUSH: create_job<await_push_job>(pkg, btm); break;
		case command::COMPUTE: create_job<compute_job>(pkg, queue, task_mngr); break;
		case command::MASTER_ACCESS: create_job<master_access_job>(pkg, task_mngr); break;
		default: { assert(false && "Unexpected command"); }
		}
	}
} // namespace detail
} // namespace celerity
