#include "worker_job.h"

#include <iostream>

#include "distr_queue.h"
#include "runtime.h"

namespace celerity {

void worker_job::update() {
	if(is_done()) return;

	if(!running) {
		for(auto it = dependencies.begin(); it != dependencies.end();) {
			auto& job = *it;
			if(job->is_done()) {
				it = dependencies.erase(it);
			} else {
				++it;
			}
		}
		if(dependencies.empty()) { running = true; }
	} else {
		done = execute(pkg);
	}
}

bool pull_job::execute(const command_pkg& pkg) {
	if(data_handle == nullptr) {
		std::cout << "PULL buffer " << pkg.data.pull.bid << " from node " << pkg.data.pull.source << std::endl;
		data_handle = btm.pull(pkg);
	}
	if(data_handle->complete) {
		std::cout << "PULL COMPLETE buffer " << pkg.data.pull.bid << " from node " << pkg.data.pull.source << std::endl;
		// TODO: Remove handle from btm
	}
	return data_handle->complete;
}

bool await_pull_job::execute(const command_pkg& pkg) {
	if(data_handle == nullptr) {
		std::cout << "AWAIT PULL of buffer " << pkg.data.await_pull.bid << " by node " << pkg.data.await_pull.target << std::endl;
		data_handle = btm.await_pull(pkg);
	}
	if(data_handle->complete) {
		std::cout << "AWAIT PULL COMPLETE of buffer " << pkg.data.await_pull.bid << " by node " << pkg.data.await_pull.target << std::endl;
	}
	return data_handle->complete;
}

job_set send_job::find_dependencies(const distr_queue& queue, const job_set& jobs) {
	// Store queue & jobs for race condition workaround
	// FIXME remove this at some point
	this->queue = &queue;
	this->jobs = &jobs;

	job_set dependencies;
	for(auto& job : jobs) {
		if(job->get_type() == command::COMPUTE) {
			if(get_task_id() != job->get_task_id() && queue.has_dependency(get_task_id(), job->get_task_id())) { dependencies.insert(job); }
		}
	}

	return dependencies;
}

bool send_job::execute(const command_pkg& pkg) {
	if(WORKAROUND_avoid_race_condition() == false) return false;
	if(data_handle == nullptr) { data_handle = btm.send(recipient, pkg); }
	return data_handle->complete;
}

/**
 * FIXME WORKAROUND
 *
 * We currently have a race-condition with regard to PULLs, made apparent by
 * master accesses, but it really can apply to all PULL commands:
 *
 * If the node issuing PULLs (e.g. the master node, for MASTER ACCESSes) does
 * so before the target node has even received the COMPUTE commands for those
 * buffer regions, it won't know that there exists a dependency for that PULL.
 * It will thus happily return garbage data.
 *
 * For now we circumvent this by requiring the corresponding AWAIT PULL
 * job to be received before a send_job can execute (i.e. a hard sync point).
 *
 * After the corresponding AWAIT PULL has been found, we scan all jobs for
 * potential dependencies. Only after those dependencies have been completed,
 * the send is executed.
 *
 * To properly avoid this issue (and with good perf), we may actually have to
 * introduce an additional command type:
 *  - PULLs need to contain a unique pull id. If the source node doesn't know
 *    about that pull id yet, it stalls the request.
 *  - After computing required results, the target nodes executes a "PULL READY"
 *	  command with the same pull id. This signals that the PULL can now be fulfilled.
 *  - Before a node would write to that buffer again, the AWAIT PULL command
 *    is issued (as it is now). This ensures that a copy of the buffer data stays
 *    around until the PULL request with the correct id has been received and
 *    processed.
 *
 * A simpler way of handling the whole issue may however be to adopt a PUSH
 * model instead:
 *  - Nodes simply PUSH computation results to the nodes that need them as soon
 *    as they're available.
 *  - Target nodes either already know that they need the data and can write it
 *    to the corresponding buffer directly, or they store it somewhere in a
 *    temporary buffer.
 *  - As they reach the corresponding AWAIT PUSH command (again, using some sort
 *    of push id), target nodes can check whether the data has already been received
 *    and if so, continue immediately - or wait otherwise.
 */
bool send_job::WORKAROUND_avoid_race_condition() {
	assert(queue != nullptr);
	assert(jobs != nullptr);

	if(corresponding_await_pull == nullptr) {
		// The await pull job won't be completed until this send job is done, so we should always find it.
		for(auto& job : *jobs) {
			if(job->get_type() == command::AWAIT_PULL) {
				auto& ap_data = job->WORKAROUND_get_pkg().data.await_pull;
				auto& send_data = this->WORKAROUND_get_pkg().data.pull;
				if(ap_data.bid == send_data.bid && ap_data.target == recipient && ap_data.target_tid == this->get_task_id()) {
					corresponding_await_pull = std::static_pointer_cast<await_pull_job>(job);
				}
			}
		}

		// Now any actual dependencies must be present, or have already been completed
		if(corresponding_await_pull != nullptr) {
			for(auto& job : *jobs) {
				if(job->get_type() == command::COMPUTE || job->get_type() == command::MASTER_ACCESS) {
					if(get_task_id() != job->get_task_id() && queue->has_dependency(get_task_id(), job->get_task_id())) { additional_dependencies.insert(job); }
				}
			}
		}
	}

	if(corresponding_await_pull != nullptr) {
		for(auto it = additional_dependencies.begin(); it != additional_dependencies.end();) {
			auto& job = *it;
			if(job->is_done()) {
				it = additional_dependencies.erase(it);
			} else {
				++it;
			}
		}
		return additional_dependencies.empty();
	}

	return false;
}

job_set compute_job::find_dependencies(const distr_queue& queue, const job_set& jobs) {
	job_set dependencies;
	for(auto& job : jobs) {
		if(job->get_type() == command::COMPUTE) {
			if(queue.has_dependency(get_task_id(), job->get_task_id())) { dependencies.insert(job); }
		}
		if(job->get_type() == command::PULL) {
			if(get_task_id() == job->get_task_id()) { dependencies.insert(job); }
		}
	}

	return dependencies;
}

bool compute_job::execute(const command_pkg& pkg) {
	if(!submitted) {
		std::cout << "COMPUTE (some range) for task " << pkg.tid << std::endl;

		auto& chunk = pkg.data.compute.chunk;
		if(chunk.range1 != 0) {
			if(chunk.range2 != 0) {
				event =
				    queue.execute(pkg.tid, subrange<3>{{chunk.offset0, chunk.offset1, chunk.offset2}, {chunk.range0, chunk.range1, chunk.range2}, {0, 0, 0}});
			} else {
				event = queue.execute(pkg.tid, subrange<2>{{chunk.offset0, chunk.offset1}, {chunk.range0, chunk.range1}, {0, 0}});
			}
		} else {
			event = queue.execute(pkg.tid, subrange<1>{{chunk.offset0}, {chunk.range0}, {0}});
		}
		submitted = true;
	}

	const auto status = event.get_info<cl::sycl::info::event::command_execution_status>();
	if(status == cl::sycl::info::event_command_status::complete) {
		std::cout << "COMPUTE COMPLETE (some range) for task " << pkg.tid << std::endl;
		return true;
	}
	return false;
}

job_set master_access_job::find_dependencies(const distr_queue& queue, const job_set& jobs) {
	job_set dependencies;
	for(auto& job : jobs) {
		if(job->get_type() == command::PULL) {
			if(get_task_id() == job->get_task_id()) { dependencies.insert(job); }
		}
	}

	return dependencies;
}

bool master_access_job::execute(const command_pkg& pkg) {
	runtime::get_instance().execute_master_access_task(pkg.tid);
	return true;
}

} // namespace celerity
