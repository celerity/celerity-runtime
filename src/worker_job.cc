#include "worker_job.h"

#include <iostream>

#include "distr_queue.h"

namespace celerity {

void worker_job::update() {
	if(is_done()) return;

	if(!running) {
		for(auto it = dependencies.begin(); it != dependencies.end();) {
			auto job = *it;
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
		// TODO: Copy data
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
	job_set dependencies;
	for(auto& job : jobs) {
		if(job->get_type() == command::COMPUTE) {
			if(get_task_id() != job->get_task_id() && queue.has_dependency(get_task_id(), job->get_task_id())) { dependencies.insert(job); }
		}
	}

	return dependencies;
}

bool send_job::execute(const command_pkg& pkg) {
	if(data_handle == nullptr) { data_handle = btm.send(recipient, pkg); }
	return data_handle->complete;
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

	auto status = event.get_info<cl::sycl::info::event::command_execution_status>();
	if(status == cl::sycl::info::event_command_status::complete) {
		std::cout << "COMPUTE COMPLETE (some range) for task " << pkg.tid << std::endl;
		return true;
	}
	return false;
}

} // namespace celerity
