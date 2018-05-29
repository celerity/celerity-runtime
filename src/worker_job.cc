#include "worker_job.h"

#include <spdlog/fmt/fmt.h>

#include "distr_queue.h"
#include "runtime.h"

namespace celerity {

template <typename DurationRep, size_t NumSamples>
void benchmark_update_moving_average(const std::array<DurationRep, NumSamples>& samples, const size_t total_sample_count, const size_t period_sample_count,
    double& avg, DurationRep& min, DurationRep& max) {
	DurationRep sum = 0;
	for(size_t i = 0; i < period_sample_count; ++i) {
		sum += samples[i];
		if(samples[i] < min) { min = samples[i]; }
		if(samples[i] > max) { max = samples[i]; }
	}
	avg = (avg * (total_sample_count - period_sample_count) + sum) / total_sample_count;
}

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
		if(dependencies.empty()) {
			running = true;
			auto job_description = get_description(pkg);
			job_logger->info(logger_map({{"event", "START"}, {"type", job_type_string[static_cast<std::underlying_type_t<job_type>>(job_description.first)]},
			    {"message", job_description.second}}));
		}
	} else {
		const auto before = bench_clock.now();
		done = execute(pkg, job_logger);
		const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(bench_clock.now() - before);

		// TODO: We may want to make benchmarking optional with a macro
		bench_samples[bench_sample_count % BENCH_MOVING_AVG_SAMPLES] = dt.count();
		if(++bench_sample_count % BENCH_MOVING_AVG_SAMPLES == 0) {
			benchmark_update_moving_average(bench_samples, bench_sample_count, BENCH_MOVING_AVG_SAMPLES, bench_avg, bench_min, bench_max);
		}
	}

	if(done) {
		// Include the remaining values into the average
		const auto remaining = bench_sample_count % BENCH_MOVING_AVG_SAMPLES;
		benchmark_update_moving_average(bench_samples, bench_sample_count, remaining, bench_avg, bench_min, bench_max);

		job_logger->info(logger_map({{"event", "STOP"}, {"message", "Performance data is measured in microseconds"},
		    {"pollDurationAvg", std::to_string(bench_avg)}, {"pollDurationMin", std::to_string(bench_min)}, {"pollDurationMax", std::to_string(bench_max)},
		    {"pollSamples", std::to_string(bench_sample_count)}}));
	}
}

std::pair<job_type, std::string> pull_job::get_description(const command_pkg& pkg) {
	return std::make_pair(job_type::PULL, fmt::format("PULL buffer {} from node {}", pkg.data.pull.bid, pkg.data.pull.source));
}

bool pull_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	if(data_handle == nullptr) { data_handle = btm.pull(pkg); }
	if(data_handle->complete) {
		// TODO: Remove handle from btm
	}
	return data_handle->complete;
}

std::pair<job_type, std::string> await_pull_job::get_description(const command_pkg& pkg) {
	return std::make_pair(job_type::AWAIT_PULL, fmt::format("AWAIT PULL of buffer {} by node {}", pkg.data.await_pull.bid, pkg.data.await_pull.target));
}

bool await_pull_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	if(data_handle == nullptr) { data_handle = btm.await_pull(pkg); }
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

std::pair<job_type, std::string> send_job::get_description(const command_pkg& pkg) {
	return std::make_pair(job_type::SEND, fmt::format("SEND buffer {} to node {}", pkg.data.pull.bid, recipient));
}

bool send_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	if(WORKAROUND_avoid_race_condition(logger) == false) return false;
	if(data_handle == nullptr) {
		logger->info("NOTE: Job duration includes waiting for corresponding AWAIT PULL (race condition workaround)");
		logger->info(logger_map({{"event", "Submit buffer to MPI"}}));
		data_handle = btm.send(recipient, pkg);
		logger->info(logger_map({{"event", "Buffer submitted to MPI"}}));
	}
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
bool send_job::WORKAROUND_avoid_race_condition(std::shared_ptr<logger> logger) {
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
			logger->info(
			    logger_map({{"event", fmt::format("Found corresponding AWAIT PULL, and {} additional dependencies", additional_dependencies.size())}}));
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

std::pair<job_type, std::string> compute_job::get_description(const command_pkg& pkg) {
	return std::make_pair(job_type::COMPUTE, "COMPUTE");
}

// TODO: SYCL should have a event::get_profiling_info call. As of ComputeCpp 0.8.0 this doesn't seem to be supported.
std::chrono::time_point<std::chrono::nanoseconds> get_profiling_info(cl_event e, cl_profiling_info param) {
	cl_ulong value;
	const auto result = clGetEventProfilingInfo(e, param, sizeof(cl_ulong), &value, nullptr);
	assert(result == CL_SUCCESS);
	return std::chrono::time_point<std::chrono::nanoseconds>(std::chrono::nanoseconds(value));
};

bool compute_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	if(!submitted) {
		// Note that we have to set the proper global size so the livepass handler can use the assigned chunk as input for range mappers
		auto gs = std::static_pointer_cast<const compute_task>(queue.get_task(pkg.tid))->get_global_size();
		auto& chunk = pkg.data.compute.chunk;
		if(chunk.range1 != 0) {
			if(chunk.range2 != 0) {
				event = queue.execute(pkg.tid,
				    subrange<3>{{chunk.offset0, chunk.offset1, chunk.offset2}, {chunk.range0, chunk.range1, chunk.range2}, boost::get<cl::sycl::range<3>>(gs)});
			} else {
				event = queue.execute(pkg.tid, subrange<2>{{chunk.offset0, chunk.offset1}, {chunk.range0, chunk.range1}, boost::get<cl::sycl::range<2>>(gs)});
			}
		} else {
			event = queue.execute(pkg.tid, subrange<1>{{chunk.offset0}, {chunk.range0}, boost::get<cl::sycl::range<1>>(gs)});
		}
		submitted = true;
	}

	// NOTE: Currently (ComputeCpp 0.8.0) there exists a bug that causes this call to deadlock
	// if the command failed to be submitted in the first place (e.g. when buffer allocation failed).
	// Codeplay has been informed about this, and they're working on it.
	const auto status = event.get_info<cl::sycl::info::event::command_execution_status>();
	if(status == cl::sycl::info::event_command_status::complete) {
		if(queue.is_ocl_profiling_enabled()) {
			// FIXME: Currently (ComputeCpp 0.8.0), the event.get() call may cause an exception within some ComputeCpp worker thread in certain situations
			// (E.g. when running on our NVIDIA Tesla K20m, but only when using more than one device).
			const auto queued = get_profiling_info(event.get(), CL_PROFILING_COMMAND_QUEUED);
			const auto submit = get_profiling_info(event.get(), CL_PROFILING_COMMAND_SUBMIT);
			const auto start = get_profiling_info(event.get(), CL_PROFILING_COMMAND_START);
			const auto end = get_profiling_info(event.get(), CL_PROFILING_COMMAND_END);

			// FIXME: The timestamps logged here don't match the actual values we just queried. Can we fix that?
			logger->info(logger_map({{"event",
			    fmt::format("Delta time queued -> submit : {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(submit - queued).count())}}));
			logger->info(logger_map(
			    {{"event", fmt::format("Delta time submit -> start: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(start - submit).count())}}));
			logger->info(logger_map(
			    {{"event", fmt::format("Delta time start -> end: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())}}));
		}
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

std::pair<job_type, std::string> master_access_job::get_description(const command_pkg& pkg) {
	return std::make_pair(job_type::MASTER_ACCESS, "MASTER ACCESS");
}

bool master_access_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	runtime::get_instance().execute_master_access_task(pkg.tid);
	return true;
}

} // namespace celerity
