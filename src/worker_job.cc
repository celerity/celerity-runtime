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
			job_logger->info(logger_map({{"cid", std::to_string(pkg.cid)}, {"event", "START"},
			    {"type", job_type_string[static_cast<std::underlying_type_t<job_type>>(job_description.first)]}, {"message", job_description.second}}));
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

		job_logger->info(logger_map({{"event", "STOP"}, {"pollDurationAvg", std::to_string(bench_avg)}, {"pollDurationMin", std::to_string(bench_min)},
		    {"pollDurationMax", std::to_string(bench_max)}, {"pollSamples", std::to_string(bench_sample_count)}}));
	}
}

std::pair<job_type, std::string> await_push_job::get_description(const command_pkg& pkg) {
	return std::make_pair(job_type::AWAIT_PUSH,
	    fmt::format("AWAIT PUSH of buffer {} by node {}", static_cast<size_t>(pkg.data.await_push.bid), static_cast<size_t>(pkg.data.await_push.source)));
}

bool await_push_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	if(data_handle == nullptr) { data_handle = btm.await_push(pkg); }
	return data_handle->complete;
}

job_set push_job::find_dependencies(const distr_queue& queue, const job_set& jobs) {
	job_set dependencies;
	for(auto& job : jobs) {
		switch(job->get_type()) {
		case command::COMPUTE:
		case command::MASTER_ACCESS:
			if(get_task_id() != job->get_task_id() && queue.has_dependency(get_task_id(), job->get_task_id())) { dependencies.insert(job); }
			break;
		default: break;
		}
	}

	return dependencies;
}

std::pair<job_type, std::string> push_job::get_description(const command_pkg& pkg) {
	return std::make_pair(
	    job_type::PUSH, fmt::format("PUSH buffer {} to node {}", static_cast<size_t>(pkg.data.push.bid), static_cast<size_t>(pkg.data.push.target)));
}

bool push_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
	if(data_handle == nullptr) {
		logger->info(logger_map({{"event", "Submit buffer to MPI"}}));
		data_handle = btm.push(pkg);
		logger->info(logger_map({{"event", "Buffer submitted to MPI"}}));
	}
	return data_handle->complete;
}

job_set compute_job::find_dependencies(const distr_queue& queue, const job_set& jobs) {
	job_set dependencies;
	for(auto& job : jobs) {
		if(job->get_type() == command::COMPUTE) {
			if(queue.has_dependency(get_task_id(), job->get_task_id())) { dependencies.insert(job); }
		}
		if(job->get_type() == command::AWAIT_PUSH) {
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
		const auto ctsk = std::static_pointer_cast<const compute_task>(queue.get_task(pkg.tid));
		const auto dimensions = ctsk->get_dimensions();
		auto gs = ctsk->get_global_size();
		auto& cmd_sr = pkg.data.compute.subrange;
		switch(dimensions) {
		default:
		case 1: event = queue.execute(pkg.tid, chunk<1>{{cmd_sr.offset[0]}, {cmd_sr.range[0]}, boost::get<cl::sycl::range<1>>(gs)}); break;
		case 2:
			event =
			    queue.execute(pkg.tid, chunk<2>{{cmd_sr.offset[0], cmd_sr.offset[1]}, {cmd_sr.range[0], cmd_sr.range[1]}, boost::get<cl::sycl::range<2>>(gs)});
			break;
		case 3:
			event = queue.execute(pkg.tid, chunk<3>{{cmd_sr.offset[0], cmd_sr.offset[1], cmd_sr.offset[2]}, {cmd_sr.range[0], cmd_sr.range[1], cmd_sr.range[2]},
			                                   boost::get<cl::sycl::range<3>>(gs)});
			break;
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
		if(job->get_type() == command::AWAIT_PUSH) {
			if(get_task_id() == job->get_task_id()) { dependencies.insert(job); }
		}
		// Wait for jobs that produce results on the master node (i.e. which don't require a transfer).
		// Note that typically this can only be master accesses, while all compute jobs run on worker nodes,
		// thus always requiring a transfer.
		if(job->get_type() == command::MASTER_ACCESS || job->get_type() == command::COMPUTE) {
			if(queue.has_dependency(get_task_id(), job->get_task_id())) { dependencies.insert(job); }
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
