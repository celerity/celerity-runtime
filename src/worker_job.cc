#include "worker_job.h"

#define FMT_HEADER_ONLY
#include <spdlog/fmt/fmt.h>

#include "device_queue.h"
#include "handler.h"
#include "runtime.h"
#include "task_manager.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- GENERAL ------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	void worker_job::update() {
		assert(running && !done);
		const auto before = std::chrono::steady_clock::now();
		done = execute(pkg, job_logger);

		// TODO: We may want to make benchmarking optional with a macro
		const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - before);
		bench_sum_execution_time += dt;
		bench_sample_count++;
		if(dt < bench_min) bench_min = dt;
		if(dt > bench_max) bench_max = dt;

		if(done) {
			const auto bench_avg = bench_sum_execution_time.count() / bench_sample_count;
			const auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_time).count();
			job_logger->trace(logger_map({{"event", "STOP"}, {"executionTime", std::to_string(execution_time)}, {"pollDurationAvg", std::to_string(bench_avg)},
			    {"pollDurationMin", std::to_string(bench_min.count())}, {"pollDurationMax", std::to_string(bench_max.count())},
			    {"pollSamples", std::to_string(bench_sample_count)}}));
		}
	}

	void worker_job::start() {
		assert(!running);
		running = true;

		auto job_description = get_description(pkg);
		job_logger->trace(logger_map({{"cid", std::to_string(pkg.cid)}, {"event", "START"},
		    {"type", command_string[static_cast<std::underlying_type_t<command_type>>(job_description.first)]}, {"message", job_description.second}}));
		start_time = std::chrono::steady_clock::now();
	}


	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- AWAIT PUSH -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> await_push_job::get_description(const command_pkg& pkg) {
		const auto data = boost::get<await_push_data>(pkg.data);
		return std::make_pair(
		    command_type::AWAIT_PUSH, fmt::format("AWAIT PUSH of buffer {} by node {}", static_cast<size_t>(data.bid), static_cast<size_t>(data.source)));
	}

	bool await_push_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		if(data_handle == nullptr) { data_handle = btm.await_push(pkg); }
		return data_handle->complete;
	}


	// --------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------- PUSH -------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> push_job::get_description(const command_pkg& pkg) {
		const auto data = boost::get<push_data>(pkg.data);
		return std::make_pair(command_type::PUSH, fmt::format("PUSH buffer {} to node {}", static_cast<size_t>(data.bid), static_cast<size_t>(data.target)));
	}

	bool push_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		if(data_handle == nullptr) {
			logger->trace(logger_map({{"event", "Submit buffer to BTM"}}));
			data_handle = btm.push(pkg);
			logger->trace(logger_map({{"event", "Buffer submitted to BTM"}}));
		}
		return data_handle->complete;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------ COMPUTE -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> compute_job::get_description(const command_pkg& pkg) { return std::make_pair(command_type::COMPUTE, "COMPUTE"); }

// While device profiling is disabled on hipSYCL anyway, we have to make sure that we don't include any OpenCL code
#if !WORKAROUND(HIPSYCL, 0)
	// TODO: SYCL should have a event::get_profiling_info call. As of ComputeCpp 0.8.0 this doesn't seem to be supported.
	std::chrono::time_point<std::chrono::nanoseconds> get_profiling_info(cl_event e, cl_profiling_info param) {
		cl_ulong value;
		const auto result = clGetEventProfilingInfo(e, param, sizeof(cl_ulong), &value, nullptr);
		assert(result == CL_SUCCESS);
		return std::chrono::time_point<std::chrono::nanoseconds>(std::chrono::nanoseconds(value));
	};
#endif

	bool compute_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		const auto data = boost::get<compute_data>(pkg.data);
		// A bit of a hack: We cannot be sure the main thread has reached the task definition yet, so we have to check it here
		if(!task_mngr.has_task(data.tid)) {
			if(!did_log_task_wait) {
				logger->trace(logger_map({{"event", "Waiting for task definition"}}));
				did_log_task_wait = true;
			}
			return false;
		}

		if(!submitted) {
			// Note that we have to set the proper global size so the livepass handler can use the assigned chunk as input for range mappers
			const auto ctsk = std::static_pointer_cast<const detail::compute_task>(task_mngr.get_task(data.tid));
			auto& cmd_sr = data.sr;
			logger->trace(logger_map({{"event", "Execute live-pass, submit kernel to SYCL"}}));
			event = queue.execute(data.tid, cmd_sr);
			submitted = true;
			logger->trace(logger_map({{"event", "Submitted"}}));

			// There currently (since 0.9.0 and up to and including 1.0.5) exists a bug that causes ComputeCpp to block when
			// querying the execution status of a compute command until it is finished. This is bad for us, as it blocks all other
			// jobs and prevents us from executing multiple compute jobs simultaneously.
			// --> See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-107 (psalz)
			// The workaround for now is to block within a worker thread.
#if WORKAROUND(COMPUTECPP, 1, 0, 5)
			computecpp_workaround_future = runtime::get_instance().execute_async_pooled([this]() {
				while(true) {
					const auto status = event.get_info<cl::sycl::info::event::command_execution_status>();
					if(status == cl::sycl::info::event_command_status::complete) { return; }
				}
			});
#endif
		}

#if WORKAROUND(COMPUTECPP, 1, 0, 5)
		assert(computecpp_workaround_future.valid());
		if(computecpp_workaround_future.wait_for(std::chrono::microseconds(1)) == std::future_status::ready) {
#else
		const auto status = event.get_info<cl::sycl::info::event::command_execution_status>();
		if(status == cl::sycl::info::event_command_status::complete) {
#endif
#if !WORKAROUND(HIPSYCL, 0)
			if(queue.is_profiling_enabled()) {
				const auto queued = get_profiling_info(event.get(), CL_PROFILING_COMMAND_QUEUED);
				const auto submit = get_profiling_info(event.get(), CL_PROFILING_COMMAND_SUBMIT);
				const auto start = get_profiling_info(event.get(), CL_PROFILING_COMMAND_START);
				const auto end = get_profiling_info(event.get(), CL_PROFILING_COMMAND_END);

				// FIXME: The timestamps logged here don't match the actual values we just queried. Can we fix that?
				logger->trace(logger_map({{"event",
				    fmt::format("Delta time queued -> submit : {}us", std::chrono::duration_cast<std::chrono::microseconds>(submit - queued).count())}}));
				logger->trace(logger_map({{"event",
				    fmt::format("Delta time submit -> start: {}us", std::chrono::duration_cast<std::chrono::microseconds>(start - submit).count())}}));
				logger->trace(logger_map(
				    {{"event", fmt::format("Delta time start -> end: {}us", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())}}));
			}
#endif
			return true;
		}
		return false;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- MASTER ACCESS --------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> master_access_job::get_description(const command_pkg& pkg) {
		return std::make_pair(command_type::MASTER_ACCESS, "MASTER ACCESS");
	}

	bool master_access_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		// In this case we can be sure that the task definition exists, as we're on the master node.
		const auto tsk = std::static_pointer_cast<const master_access_task>(task_mngr.get_task(boost::get<master_access_data>(pkg.data).tid));
		auto cgh = std::make_unique<master_access_task_handler<false>>();
		tsk->get_functor()(*cgh);
		return true;
	}

} // namespace detail
} // namespace celerity
