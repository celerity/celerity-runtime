#include "worker_job.h"

#include <spdlog/fmt/fmt.h>

#include "device_queue.h"
#include "handler.h"
#include "reduction_manager.h"
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
	// --------------------------------------------------- HORIZON --------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<celerity::detail::command_type, std::string> horizon_job::get_description(const command_pkg& pkg) {
		return std::make_pair(command_type::HORIZON, "HORIZON");
	}

	bool horizon_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		const auto data = std::get<horizon_data>(pkg.data);
		task_mngr.notify_horizon_executed(data.tid);
		return true;
	};

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- AWAIT PUSH -----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> await_push_job::get_description(const command_pkg& pkg) {
		const auto data = std::get<await_push_data>(pkg.data);
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
		const auto data = std::get<push_data>(pkg.data);
		return std::make_pair(command_type::PUSH, fmt::format("PUSH buffer {} to node {}", static_cast<size_t>(data.bid), static_cast<size_t>(data.target)));
	}

	bool push_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		if(data_handle == nullptr) {
			const auto data = std::get<push_data>(pkg.data);
			// Getting buffer data from the buffer manager may incur a host-side buffer reallocation.
			// If any other tasks are currently using this buffer for reading, we run into problems.
			// To avoid this, we use a very crude buffer locking mechanism for now.
			// FIXME: Get rid of this, replace with finer grained approach.
			if(buffer_mngr.is_locked(data.bid)) { return false; }

			logger->trace(logger_map({{"event", "Submit buffer to BTM"}}));
			data_handle = btm.push(pkg);
			logger->trace(logger_map({{"event", "Buffer submitted to BTM"}}));
		}

		return data_handle->complete;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ----------------------------------------------------- REDUCTION ----------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	bool reduction_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		const auto& data = std::get<reduction_data>(pkg.data);
		rm.finish_reduction(data.rid);
		return true;
	}

	std::pair<command_type, std::string> reduction_job::get_description(const command_pkg& pkg) { return {command_type::REDUCTION, "REDUCTION"}; }

	// --------------------------------------------------------------------------------------------------------------------
	// --------------------------------------------------- HOST_EXECUTE ---------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> host_execute_job::get_description(const command_pkg& pkg) {
		return std::make_pair(command_type::EXECUTION, "HOST_EXECUTE");
	}

	bool host_execute_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		const auto data = std::get<execution_data>(pkg.data);

		if(!submitted) {
			auto tsk = task_mngr.get_task(data.tid);
			assert(tsk->get_execution_target() == execution_target::HOST);
			assert(!data.initialize_reductions); // For now, we do not support reductions in host tasks

			if(!buffer_mngr.try_lock(pkg.cid, tsk->get_buffer_access_map().get_accessed_buffers())) { return false; }

			logger->trace(logger_map({{"event", "Execute live-pass, scheduling host task in thread pool"}}));

			// Note that for host tasks, there is no indirection through a queue->submit step like there is for SYCL tasks. The CGF is executed directly,
			// which then schedules task in the thread pool through the host_queue.
			auto& cgf = tsk->get_command_group();
			live_pass_host_handler cgh(tsk, data.sr, data.initialize_reductions, queue);
			cgf(cgh);
			future = cgh.into_future();

			assert(future.valid());
			submitted = true;
			logger->trace(logger_map({{"event", "Submitted"}}));
		}

		assert(future.valid());
		if(future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			buffer_mngr.unlock(pkg.cid);

			auto info = future.get();
			logger->trace(logger_map({{"event", fmt::format("Delta time submit -> start: {}us",
			                                        std::chrono::duration_cast<std::chrono::microseconds>(info.start_time - info.submit_time).count())}}));
			logger->trace(logger_map({{"event", fmt::format("Delta time start -> end: {}us",
			                                        std::chrono::duration_cast<std::chrono::microseconds>(info.end_time - info.start_time).count())}}));
			return true;
		}
		return false;
	}

	// --------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------- DEVICE_EXECUTE ------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------

	std::pair<command_type, std::string> device_execute_job::get_description(const command_pkg& pkg) {
		return std::make_pair(command_type::EXECUTION, "DEVICE_EXECUTE");
	}

	bool device_execute_job::execute(const command_pkg& pkg, std::shared_ptr<logger> logger) {
		const auto data = std::get<execution_data>(pkg.data);

		if(!submitted) {
			auto tsk = task_mngr.get_task(data.tid);
			assert(tsk->get_execution_target() == execution_target::DEVICE);

			if(!buffer_mngr.try_lock(pkg.cid, tsk->get_buffer_access_map().get_accessed_buffers())) { return false; }

			logger->trace(logger_map({{"event", "Execute live-pass, submit kernel to SYCL"}}));

			live_pass_device_handler cgh(tsk, data.sr, data.initialize_reductions, queue);
			auto& cgf = tsk->get_command_group();
			cgf(cgh);
			event = cgh.get_submission_event();

			submitted = true;
			logger->trace(logger_map({{"event", "Submitted"}}));
		}

		const auto status = event.get_info<cl::sycl::info::event::command_execution_status>();
		if(status == cl::sycl::info::event_command_status::complete) {
			buffer_mngr.unlock(pkg.cid);

			auto tsk = task_mngr.get_task(data.tid);
			for(auto rid : tsk->get_reductions()) {
				auto reduction = reduction_mngr.get_reduction(rid);
				reduction_mngr.push_overlapping_reduction_data(rid, local_nid, buffer_mngr.get_buffer_data(reduction.output_buffer_id, {}, {1, 1, 1}));
			}

			if(queue.is_profiling_enabled()) {
				const auto submit = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>());
				const auto start = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_start>());
				const auto end = std::chrono::nanoseconds(event.get_profiling_info<cl::sycl::info::event_profiling::command_end>());

				logger->trace(logger_map({{"event",
				    fmt::format("Delta time submit -> start: {}us", std::chrono::duration_cast<std::chrono::microseconds>(start - submit).count())}}));
				logger->trace(logger_map(
				    {{"event", fmt::format("Delta time start -> end: {}us", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())}}));
			}
			return true;
		}
		return false;
	}

} // namespace detail
} // namespace celerity
