#include "executor.h"

#include <queue>

#include "distr_queue.h"
#include "frame.h"
#include "log.h"
#include "mpi_support.h"
#include "named_threads.h"
#include "task_hydrator.h"

// TODO: Get rid of this. (This could potentialy even cause deadlocks on large clusters)
constexpr size_t MAX_CONCURRENT_JOBS = 20;

namespace celerity {
namespace detail {
	void duration_metric::resume() {
		assert(!m_running);
		m_current_start = m_clock.now();
		m_running = true;
	}

	void duration_metric::pause() {
		assert(m_running);
		m_duration += std::chrono::duration_cast<std::chrono::microseconds>(m_clock.now() - m_current_start);
		m_running = false;
	}

	executor::executor(
	    node_id local_nid, host_queue& h_queue, device_queue& d_queue, task_manager& tm, buffer_manager& buffer_mngr, reduction_manager& reduction_mngr)
	    : m_local_nid(local_nid), m_h_queue(h_queue), m_d_queue(d_queue), m_task_mngr(tm), m_buffer_mngr(buffer_mngr), m_reduction_mngr(reduction_mngr) {
		m_btm = std::make_unique<buffer_transfer_manager>();
		m_metrics.initial_idle.resume();
	}

	void executor::startup() {
		m_exec_thrd = std::thread(&executor::run, this);
		set_thread_name(m_exec_thrd.native_handle(), "cy-executor");
	}

	void executor::shutdown() {
		if(m_exec_thrd.joinable()) { m_exec_thrd.join(); }

		CELERITY_DEBUG("Executor initial idle time = {}us, compute idle time = {}us, starvation time = {}us", m_metrics.initial_idle.get().count(),
		    m_metrics.device_idle.get().count(), m_metrics.starvation.get().count());
	}

	void executor::run() {
		task_hydrator::make_available();
		bool done = false;

		while(!done || !m_jobs.empty()) {
			// Bail if a device error ocurred.
			if(m_running_device_compute_jobs > 0) { m_d_queue.get_sycl_queue().throw_asynchronous(); }

			// We poll transfers from here (in the same thread, interleaved with job updates),
			// as it allows us to omit any sort of locking when interacting with the BTM through jobs.
			// This actually makes quite a big difference, especially for lots of small transfers.
			// The BTM uses non-blocking MPI routines internally, making this a relatively cheap operation.
			m_btm->poll();

			std::vector<command_id> ready_jobs;
			for(auto it = m_jobs.begin(); it != m_jobs.end();) {
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
					continue;
				}

				for(const auto& d : job_handle.dependents) {
					assert(m_jobs.count(d) == 1);
					m_jobs[d].unsatisfied_dependencies--;
					if(m_jobs[d].unsatisfied_dependencies == 0) { ready_jobs.push_back(d); }
				}

				if(isa<device_execute_job>(job_handle.job.get())) {
					m_running_device_compute_jobs--;
				} else if(const auto epoch = dynamic_cast<epoch_job*>(job_handle.job.get()); epoch && epoch->get_epoch_action() == epoch_action::shutdown) {
					assert(m_command_queue.empty());
					done = true;
				}

				it = m_jobs.erase(it);
			}

			// Process newly available jobs
			if(!ready_jobs.empty()) {
				// Make sure to start any push jobs before other jobs, as on some platforms copying data from a compute device while
				// also reading it from within a kernel is not supported. To avoid stalling other nodes, we thus perform the push first.
				std::sort(ready_jobs.begin(), ready_jobs.end(),
				    [this](command_id a, command_id b) { return m_jobs[a].cmd == command_type::push && m_jobs[b].cmd != command_type::push; });
				for(command_id cid : ready_jobs) {
					auto* job = m_jobs.at(cid).job.get();
					job->start();
					job->update();
					if(isa<device_execute_job>(job)) { m_running_device_compute_jobs++; }
				}
			}

			if(m_jobs.size() < MAX_CONCURRENT_JOBS) {
				// TODO: Double-buffer command queue?
				// FIXME: Don't hold lock while calling handle_command (it's annoying b/c handling can fail)
				std::unique_lock lk(m_command_queue_mutex);
				if(!m_command_queue.empty()) {
					if(!handle_command(*m_command_queue.front())) {
						// In case the command couldn't be handled, don't pop it from the queue.
						continue;
					}
					m_command_queue.pop();
				}
			}

			if(m_first_command_received) { update_metrics(); }
		}

		assert(m_running_device_compute_jobs == 0);
		task_hydrator::teardown();
	}

	bool executor::handle_command(const command_frame& frame) {
		// A worker might receive a task command before creating the corresponding task graph node
		if(const auto tid = frame.pkg.get_tid()) {
			if(!m_task_mngr.has_task(*tid)) { return false; }
		}

		switch(frame.pkg.get_command_type()) {
		case command_type::horizon: create_job<horizon_job>(frame, m_task_mngr); break;
		case command_type::epoch: create_job<epoch_job>(frame, m_task_mngr); break;
		case command_type::push: create_job<push_job>(frame, *m_btm, m_buffer_mngr); break;
		case command_type::await_push: create_job<await_push_job>(frame, *m_btm); break;
		case command_type::reduction: create_job<reduction_job>(frame, m_reduction_mngr); break;
		case command_type::execution:
			if(m_task_mngr.get_task(frame.pkg.get_tid().value())->get_execution_target() == execution_target::host) {
				create_job<host_execute_job>(frame, m_h_queue, m_task_mngr, m_buffer_mngr);
			} else {
				create_job<device_execute_job>(frame, m_d_queue, m_task_mngr, m_buffer_mngr, m_reduction_mngr, m_local_nid);
			}
			break;
		case command_type::data_request: create_job<data_request_job>(frame, *m_btm); break;
		default: assert(!"Unexpected command");
		}
		return true;
	}

	void executor::update_metrics() {
		if(m_running_device_compute_jobs == 0) {
			if(!m_metrics.device_idle.is_running()) { m_metrics.device_idle.resume(); }
		} else {
			if(m_metrics.device_idle.is_running()) { m_metrics.device_idle.pause(); }
		}
		if(m_jobs.empty()) {
			if(!m_metrics.starvation.is_running()) { m_metrics.starvation.resume(); }
		} else {
			if(m_metrics.starvation.is_running()) { m_metrics.starvation.pause(); }
		}
	}
} // namespace detail
} // namespace celerity
