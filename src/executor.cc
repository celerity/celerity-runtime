/**

Thoughts on executor deadlock w/ PeterTh 2022-10-14:

- Short term try sorting all commands before inserting into the executor queue: Push > await push > execute (and other commands). Or try AP > P (would still
deadlock if #buffers > 20 though, so probably not a good idea)
- Various longer term ideas:
    - Have multiple queues for each command type (or at least the big 3). Limit the number of in-flight commands per type.
      Hard limit on execution commands, "soft limit" on others: Start all P/AP commands that come before the first execution command currently in-flight
      (here "before" means P/APs generated as part of the same task, would need to either generate them all before exec commands or somehow re-assign IDs).
    - Polling on P/AP jobs is redundant w/ BTM.poll(). The BTM could just decrease all successor's dependency count once a transfer is done (e.g. via callback).
    - Use pointers instead of IDs to cross reference inside the executor. Maybe even embed execution status into the CDAG nodes themselves.
    - Distinguish between active (execute, push, ...) commands and passive (await push) commands. If there are no active commands running, keep adding
      additional commands from the queue. Might need additional mechanism to avoid unbounded growth though (horizons?). It would be nice if we
      could theoretically guarantee deadlock-free execution.
*/

#include "executor.h"

#include <queue>

#include "distr_queue.h"
#include "frame.h"
#include "local_devices.h"
#include "log.h"
#include "mpi_support.h"
#include "named_threads.h"
#include "task_hydrator.h"

// TODO: Get rid of this. (This could potentialy even cause deadlocks on large clusters)
constexpr size_t MAX_CONCURRENT_JOBS = 50;
constexpr size_t MAX_CONCURRENT_COMPUTES_PER_DEVICE = 1;

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

	executor::executor(node_id local_nid, local_devices& devices, task_manager& tm, buffer_manager& buffer_mngr, reduction_manager& reduction_mngr)
	    : m_local_nid(local_nid), m_local_devices(devices), m_active_compute_jobs_by_device(devices.num_compute_devices()), m_task_mngr(tm),
	      m_buffer_mngr(buffer_mngr), m_reduction_mngr(reduction_mngr) {
		m_btm = std::make_unique<buffer_transfer_manager>(m_buffer_mngr, m_reduction_mngr);
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

// It's called "attempt" b/c we might produce false positives for long-running compute jobs
// NOTE: Enabling this is not enough atm, requires minor changes to worker_job and create_job().
#define ATTEMPT_DEADLOCK_DETECTION 0
#if ATTEMPT_DEADLOCK_DETECTION
		namespace chr = std::chrono;
		using namespace std::chrono_literals;
		chr::steady_clock::time_point ts_last_change = chr::steady_clock::now() + 1s;
#endif

		while(!done || !m_jobs.empty()) {
			// Bail if a device error ocurred.
			if(m_running_device_compute_jobs > 0) {
				// NOCOMMIT FIXME: Ugh, that's not ideal. Maybe not check in every iteration.
				for(device_id did = 0; did < m_local_devices.num_compute_devices(); ++did) {
					if(m_active_compute_jobs_by_device[did] > 0) { m_local_devices.get_device_queue(did).get_sycl_queue().throw_asynchronous(); }
				}
			}

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
					const device_id did = static_cast<device_execute_job*>(job_handle.job.get())->get_device_id();
					m_active_compute_jobs_by_device[did]--;
					m_running_device_compute_jobs--;
				} else if(const auto epoch = dynamic_cast<epoch_job*>(job_handle.job.get()); epoch && epoch->get_epoch_action() == epoch_action::shutdown) {
					assert(m_command_queue.empty());
					done = true;
				}

#if ATTEMPT_DEADLOCK_DETECTION
				ts_last_change = chr::steady_clock::now();
#endif
				it = m_jobs.erase(it);
			}

			// Process newly available jobs
			if(!ready_jobs.empty()) {
				// Make sure to start any push jobs before other jobs, as on some platforms copying data from a compute device while
				// also reading it from within a kernel is not supported. To avoid stalling other nodes, we thus perform the push first.
				std::sort(ready_jobs.begin(), ready_jobs.end(),
				    [this](command_id a, command_id b) { return m_jobs[a].cmd == command_type::push && m_jobs[b].cmd != command_type::push; });
				for(command_id cid : ready_jobs) {
#if ATTEMPT_DEADLOCK_DETECTION
					ts_last_change = chr::steady_clock::now();
#endif
					auto* job = m_jobs.at(cid).job.get();

					// TODO: We probably also want to limit the number of concurrent jobs that can be in preparation phase
					if(!job->prepare()) continue;

					if(isa<device_execute_job>(job)) {
						const device_id did = static_cast<device_execute_job*>(job)->get_device_id();
						if(m_active_compute_jobs_by_device[did] >= MAX_CONCURRENT_COMPUTES_PER_DEVICE) continue;

						m_active_compute_jobs_by_device[did]++;
						m_running_device_compute_jobs++;
					}

					job->start();
					job->update();
				}
			}

			if(m_jobs.size() < MAX_CONCURRENT_JOBS) {
				// TODO: Double-buffer command queue?
				// FIXME: Don't hold lock while calling handle_command (it's annoying b/c handling can fail)
				std::unique_lock lk(m_command_queue_mutex);
				if(!m_command_queue.empty()) {
					if(!handle_command(m_command_queue.front())) {
						// In case the command couldn't be handled, don't pop it from the queue.
						continue;
					}
#if ATTEMPT_DEADLOCK_DETECTION
					ts_last_change = chr::steady_clock::now();
#endif
					m_command_queue.pop();
				}
			}

			if(m_first_command_received) { update_metrics(); }

#if ATTEMPT_DEADLOCK_DETECTION
			// NOCOMMIT Add something like this, at least as a function that can be called from a GDB session (it might be hard to automatically detect
			// deadlocks reliably).
			if((chr::steady_clock::now() - ts_last_change) > 10s) {
				std::this_thread::sleep_for(size_t(m_local_nid) * 100ms);

				fmt::print("Executor is stuck!\n");
				fmt::print("Jobs:\n");
				for(auto& [id, jh] : m_jobs) {
					fmt::print("\t{} [running={}]: {} | {} unfulfilled dependencies, successors: [{}]\n", id, jh.job->is_running(),
					    jh.job->get_description(jh.job->m_pkg), jh.unsatisfied_dependencies, fmt::join(jh.dependents, ", "));
				}

				const auto foo = [this](const command_frame& frame) -> job_handle* {
					switch(frame.pkg.get_command_type()) {
					case command_type::horizon: return create_job<horizon_job>(frame, m_task_mngr); break;
					case command_type::epoch: return create_job<epoch_job>(frame, m_task_mngr); break;
					case command_type::push: return create_job<push_job>(frame, *m_btm, m_buffer_mngr); break;
					case command_type::await_push: return create_job<await_push_job>(frame, *m_btm); break;
					case command_type::reduction: return create_job<reduction_job>(frame, m_reduction_mngr); break;
					case command_type::execution:
						if(m_task_mngr.get_task(frame.pkg.get_tid().value())->get_execution_target() == execution_target::host) {
							return create_job<host_execute_job>(frame, m_local_devices.get_host_queue(), m_task_mngr, m_buffer_mngr);
						} else {
							return create_job<device_execute_job>(frame, m_local_devices.get_device_queue(std::get<execution_data>(frame.pkg.data).did),
							    m_task_mngr, m_buffer_mngr, m_reduction_mngr, m_local_nid);
						}
						break;
					case command_type::data_request: create_job<data_request_job>(frame, *m_btm); break;
					default: assert(!"Unexpected command");
					}
					return nullptr;
				};

				fmt::print("Next 5 commands in queue:\n");
				for(int i = 0; i < 5 && !m_command_queue.empty(); ++i) {
					auto& frame = *m_command_queue.front();
					fmt::print("\t{}: {}\n", frame.pkg.cid, foo(frame)->job->get_description(frame.pkg));
					m_command_queue.pop();
				}

				std::this_thread::sleep_for(
				    (runtime::get_instance().get_num_nodes() - size_t(m_local_nid)) * 100ms); // Prevent mpirun to kill other jobs if we exit on this rank
				raise(SIGKILL);
			}
#endif
		}

		assert(m_running_device_compute_jobs == 0);
		task_hydrator::teardown();
	}

	bool executor::handle_command(const command_pkg& pkg) {
		// A worker might receive a task command before creating the corresponding task graph node
		if(const auto tid = pkg.get_tid()) {
			if(!m_task_mngr.has_task(*tid)) { return false; }
		}

		switch(pkg.get_command_type()) {
		case command_type::horizon: create_job<horizon_job>(pkg, m_task_mngr); break;
		case command_type::epoch: create_job<epoch_job>(pkg, m_task_mngr); break;
		case command_type::push: create_job<push_job>(pkg, *m_btm, m_buffer_mngr); break;
		case command_type::await_push: create_job<await_push_job>(pkg, *m_btm); break;
		case command_type::reduction: create_job<reduction_job>(pkg, m_reduction_mngr); break;
		case command_type::execution:
			if(m_task_mngr.get_task(pkg.get_tid().value())->get_execution_target() == execution_target::host) {
				create_job<host_execute_job>(pkg, m_local_devices.get_host_queue(), m_task_mngr, m_buffer_mngr);
			} else {
				const device_id did = std::get<execution_data>(pkg.data).did;
				assert(did < m_local_devices.num_compute_devices());
				create_job<device_execute_job>(pkg, m_local_devices.get_device_queue(did), m_task_mngr, m_buffer_mngr, m_reduction_mngr, m_local_nid);
			}
			break;
		case command_type::data_request: create_job<data_request_job>(pkg, *m_btm); break;
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
