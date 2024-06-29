#pragma once

#include <cassert>
#include <chrono>
#include <future>
#include <limits>
#include <utility>

#include "buffer_transfer_manager.h"
#include "closure_hydrator.h"
#include "command.h"
#include "host_queue.h"
#include "log.h"

namespace celerity {
namespace detail {

	class device_queue;
	class executor;
	class task_manager;
	class reduction_manager;
	class buffer_manager;

	class worker_job;

	class worker_job {
	  public:
		worker_job(const worker_job&) = delete;
		worker_job(worker_job&&) = delete;

		virtual ~worker_job() = default;

		void start();
		void update();

		bool is_running() const { return m_running; }
		bool is_done() const { return m_done; }

	  protected:
		template <typename... Es>
		explicit worker_job(command_pkg pkg, std::tuple<Es...> ctx = {}) : m_pkg(pkg), m_lctx(make_log_context(pkg, ctx)) {}

	  private:
		command_pkg m_pkg;
		log_context m_lctx;
		bool m_running = false;
		bool m_done = false;

		// Benchmarking
		std::chrono::steady_clock::time_point m_start_time;
		std::chrono::microseconds m_bench_sum_execution_time = {};
		size_t m_bench_sample_count = 0;
		std::chrono::microseconds m_bench_min = std::numeric_limits<std::chrono::microseconds>::max();
		std::chrono::microseconds m_bench_max = std::numeric_limits<std::chrono::microseconds>::min();

		template <typename... Es>
		log_context make_log_context(const command_pkg& pkg, const std::tuple<Es...>& ctx = {}) {
			if(const auto tid = pkg.get_tid()) {
				return log_context{std::tuple_cat(std::tuple{"task", *tid, "job", pkg.cid}, ctx)};
			} else {
				return log_context{std::tuple_cat(std::tuple{"job", pkg.cid}, ctx)};
			}
		}

		virtual bool execute(const command_pkg& pkg) = 0;

		/**
		 * Returns a human-readable job description for logging.
		 */
		virtual std::string get_description(const command_pkg& pkg) = 0;
	};

	class horizon_job : public worker_job {
	  public:
		horizon_job(command_pkg pkg, task_manager& tm) : worker_job(pkg), m_task_mngr(tm) { assert(pkg.get_command_type() == command_type::horizon); }

	  private:
		task_manager& m_task_mngr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class epoch_job : public worker_job {
	  public:
		epoch_job(command_pkg pkg, task_manager& tm) : worker_job(pkg), m_task_mngr(tm), m_action(std::get<epoch_data>(pkg.data).action) {
			assert(pkg.get_command_type() == command_type::epoch);
		}

		epoch_action get_epoch_action() const { return m_action; }

	  private:
		task_manager& m_task_mngr;
		epoch_action m_action;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	/**
	 * Informs the data_transfer_manager about the awaited push, then waits until the transfer has been received and completed.
	 */
	class await_push_job : public worker_job {
	  public:
		await_push_job(command_pkg pkg, buffer_transfer_manager& btm) : worker_job(pkg), m_btm(btm) {
			assert(pkg.get_command_type() == command_type::await_push);
		}

	  private:
		buffer_transfer_manager& m_btm;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> m_data_handle = nullptr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class push_job : public worker_job {
	  public:
		push_job(command_pkg pkg, buffer_transfer_manager& btm, buffer_manager& bm) : worker_job(pkg), m_btm(btm), m_buffer_mngr(bm) {
			assert(pkg.get_command_type() == command_type::push);
		}

	  private:
		buffer_transfer_manager& m_btm;
		buffer_manager& m_buffer_mngr;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> m_data_handle = nullptr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class reduction_job : public worker_job {
	  public:
		reduction_job(command_pkg pkg, reduction_manager& rm) : worker_job(pkg, std::tuple{"rid", std::get<reduction_data>(pkg.data).rid}), m_rm(rm) {
			assert(pkg.get_command_type() == command_type::reduction);
		}

	  private:
		reduction_manager& m_rm;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	// host-compute jobs, master-node tasks and collective host tasks
	class host_execute_job : public worker_job {
	  public:
		host_execute_job(command_pkg pkg, host_queue& queue, task_manager& tm, buffer_manager& bm)
		    : worker_job(pkg), m_queue(queue), m_task_mngr(tm), m_buffer_mngr(bm) {
			assert(pkg.get_command_type() == command_type::execution);
		}

	  private:
		host_queue& m_queue;
		task_manager& m_task_mngr;
		buffer_manager& m_buffer_mngr;
		std::future<host_queue::execution_info> m_future;
		bool m_submitted = false;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		std::vector<std::unique_ptr<oob_bounding_box>> m_oob_indices_per_accessor;
#endif

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	/**
	 * TODO: Optimization opportunity: If we don't have any outstanding await-pushes, submitting the kernel to SYCL right away may be faster,
	 * as it can already start copying buffers to the device (i.e. let SYCL do the scheduling).
	 */
	class device_execute_job : public worker_job {
	  public:
		device_execute_job(command_pkg pkg, device_queue& queue, task_manager& tm, buffer_manager& bm, reduction_manager& rm, node_id local_nid)
		    : worker_job(pkg), m_queue(queue), m_task_mngr(tm), m_buffer_mngr(bm), m_reduction_mngr(rm), m_local_nid(local_nid) {
			assert(pkg.get_command_type() == command_type::execution);
		}

	  private:
		device_queue& m_queue;
		task_manager& m_task_mngr;
		buffer_manager& m_buffer_mngr;
		reduction_manager& m_reduction_mngr;
		node_id m_local_nid;
		cl::sycl::event m_event;
		bool m_submitted = false;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		std::vector<oob_bounding_box*> m_oob_indices_per_accessor;
#endif

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class fence_job : public worker_job {
	  public:
		fence_job(command_pkg pkg, task_manager& tm) : worker_job(pkg), m_task_mngr(tm) { assert(pkg.get_command_type() == command_type::fence); }

	  private:
		task_manager& m_task_mngr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

} // namespace detail
} // namespace celerity
