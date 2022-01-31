#pragma once

#include <cassert>
#include <chrono>
#include <future>
#include <limits>
#include <utility>

#include "buffer_manager.h"
#include "buffer_transfer_manager.h"
#include "command.h"
#include "host_queue.h"
#include "log.h"

namespace celerity {
namespace detail {

	class device_queue;
	class executor;
	class task_manager;
	class reduction_manager;

	class worker_job;

	class worker_job {
	  public:
		worker_job(const worker_job&) = delete;
		worker_job(worker_job&&) = delete;

		virtual ~worker_job() = default;

		void start();
		void update();

		bool is_running() const { return running; }
		bool is_done() const { return done; }

	  protected:
		template <typename... Es>
		worker_job(command_pkg pkg, std::tuple<Es...> ctx = {}) : pkg(pkg), lctx(std::tuple_cat(std::make_tuple("job", pkg.cid), ctx)) {}

	  private:
		command_pkg pkg;
		log_context lctx;
		bool running = false;
		bool done = false;

		// Benchmarking
		std::chrono::steady_clock::time_point start_time;
		std::chrono::microseconds bench_sum_execution_time = {};
		size_t bench_sample_count = 0;
		std::chrono::microseconds bench_min = std::numeric_limits<std::chrono::microseconds>::max();
		std::chrono::microseconds bench_max = std::numeric_limits<std::chrono::microseconds>::min();

		virtual bool execute(const command_pkg& pkg) = 0;

		/**
		 * Returns a human-readable job description for logging.
		 */
		virtual std::string get_description(const command_pkg& pkg) = 0;
	};

	class horizon_job : public worker_job {
	  public:
		horizon_job(command_pkg pkg, task_manager& tm) : worker_job(pkg, std::make_tuple("tid", std::get<horizon_data>(pkg.data).tid)), task_mngr(tm) {
			assert(pkg.cmd == command_type::HORIZON);
		}

	  private:
		task_manager& task_mngr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class epoch_job : public worker_job {
	  public:
		epoch_job(command_pkg pkg, task_manager& tm) : worker_job(pkg), task_mngr(tm) { assert(pkg.cmd == command_type::EPOCH); }

	  private:
		task_manager& task_mngr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	/**
	 * Informs the data_transfer_manager about the awaited push, then waits until the transfer has been received and completed.
	 */
	class await_push_job : public worker_job {
	  public:
		await_push_job(command_pkg pkg, buffer_transfer_manager& btm) : worker_job(pkg), btm(btm) { assert(pkg.cmd == command_type::AWAIT_PUSH); }

	  private:
		buffer_transfer_manager& btm;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class push_job : public worker_job {
	  public:
		push_job(command_pkg pkg, buffer_transfer_manager& btm, buffer_manager& bm) : worker_job(pkg), btm(btm), buffer_mngr(bm) {
			assert(pkg.cmd == command_type::PUSH);
		}

	  private:
		buffer_transfer_manager& btm;
		buffer_manager& buffer_mngr;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	class reduction_job : public worker_job {
	  public:
		reduction_job(command_pkg pkg, reduction_manager& rm) : worker_job(pkg, std::make_tuple("rid", std::get<reduction_data>(pkg.data).rid)), rm(rm) {
			assert(pkg.cmd == command_type::REDUCTION);
		}

	  private:
		reduction_manager& rm;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	// host-compute jobs, master-node tasks and collective host tasks
	class host_execute_job : public worker_job {
	  public:
		host_execute_job(command_pkg pkg, detail::host_queue& queue, detail::task_manager& tm, buffer_manager& bm)
		    : worker_job(pkg, std::make_tuple("tid", std::get<execution_data>(pkg.data).tid)), queue(queue), task_mngr(tm), buffer_mngr(bm) {
			assert(pkg.cmd == command_type::EXECUTION);
		}

	  private:
		detail::host_queue& queue;
		detail::task_manager& task_mngr;
		detail::buffer_manager& buffer_mngr;
		std::future<detail::host_queue::execution_info> future;
		bool submitted = false;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

	/**
	 * TODO: Optimization opportunity: If we don't have any outstanding await-pushes, submitting the kernel to SYCL right away may be faster,
	 * as it can already start copying buffers to the device (i.e. let SYCL do the scheduling).
	 */
	class device_execute_job : public worker_job {
	  public:
		device_execute_job(command_pkg pkg, detail::device_queue& queue, detail::task_manager& tm, buffer_manager& bm, reduction_manager& rm, node_id local_nid)
		    : worker_job(pkg, std::make_tuple("tid", std::get<execution_data>(pkg.data).tid)), queue(queue), task_mngr(tm), buffer_mngr(bm), reduction_mngr(rm),
		      local_nid(local_nid) {
			assert(pkg.cmd == command_type::EXECUTION);
		}

	  private:
		detail::device_queue& queue;
		detail::task_manager& task_mngr;
		detail::buffer_manager& buffer_mngr;
		detail::reduction_manager& reduction_mngr;
		node_id local_nid;
		cl::sycl::event event;
		bool submitted = false;

		bool execute(const command_pkg& pkg) override;
		std::string get_description(const command_pkg& pkg) override;
	};

} // namespace detail
} // namespace celerity
