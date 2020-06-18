#pragma once

#include <memory>

#include <CL/sycl.hpp>

#include "config.h"
#include "logger.h"

namespace celerity {
namespace detail {

	class task;

	/**
	 * The @p device_queue wraps the actual SYCL queue and is used to submit kernels.
	 */
	class device_queue {
	  public:
		device_queue(logger& queue_logger) : queue_logger(queue_logger){};

		/**
		 * @brief Initializes the @p device_queue, selecting an appropriate device in the process.
		 *
		 * @param cfg The configuration is used to select the appropriate SYCL device.
		 * @param user_device Optionally a device can be provided, which will take precedence over any configuration.
		 */
		void init(const config& cfg, cl::sycl::device* user_device);

		/**
		 * @brief Executes the kernel associated with task @p ctsk over the chunk @p chnk.
		 */
		template <typename Fn>
		cl::sycl::event submit(Fn&& fn) {
			// FIXME: Get rid of forced_work_group_size
			return sycl_queue->submit([fn = std::forward<Fn>(fn), fwgs = forced_work_group_size](cl::sycl::handler& sycl_handler) { fn(sycl_handler, fwgs); });
		}

		/**
		 * @brief Waits until all currently submitted operations have completed.
		 */
		void wait() { sycl_queue->wait_and_throw(); }

		/**
		 * @brief Returns whether device profiling is enabled.
		 */
		bool is_profiling_enabled() const { return device_profiling_enabled; }

		cl::sycl::queue& get_sycl_queue() const {
			assert(sycl_queue != nullptr);
			return *sycl_queue;
		}

	  private:
		logger& queue_logger;
		std::unique_ptr<cl::sycl::queue> sycl_queue;
		bool device_profiling_enabled = false;
		// FIXME: Get rid of this
		size_t forced_work_group_size = 0;

		cl::sycl::device pick_device(const config& cfg, cl::sycl::device* user_device) const;
		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

} // namespace detail
} // namespace celerity
