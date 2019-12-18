#pragma once

#include <memory>

#include <CL/sycl.hpp>

#include "config.h"
#include "handler.h"
#include "logger.h"
#include "ranges.h"
#include "task.h"
#include "types.h"

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
		void init(config& cfg, cl::sycl::device* user_device);

		/**
		 * @brief Executes the kernel associated with task @p ctsk over the chunk @p chnk.
		 */
		cl::sycl::event execute(std::shared_ptr<const compute_task> ctsk, subrange<3> sr) const {
			auto& cgf = ctsk->get_command_group();
			return sycl_queue->submit([&cgf, ctsk, sr, this](cl::sycl::handler& sycl_handler) {
				auto cgh = std::make_unique<compute_task_handler<false>>(ctsk, sr, &sycl_handler, forced_work_group_size);
				cgf(*cgh);
			});
		}

		/**
		 * @brief Waits until all currently submitted operations have completed.
		 */
		void wait() const { sycl_queue->wait_and_throw(); }

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

		cl::sycl::device pick_device(config& cfg, cl::sycl::device* user_device) const;
		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

} // namespace detail
} // namespace celerity
