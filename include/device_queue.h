#pragma once

#include <memory>

#include "config.h"
#include "handler.h"
#include "logger.h"
#include "ranges.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

namespace cl {
namespace sycl {
	class device;
	class event;
	class queue;
} // namespace sycl
} // namespace cl

namespace celerity {
namespace detail {

	class task;
	class task_manager;

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
		 * @param task_mngr The @p device_queue does not take ownership of @p task_mngr, but it is expected to exist for @p device_queue's entire lifetime.
		 * @param user_device Optionally a device can be provided, which will take precedence over any configuration.
		 */
		void init(config& cfg, task_manager* task_mngr, cl::sycl::device* user_device);

		/**
		 * @brief Executes the kernel associated with task @p tid over the chunk @p chnk.
		 */
		cl::sycl::event execute(task_id tid, subrange<3> sr) const {
			assert(task_mngr->has_task(tid));
			assert(task_mngr->get_task(tid)->get_type() == task_type::COMPUTE);
			auto task = std::static_pointer_cast<const compute_task>(task_mngr->get_task(tid));
			auto& cgf = task->get_command_group();
			return sycl_queue->submit([&cgf, task, sr, this](cl::sycl::handler& sycl_handler) {
				auto cgh = std::make_unique<compute_task_handler<false>>(task, sr, &sycl_handler, forced_work_group_size);
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
		task_manager* task_mngr = nullptr;
		std::unique_ptr<cl::sycl::queue> sycl_queue;
		bool device_profiling_enabled = false;
		// FIXME: Get rid of this
		size_t forced_work_group_size = 0;

		cl::sycl::device pick_device(config& cfg, cl::sycl::device* user_device) const;
		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

} // namespace detail
} // namespace celerity
