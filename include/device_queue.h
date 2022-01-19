#pragma once

#include <memory>

#include <CL/sycl.hpp>

#include "config.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	class task;

	/**
	 * The @p device_queue wraps the actual SYCL queue and is used to submit kernels.
	 */
	class device_queue {
	  public:
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
			auto evt = sycl_queue->submit([fn = std::forward<Fn>(fn)](cl::sycl::handler& sycl_handler) { fn(sycl_handler); });
#if WORKAROUND_HIPSYCL
			// hipSYCL does not guarantee that command groups are actually scheduled until an explicit await operation, which we cannot insert without
			// blocking the executor loop (See https://github.com/illuhad/hipSYCL/issues/599). Instead, we explicitly flush the queue to be able to continue
			// using our polling-based approach.
			hipsycl::rt::application::dag().flush_async();
#endif
			return evt;
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
		std::unique_ptr<cl::sycl::queue> sycl_queue;
		bool device_profiling_enabled = false;

		cl::sycl::device pick_device(const config& cfg, cl::sycl::device* user_device) const;
		void handle_async_exceptions(cl::sycl::exception_list el) const;
	};

} // namespace detail
} // namespace celerity
