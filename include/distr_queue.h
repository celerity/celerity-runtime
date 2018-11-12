#pragma once

#include <memory>
#include <mutex>

#include <SYCL/sycl.hpp>

#include "handler.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

namespace celerity {

namespace detail {
	class task_manager;
}

template <typename DataT, int Dims>
class buffer;

class distr_queue {
  public:
	distr_queue();
	distr_queue(cl::sycl::device& device);

	~distr_queue();

	distr_queue(const distr_queue& other) = delete;
	distr_queue(distr_queue&& other) = delete;

	template <typename CGF>
	void submit(CGF cgf) {
		task_mngr->create_compute_task(cgf);
	}

	/**
	 * @internal
	 */
	void set_task_manager(std::shared_ptr<detail::task_manager> task_mngr) { this->task_mngr = task_mngr; }

	/**
	 * @brief Executes the kernel associated with task @p tid over the chunk @p chnk.
	 * @internal
	 */
	cl::sycl::event execute(task_id tid, subrange<3> sr) {
		assert(task_mngr->has_task(tid));
		assert(task_mngr->get_task(tid)->get_type() == task_type::COMPUTE);
		auto task = std::static_pointer_cast<const detail::compute_task>(task_mngr->get_task(tid));
		auto& cgf = task->get_command_group();
		return sycl_queue->submit([&cgf, task, sr](cl::sycl::handler& sycl_handler) {
			compute_livepass_handler h(*task, sr, &sycl_handler);
			cgf(h);
		});
	}

	/**
	 * @internal
	 */
	bool is_ocl_profiling_enabled() const { return ocl_profiling_enabled; }

  private:
	std::unique_ptr<cl::sycl::queue> sycl_queue;
	bool ocl_profiling_enabled = false;
	std::shared_ptr<detail::task_manager> task_mngr;

	void init(cl::sycl::device* device_ptr);

	static void handle_async_exceptions(cl::sycl::exception_list el);
};

} // namespace celerity
