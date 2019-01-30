#pragma once

#include "runtime.h"

namespace celerity {

class distr_queue {
  public:
	distr_queue() { init(nullptr); };
	distr_queue(cl::sycl::device& device) { init(&device); };
	~distr_queue() { runtime::get_instance().shutdown(); }

	distr_queue(const distr_queue& other) = delete;
	distr_queue(distr_queue&& other) = delete;

	template <typename CGF>
	void submit(CGF cgf) {
		// (Note while this function could be made static, it must not be!)
		runtime::get_instance().get_task_manager().create_compute_task(cgf);
	}

  private:
	void init(cl::sycl::device* user_device) {
		if(!runtime::is_initialized()) { runtime::init(nullptr, nullptr); }
		runtime::get_instance().startup(user_device);
	}
};

} // namespace celerity
