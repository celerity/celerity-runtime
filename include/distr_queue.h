#pragma once

#include "runtime.h"

namespace celerity {

class distr_queue {
  public:
	distr_queue() { init(nullptr); };
	distr_queue(cl::sycl::device& device) { init(&device); };
	~distr_queue() { detail::runtime::get_instance().shutdown(); }

	distr_queue(const distr_queue& other) = delete;
	distr_queue(distr_queue&& other) = delete;

	template <typename CGF>
	void submit(CGF cgf) {
		// (Note while this function could be made static, it must not be!)
		detail::runtime::get_instance().get_task_manager().create_compute_task(cgf);
	}

	template <typename MAF>
	void with_master_access(MAF maf) {
		detail::runtime::get_instance().get_task_manager().create_master_access_task(maf);
	}

  private:
	void init(cl::sycl::device* user_device) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }
		detail::runtime::get_instance().startup(user_device);
	}
};

} // namespace celerity
