#pragma once

#include <memory>
#include <type_traits>

#include "runtime.h"

#if !defined(CELERITY_STRICT_CGF_SAFETY)
#define CELERITY_STRICT_CGF_SAFETY 1
#endif

namespace celerity {
namespace detail {

	class distr_queue_tracker {
	  public:
		~distr_queue_tracker() { runtime::get_instance().shutdown(); }
	};

	template <typename CGF>
	constexpr bool is_safe_cgf = std::is_standard_layout<CGF>::value;

} // namespace detail

class distr_queue {
  public:
	distr_queue() { init(nullptr); }
	distr_queue(cl::sycl::device& device) {
		if(detail::runtime::is_initialized()) { throw std::runtime_error("Passing explicit device not possible, runtime has already been initialized."); }
		init(&device);
	}

	distr_queue(const distr_queue&) = default;
	distr_queue(distr_queue&&) = default;

	distr_queue& operator=(const distr_queue&) = delete;
	distr_queue& operator=(distr_queue&&) = delete;

	template <typename CGF>
	void submit(CGF cgf) {
#if CELERITY_STRICT_CGF_SAFETY
		static_assert(
		    detail::is_safe_cgf<CGF>, "The provided command group function is not multi-pass execution safe. Please make sure to only capture by value.");
#endif

		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		detail::runtime::get_instance().get_task_manager().create_compute_task(cgf);
	}

	template <typename MAF>
	void with_master_access(MAF maf) {
		detail::runtime::get_instance().get_task_manager().create_master_access_task(maf);
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes accross the entire cluster. }
	 */
	void slow_full_sync() { detail::runtime::get_instance().sync(); }

  private:
	std::shared_ptr<detail::distr_queue_tracker> tracker;

	void init(cl::sycl::device* user_device) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr, user_device); }
		try {
			detail::runtime::get_instance().startup();
		} catch(detail::runtime_already_started_error&) {
			throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)");
		}
		tracker = std::make_shared<detail::distr_queue_tracker>();
	}
};

} // namespace celerity
