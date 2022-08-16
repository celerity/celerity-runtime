#pragma once

#include <memory>
#include <type_traits>

#include "capture.h"
#include "device_queue.h"
#include "host_object.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	class distr_queue_tracker {
	  public:
		~distr_queue_tracker() {
			if(!m_drained) { detail::runtime::get_instance().shutdown({}, {}); }
		}

		friend void check_not_drained(const std::shared_ptr<distr_queue_tracker>& tracker) {
			if(!tracker) { throw std::runtime_error("distr_queue has been moved from"); }
			if(tracker->m_drained) { throw std::runtime_error("distr_queue has already been drained"); }
		}

		void set_drained() { m_drained = true; }

	  private:
		bool m_drained = false;
	};

	template <typename CGF>
	constexpr bool is_safe_cgf = std::is_standard_layout<CGF>::value;

} // namespace detail

struct allow_by_ref_t {};

inline constexpr allow_by_ref_t allow_by_ref{};

class distr_queue {
  public:
	distr_queue() { init(detail::auto_select_device{}); }

	[[deprecated("Use the overload with device selector instead, this will be removed in future release")]] distr_queue(cl::sycl::device& device) {
		if(detail::runtime::is_initialized()) { throw std::runtime_error("Passing explicit device not possible, runtime has already been initialized."); }
		init(device);
	}

	template <typename DeviceSelector>
	distr_queue(const DeviceSelector& device_selector) {
		if(detail::runtime::is_initialized()) {
			throw std::runtime_error("Passing explicit device selector not possible, runtime has already been initialized.");
		}
		init(device_selector);
	}

	/**
	 * Submits a command group to the queue.
	 *
	 * Invoke via `q.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {...})`.
	 *
	 * With this overload, CGF may capture by-reference. This may lead to lifetime issues with asynchronous execution, so using the `submit(cgf)` overload is
	 * preferred in the common case.
	 */
	template <typename CGF>
	void submit(allow_by_ref_t, CGF cgf) { // NOLINT(readability-convert-member-functions-to-static)
		check_not_drained(m_tracker);
		detail::runtime::get_instance().get_task_manager().submit_command_group(std::move(cgf));
	}

	/**
	 * Submits a command group to the queue.
	 *
	 * CGF must not capture by reference. This is a conservative safety check to avoid lifetime issues when command groups are executed asynchronously.
	 *
	 * If you know what you are doing, you can use the `allow_by_ref` overload of `submit` to bypass this check.
	 */
	template <typename CGF>
	void submit(CGF cgf) {
		static_assert(detail::is_safe_cgf<CGF>, "The provided command group function is not multi-pass execution safe. Please make sure to only capture by "
		                                        "value. If you know what you're doing, use submit(celerity::allow_by_ref, ...).");
		submit(allow_by_ref, std::move(cgf));
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes accross the entire cluster. }
	 */
	void slow_full_sync() { (void)slow_full_sync({}, {}); }

	template <typename T>
	typename experimental::capture<T>::value_type slow_full_sync(const experimental::capture<T>& cap) {
		auto [buffer_captures, side_effects] = detail::capture_inspector::collect_requirements(std::tuple{cap});
		const auto sync_guard = slow_full_sync(std::move(buffer_captures), std::move(side_effects));
		return std::get<0>(detail::capture_inspector::exfiltrate_by_copy(cap));
	}

	template <typename... Ts>
	std::tuple<typename experimental::capture<Ts>::value_type...> slow_full_sync(const std::tuple<experimental::capture<Ts>...>& caps) {
		auto [buffer_captures, side_effects] = detail::capture_inspector::collect_requirements(caps);
		const auto sync_guard = slow_full_sync(std::move(buffer_captures), std::move(side_effects));
		return detail::capture_inspector::exfiltrate_by_copy(caps);
	}

	void drain() { drain_internal({}, {}); }

	template <typename T>
	typename experimental::capture<T>::value_type drain(const experimental::capture<T>& cap) {
		auto [buffer_captures, side_effects] = detail::capture_inspector::collect_requirements(std::tuple{cap});
		drain_internal(std::move(buffer_captures), std::move(side_effects));
		return std::get<0>(detail::capture_inspector::exfiltrate_by_move(std::tuple{cap}));
	}

	template <typename... Ts>
	std::tuple<typename experimental::capture<Ts>::value_type...> drain(const std::tuple<experimental::capture<Ts>...>& caps) {
		auto [buffer_captures, side_effects] = detail::capture_inspector::collect_requirements(caps);
		drain_internal(std::move(buffer_captures), std::move(side_effects));
		return detail::capture_inspector::exfiltrate_by_move(caps);
	}

  private:
	std::shared_ptr<detail::distr_queue_tracker> m_tracker;

	void init(detail::device_or_selector device_or_selector) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr, device_or_selector); }
		try {
			detail::runtime::get_instance().startup();
		} catch(detail::runtime_already_started_error&) {
			throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)");
		}
		m_tracker = std::make_shared<detail::distr_queue_tracker>();
	}

	detail::runtime::sync_guard slow_full_sync(detail::buffer_capture_map buffer_captures, detail::side_effect_map side_effects) {
		check_not_drained(m_tracker);
		return detail::runtime::get_instance().sync(std::move(buffer_captures), std::move(side_effects));
	}

	void drain_internal(detail::buffer_capture_map buffer_captures, detail::side_effect_map side_effects) {
		check_not_drained(m_tracker);
		detail::runtime::get_instance().shutdown(std::move(buffer_captures), std::move(side_effects));
		m_tracker->set_drained();
	}
};

} // namespace celerity
