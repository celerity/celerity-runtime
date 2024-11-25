#pragma once

#include "buffer.h"
#include "device_selector.h"
#include "fence.h"
#include "handler.h"
#include "host_object.h"
#include "ranges.h"
#include "runtime.h"
#include "tracy.h"
#include "types.h"

#include <future>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <sycl/sycl.hpp>


namespace celerity {

template <typename T, int Dims>
class buffer_snapshot;

struct [[deprecated("This tag type is no longer required to capture by reference")]] allow_by_ref_t {};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
inline constexpr allow_by_ref_t allow_by_ref{};
#pragma GCC diagnostic pop

class [[deprecated("Use celerity::queue instead")]] distr_queue {
  public:
	distr_queue() : distr_queue(ctor_internal_tag{}, detail::auto_select_devices{}) {}

	/**
	 * @brief Creates a distr_queue and instructs it to use a particular set of devices.
	 *
	 * @param devices The devices to be used on the current node. This can vary between nodes.
	 *                If there are multiple nodes running on the same host, the list of devices must be the same across nodes on the same host.
	 */
	distr_queue(const std::vector<sycl::device>& devices) : distr_queue(ctor_internal_tag{}, devices) {}

	/**
	 * @brief Creates a distr_queue and instructs it to use a particular set of devices.
	 *
	 * @param device_selector The device selector to be used on the current node. This can vary between nodes.
	 *                        If there are multiple nodes running on the same host, the selector must be the same across nodes on the same host.
	 */
	template <typename DeviceSelector>
	distr_queue(const DeviceSelector& device_selector) : distr_queue(ctor_internal_tag{}, device_selector) {}

	/**
	 * Submits a command group to the queue.
	 *
	 * Invoke via `q.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {...})`.
	 */
	template <typename CGF>
	[[deprecated("This overload is no longer required to capture by reference")]] void submit(allow_by_ref_t /* tag */, CGF cgf) {
		submit(std::move(cgf));
	}

	/**
	 * Submits a command group to the queue.
	 */
	template <typename CGF>
	void submit(CGF cgf) { // NOLINT(readability-convert-member-functions-to-static)
		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("distr_queue::submit", Orange3);
		auto cg = detail::invoke_command_group_function(std::move(cgf));
		[[maybe_unused]] const auto tid = detail::runtime::get_instance().submit(std::move(cg));
		CELERITY_DETAIL_TRACY_ZONE_NAME("T{} submit", tid);
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes across the entire cluster. }
	 */
	void slow_full_sync() { // NOLINT(readability-convert-member-functions-to-static)
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("distr_queue::slow_full_sync", Red2);
		[[maybe_unused]] const auto tid = detail::runtime::get_instance().sync(detail::epoch_action::barrier);
		CELERITY_DETAIL_TRACY_ZONE_NAME("T{} slow_full_sync", tid);
	}

	/**
	 * Asynchronously captures the value of a host object by copy, introducing the same dependencies as a side-effect would.
	 *
	 * Waiting on the returned future in the application thread can stall scheduling of more work. To hide latency, either submit more command groups between
	 * fence and wait operations or ensure that other independent command groups are eligible to run while the fence is executed.
	 */
	template <typename T>
	[[nodiscard]] std::future<T> fence(const experimental::host_object<T>& obj) {
		return detail::fence(obj);
	}

	/**
	 * Asynchronously captures the contents of a buffer subrange, introducing the same dependencies as a read-accessor would.
	 *
	 * Waiting on the returned future in the application thread can stall scheduling of more work. To hide latency, either submit more command groups between
	 * fence and wait operations or ensure that other independent command groups are eligible to run while the fence is executed.
	 */
	template <typename DataT, int Dims>
	[[nodiscard]] std::future<buffer_snapshot<DataT, Dims>> fence(const buffer<DataT, Dims>& buf, const subrange<Dims>& sr) {
		return detail::fence(buf, sr);
	}

	/**
	 * Asynchronously captures the contents of an entire buffer, introducing the same dependencies as a read-accessor would.
	 *
	 * Waiting on the returned future in the application thread can stall scheduling of more work. To hide latency, either submit more command groups between
	 * fence and wait operations or ensure that other independent command groups are eligible to run while the fence is executed.
	 */
	template <typename DataT, int Dims>
	[[nodiscard]] std::future<buffer_snapshot<DataT, Dims>> fence(const buffer<DataT, Dims>& buf) {
		return detail::fence(buf, {{}, buf.get_range()});
	}

  private:
	/// A `tacker` instance is shared by all copies of this `distr_queue` via a `std::shared_ptr` to implement (SYCL) reference semantics.
	/// It notifies the runtime of queue creation and destruction, which might trigger runtime initialization if it is the first such object.
	struct tracker {
		tracker(const detail::devices_or_selector& devices_or_selector) {
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("distr_queue::distr_queue", DarkSlateBlue);

			if(!detail::runtime::has_instance()) {
				detail::runtime::init(nullptr, nullptr, devices_or_selector);
			} else if(!std::holds_alternative<detail::auto_select_devices>(devices_or_selector)) {
				throw std::runtime_error(fmt::format("Passing explicit device {} not possible, runtime has already been initialized.",
				    std::holds_alternative<detail::device_selector>(devices_or_selector) ? "selector" : "list"));
			}

			detail::runtime::get_instance().create_queue();
		}

		tracker(const tracker&) = delete;
		tracker(tracker&&) = delete;
		tracker& operator=(const tracker&) = delete;
		tracker& operator=(tracker&&) = delete;

		~tracker() {
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("distr_queue::~distr_queue", DarkCyan);

			detail::runtime::get_instance().destroy_queue();

			// ~distr_queue() guarantees that all operations on that particular queue have finished executing, which we simply guarantee by waiting on all
			// operations on all live queues.
			if(detail::runtime::has_instance()) { detail::runtime::get_instance().sync(detail::epoch_action::none); }
		}
	};

	struct ctor_internal_tag {};

	std::shared_ptr<tracker> m_tracker;

	distr_queue(const ctor_internal_tag /* tag */, const detail::devices_or_selector& devices_or_selector)
	    : m_tracker(std::make_shared<tracker>(devices_or_selector)) {}
};

} // namespace celerity
