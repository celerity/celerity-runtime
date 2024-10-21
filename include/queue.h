#pragma once

#include <future>
#include <memory>

#include "fence.h"
#include "runtime.h"
#include "tracy.h"
#include "types.h"

namespace celerity::detail {
struct barrier_tag {};
} // namespace celerity::detail

namespace celerity::experimental {
/// Pass this tag to `queue::wait` to issue a barrier synchronization across the entire cluster.
inline constexpr detail::barrier_tag barrier{};
} // namespace celerity::experimental

namespace celerity {

class queue {
  public:
	/// Constructs a queue which distributes work across all devices associated with the runtime.
	///
	/// To manually select a subset of devices in the system, call `runtime::init` with an appropriate selector before constructing the first Celerity object.
	queue() : m_tracker(std::make_shared<tracker>()) {}

	/// Submits a command group to the queue.
	template <typename CGF>
	void submit(CGF&& cgf) { // NOLINT(readability-convert-member-functions-to-static)
		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::submit", Orange3);
		[[maybe_unused]] const auto tid = detail::runtime::get_instance().submit(std::forward<CGF>(cgf));
		CELERITY_DETAIL_TRACY_ZONE_NAME("T{} submit", tid);
	}

	/// Waits for all tasks submitted to the queue to complete.
	///
	/// Since waiting will stall the scheduling of more work, this should be used sparingly - more so than on a single-node SYCL program.
	///
	/// Note that this overload of `wait` does not issue a global barrier, so when using this for simple user-side benchmarking, cluster nodes might disagree on
	/// start time measurements. Use `wait(experimental::barrier)` instead for benchmarking purposes.
	void wait() { // NOLINT(readability-convert-member-functions-to-static)
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::wait", Red2);
		[[maybe_unused]] const auto tid = detail::runtime::get_instance().sync(detail::epoch_action::none);
		CELERITY_DETAIL_TRACY_ZONE_NAME("T{} wait", tid);
	}

	/// Waits for all tasks submitted to the queue to complete, then barrier-synchronizes across the entire cluster.
	///
	/// This has an even higher latency than `wait()`, but may be useful for user-side performance measurements.
	void wait(detail::barrier_tag /* barrier */) { // NOLINT(readability-convert-member-functions-to-static)
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::wait", Red2);
		[[maybe_unused]] const auto tid = detail::runtime::get_instance().sync(detail::epoch_action::barrier);
		CELERITY_DETAIL_TRACY_ZONE_NAME("T{} wait (barrier)", tid);
	}

	/// Asynchronously captures the value of a host object by copy, introducing the same dependencies as a side-effect would.
	///
	/// Waiting on the returned future in the application thread can stall scheduling of more work. To hide latency, either submit more command groups between
	/// fence and wait operations or ensure that other independent command groups are eligible to run while the fence is executed.
	template <typename T>
	[[nodiscard]] std::future<T> fence(const experimental::host_object<T>& obj) {
		return detail::fence(obj);
	}

	/// Asynchronously captures the contents of a buffer subrange, introducing the same dependencies as a read-accessor would.
	///
	/// Waiting on the returned future in the application thread can stall scheduling of more work. To hide latency, either submit more command groups between
	/// fence and wait operations or ensure that other independent command groups are eligible to run while the fence is executed.
	template <typename DataT, int Dims>
	[[nodiscard]] std::future<buffer_snapshot<DataT, Dims>> fence(const buffer<DataT, Dims>& buf, const subrange<Dims>& sr) {
		return detail::fence(buf, sr);
	}

	/// Asynchronously captures the contents of an entire buffer, introducing the same dependencies as a read-accessor would.
	///
	/// Waiting on the returned future in the application thread can stall scheduling of more work. To hide latency, either submit more command groups between
	/// fence and wait operations or ensure that other independent command groups are eligible to run while the fence is executed.
	template <typename DataT, int Dims>
	[[nodiscard]] std::future<buffer_snapshot<DataT, Dims>> fence(const buffer<DataT, Dims>& buf) {
		return detail::fence(buf, {{}, buf.get_range()});
	}

  private:
	/// A `tacker` instance is shared by all copies of this `queue` via a `std::shared_ptr` to implement (SYCL) reference semantics.
	/// It notifies the runtime of queue creation and destruction, which might trigger runtime initialization if it is the first such object.
	struct tracker {
		tracker() {
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::queue", DarkSlateBlue);
			if(!detail::runtime::has_instance()) { detail::runtime::init(nullptr, nullptr, detail::auto_select_devices{}); }
			detail::runtime::get_instance().create_queue();
		}

		tracker(const tracker&) = delete;
		tracker(tracker&&) = delete;
		tracker& operator=(const tracker&) = delete;
		tracker& operator=(tracker&&) = delete;

		~tracker() {
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::~queue", DarkCyan);

			detail::runtime::get_instance().destroy_queue();

			// ~queue() guarantees that all operations on that particular queue have finished executing, which we simply guarantee by waiting on all operations
			// on all live queues.
			if(detail::runtime::has_instance()) { detail::runtime::get_instance().sync(detail::epoch_action::none); }
		}
	};

	std::shared_ptr<tracker> m_tracker;
};

} // namespace celerity

namespace celerity::experimental {

/// Controls the lookahead window size for all future submissions on the queue. The default setting is `lookahead::automatic`.
///
/// Use this function if the default configuration either does not eliminate enough buffer resizes in your application (`lookahead::infinite`), or host tasks
/// and kernels interact with the rest of your application in ways that require immediate flushing of every submitted command group (`lookahead::none`).
//
// Attached to the queue to signal that semantics are in-order with other submissions. Still applies to all queues.
// Experimental: This is only applicable to fully static work assignment, which might not remain the default forever.
inline void set_lookahead(celerity::queue& /* queue */, const experimental::lookahead lookahead) {
	detail::runtime::get_instance().set_scheduler_lookahead(lookahead);
}

/// Flushes all command groups asynchronously enqueued in the scheduler.
///
/// This is beneficial only in rare situations where host-side code needs to synchronize with kernels or host tasks in a manner that is opaque to the runtime.
inline void flush(celerity::queue& /* queue */) { detail::runtime::get_instance().flush_scheduler(); }

} // namespace celerity::experimental
