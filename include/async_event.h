#pragma once

#include <cassert>
#include <memory>
#include <type_traits>


namespace celerity::detail {

/// Abstract base class for `async_event` implementations.
class async_event_impl {
  public:
	async_event_impl() = default;
	async_event_impl(const async_event_impl&) = delete;
	async_event_impl(async_event_impl&&) = delete;
	async_event_impl& operator=(const async_event_impl&) = delete;
	async_event_impl& operator=(async_event_impl&&) = delete;
	virtual ~async_event_impl() = default;

	/// If this function returns true once, the implementation must guarantee that it will always do so in the future.
	/// The event is expected to be cheap to poll repeatedly, and the operation must proceed in the background even while not being polled.
	virtual bool is_complete() const = 0;
};

/// `async_event` implementation that is immediately complete. Used to report synchronous completion of some operations within an otherwise asynchronous
/// context.
class complete_event final : public async_event_impl {
  public:
	complete_event() = default;
	bool is_complete() const override { return true; }
};

/// Type-erased event signalling completion of events at the executor layer. These may wrap SYCL events, asynchronous MPI requests, or similar.
class [[nodiscard]] async_event {
  public:
	async_event() = default;
	async_event(std::unique_ptr<async_event_impl> impl) noexcept : m_impl(std::move(impl)) {}

	/// Polls the underlying event operation to check if it has completed. This function is cheap to call repeatedly.
	bool is_complete() const {
		assert(m_impl != nullptr);
		return m_impl->is_complete();
	}

  private:
	std::unique_ptr<async_event_impl> m_impl;
};

/// Shortcut to create an `async_event` using an `async_event_impl`-derived type `Event`.
template <typename Event, typename... CtorParams>
async_event make_async_event(CtorParams&&... ctor_args) {
	static_assert(std::is_base_of_v<async_event_impl, Event>);
	return async_event(std::make_unique<Event>(std::forward<CtorParams>(ctor_args)...));
}

/// Shortcut to create an `async_event(complete_event)`.
inline async_event make_complete_event() { return make_async_event<complete_event>(); }

} // namespace celerity::detail
