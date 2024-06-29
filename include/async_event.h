#pragma once

#include <cassert>
#include <chrono>
#include <memory>
#include <optional>
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
	virtual bool is_complete() = 0;

	/// There is only one instruction type which returns a result, namely alloc_instruction returning a pointer to the allocated memory, i.e. a void*. Having a
	/// void* return type on async_event_impl is somewhat leaky, but we don't gain much by wrapping it in a std::any.
	virtual void* get_result() { return nullptr; }

	/// Returns the time execution time as measured if profiling was enabled in the issuing component. Requires `is_complete()` to be true.
	virtual std::optional<std::chrono::nanoseconds> get_native_execution_time() { return std::nullopt; }
};

/// `async_event` implementation that is immediately complete. Used to report synchronous completion of some operations within an otherwise asynchronous
/// context.
class complete_event final : public async_event_impl {
  public:
	complete_event() = default;
	explicit complete_event(void* const result) : m_result(result) {}
	bool is_complete() override { return true; }
	void* get_result() override { return m_result; }

  private:
	void* m_result = nullptr;
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

	void* get_result() const {
		assert(m_impl != nullptr);
		return m_impl->get_result();
	}

	std::optional<std::chrono::nanoseconds> get_native_execution_time() const {
		assert(m_impl != nullptr);
		return m_impl->get_native_execution_time();
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
inline async_event make_complete_event(void* const result) { return make_async_event<complete_event>(result); }

} // namespace celerity::detail
