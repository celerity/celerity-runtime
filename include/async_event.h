#pragma once

#include <any>
#include <cassert>
#include <memory>
#include <type_traits>


namespace celerity::detail {

class async_event_base {
  public:
	async_event_base() = default;
	async_event_base(const async_event_base&) = delete;
	async_event_base(async_event_base&&) = delete;
	async_event_base& operator=(const async_event_base&) = delete;
	async_event_base& operator=(async_event_base&&) = delete;
	virtual ~async_event_base() = default;

	virtual bool is_complete() const = 0;
	virtual std::any take_result() { return {}; }
};

class complete_event final : public async_event_base {
  public:
	complete_event() = default;
	complete_event(std::any result) : m_result(std::move(result)) {}

	bool is_complete() const override { return true; }
	std::any take_result() override { return std::move(m_result); }

  private:
	std::any m_result;
};

class [[nodiscard]] async_event {
  public:
	async_event() = default;
	async_event(std::unique_ptr<async_event_base> impl) noexcept : m_impl(std::move(impl)) {}

	bool is_complete() const {
		assert(m_impl != nullptr);
		return m_impl->is_complete();
	}

	std::any take_result() {
		assert(m_impl != nullptr);
		assert(m_impl->is_complete());
		return m_impl->take_result();
	}

  private:
	std::unique_ptr<async_event_base> m_impl;
};

template <typename Event, typename... CtorParams>
async_event make_async_event(CtorParams&&... ctor_args) {
	static_assert(std::is_base_of_v<async_event_base, Event>);
	return async_event(std::make_unique<Event>(std::forward<CtorParams>(ctor_args)...));
}

inline async_event make_complete_event() { return make_async_event<complete_event>(); }

template <typename Result>
async_event make_complete_event(Result result) {
	return make_async_event<complete_event>(std::any(std::move(result)));
}

} // namespace celerity::detail
