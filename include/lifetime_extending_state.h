#pragma once

#include <memory>

namespace celerity::detail {

/**
 * Helper type for creating objects with reference semantics, whose lifetime can be extended by tasks.
 */
class lifetime_extending_state {
  public:
	lifetime_extending_state() = default;
	lifetime_extending_state(const lifetime_extending_state&) = delete;
	lifetime_extending_state(lifetime_extending_state&&) = delete;
	lifetime_extending_state& operator=(const lifetime_extending_state&) = delete;
	lifetime_extending_state& operator=(lifetime_extending_state&&) = delete;

	virtual ~lifetime_extending_state() = default;
};

/**
 * Wrapper type that allows to retrieve the contained lifetime extending state (creation and storage of which is to be implemented by sub-classes).
 */
class lifetime_extending_state_wrapper {
  public:
	lifetime_extending_state_wrapper() = default;
	lifetime_extending_state_wrapper(const lifetime_extending_state_wrapper&) = default;
	lifetime_extending_state_wrapper(lifetime_extending_state_wrapper&&) noexcept = default;
	lifetime_extending_state_wrapper& operator=(const lifetime_extending_state_wrapper&) = default;
	lifetime_extending_state_wrapper& operator=(lifetime_extending_state_wrapper&&) noexcept = default;

	virtual ~lifetime_extending_state_wrapper() = default;

  protected:
	friend std::shared_ptr<lifetime_extending_state> get_lifetime_extending_state(const lifetime_extending_state_wrapper& wrapper);
	virtual std::shared_ptr<lifetime_extending_state> get_lifetime_extending_state() const = 0;
};

inline std::shared_ptr<lifetime_extending_state> get_lifetime_extending_state(const lifetime_extending_state_wrapper& wrapper) {
	return wrapper.get_lifetime_extending_state();
}

} // namespace celerity::detail