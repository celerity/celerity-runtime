#pragma once

#include <memory>

namespace celerity::detail {

class lifetime_extending_state {
  public:
	virtual ~lifetime_extending_state() = default;
};

class lifetime_extending_state_wrapper {
  public:
	virtual ~lifetime_extending_state_wrapper() = default;

  protected:
	friend std::shared_ptr<lifetime_extending_state> get_lifetime_extending_state(const lifetime_extending_state_wrapper& wrapper);
	virtual std::shared_ptr<lifetime_extending_state> get_lifetime_extending_state() const = 0;
};

inline std::shared_ptr<lifetime_extending_state> get_lifetime_extending_state(const lifetime_extending_state_wrapper& wrapper) {
	return wrapper.get_lifetime_extending_state();
}

} // namespace celerity::detail