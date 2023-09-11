#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

namespace celerity {
class handler;
}

// Definition is in handler.h to avoid circular dependency
namespace celerity::experimental {
template <typename Hint>
void hint(handler& cgh, Hint&& hint);
}

namespace celerity::detail {

class hint_base {
  public:
	hint_base() = default;
	hint_base(const hint_base&) = default;
	hint_base& operator=(const hint_base&) = default;
	hint_base(hint_base&&) = default;
	hint_base& operator=(hint_base&&) = default;
	virtual ~hint_base() = default;

  private:
	friend class celerity::handler;
	virtual void validate(const hint_base& other) const {}
};

} // namespace celerity::detail

namespace celerity::experimental::hints {}; // namespace celerity::experimental::hints
