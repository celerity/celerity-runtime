#pragma once

#include <cstddef>
#include <stdexcept>


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

namespace celerity::experimental::hints {

/**
 * Suggests that the task should be split into 1D chunks.
 * This is currently the default behavior.
 */
class split_1d final : public detail::hint_base {
  private:
	void validate(const hint_base& other) const override;
};

/**
 * Suggests that the task should be split into 2D chunks.
 */
class split_2d final : public detail::hint_base {
  private:
	void validate(const hint_base& other) const override;
};

inline void split_1d::validate(const hint_base& other) const {
	if(dynamic_cast<const split_2d*>(&other) != nullptr) { throw std::runtime_error("Cannot combine split_1d and split_2d hints"); }
}

inline void split_2d::validate(const hint_base& other) const {
	if(dynamic_cast<const split_1d*>(&other) != nullptr) { throw std::runtime_error("Cannot combine split_1d and split_2d hints"); }
}

/**
 * Suggests that more than chunks of the execution space assigned to individual devices should be subdivided further.
 * Oversubscription can improve utilization by increasing the potential for computation-communication overlap.
 */
class oversubscribe final : public detail::hint_base {
  public:
	explicit oversubscribe(const size_t factor) : m_factor(factor) {
		if(factor == 0) throw std::runtime_error("Oversubscription cannot be 0");
	}

	[[nodiscard]] size_t get_factor() const { return m_factor; }

  private:
	size_t m_factor;
};

} // namespace celerity::experimental::hints
