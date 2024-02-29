#pragma once

#include "types.h"

#include <memory>
#include <numeric>

namespace celerity::detail {

class runtime_reduction {
  public:
	virtual ~runtime_reduction() = default;

	virtual void reduce(void* dest, const void* src, size_t src_count, bool include_dest) const = 0;
	virtual void fill_identity(void* dest, size_t count) const = 0;

  protected:
	runtime_reduction() = default;
	runtime_reduction(const runtime_reduction&) = default;
	runtime_reduction(runtime_reduction&&) = default;
	runtime_reduction& operator=(const runtime_reduction&) = default;
	runtime_reduction& operator=(runtime_reduction&&) = default;
};

template <typename Scalar, typename BinaryOp>
class runtime_reduction_impl final : public runtime_reduction {
  public:
	runtime_reduction_impl(const BinaryOp& op, const Scalar& identity) : m_op(op), m_identity(identity) {}

	void reduce(void* const dest, const void* const src, const size_t src_count, const bool include_dest) const override {
		const auto v_dest = static_cast<Scalar*>(dest);
		const auto v_src = static_cast<const Scalar*>(src);
		*v_dest = std::reduce(v_src, v_src + src_count, include_dest ? *v_dest : m_identity, m_op);
	}

	void fill_identity(void* const dest, const size_t count) const override { //
		std::uninitialized_fill_n(static_cast<Scalar*>(dest), count, m_identity);
	}

  private:
	BinaryOp m_op;
	Scalar m_identity;
};

template <typename Scalar, typename BinaryOp>
std::unique_ptr<runtime_reduction> make_runtime_reduction(const BinaryOp& op, const Scalar& identity) {
	return std::make_unique<runtime_reduction_impl<Scalar, BinaryOp>>(op, identity);
}

struct reduction_info {
	reduction_id rid = 0;
	buffer_id bid = 0;
	bool init_from_buffer = false;
};

} // namespace celerity::detail
