#pragma once

#include <SYCL/sycl.hpp>

#include "subrange.h"

namespace celerity {

namespace detail {
	// FIXME: The input dimensions must match the kernel size, not the buffer
	template <int Dims>
	using range_mapper_fn = std::function<subrange<Dims>(subrange<Dims> range)>;

	class range_mapper_base {
	  public:
		range_mapper_base(cl::sycl::access::mode am) : access_mode(am) {}
		cl::sycl::access::mode get_access_mode() const { return access_mode; }

		virtual size_t get_dimensions() const = 0;
		virtual subrange<1> operator()(subrange<1> range) { return subrange<1>(); }
		virtual subrange<2> operator()(subrange<2> range) { return subrange<2>(); }
		virtual subrange<3> operator()(subrange<3> range) { return subrange<3>(); }
		virtual ~range_mapper_base() {}

	  private:
		cl::sycl::access::mode access_mode;
	};

	template <int Dims>
	class range_mapper : public range_mapper_base {
	  public:
		range_mapper(range_mapper_fn<Dims> fn, cl::sycl::access::mode am) : range_mapper_base(am), rmfn(fn) {}
		size_t get_dimensions() const override { return Dims; }
		subrange<Dims> operator()(subrange<Dims> range) override { return rmfn(range); }

	  private:
		range_mapper_fn<Dims> rmfn;
	};
} // namespace detail


// --------------------------- Convenience range mappers ---------------------------


namespace access {
	template <int Dims>
	struct one_to_one {
		subrange<Dims> operator()(subrange<Dims> range) const { return range; }
	};
} // namespace access

} // namespace celerity
