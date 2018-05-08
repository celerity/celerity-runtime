#pragma once

#include <SYCL/sycl.hpp>

namespace celerity {

// FIXME: Naming; could be clearer
template <int Dims>
struct subrange {
	static constexpr int dims = Dims;

	cl::sycl::id<Dims> start;
	cl::sycl::range<Dims> range;
	cl::sycl::range<Dims> global_size;
};

} // namespace celerity
