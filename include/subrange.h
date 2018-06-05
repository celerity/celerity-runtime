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

	subrange() = default;

	subrange(cl::sycl::id<Dims> start, cl::sycl::range<Dims> range, cl::sycl::range<Dims> global_size) : start(start), range(range), global_size(global_size) {}

	template <int OtherDims>
	subrange(subrange<OtherDims> other) : start(other.start), range(other.range), global_size(other.global_size) {}
};

} // namespace celerity
