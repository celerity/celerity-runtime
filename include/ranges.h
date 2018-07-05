#pragma once

#include <SYCL/sycl.hpp>

namespace celerity {

template <int Dims>
struct chunk {
	static constexpr int dims = Dims;

	cl::sycl::id<Dims> offset;
	cl::sycl::range<Dims> range;
	cl::sycl::range<Dims> global_size;

	chunk() = default;

	chunk(cl::sycl::id<Dims> offset, cl::sycl::range<Dims> range, cl::sycl::range<Dims> global_size) : offset(offset), range(range), global_size(global_size) {}

	template <int OtherDims>
	chunk(chunk<OtherDims> other) : offset(other.offset), range(other.range), global_size(other.global_size) {}
};

template <int Dims>
struct subrange {
	static constexpr int dims = Dims;

	cl::sycl::id<Dims> offset;
	cl::sycl::range<Dims> range;

	subrange() = default;

	subrange(cl::sycl::id<Dims> offset, cl::sycl::range<Dims> range) : offset(offset), range(range) {}

	template <int OtherDims>
	subrange(subrange<OtherDims> other) : offset(other.offset), range(other.range) {}

	subrange(chunk<Dims> other) : offset(other.offset), range(other.range) {}
};

} // namespace celerity
