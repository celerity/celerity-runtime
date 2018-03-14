#pragma once

#include <SYCL/sycl.hpp>

namespace celerity {

// FIXME: Naming; could be clearer
template <int Dims>
struct subrange {
	// TODO: Should "start" be a cl::sycl::id instead? (What's the difference?)
	// We'll leave it a range for now so we don't have to provide conversion
	// overloads below
	cl::sycl::range<Dims> start;
	cl::sycl::range<Dims> range;
	cl::sycl::range<Dims> global_size;
};

} // namespace celerity
