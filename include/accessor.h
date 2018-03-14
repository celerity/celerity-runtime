#pragma once

#include <SYCL/sycl.hpp>

namespace celerity {

// FIXME: Type, dimensions
template <cl::sycl::access::mode Mode>
using accessor = cl::sycl::accessor<float, 1, Mode, cl::sycl::access::target::global_buffer>;

// TODO: Looks like we will have to provide the full accessor API
// FIXME: Type, dimensions
template <cl::sycl::access::mode Mode>
class prepass_accessor {
  public:
	float& operator[](cl::sycl::id<1> index) const { return value; }

  private:
	mutable float value = 0.f;
};

} // namespace celerity
