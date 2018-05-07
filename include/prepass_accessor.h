#pragma once

#include <SYCL/sycl.hpp>

namespace celerity {

// TODO: Looks like we will have to provide the full (mocked) accessor API
template <typename DataT, int Dims, cl::sycl::access::mode Mode>
class prepass_accessor {
  public:
	DataT& operator[](cl::sycl::id<Dims> index) const { throw std::runtime_error("Accessor used outside kernel / functor"); }
};

} // namespace celerity
