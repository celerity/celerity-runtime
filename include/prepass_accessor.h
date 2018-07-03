#pragma once

#include <type_traits>

#include <SYCL/sycl.hpp>

namespace celerity {

// TODO: Looks like we will have to provide the full (mocked) accessor API
template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
class prepass_accessor {
  public:
	DataT& operator[](cl::sycl::id<Dims> index) const { throw std::runtime_error("Accessor used outside kernel / functor"); }

	template <cl::sycl::access::target T = Target>
	std::enable_if_t<T == cl::sycl::access::target::host_buffer, DataT*> get_pointer() const {
		throw std::runtime_error("Accessor used outside kernel / functor");
	}
};

} // namespace celerity
