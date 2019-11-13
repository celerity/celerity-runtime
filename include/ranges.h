#pragma once

#include <CL/sycl.hpp>

#include "workaround.h"

#if !WORKAROUND_HIPSYCL
#define __host__
#define __device__
#endif

namespace celerity {
namespace detail {

// The hipSYCL source transformation seems to have difficulties annotating this
// definition, which is why we provide the __device__ and __host__ attributes
// manually.
#define MAKE_ARRAY_CAST_FN(name, default_value, out_type)                                                                                                      \
	template <int DimsOut, template <int> class InType, int DimsIn>                                                                                            \
	__device__ __host__ out_type<DimsOut> name(const InType<DimsIn>& other) {                                                                                  \
		out_type<DimsOut> result;                                                                                                                              \
		for(int o = 0; o < DimsOut; ++o) {                                                                                                                     \
			result[o] = o < DimsIn ? other[o] : default_value;                                                                                                 \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	MAKE_ARRAY_CAST_FN(range_cast, 1, cl::sycl::range);
	MAKE_ARRAY_CAST_FN(id_cast, 0, cl::sycl::id);

} // namespace detail

template <int Dims>
struct chunk {
	static constexpr int dims = Dims;

	cl::sycl::id<Dims> offset;
	cl::sycl::range<Dims> range;
	cl::sycl::range<Dims> global_size;

	chunk() = default;

	chunk(cl::sycl::id<Dims> offset, cl::sycl::range<Dims> range, cl::sycl::range<Dims> global_size) : offset(offset), range(range), global_size(global_size) {}

	template <int OtherDims>
	chunk(chunk<OtherDims> other)
	    : offset(detail::id_cast<Dims>(other.offset)), range(detail::range_cast<Dims>(other.range)), global_size(detail::range_cast<Dims>(other.global_size)) {}
};

template <int Dims>
struct subrange {
	static constexpr int dims = Dims;

	cl::sycl::id<Dims> offset;
	cl::sycl::range<Dims> range;

	subrange() = default;

	subrange(cl::sycl::id<Dims> offset, cl::sycl::range<Dims> range) : offset(offset), range(range) {}

	template <int OtherDims>
	subrange(subrange<OtherDims> other) : offset(detail::id_cast<Dims>(other.offset)), range(detail::range_cast<Dims>(other.range)) {}

	subrange(chunk<Dims> other) : offset(other.offset), range(other.range) {}

	bool operator==(const subrange& rhs) { return offset == rhs.offset && range == rhs.range; }
};

} // namespace celerity
