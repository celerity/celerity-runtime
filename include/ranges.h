#pragma once

#include <CL/sycl.hpp>

namespace celerity {
namespace detail {

#define MAKE_ARRAY_CAST_FN(name, default_value, out_type)                                                                                                      \
	template <int DimsOut, template <int> class InType, int DimsIn>                                                                                            \
	out_type<DimsOut> name(const InType<DimsIn>& other) {                                                                                                      \
		static_assert(DimsOut > 0 && DimsOut < 4, "SYCL only supports 1, 2, or 3 dimensions for range / id");                                                  \
		out_type<DimsOut> result;                                                                                                                              \
		for(int o = 0; o < DimsOut; ++o) {                                                                                                                     \
			result[o] = o < DimsIn ? other[o] : default_value;                                                                                                 \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	MAKE_ARRAY_CAST_FN(range_cast, 1, cl::sycl::range)
	MAKE_ARRAY_CAST_FN(id_cast, 0, cl::sycl::id)

#undef MAKE_ARRAY_CAST_FN

#define MAKE_COMPONENT_WISE_BINARY_FN(name, range_type, op)                                                                                                    \
	template <int Dims>                                                                                                                                        \
	range_type<Dims> name(const range_type<Dims>& a, const range_type<Dims>& b) {                                                                              \
		range_type<Dims> result;                                                                                                                               \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = op(a[d], b[d]);                                                                                                                        \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	MAKE_COMPONENT_WISE_BINARY_FN(min_range, cl::sycl::range, std::min)
	MAKE_COMPONENT_WISE_BINARY_FN(max_range, cl::sycl::range, std::max)
	MAKE_COMPONENT_WISE_BINARY_FN(min_id, cl::sycl::id, std::min)
	MAKE_COMPONENT_WISE_BINARY_FN(max_id, cl::sycl::id, std::max)

#undef MAKE_COMPONENT_WISE_BINARY_FN

} // namespace detail

template <int Dims>
struct chunk {
	static_assert(Dims > 0);

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
	static_assert(Dims > 0);

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
