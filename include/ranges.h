#pragma once

#include "sycl_wrappers.h"

namespace celerity {
namespace detail {

	template <int TargetDims, typename Target, int SubscriptDim = 0>
	class subscript_proxy;

	template <int TargetDims, typename Target, int SubscriptDim = 0>
	struct subscript_result {
		using type = subscript_proxy<TargetDims, Target, SubscriptDim + 1>;
	};

	template <int TargetDims, typename Target>
	struct subscript_result<TargetDims, Target, TargetDims - 1> {
		using type = decltype(std::declval<Target&>()[std::declval<const id<TargetDims>&>()]);
	};

	// Workaround for old ComputeCpp "stable" compiler: We cannot use decltype(auto), because it will not infer lvalue references correctly
	// TODO replace subscript_result and all its uses with decltype(auto) once we require the new ComputeCpp (experimental) compiler.
	template <int TargetDims, typename Target, int SubscriptDim = 0>
	using subscript_result_t = typename subscript_result<TargetDims, Target, SubscriptDim>::type;

	template <int TargetDims, typename Target, int SubscriptDim>
	inline subscript_result_t<TargetDims, Target, SubscriptDim> subscript(Target& tgt, id<TargetDims> id, const size_t index) {
		static_assert(SubscriptDim < TargetDims);
		id[SubscriptDim] = index;
		if constexpr(SubscriptDim == TargetDims - 1) {
			return tgt[std::as_const(id)];
		} else {
			return subscript_proxy<TargetDims, Target, SubscriptDim + 1>{tgt, id};
		}
	}

	template <int TargetDims, typename Target>
	inline subscript_result_t<TargetDims, Target> subscript(Target& tgt, const size_t index) {
		return subscript<TargetDims, Target, 0>(tgt, id<TargetDims>{}, index);
	}

	template <int TargetDims, typename Target, int SubscriptDim>
	class subscript_proxy {
	  public:
		subscript_proxy(Target& tgt, const id<TargetDims> id) : m_tgt(tgt), m_id(id) {}

		inline subscript_result_t<TargetDims, Target, SubscriptDim> operator[](const size_t index) const {
			return subscript<TargetDims, Target, SubscriptDim>(m_tgt, m_id, index);
		}

	  private:
		Target& m_tgt;
		id<TargetDims> m_id{};
	};

	inline size_t get_linear_index(const cl::sycl::range<1>& range, const cl::sycl::id<1>& index) { return index[0]; }

	inline size_t get_linear_index(const cl::sycl::range<2>& range, const cl::sycl::id<2>& index) { return index[0] * range[1] + index[1]; }

	inline size_t get_linear_index(const cl::sycl::range<3>& range, const cl::sycl::id<3>& index) {
		return index[0] * range[1] * range[2] + index[1] * range[2] + index[2];
	}

	template <int Dims, template <int> class Type>
	auto make_range_type() {
		if constexpr(Dims == 1) return Type<Dims>{0};
		if constexpr(Dims == 2) return Type<Dims>{0, 0};
		if constexpr(Dims == 3) return Type<Dims>{0, 0, 0};
	}

#define MAKE_ARRAY_CAST_FN(name, default_value, out_type)                                                                                                      \
	template <int DimsOut, template <int> class InType, int DimsIn>                                                                                            \
	out_type<DimsOut> name(const InType<DimsIn>& other) {                                                                                                      \
		static_assert(DimsOut > 0 && DimsOut < 4, "SYCL only supports 1, 2, or 3 dimensions for range / id");                                                  \
		out_type<DimsOut> result = make_range_type<DimsOut, out_type>();                                                                                       \
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
		range_type<Dims> result = make_range_type<Dims, range_type>();                                                                                         \
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

	struct {
		size_t value;
		operator cl::sycl::range<1>() const { return {value}; }
		operator cl::sycl::range<2>() const { return {value, value}; }
		operator cl::sycl::range<3>() const { return {value, value, value}; }
	} inline constexpr zero_range{0}, unit_range{1};

}; // namespace detail

template <int Dims>
struct chunk {
	static_assert(Dims > 0);

	static constexpr int dims = Dims;

	celerity::id<Dims> offset;
	celerity::range<Dims> range = detail::zero_range;
	celerity::range<Dims> global_size = detail::zero_range;

	chunk() = default;

	chunk(celerity::id<Dims> offset, celerity::range<Dims> range, celerity::range<Dims> global_size) : offset(offset), range(range), global_size(global_size) {}

	friend bool operator==(const chunk& lhs, const chunk& rhs) {
		return lhs.offset == rhs.offset && lhs.range == rhs.range && lhs.global_size == rhs.global_size;
	}
	friend bool operator!=(const chunk& lhs, const chunk& rhs) { return !operator==(lhs, rhs); }
};

template <int Dims>
struct subrange {
	static_assert(Dims > 0);

	static constexpr int dims = Dims;

	celerity::id<Dims> offset;
	celerity::range<Dims> range = detail::range_cast<Dims>(celerity::range<3>(0, 0, 0));

	subrange() = default;

	// Due to an apparent bug in the ComputeCpp 2.6.0 compiler, the implicit copy constructor of subrange instantiates a copy constructor of the "array" type
	// underlying ComputeCpp's implementation of sycl::range. This generated array copy constructor receives a *non-const* lvalue instead of a const lvalue,
	// which causes a const-mismatch -- but only when transitively called from the host_memory_layout runtime_test. Explicitly defining the constructor without
	// delegating to range(const range&) seems to fix this.
#if CELERITY_WORKAROUND(COMPUTECPP)
	subrange(const subrange& other) { *this = other; }

	subrange& operator=(const subrange& other) {
		offset = detail::id_cast<Dims>(other.offset);
		range = detail::range_cast<Dims>(other.range);
		return *this;
	}
#endif

	subrange(celerity::id<Dims> offset, celerity::range<Dims> range) : offset(offset), range(range) {}

	subrange(chunk<Dims> other) : offset(other.offset), range(other.range) {}

	friend bool operator==(const subrange& lhs, const subrange& rhs) { return lhs.offset == rhs.offset && lhs.range == rhs.range; }
	friend bool operator!=(const subrange& lhs, const subrange& rhs) { return !operator==(lhs, rhs); }
};

namespace detail {

	template <int Dims, int OtherDims>
	chunk<Dims> chunk_cast(const chunk<OtherDims>& other) {
		return chunk{detail::id_cast<Dims>(other.offset), detail::range_cast<Dims>(other.range), detail::range_cast<Dims>(other.global_size)};
	}

	template <int Dims, int OtherDims>
	subrange<Dims> subrange_cast(const subrange<OtherDims>& other) {
		return subrange{detail::id_cast<Dims>(other.offset), detail::range_cast<Dims>(other.range)};
	}

} // namespace detail

} // namespace celerity
