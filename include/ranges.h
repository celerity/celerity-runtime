#pragma once

#include "sycl_wrappers.h"

namespace celerity {

// clang-format off
template <int Dims = 1> class range;
template <int Dims = 1> class id;
template <int Dims = 1> class nd_range;
template <int Dims = 1> struct subrange;
template <int Dims = 1> struct chunk;
// clang-format on

} // namespace celerity

namespace celerity::detail {

struct make_from_t {
} inline static constexpr make_from;

// We need a specialized storage type to avoid generating a `size_t values[0]` array which clang interprets as dynamically-sized
template <int Dims>
struct coordinate_storage {
	constexpr size_t operator[](int dimension) const { return values[dimension]; }
	constexpr size_t& operator[](int dimension) { return values[dimension]; }
	size_t values[Dims] = {};
};

template <>
struct coordinate_storage<0> {
	constexpr size_t operator[](int /* dimension */) const { return 0; }
	// This is UB, but also unreachable. Unfortunately we can't call __builtin_unreachable from a constexpr function.
	constexpr size_t& operator[](int /* dimension */) { return *static_cast<size_t*>(static_cast<void*>(this)); }
};

// We implement range and id from scratch to allow zero-dimensional structures.
template <typename Interface, int Dims>
class coordinate {
  public:
	constexpr static int dimensions = Dims;

	coordinate() = default;

	template <typename InterfaceIn, int DimsIn>
	constexpr coordinate(const make_from_t /* tag */, const coordinate<InterfaceIn, DimsIn>& other, const size_t default_value) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = d < DimsIn ? other[d] : default_value;
		}
	}

	template <typename InterfaceIn>
	constexpr coordinate(const make_from_t /* tag */, const coordinate<InterfaceIn, Dims>& other) : coordinate(make_from, other, 0) {}

	template <typename... Values, typename = std::enable_if_t<sizeof...(Values) + 1 == Dims && (... && std::is_convertible_v<Values, size_t>)>>
	constexpr coordinate(const size_t dim_0, const Values... dim_n) : m_values{{dim_0, static_cast<size_t>(dim_n)...}} {}

	constexpr size_t get(int dimension) { return m_values[dimension]; }
	constexpr size_t& operator[](int dimension) { return m_values[dimension]; }
	constexpr size_t operator[](int dimension) const { return m_values[dimension]; }

	friend constexpr bool operator==(const Interface& lhs, const Interface& rhs) {
		bool equal = true;
		for(int d = 0; d < Dims; ++d) {
			equal &= lhs[d] == rhs[d];
		}
		return equal;
	}

	friend constexpr bool operator!=(const Interface& lhs, const Interface& rhs) { return !(lhs == rhs); }

#define CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(op)                                                                                                \
	friend constexpr Interface operator op(const Interface& lhs, const Interface& rhs) {                                                                       \
		Interface result;                                                                                                                                      \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = lhs.m_values[d] op rhs.m_values[d];                                                                                                    \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}                                                                                                                                                          \
	friend constexpr Interface operator op(const Interface& lhs, const size_t& rhs) {                                                                          \
		Interface result;                                                                                                                                      \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = lhs.m_values[d] op rhs;                                                                                                                \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(+)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(-)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(*)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(/)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(%)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<<)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>>)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(&)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(|)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(^)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(&&)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(||)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<=)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>=)

#define CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(op)                                                                                             \
	friend constexpr Interface& operator op(Interface& lhs, const Interface& rhs) {                                                                            \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			lhs.m_values[d] op rhs.m_values[d];                                                                                                                \
		}                                                                                                                                                      \
		return lhs;                                                                                                                                            \
	}                                                                                                                                                          \
	friend constexpr Interface& operator op(Interface& lhs, const size_t& rhs) {                                                                               \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			lhs.m_values[d] op rhs;                                                                                                                            \
		}                                                                                                                                                      \
		return lhs;                                                                                                                                            \
	}

	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(+=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(-=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(*=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(/=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(%=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(<<=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(>>=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(&=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(|=)
	CELERITY_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(^=)

#define CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(op)                                                                                                \
	friend constexpr Interface operator op(const size_t& lhs, const Interface& rhs) {                                                                          \
		Interface result;                                                                                                                                      \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = lhs op rhs.m_values[d];                                                                                                                \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(+)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(-)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(*)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(/)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(%)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<<)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>>)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(&)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(|)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(^)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(&&)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(||)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<=)
	CELERITY_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>=)

#define CELERITY_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(op)                                                                                                     \
	friend constexpr Interface operator op(const Interface& rhs) {                                                                                             \
		Interface result;                                                                                                                                      \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = op rhs;                                                                                                                                \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(+)
	CELERITY_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(-)

#define CELERITY_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(op)                                                                                                   \
	friend constexpr Interface& operator op(Interface& rhs) {                                                                                                  \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			op rhs[d];                                                                                                                                         \
		}                                                                                                                                                      \
		return rhs;                                                                                                                                            \
	}

	CELERITY_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(++)
	CELERITY_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(--)

#define CELERITY_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(op)                                                                                                  \
	friend constexpr Interface operator op(Interface& lhs, int) {                                                                                              \
		Interface result = lhs;                                                                                                                                \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			lhs[d] op;                                                                                                                                         \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(++)
	CELERITY_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(--)

  private:
	coordinate_storage<Dims> m_values;
};

template <typename InterfaceOut, typename InterfaceIn, int DimsIn>
InterfaceOut coordinate_cast(const coordinate<InterfaceIn, DimsIn>& in) {
	return InterfaceOut(make_from, in);
}

template <int DimsOut, typename InterfaceIn, int DimsIn>
range<DimsOut> range_cast(const coordinate<InterfaceIn, DimsIn>& in) {
	return coordinate_cast<range<DimsOut>>(in);
}

template <int DimsOut, typename InterfaceIn, int DimsIn>
id<DimsOut> id_cast(const coordinate<InterfaceIn, DimsIn>& in) {
	return coordinate_cast<id<DimsOut>>(in);
}

struct zero_range_t {
} inline static constexpr zero_range;
struct unit_range_t {
} inline static constexpr unit_range;


}; // namespace celerity::detail

namespace celerity {

template <int Dims>
class range : public detail::coordinate<range<Dims>, Dims> {
  private:
	using coordinate = detail::coordinate<range<Dims>, Dims>;

  public:
	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	constexpr range() noexcept {}

	template <typename... Values, typename = std::enable_if_t<sizeof...(Values) + 1 == Dims>>
	constexpr range(const size_t dim_0, const Values... dim_n) : coordinate(dim_0, dim_n...) {}

	constexpr range(const detail::zero_range_t /* tag */) {}

	constexpr range(const detail::unit_range_t /* tag */) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = 1;
		}
	}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	explicit range(const sycl::range<Dims>& sycl_range) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = sycl_range[d];
		}
	}

	constexpr size_t size() const {
		size_t s = 1;
		for(int d = 0; d < Dims; ++d) {
			s *= (*this)[d];
		}
		return s;
	}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	explicit operator sycl::range<Dims>() const {
		if constexpr(Dims == 1) {
			return {(*this)[0]};
		} else if constexpr(Dims == 2) {
			return {(*this)[0], (*this)[1]};
		} else {
			return {(*this)[0], (*this)[1], (*this)[2]};
		}
	}

  private:
	friend class detail::coordinate<range<Dims>, Dims>;

	template <typename InterfaceOut, typename InterfaceIn, int DimsIn>
	friend InterfaceOut detail::coordinate_cast(const detail::coordinate<InterfaceIn, DimsIn>& in);

	template <typename Default = void, int D = Dims, typename = std::enable_if_t<D != 0>>
	constexpr range() noexcept {}

	template <typename InterfaceIn, int DimsIn>
	constexpr range(const detail::make_from_t /* tag */, const detail::coordinate<InterfaceIn, DimsIn>& in)
	    : coordinate(detail::make_from, in, /* default_value= */ 1) {}
};

range() -> range<0>;
range(size_t) -> range<1>;
range(size_t, size_t) -> range<2>;
range(size_t, size_t, size_t) -> range<3>;

template <int Dims>
class id : public detail::coordinate<id<Dims>, Dims> {
  private:
	using coordinate = detail::coordinate<id<Dims>, Dims>;

  public:
	constexpr id() noexcept = default;

	template <typename... Values, typename = std::enable_if_t<sizeof...(Values) + 1 == Dims>>
	constexpr id(const size_t dim_0, const Values... dim_n) : coordinate(dim_0, dim_n...) {}

	constexpr id(const range<Dims>& range) : coordinate(detail::make_from, range) {}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	explicit id(const sycl::id<Dims>& sycl_id) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = sycl_id[d];
		}
	}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	explicit operator sycl::id<Dims>() const {
		if constexpr(Dims == 1) {
			return {(*this)[0]};
		} else if constexpr(Dims == 2) {
			return {(*this)[0], (*this)[1]};
		} else {
			return {(*this)[0], (*this)[1], (*this)[2]};
		}
	}

  private:
	template <typename InterfaceOut, typename InterfaceIn, int DimsIn>
	friend InterfaceOut detail::coordinate_cast(const detail::coordinate<InterfaceIn, DimsIn>& in);

	template <typename InterfaceIn, int DimsIn>
	constexpr id(const detail::make_from_t /* tag */, const detail::coordinate<InterfaceIn, DimsIn>& in)
	    : coordinate(detail::make_from, in, /* default_value= */ 0) {}
};

id() -> id<0>;
id(size_t) -> id<1>;
id(size_t, size_t) -> id<2>;
id(size_t, size_t, size_t) -> id<3>;

// We re-implement nd_range to un-deprecate kernel offsets
template <int Dims>
class nd_range {
  public:
	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	constexpr nd_range() noexcept {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	nd_range(const cl::sycl::nd_range<Dims>& s_range)
	    : m_global_range(s_range.get_global_range()), m_local_range(s_range.get_local_range()), m_offset(s_range.get_offset()) {}
#pragma GCC diagnostic pop

	nd_range(const range<Dims>& global_range, const range<Dims>& local_range, const id<Dims>& offset = {})
	    : m_global_range(global_range), m_local_range(local_range), m_offset(offset) {
#ifndef __SYCL_DEVICE_ONLY__
		for(int d = 0; d < Dims; ++d) {
			if(local_range[d] == 0 || global_range[d] % local_range[d] != 0) { throw std::invalid_argument("global_range is not divisible by local_range"); }
		}
#endif
	}

	operator cl::sycl::nd_range<Dims>() const { return cl::sycl::nd_range<Dims>{m_global_range, m_local_range, m_offset}; }

	const range<Dims>& get_global_range() const { return m_global_range; }
	const range<Dims>& get_local_range() const { return m_local_range; }
	const range<Dims>& get_group_range() const { return m_global_range / m_local_range; }
	const id<Dims>& get_offset() const { return m_offset; }

	friend bool operator==(const nd_range& lhs, const nd_range& rhs) {
		return lhs.m_global_range == rhs.m_global_range && lhs.m_local_range == rhs.m_local_range && lhs.m_offset == rhs.m_offset;
	}

	friend bool operator!=(const nd_range& lhs, const nd_range& rhs) { return !(lhs == rhs); }

  private:
	range<Dims> m_global_range;
	range<Dims> m_local_range;
	id<Dims> m_offset;
};

// Non-templated deduction guides allow construction of nd_range from range initializer lists like so: nd_range{{1, 2}, {3, 4}}
// ... except, currently, for ComputeCpp which uses an outdated Clang (TODO)
nd_range(range<1> global_range, range<1> local_range, id<1> offset) -> nd_range<1>;
nd_range(range<1> global_range, range<1> local_range) -> nd_range<1>;
nd_range(range<2> global_range, range<2> local_range, id<2> offset) -> nd_range<2>;
nd_range(range<2> global_range, range<2> local_range) -> nd_range<2>;
nd_range(range<3> global_range, range<3> local_range, id<3> offset) -> nd_range<3>;
nd_range(range<3> global_range, range<3> local_range) -> nd_range<3>;

} // namespace celerity

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

	inline size_t get_linear_index(const celerity::range<1>& range, const celerity::id<1>& index) { return index[0]; }

	inline size_t get_linear_index(const celerity::range<2>& range, const celerity::id<2>& index) { return index[0] * range[1] + index[1]; }

	inline size_t get_linear_index(const celerity::range<3>& range, const celerity::id<3>& index) {
		return index[0] * range[1] * range[2] + index[1] * range[2] + index[2];
	}

#define MAKE_COMPONENT_WISE_BINARY_FN(name, range_type, op)                                                                                                    \
	template <int Dims>                                                                                                                                        \
	range_type<Dims> name(const range_type<Dims>& a, const range_type<Dims>& b) {                                                                              \
		auto result = a;                                                                                                                                       \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = op(result[d], b[d]);                                                                                                                   \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	MAKE_COMPONENT_WISE_BINARY_FN(min_range, range, std::min)
	MAKE_COMPONENT_WISE_BINARY_FN(max_range, range, std::max)
	MAKE_COMPONENT_WISE_BINARY_FN(min_id, id, std::min)
	MAKE_COMPONENT_WISE_BINARY_FN(max_id, id, std::max)

#undef MAKE_COMPONENT_WISE_BINARY_FN

} // namespace detail

template <int Dims>
struct chunk {
	static constexpr int dimensions = Dims;

	celerity::id<Dims> offset;
	celerity::range<Dims> range = detail::zero_range;
	celerity::range<Dims> global_size = detail::zero_range;

	chunk() = default;

	chunk(const celerity::id<Dims>& offset, const celerity::range<Dims>& range, const celerity::range<Dims>& global_size)
	    : offset(offset), range(range), global_size(global_size) {}

	friend bool operator==(const chunk& lhs, const chunk& rhs) {
		return lhs.offset == rhs.offset && lhs.range == rhs.range && lhs.global_size == rhs.global_size;
	}
	friend bool operator!=(const chunk& lhs, const chunk& rhs) { return !operator==(lhs, rhs); }
};

template <int Dims>
struct subrange {
	static constexpr int dimensions = Dims;

	celerity::id<Dims> offset;
	celerity::range<Dims> range = detail::zero_range;

	subrange() = default;

	subrange(const celerity::id<Dims>& offset, const celerity::range<Dims>& range) : offset(offset), range(range) {}

	subrange(const chunk<Dims>& other) : offset(other.offset), range(other.range) {}

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
