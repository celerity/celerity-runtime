#pragma once

#include "sycl_wrappers.h"
#include "workaround.h"

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

// We need a specialized storage type for coordinates to avoid generating a `size_t values[0]` array which clang interprets as dynamically-sized.
// By specializing on the Interface type, id<> and range<> become distinct types "all the way down", so both an id<0> and a range<0> can be included as struct
// members with [[no_unique_address]] within another struct and to actually overlap. This is required to ensure that 0-dimensional accessors are pointer-sized,
// and would otherwise be prohibited by strict-aliasing rules (because two identical pointers with the same type must point to the same object).
template <typename Interface, int Dims>
struct coordinate_storage {
	constexpr size_t operator[](int dimension) const {
		CELERITY_DETAIL_ASSERT_ON_HOST(dimension < Dims);
		return values[dimension];
	}

	constexpr size_t& operator[](int dimension) {
		CELERITY_DETAIL_ASSERT_ON_HOST(dimension < Dims);
		return values[dimension];
	}

	size_t values[Dims] = {};
};

template <typename Interface>
struct coordinate_storage<Interface, 0> {
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

	constexpr size_t get(int dimension) const { return m_values[dimension]; }
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

#define CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(op)                                                                                         \
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

	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(+)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(-)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(*)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(/)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(%)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<<)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>>)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(&)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(|)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(^)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(&&)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(||)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(<=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR(>=)

#undef CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_LHS_OPERATOR

#define CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(op)                                                                                      \
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

	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(+=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(-=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(*=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(/=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(%=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(<<=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(>>=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(&=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(|=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR(^=)

#undef CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_INPLACE_LHS_OPERATOR

#define CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(op)                                                                                         \
	friend constexpr Interface operator op(const size_t& lhs, const Interface& rhs) {                                                                          \
		Interface result;                                                                                                                                      \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = lhs op rhs.m_values[d];                                                                                                                \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(+)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(-)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(*)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(/)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(%)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<<)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>>)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(&)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(|)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(^)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(&&)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(||)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(<=)
	CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR(>=)

#undef CELERITY_DETAIL_DEFINE_COORDINATE_BINARY_COPY_RHS_OPERATOR

#define CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(op)                                                                                              \
	friend constexpr Interface operator op(const Interface& rhs) {                                                                                             \
		Interface result;                                                                                                                                      \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = op rhs[d];                                                                                                                             \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(+)
	CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR(-)

#undef CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_COPY_OPERATOR

#define CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(op)                                                                                            \
	friend constexpr Interface& operator op(Interface& rhs) {                                                                                                  \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			op rhs[d];                                                                                                                                         \
		}                                                                                                                                                      \
		return rhs;                                                                                                                                            \
	}

	CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(++)
	CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR(--)

#undef CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_PREFIX_OPERATOR

#define CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(op)                                                                                           \
	friend constexpr Interface operator op(Interface& lhs, int) {                                                                                              \
		Interface result = lhs;                                                                                                                                \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			lhs[d] op;                                                                                                                                         \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(++)
	CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR(--)

#undef CELERITY_DETAIL_DEFINE_COORDINATE_UNARY_POSTFIX_OPERATOR

  private:
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS coordinate_storage<Interface, Dims> m_values;
};

template <int DimsOut, typename InterfaceIn>
range<DimsOut> range_cast(const InterfaceIn& in);

template <int DimsOut, typename InterfaceIn>
id<DimsOut> id_cast(const InterfaceIn& in);

struct zeros_t {
} inline static constexpr zeros;
struct ones_t {
} inline static constexpr ones;

}; // namespace celerity::detail

namespace celerity {

template <int Dims>
class range : public detail::coordinate<range<Dims>, Dims> {
  private:
	using coordinate = detail::coordinate<range<Dims>, Dims>;

  public:
	constexpr range() noexcept = default;

	template <typename... Values, typename = std::enable_if_t<sizeof...(Values) + 1 == Dims>>
	constexpr range(const size_t dim_0, const Values... dim_n) : coordinate(dim_0, dim_n...) {}

	constexpr range(const detail::zeros_t /* tag */) {}

	constexpr range(const detail::ones_t /* tag */) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = 1;
		}
	}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	range(const sycl::range<Dims>& sycl_range) {
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
	operator sycl::range<Dims>() const {
		if constexpr(Dims == 1) {
			return {(*this)[0]};
		} else if constexpr(Dims == 2) {
			return {(*this)[0], (*this)[1]};
		} else {
			return {(*this)[0], (*this)[1], (*this)[2]};
		}
	}

  private:
	template <int DimsOut, typename InterfaceIn>
	friend range<DimsOut> detail::range_cast(const InterfaceIn& in);

	template <typename InterfaceIn, int DimsIn>
	constexpr range(const detail::make_from_t /* tag */, const detail::coordinate<InterfaceIn, DimsIn>& in)
	    : coordinate(detail::make_from, in, /* default_value= */ 1) {}
};

range()->range<0>;
range(size_t)->range<1>;
range(size_t, size_t)->range<2>;
range(size_t, size_t, size_t)->range<3>;

template <int Dims>
class id : public detail::coordinate<id<Dims>, Dims> {
  private:
	using coordinate = detail::coordinate<id<Dims>, Dims>;

  public:
	constexpr id() noexcept = default;

	template <typename... Values, typename = std::enable_if_t<sizeof...(Values) + 1 == Dims>>
	constexpr id(const size_t dim_0, const Values... dim_n) : coordinate(dim_0, dim_n...) {}

	constexpr id(const range<Dims>& range) : coordinate(detail::make_from, range) {}

	constexpr id(const detail::zeros_t /* tag */) {}

	constexpr id(const detail::ones_t /* tag */) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = 1;
		}
	}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	id(const sycl::id<Dims>& sycl_id) {
		for(int d = 0; d < Dims; ++d) {
			(*this)[d] = sycl_id[d];
		}
	}

	template <int D = Dims, typename = std::enable_if_t<D >= 1 && D <= 3>>
	operator sycl::id<Dims>() const {
		if constexpr(Dims == 1) {
			return {(*this)[0]};
		} else if constexpr(Dims == 2) {
			return {(*this)[0], (*this)[1]};
		} else {
			return {(*this)[0], (*this)[1], (*this)[2]};
		}
	}

  private:
	template <int DimsOut, typename InterfaceIn>
	friend id<DimsOut> detail::id_cast(const InterfaceIn& in);

	template <typename InterfaceIn, int DimsIn>
	constexpr id(const detail::make_from_t /* tag */, const detail::coordinate<InterfaceIn, DimsIn>& in)
	    : coordinate(detail::make_from, in, /* default_value= */ 0) {}
};

id()->id<0>;
id(size_t)->id<1>;
id(size_t, size_t)->id<2>;
id(size_t, size_t, size_t)->id<3>;

// Celerity nd_range does not deprecate kernel offsets since we can support them at no additional cost in the distributed model.
template <int Dims>
class nd_range {
  public:
	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	constexpr nd_range() noexcept {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	nd_range(const sycl::nd_range<Dims>& s_range)
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

	operator sycl::nd_range<Dims>() const { return sycl::nd_range<Dims>{m_global_range, m_local_range, m_offset}; }

	const range<Dims>& get_global_range() const { return m_global_range; }
	const range<Dims>& get_local_range() const { return m_local_range; }
	const range<Dims>& get_group_range() const { return m_global_range / m_local_range; }
	const id<Dims>& get_offset() const { return m_offset; }

	friend bool operator==(const nd_range& lhs, const nd_range& rhs) {
		return lhs.m_global_range == rhs.m_global_range && lhs.m_local_range == rhs.m_local_range && lhs.m_offset == rhs.m_offset;
	}

	friend bool operator!=(const nd_range& lhs, const nd_range& rhs) { return !(lhs == rhs); }

  private:
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_global_range;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_local_range;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> m_offset;
};

// Non-templated deduction guides allow construction of nd_range from range initializer lists like so: nd_range{{1, 2}, {3, 4}}
nd_range(range<1> global_range, range<1> local_range, id<1> offset)->nd_range<1>;
nd_range(range<1> global_range, range<1> local_range)->nd_range<1>;
nd_range(range<2> global_range, range<2> local_range, id<2> offset)->nd_range<2>;
nd_range(range<2> global_range, range<2> local_range)->nd_range<2>;
nd_range(range<3> global_range, range<3> local_range, id<3> offset)->nd_range<3>;
nd_range(range<3> global_range, range<3> local_range)->nd_range<3>;

} // namespace celerity

namespace celerity {
namespace detail {

	template <int TargetDims, typename Target, int SubscriptDim = 0>
	class subscript_proxy;

	template <int TargetDims, typename Target, int SubscriptDim>
	inline decltype(auto) subscript(Target& tgt, id<TargetDims> id, const size_t index) {
		static_assert(SubscriptDim < TargetDims);
		id[SubscriptDim] = index;
		if constexpr(SubscriptDim == TargetDims - 1) {
			return tgt[std::as_const(id)];
		} else {
			return subscript_proxy<TargetDims, Target, SubscriptDim + 1>{tgt, id};
		}
	}

	template <int TargetDims, typename Target>
	inline decltype(auto) subscript(Target& tgt, const size_t index) {
		return subscript<TargetDims, Target, 0>(tgt, id<TargetDims>{}, index);
	}

	template <int TargetDims, typename Target, int SubscriptDim>
	class subscript_proxy {
	  public:
		subscript_proxy(Target& tgt, const id<TargetDims> id) : m_tgt(tgt), m_id(id) {}

		inline decltype(auto) operator[](const size_t index) const { //
			return subscript<TargetDims, Target, SubscriptDim>(m_tgt, m_id, index);
		}

	  private:
		Target& m_tgt;
		id<TargetDims> m_id{};
	};

	inline size_t get_linear_index(const range<0>& /* range */, const id<0>& /* index */) { return 0; }

	inline size_t get_linear_index(const range<1>& range, const id<1>& index) { return index[0]; }

	inline size_t get_linear_index(const range<2>& range, const id<2>& index) { return index[0] * range[1] + index[1]; }

	inline size_t get_linear_index(const range<3>& range, const id<3>& index) { return index[0] * range[1] * range[2] + index[1] * range[2] + index[2]; }

#define CELERITY_DETAIL_MAKE_COMPONENT_WISE_FN(name, coord, op)                                                                                                \
	template <int Dims>                                                                                                                                        \
	coord<Dims> name(const coord<Dims>& a, const coord<Dims>& b) {                                                                                             \
		auto result = a;                                                                                                                                       \
		for(int d = 0; d < Dims; ++d) {                                                                                                                        \
			result[d] = op(result[d], b[d]);                                                                                                                   \
		}                                                                                                                                                      \
		return result;                                                                                                                                         \
	}

	CELERITY_DETAIL_MAKE_COMPONENT_WISE_FN(range_min, range, std::min)
	CELERITY_DETAIL_MAKE_COMPONENT_WISE_FN(range_max, range, std::max)
	CELERITY_DETAIL_MAKE_COMPONENT_WISE_FN(id_min, id, std::min)
	CELERITY_DETAIL_MAKE_COMPONENT_WISE_FN(id_max, id, std::max)

#undef CELERITY_DETAIL_MAKE_COMPONENT_WISE_FN

	template <typename Interface, int Dims>
	bool all_true(const coordinate<Interface, Dims>& bools) {
		for(int d = 0; d < Dims; ++d) {
			CELERITY_DETAIL_ASSERT_ON_HOST(bools[d] == 0 || bools[d] == 1);
			if(bools[d] == 0) return false;
		}
		return true;
	}

} // namespace detail

template <int Dims>
struct chunk {
	static constexpr int dimensions = Dims;

	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> offset;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS celerity::range<Dims> range = detail::zeros;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS celerity::range<Dims> global_size = detail::zeros;

	constexpr chunk() = default;

	constexpr chunk(const id<Dims>& offset, const celerity::range<Dims>& range, const celerity::range<Dims>& global_size)
	    : offset(offset), range(range), global_size(global_size) {}

	friend bool operator==(const chunk& lhs, const chunk& rhs) {
		return lhs.offset == rhs.offset && lhs.range == rhs.range && lhs.global_size == rhs.global_size;
	}
	friend bool operator!=(const chunk& lhs, const chunk& rhs) { return !operator==(lhs, rhs); }
};

template <int Dims>
struct subrange {
	static constexpr int dimensions = Dims;

	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> offset;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS celerity::range<Dims> range = detail::zeros;

	constexpr subrange() = default;
	constexpr subrange(const id<Dims>& offset, const celerity::range<Dims>& range) : offset(offset), range(range) {}
	constexpr subrange(const chunk<Dims>& other) : offset(other.offset), range(other.range) {}

	friend bool operator==(const subrange& lhs, const subrange& rhs) { return lhs.offset == rhs.offset && lhs.range == rhs.range; }
	friend bool operator!=(const subrange& lhs, const subrange& rhs) { return !operator==(lhs, rhs); }
};

} // namespace celerity

namespace celerity::detail {

/// Returns the smallest dimensionality that the range can be `range_cast` to.
template <int Dims>
int get_effective_dims(const range<Dims>& range) {
	for(int dims = Dims; dims > 0; --dims) {
		if(range[dims - 1] > 1) return dims;
	}
	return 0;
}

template <int DimsOut, typename InterfaceIn>
range<DimsOut> range_cast(const InterfaceIn& in) {
	CELERITY_DETAIL_ASSERT_ON_HOST(get_effective_dims(in) <= DimsOut);
	return range<DimsOut>(make_from, in);
}

/// Returns the smallest dimensionality that the id can be `id_cast` to.
template <int Dims>
int get_effective_dims(const id<Dims>& id) {
	for(int dims = Dims; dims > 0; --dims) {
		if(id[dims - 1] > 0) { return dims; }
	}
	return 0;
}

template <int DimsOut, typename InterfaceIn>
id<DimsOut> id_cast(const InterfaceIn& in) {
	CELERITY_DETAIL_ASSERT_ON_HOST(get_effective_dims(in) <= DimsOut);
	return id<DimsOut>(make_from, in);
}

/// Returns the smallest dimensionality that the chunk can be `chunk_cast` to.
template <int Dims>
int get_effective_dims(const chunk<Dims>& ck) {
	return std::max({get_effective_dims(ck.offset), get_effective_dims(ck.range), get_effective_dims(ck.global_size)});
}

/// Returns the smallest dimensionality that the subrange can be `subrange_cast` to.
template <int Dims>
int get_effective_dims(const subrange<Dims>& sr) {
	return std::max(get_effective_dims(sr.offset), get_effective_dims(sr.range));
}

template <int Dims, int OtherDims>
chunk<Dims> chunk_cast(const chunk<OtherDims>& other) {
	CELERITY_DETAIL_ASSERT_ON_HOST(get_effective_dims(other) <= Dims);
	return chunk{detail::id_cast<Dims>(other.offset), detail::range_cast<Dims>(other.range), detail::range_cast<Dims>(other.global_size)};
}

template <int Dims, int OtherDims>
subrange<Dims> subrange_cast(const subrange<OtherDims>& other) {
	CELERITY_DETAIL_ASSERT_ON_HOST(get_effective_dims(other) <= Dims);
	return subrange{detail::id_cast<Dims>(other.offset), detail::range_cast<Dims>(other.range)};
}

} // namespace celerity::detail
