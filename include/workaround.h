#pragma once

#include <cassert>

#include <CL/sycl.hpp>
#include <sycl/sycl.hpp>

#if defined(CELERITY_DPCPP)
#define CELERITY_WORKAROUND_DPCPP 1
#else
#define CELERITY_WORKAROUND_DPCPP 0
#endif

#if defined(__HIPSYCL__)
#define CELERITY_WORKAROUND_HIPSYCL 1
#define CELERITY_WORKAROUND_VERSION_MAJOR HIPSYCL_VERSION_MAJOR
#define CELERITY_WORKAROUND_VERSION_MINOR HIPSYCL_VERSION_MINOR
#define CELERITY_WORKAROUND_VERSION_PATCH HIPSYCL_VERSION_PATCH
#else
#define CELERITY_WORKAROUND_HIPSYCL 0
#endif

#define CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL_1(major) (CELERITY_WORKAROUND_VERSION_MAJOR <= major)
#define CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL_2(major, minor)                                                                                              \
	(CELERITY_WORKAROUND_VERSION_MAJOR < major) || (CELERITY_WORKAROUND_VERSION_MAJOR == major && CELERITY_WORKAROUND_VERSION_MINOR <= minor)
#define CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL_3(major, minor, patch)                                                                                       \
	(CELERITY_WORKAROUND_VERSION_MAJOR < major) || (CELERITY_WORKAROUND_VERSION_MAJOR == major && CELERITY_WORKAROUND_VERSION_MINOR < minor)                   \
	    || (CELERITY_WORKAROUND_VERSION_MAJOR == major && CELERITY_WORKAROUND_VERSION_MINOR == minor && CELERITY_WORKAROUND_VERSION_PATCH <= patch)

#define CELERITY_WORKAROUND_GET_OVERLOAD(_1, _2, _3, NAME, ...) NAME
#define CELERITY_WORKAROUND_MSVC_VA_ARGS_EXPANSION(x) x // Workaround for MSVC PP expansion behavior of __VA_ARGS__
#define CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL(...)                                                                                                         \
	CELERITY_WORKAROUND_MSVC_VA_ARGS_EXPANSION(CELERITY_WORKAROUND_GET_OVERLOAD(__VA_ARGS__, CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL_3,                      \
	    CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL_2, CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL_1)(__VA_ARGS__))

#define CELERITY_WORKAROUND(impl) (CELERITY_WORKAROUND_##impl == 1)
#define CELERITY_WORKAROUND_LESS_OR_EQUAL(impl, ...) (CELERITY_WORKAROUND(impl) && CELERITY_WORKAROUND_VERSION_LESS_OR_EQUAL(__VA_ARGS__))

#if __has_cpp_attribute(no_unique_address) // C++20, but implemented as an extension for earlier standards in Clang
#define CELERITY_DETAIL_HAS_NO_UNIQUE_ADDRESS true
#define CELERITY_DETAIL_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define CELERITY_DETAIL_HAS_NO_UNIQUE_ADDRESS false
#define CELERITY_DETAIL_NO_UNIQUE_ADDRESS
#endif

#if CELERITY_DETAIL_ENABLE_DEBUG && !defined(__SYCL_DEVICE_ONLY__)
#define CELERITY_DETAIL_ASSERT_ON_HOST(...) assert(__VA_ARGS__)
#else
#define CELERITY_DETAIL_ASSERT_ON_HOST(...)
#endif

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
#define CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(...) __VA_ARGS__
#else
#define CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(...)
#endif


#if CELERITY_WORKAROUND(HIPSYCL)

namespace hipsycl::sycl::detail {

template <class T>
struct element_type {
	using type = T;
};

template <class T, int Dim>
struct element_type<vec<T, Dim>> {
	using type = T;
};

template <typename T>
using element_type_t = typename element_type<T>::type;

template <typename BinaryOperation, typename AccumulatorT, typename Enable = void>
struct known_identity_trait {
	static constexpr bool has_known_identity = false;
};

template <typename T, typename Enable = void>
struct minmax_identity {
	inline static constexpr T max_id = static_cast<T>(std::numeric_limits<T>::lowest());
	inline static constexpr T min_id = static_cast<T>(std::numeric_limits<T>::max());
};

template <typename T>
struct minmax_identity<T, std::enable_if_t<std::numeric_limits<T>::has_infinity>> {
	inline static constexpr T max_id = static_cast<T>(-std::numeric_limits<T>::infinity());
	inline static constexpr T min_id = static_cast<T>(std::numeric_limits<T>::infinity());
};

#define HIPSYCL_DEFINE_IDENTITY(op, cond, identity)                                                                                                            \
	template <typename T, typename U>                                                                                                                          \
	struct known_identity_trait<op<T>, U, std::enable_if_t<cond>> {                                                                                            \
		inline static constexpr bool has_known_identity = true;                                                                                                \
		inline static constexpr std::remove_cv_t<T> known_identity = (identity);                                                                               \
	};                                                                                                                                                         \
	template <typename T>                                                                                                                                      \
	struct known_identity_trait<op<void>, T, std::enable_if_t<cond>> {                                                                                         \
		inline static constexpr bool has_known_identity = true;                                                                                                \
		inline static constexpr std::remove_cv_t<T> known_identity = (identity);                                                                               \
	}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbool-operation" // allow ~bool, bool & bool
#endif

HIPSYCL_DEFINE_IDENTITY(plus, std::is_arithmetic_v<element_type_t<T>>, T{});
HIPSYCL_DEFINE_IDENTITY(multiplies, std::is_arithmetic_v<element_type_t<T>>, T{static_cast<element_type_t<T>>(1)});
HIPSYCL_DEFINE_IDENTITY(bit_or, std::is_integral_v<element_type_t<T>>, T{});
HIPSYCL_DEFINE_IDENTITY(bit_and, std::is_integral_v<element_type_t<T>>, T{static_cast<element_type_t<T>>(~element_type_t<T>{})});
HIPSYCL_DEFINE_IDENTITY(bit_xor, std::is_integral_v<element_type_t<T>>, T{});
HIPSYCL_DEFINE_IDENTITY(logical_or, (std::is_same_v<element_type_t<std::remove_cv_t<T>>, bool>), T{false});
HIPSYCL_DEFINE_IDENTITY(logical_and, (std::is_same_v<element_type_t<std::remove_cv_t<T>>, bool>), T{true});
HIPSYCL_DEFINE_IDENTITY(minimum, std::is_arithmetic_v<element_type_t<T>>, T{minmax_identity<element_type_t<std::remove_cv_t<T>>>::min_id});
HIPSYCL_DEFINE_IDENTITY(maximum, std::is_arithmetic_v<element_type_t<T>>, T{minmax_identity<element_type_t<std::remove_cv_t<T>>>::max_id});

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#undef HIPSYCL_DEFINE_IDENTITY

} // namespace hipsycl::sycl::detail

namespace hipsycl::sycl {

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity {
	static constexpr AccumulatorT value = detail::known_identity_trait<BinaryOperation, AccumulatorT>::known_identity;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v = known_identity<BinaryOperation, AccumulatorT>::value;

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity {
	static constexpr bool value = detail::known_identity_trait<BinaryOperation, AccumulatorT>::has_known_identity;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v = has_known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace hipsycl::sycl

namespace hipsycl::sycl::property::reduction {
class initialize_to_identity : public hipsycl::sycl::detail::property {};
} // namespace hipsycl::sycl::property::reduction

#endif
