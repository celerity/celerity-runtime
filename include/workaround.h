#pragma once

#include <CL/sycl.hpp>
#include <sycl/sycl.hpp>

#if defined(CELERITY_DPCPP)
#define CELERITY_WORKAROUND_DPCPP 1
#else
#define CELERITY_WORKAROUND_DPCPP 0
#endif

#if defined(__COMPUTECPP__)
#define CELERITY_WORKAROUND_COMPUTECPP 1
#define CELERITY_WORKAROUND_VERSION_MAJOR COMPUTECPP_VERSION_MAJOR
#define CELERITY_WORKAROUND_VERSION_MINOR COMPUTECPP_VERSION_MINOR
#define CELERITY_WORKAROUND_VERSION_PATCH COMPUTECPP_VERSION_PATCH
#else
#define CELERITY_WORKAROUND_COMPUTECPP 0
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


namespace celerity::detail {

// "normal" accessors return T& references to data, whereas atomic accessors return sycl::atomic<T> pointer wrappers. decltype(auto) is not enough to give
// ranged_sycl_access() a correct generic return type, so we specify it manually through the following trait.

template <typename T, sycl::access_mode Mode, sycl::access::target Target>
struct access_value_trait {
	using type = T;
	using reference = T&;
};

template <typename T, sycl::access::target Target>
struct access_value_trait<T, sycl::access_mode::read, Target> {
	using type = const T;
	using reference = const T&;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // mode::atomic and target::local are deprecated

template <typename T, sycl::access::target Target>
struct access_value_trait<T, sycl::access_mode::atomic, Target> {
	using type = T;
	using reference = sycl::atomic<T, sycl::access::address_space::global_space>;
};

template <typename T>
struct access_value_trait<T, sycl::access_mode::atomic, sycl::access::target::local> {
	using type = T;
	using reference = sycl::atomic<T, sycl::access::address_space::local_space>;
};

#pragma GCC diagnostic pop

template <typename T, sycl::access_mode Mode, sycl::access::target Target>
using access_value_type = typename access_value_trait<T, Mode, Target>::type;

template <typename T, sycl::access_mode Mode, sycl::access::target Target>
using access_value_reference = typename access_value_trait<T, Mode, Target>::reference;

} // namespace celerity::detail