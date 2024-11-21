#pragma once

#include "version.h"

#include <cassert>

#include <sycl/sycl.hpp>


#if CELERITY_SYCL_IS_DPCPP
#define CELERITY_WORKAROUND_DPCPP 1
#else
#define CELERITY_WORKAROUND_DPCPP 0
#endif

#if CELERITY_SYCL_IS_ACPP
#define CELERITY_WORKAROUND_ACPP 1
#define CELERITY_WORKAROUND_VERSION_MAJOR HIPSYCL_VERSION_MAJOR
#define CELERITY_WORKAROUND_VERSION_MINOR HIPSYCL_VERSION_MINOR
#define CELERITY_WORKAROUND_VERSION_PATCH HIPSYCL_VERSION_PATCH
#else
#define CELERITY_WORKAROUND_ACPP 0
#endif

#if CELERITY_SYCL_IS_SIMSYCL
#define CELERITY_WORKAROUND_SIMSYCL 1
#else
#define CELERITY_WORKAROUND_SIMSYCL 0
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

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
#define CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(...) __VA_ARGS__
#else
#define CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(...)
#endif

// SYCL implementations (compiler frontends) provide different means of distinguishing host and device code. Multi-pass compilers define __SYCL_DEVICE_ONLY__ in
// the device pass, as per the standard. Single-pass compilers however cannot expose this information at preprocessing time or even as a constant expression,
// since they will only inject that information at code generation (and optimization) time. Below we define a set of implementation-independent macros for both
// cases. This is primarily used to provide accessor copy constructors conditionally for host-side accessor hydration.

#if CELERITY_SYCL_IS_ACPP && ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
#define CELERITY_DETAIL_COMPILE_TIME_TARGET_DEVICE_ONLY 0
#define CELERITY_DETAIL_COMPILE_TIME_TARGET_HOST_ONLY 0
#define CELERITY_DETAIL_IF_RUNTIME_TARGET_DEVICE(...) __acpp_if_target_device(__VA_ARGS__)
#define CELERITY_DETAIL_IF_RUNTIME_TARGET_HOST(...) __acpp_if_target_host(__VA_ARGS__)
#elif defined(__SYCL_DEVICE_ONLY__)
#define CELERITY_DETAIL_COMPILE_TIME_TARGET_DEVICE_ONLY 1
#define CELERITY_DETAIL_COMPILE_TIME_TARGET_HOST_ONLY 0
#define CELERITY_DETAIL_IF_RUNTIME_TARGET_DEVICE(...) __VA_ARGS__
#define CELERITY_DETAIL_IF_RUNTIME_TARGET_HOST(...)
#else
#define CELERITY_DETAIL_COMPILE_TIME_TARGET_DEVICE_ONLY 0
#define CELERITY_DETAIL_COMPILE_TIME_TARGET_HOST_ONLY 1
#define CELERITY_DETAIL_IF_RUNTIME_TARGET_DEVICE(...)
#define CELERITY_DETAIL_IF_RUNTIME_TARGET_HOST(...) __VA_ARGS__
#endif

#define CELERITY_DETAIL_ASSERT_ON_HOST(...) CELERITY_DETAIL_IF_RUNTIME_TARGET_HOST(assert(__VA_ARGS__);)

#if CELERITY_DETAIL_COMPILE_TIME_TARGET_HOST_ONLY
#define CELERITY_DETAIL_CONSTEXPR_ASSERT_ON_HOST(...) assert(__VA_ARGS__);
#else
#define CELERITY_DETAIL_CONSTEXPR_ASSERT_ON_HOST(...)
#endif
