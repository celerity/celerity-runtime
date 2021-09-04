#pragma once

#include <CL/sycl.hpp>

// TODO: Don't pollute preprocessor namespace with generic "WORKAROUND" names, prefix with "CELERITY".

#if defined(CELERITY_DPCPP)
#define WORKAROUND_DPCPP 1
#else
#define WORKAROUND_DPCPP 0
#endif

#if defined(__COMPUTECPP__)
#define WORKAROUND_COMPUTECPP 1
#define CELERITY_DETAIL_WA_VERSION_MAJOR COMPUTECPP_VERSION_MAJOR
#define CELERITY_DETAIL_WA_VERSION_MINOR COMPUTECPP_VERSION_MINOR
#define CELERITY_DETAIL_WA_VERSION_PATCH COMPUTECPP_VERSION_PATCH
#else
#define WORKAROUND_COMPUTECPP 0
#endif

#if defined(__HIPSYCL__)
#define WORKAROUND_HIPSYCL 1
#define CELERITY_DETAIL_WA_VERSION_MAJOR HIPSYCL_VERSION_MAJOR
#define CELERITY_DETAIL_WA_VERSION_MINOR HIPSYCL_VERSION_MINOR
#define CELERITY_DETAIL_WA_VERSION_PATCH HIPSYCL_VERSION_PATCH
// Works around a weird hipSYCL bug that causes some functions not to be auto-annotated with __host__ __device__ on CUDA targets
#define WORKAROUND_HIPSYCL_UNIVERSAL_TARGET HIPSYCL_UNIVERSAL_TARGET
#else
#define WORKAROUND_HIPSYCL 0
#define WORKAROUND_HIPSYCL_UNIVERSAL_TARGET
#endif

#define CELERITY_DETAIL_WA_CHECK_VERSION_1(major) (CELERITY_DETAIL_WA_VERSION_MAJOR <= major)
#define CELERITY_DETAIL_WA_CHECK_VERSION_2(major, minor)                                                                                                       \
	(CELERITY_DETAIL_WA_VERSION_MAJOR < major) || (CELERITY_DETAIL_WA_VERSION_MAJOR == major && CELERITY_DETAIL_WA_VERSION_MINOR <= minor)
#define CELERITY_DETAIL_WA_CHECK_VERSION_3(major, minor, patch)                                                                                                \
	(CELERITY_DETAIL_WA_VERSION_MAJOR < major) || (CELERITY_DETAIL_WA_VERSION_MAJOR == major && CELERITY_DETAIL_WA_VERSION_MINOR < minor)                      \
	    || (CELERITY_DETAIL_WA_VERSION_MAJOR == major && CELERITY_DETAIL_WA_VERSION_MINOR == minor && CELERITY_DETAIL_WA_VERSION_PATCH <= patch)

#define CELERITY_DETAIL_WA_GET_OVERLOAD(_1, _2, _3, NAME, ...) NAME
#define CELERITY_DETAIL_WA_MSVC_WORKAROUND(x) x // Workaround for MSVC PP expansion behavior of __VA_ARGS__
#define CELERITY_DETAIL_WA_CHECK_VERSION(...)                                                                                                                  \
	CELERITY_DETAIL_WA_MSVC_WORKAROUND(CELERITY_DETAIL_WA_GET_OVERLOAD(                                                                                        \
	    __VA_ARGS__, CELERITY_DETAIL_WA_CHECK_VERSION_3, CELERITY_DETAIL_WA_CHECK_VERSION_2, CELERITY_DETAIL_WA_CHECK_VERSION_1)(__VA_ARGS__))

#define WORKAROUND(impl, ...) (WORKAROUND_##impl == 1 && CELERITY_DETAIL_WA_CHECK_VERSION(__VA_ARGS__))
