#pragma once

#include <CL/sycl.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/facilities/overload.hpp>

#if defined(__COMPUTECPP__)
#define WORKAROUND_COMPUTECPP 1
#define _WA_VERSION_MAJOR COMPUTECPP_VERSION_MAJOR
#define _WA_VERSION_MINOR COMPUTECPP_VERSION_MINOR
#define _WA_VERSION_PATCH COMPUTECPP_VERSION_PATCH
#else
#define WORKAROUND_COMPUTECPP 0
#endif

#if defined(__HIPSYCL__) || defined(__HIPSYCL_TRANSFORM__)
#define WORKAROUND_HIPSYCL 1
#define _WA_VERSION_MAJOR HIPSYCL_VERSION_MAJOR
#define _WA_VERSION_MINOR HIPSYCL_VERSION_MINOR
#define _WA_VERSION_PATCH HIPSYCL_VERSION_PATCH
#else
#define WORKAROUND_HIPSYCL 0
#endif

#define _WA_CHECK_VERSION_1(major) (_WA_VERSION_MAJOR <= major)
#define _WA_CHECK_VERSION_2(major, minor) (_WA_VERSION_MAJOR < major) || (_WA_VERSION_MAJOR == major && _WA_VERSION_MINOR <= minor)
#define _WA_CHECK_VERSION_3(major, minor, patch)                                                                                                               \
	(_WA_VERSION_MAJOR < major) || (_WA_VERSION_MAJOR == major && _WA_VERSION_MINOR < minor)                                                                   \
	    || (_WA_VERSION_MAJOR == major && _WA_VERSION_MINOR == minor && _WA_VERSION_PATCH <= patch)

#if !BOOST_PP_VARIADICS_MSVC
#define _WA_CHECK_VERSION(...) BOOST_PP_OVERLOAD(_WA_CHECK_VERSION_, __VA_ARGS__)(__VA_ARGS__)
#else
#define _WA_CHECK_VERSION(...) BOOST_PP_CAT(BOOST_PP_OVERLOAD(_WA_CHECK_VERSION_, __VA_ARGS__)(__VA_ARGS__), BOOST_PP_EMPTY())
#endif

#define WORKAROUND(impl, ...) (WORKAROUND_##impl == 1 && _WA_CHECK_VERSION(__VA_ARGS__))
