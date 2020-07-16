#pragma once

/**
 * As of April 2020, Boost (1.73) does not have explicit support for CUDA compilation using Clang.
 * Instead, Boost assumes (in most places) that CUDA code is always compiled with NVIDIA's NVCC.
 * Detection of CUDA compilation is based on the __CUDACC__ macro. Unfortunately, this macro is also
 * set by Clang, alongside the __clang__ macro. This combination is not anticipated by Boost, and
 * thus, chaos ensues.
 *
 * This header contains several workarounds for various issues resulting from this. It is very
 * likely that similar issues exist for other Boost components that have not yet been used in
 * Celerity. In that case, additional workarounds will have to be added here (if possible)
 * and ideally the underlying issue is reported upstream.
 *
 * All of this is only required when compiling with hipSYCL targeting the CUDA backend
 * (and possibly HIP). For the Celerity runtime itself, this header is automatically included in
 * every translation unit (configured through CMake). For users, it is included from within the main
 * entry header (celerity.h).
 *
 * Note that Boost 1.65 is the earliest version considered here (older versions might still work).
 */

/**
 * Ensure that the Clang compiler configuration is being used (otherwise GCC is assumed for CUDA).
 *
 * Required for Boost 1.69 - 1.73.
 */
#define BOOST_COMPILER_CONFIG "boost/config/compiler/clang.hpp"

/**
 * The __is_base_of intrinsic is used by Boost type_traits,
 * which in turn is used all over the place internally.
 *
 * Bug repro:
 * ```
 *  struct bar {};
 *  boost::optional<bar> foo;
 *  foo = bar{};
 * ```
 *
 * Unfortunately there is no easy way to enable all Clang intrinsics, as that particular
 * header (boost/type_traits/intrinsics.hpp) checks for __CUDACC__ directly.
 *
 * Required for Boost 1.65 - 1.68 (if using the Clang config above), 1.69 - 1.73 (not fixed yet).
 */
#define BOOST_IS_BASE_OF(T, U) (__is_base_of(T, U) && !is_same<T, U>::value)

/**
 * Enable variadic preprocessor macros. Used by the WORKAROUND macros in workaround.h.
 *
 * Required for Boost 1.65, 1.66, 1.69 - 1.72 (fixed in 1.73).
 * See https://github.com/boostorg/preprocessor/issues/24
 */
#define BOOST_PP_VARIADICS 1
