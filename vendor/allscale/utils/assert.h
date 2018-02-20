#pragma once

/**
 * This header file defines a set of macros to define more readable and flexible assertions within
 * program code. Also, macros supporting the declaration of variables only required for checking
 * assertions are supported. As all assertions, in case the macro NDEBUG is defined, they will be
 * ignored. In those cases, variables declared using the 'assert_decl' macro will not be declared.
 */

#include <iostream>

#define __allscale_xstr_(a) __allscale_str_(a)
#define __allscale_str_(a) #a

#include "allscale/utils/unused.h"

#if defined(NDEBUG)

#define _assert_ignore                                                                                                                                         \
	if(false) std::cerr << ""

#define assert_decl(_DECL) ((void)0)
#define assert_true(_COND) _assert_ignore
#define assert_eq(_a, _b) _assert_ignore
#define assert_ne(_a, _b) _assert_ignore
#define assert_lt(_a, _b) _assert_ignore
#define assert_le(_a, _b) _assert_ignore
#define assert_gt(_a, _b) _assert_ignore
#define assert_ge(_a, _b) _assert_ignore
#define assert_fail() _assert_ignore
#define assert_pred1(_a, _b) _assert_ignore
#define assert_not_pred1(_a, _b) _assert_ignore
#define assert_pred2(_a, _b, _c) _assert_ignore
#define assert_not_pred2(_a, _b, _c) _assert_ignore

#else
#include <iostream>


namespace insieme {
namespace utils {
	namespace detail {

		struct LazyAssertion {
			bool value;
			LazyAssertion(bool value) : value(value) {}
			~LazyAssertion() {
				if(!value) {
					std::cerr << "\n";
					abort();
				}
			}
			operator bool() const {
				return !value;
			}
		};

	} // end namespace detail
} // end namespace utils
} // end namespace insieme

#define assert_decl(_DECL) _DECL

#define assert_true(_COND)                                                                                                                                     \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((bool)(_COND)))                                                                                 \
	std::cerr << "\nAssertion " #_COND " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n"

#define assert_eq(_A, _B)                                                                                                                                      \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((_A) == (_B)))                                                                                  \
	std::cerr << "\nAssertion " #_A " == " #_B " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n\t" #_A " = " << (_A) << "\n\t" #_B " = " << (_B) << "\n"

#define assert_ne(_A, _B)                                                                                                                                      \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((_A) != (_B)))                                                                                  \
	std::cerr << "\nAssertion " #_A " != " #_B " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n\t" #_A " = " << (_A) << "\n\t" #_B " = " << (_B) << "\n"

#define assert_lt(_A, _B)                                                                                                                                      \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((_A) < (_B)))                                                                                   \
	std::cerr << "\nAssertion " #_A " < " #_B " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n\t" #_A " = " << (_A) << "\n\t" #_B " = " << (_B) << "\n"

#define assert_le(_A, _B)                                                                                                                                      \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((_A) <= (_B)))                                                                                  \
	std::cerr << "\nAssertion " #_A " <= " #_B " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n\t" #_A " = " << (_A) << "\n\t" #_B " = " << (_B) << "\n"

#define assert_gt(_A, _B)                                                                                                                                      \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((_A) > (_B)))                                                                                   \
	std::cerr << "\nAssertion " #_A " > " #_B " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n\t" #_A " = " << (_A) << "\n\t" #_B " = " << (_B) << "\n"

#define assert_ge(_A, _B)                                                                                                                                      \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((_A) >= (_B)))                                                                                  \
	std::cerr << "\nAssertion " #_A " >= " #_B " of " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n\t" #_A " = " << (_A) << "\n\t" #_B " = " << (_B) << "\n"

#define assert_fail()                                                                                                                                          \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion(false)) std::cerr << "\nAssertion failed in " __FILE__ ":" __allscale_xstr_(__LINE__) " - "

#define assert_pred1(_P, _A)                                                                                                                                   \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((bool)((_P)(_A))))                                                                              \
	std::cerr << "\nAssertion " #_P "(" #_A ") with " #_A " = " << (_A) << " in " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n"

#define assert_not_pred1(_P, _A)                                                                                                                               \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion(!(bool)((_P)(_A))))                                                                             \
	std::cerr << "\nAssertion !" #_P "(" #_A ") with " #_A " = " << (_A) << " in " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n"

#define assert_pred2(_P, _A, _B)                                                                                                                               \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion((bool)((_P)(_A, _B))))                                                                          \
	std::cerr << "\nAssertion " #_P "(" #_A ", " #_B ") with\n " #_A " = " << (_A) << "\n " #_B " = " << (_B)                                                  \
	          << "\n in " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n"

#define assert_not_pred2(_P, _A, _B)                                                                                                                           \
	if(__allscale_unused auto __allscale_temp_object_ = insieme::utils::detail::LazyAssertion(!(bool)((_P)(_A, _B))))                                                                         \
	std::cerr << "\nAssertion !" #_P "(" #_A ", " #_B ") with\n " #_A " = " << (_A) << "\n " #_B " = " << (_B)                                                 \
	          << "\n in " __FILE__ ":" __allscale_xstr_(__LINE__) " failed!\n"

#endif

// ------ derived definitions ------

#define assert_false(_COND) assert_true(!(_COND))
#define assert_not_implemented() assert_fail() << "Not implemented functionality in " __FILE__ ":" __allscale_xstr_(__LINE__) "\n"

// --------- bounds checks ---------

#if defined(ALLSCALE_CHECK_BOUNDS)

#define allscale_check_bounds(_INDEX, _CONTAINER)																												\
	assert_true((_INDEX) >= 0 && (_INDEX) < (_CONTAINER).size()) << "Index " << (_INDEX) << " out of bounds " << (_CONTAINER).size();

#else

#define allscale_check_bounds(_INDEX, _CONTAINER)																												\
	if(false) std::cerr << ""

#endif