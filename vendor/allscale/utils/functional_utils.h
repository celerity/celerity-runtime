#pragma once

#include <functional>
#include <type_traits>

#include "allscale/utils/type_list.h"

namespace allscale {
namespace utils {


	// -------------------- Function Traits for Lambdas ----------------------------

	namespace detail {

		template<typename Function> struct lambda_traits_helper { };

		// get rid of const modifier
		template<typename T>
		struct lambda_traits_helper<const T> : public lambda_traits_helper<T> {};

		// get rid of pointers
		template<typename T>
		struct lambda_traits_helper<T*> : public lambda_traits_helper<T> {};

		// handle class of member function pointers
		template<typename R, typename C, typename ... A>
		struct lambda_traits_helper<R(C::*)(A...)> : public lambda_traits_helper<R(*)(A...)> {
			typedef C class_type;
		};

		// get rid of const modifier
		template<typename R, typename C, typename ... A>
		struct lambda_traits_helper<R(C::*)(A...) const> : public lambda_traits_helper<R(C::*)(A...)> {};

		template<typename R>
		struct lambda_traits_helper<R(void)>
		{
		  enum { arity = 0 };
		  typedef R result_type;
		  typedef type_list<> argument_types;
		};

		template<typename R, typename T1>
		struct lambda_traits_helper<R(T1)>
		{
		  enum { arity = 1 };
		  typedef R result_type;
		  typedef T1 arg1_type;
		  typedef T1 argument_type;
		  typedef type_list<T1> argument_types;
		};

		template<typename R, typename T1, typename T2>
		struct lambda_traits_helper<R(T1, T2)>
		{
		  enum { arity = 2 };
		  typedef R result_type;
		  typedef T1 arg1_type;
		  typedef T2 arg2_type;
		  typedef T1 first_argument_type;
		  typedef T2 second_argument_type;
		  typedef type_list<T1,T2> argument_types;
		};

		template <typename R, typename T1, typename T2, typename T3, typename ... A >
		struct lambda_traits_helper<R( T1, T2, T3, A ... )>  {
			enum { arity = 3 + sizeof...(A) };
			typedef R result_type;
			typedef T1 arg1_type;
			typedef T2 arg2_type;
			typedef T3 arg3_type;
			typedef type_list<T1,T2,T3,A...> argument_types;
		};


		template<typename Lambda>
		struct call_operator_type {
			using type = decltype(Lambda::operator());
		};

		template<typename Lambda>
		decltype(&Lambda::operator()) getCallOperator() {
			return &Lambda::operator();
		}

		template<typename Lambda>
		decltype(&Lambda::template operator()<int>) getCallOperator() {
			return &Lambda::template operator()<int>;
		}

		template<typename Lambda>
		decltype(&Lambda::template operator()<int,int>) getCallOperator() {
			return &Lambda::template operator()<int,int>;
		}

		template<typename Lambda>
		decltype(&Lambda::template operator()<int,int,int>) getCallOperator() {
			return &Lambda::template operator()<int,int,int>;
		}

	} // end namespace detail


	template <typename Lambda>
	struct lambda_traits : public detail::lambda_traits_helper<decltype(detail::getCallOperator<Lambda>())> { };

	template<typename R, typename ... P>
	struct lambda_traits<R(P...)> : public detail::lambda_traits_helper<R(P...)> { };

	template<typename R, typename ... P>
	struct lambda_traits<R(*)(P...)> : public lambda_traits<R(P...)> { };

	template<typename R, typename ... P>
	struct lambda_traits<R(* const)(P...)> : public lambda_traits<R(P...)> { };

	template<typename R, typename C, typename ... P>
	struct lambda_traits<R(C::*)(P...)> : public detail::lambda_traits_helper<R(C::*)(P...)> { };

	template<typename R, typename C, typename ... P>
	struct lambda_traits<R(C::* const)(P...)> : public lambda_traits<R(C::*)(P...)> { };



	template<typename T>
	struct is_std_function : public std::false_type {};

	template<typename E>
	struct is_std_function<std::function<E>> : public std::true_type {};

	template<typename T>
	struct is_std_function<const T> : public is_std_function<T> {};

	template<typename T>
	struct is_std_function<T&> : public is_std_function<T> {};


} // end namespace utils
} // end namespace allscale
