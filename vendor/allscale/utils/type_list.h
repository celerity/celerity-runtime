#pragma once

#include <type_traits>

namespace allscale {
namespace utils {


	// -------------------- Type List traits ----------------------------

	template <typename ... Ts>
	struct type_list {
		enum { length = sizeof...(Ts) };
		enum { empty = (length == 0) };
	};


	// -- test whether a given list contains a given type --

	template<typename T, typename List>
	struct type_list_contains;

	template<typename H, typename ... R>
	struct type_list_contains<H,type_list<H,R...>> : public std::true_type {};

	template<typename T, typename H, typename ... R>
	struct type_list_contains<T,type_list<H,R...>> : public type_list_contains<T,type_list<R...>> {};

	template<typename T>
	struct type_list_contains<T,type_list<>> : public std::false_type {};


	// -- extracts a type at a given position --

	template<std::size_t pos, typename List>
	struct type_at;

	template<typename H, typename ...R>
	struct type_at<0, type_list<H,R...>> {
		typedef H type;
	};

	template<std::size_t pos, typename H, typename ...R>
	struct type_at<pos, type_list<H,R...>> {
		typedef typename type_at<pos-1, type_list<R...>>::type type;
	};


	// -- obtains the index of a given type --

	template<typename T, typename List>
	struct type_index;

	template<typename H, typename ... R>
	struct type_index<H,type_list<H,R...>> {
		enum { value = 0 };
	};

	template<typename T, typename H, typename ... R>
	struct type_index<T,type_list<H,R...>> {
		enum { value = type_index<T,type_list<R...>>::value + 1 };
	};


} // end namespace utils
} // end namespace allscale
