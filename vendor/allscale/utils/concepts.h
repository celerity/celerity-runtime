#pragma once

#include <type_traits>

namespace allscale {
namespace utils {

	template<typename T, typename _ = void>
	struct is_equality_comparable : public std::false_type {};

	template<typename T>
	struct is_equality_comparable<T,typename std::enable_if<
			std::is_convertible<decltype(std::declval<const T&>() == std::declval<const T&>()),bool>::value &&
			std::is_convertible<decltype(std::declval<const T&>() != std::declval<const T&>()),bool>::value,
		void>::type> : public std::true_type {};


	template<typename T, typename _ = void>
	struct is_value : public std::false_type {};

	template<typename T>
	struct is_value<T,typename std::enable_if<

			// regions need to be default-constructible
			std::is_default_constructible<T>::value &&

			// regions need to be default-constructible
			std::is_copy_constructible<T>::value &&

			// regions need to be default-constructible
			std::is_copy_assignable<T>::value &&

			// regions need to be destructible
			std::is_destructible<T>::value &&

			// regions need to be equality comparable
			utils::is_equality_comparable<T>::value,

		void>::type> : public std::true_type {};


} // end namespace utils
} // end namespace allscale
