#pragma once

#include <vector>

namespace allscale {
namespace utils {

	namespace {

		/**
		 * The terminal case of a function where a variable number of arguments is written into a vector in proper order.
		 *
		 * @tparam T the element type maintained within the extended vector
		 * @param vector the vector to which nothing is written to
		 */
		template<typename T>
		inline void appendToVector(std::vector<T>&) {}

		/**
		 * A variable-argument function writing elements into a vector in the given order.
		 *
		 * @tparam T the type of element maintained within the modified vector
		 * @tparam Elements the types of the remaining elements (need to be convertible to T)
		 * @param vector the vector to be written to
		 * @param first the next element to be added
		 * @param rest the remaining elements to be added
		 */
		template<typename T, typename ... Elements>
		inline void appendToVector(std::vector<T>& vector, const T& first, const Elements& ... rest) {
			vector.push_back(first);
			appendToVector<T>(vector, rest...);
		}

	}

	/**
	 * Create an empty vector containing no elements.
	 *
	 * @tparam T the type of element to be stored in the resulting vector
	 * @return the resulting vector
	 */
	template<typename T>
	inline std::vector<T> toVector() {
		return std::vector<T> ();
	}

	/**
	 * Creates a vector containing the given elements.
	 *
	 * @tparam T the type of element to be stored in the resulting vector
	 * @tparam Elements the types of the remaining elements (need to be convertible to T)
	 * @param first the first element to be within the list
	 * @param rest the remaining elements to be stored within the list
	 * @return the resulting vector
	 */
	template<typename T, typename ... Elements>
	inline std::vector<T> toVector(const T& first, const Elements& ... rest) {
		std::vector<T> res;
		res.reserve(1 + sizeof...(rest));
		appendToVector<T>(res, first, rest...);
		return res;
	}


	template<typename T>
	struct is_vector : public std::false_type {};

	template<typename E>
	struct is_vector<std::vector<E>> : public std::true_type {};

	template<typename T>
	struct is_vector<const T> : public is_vector<T> {};

	template<typename T>
	struct is_vector<T&> : public is_vector<T> {};

} // end namespace utils
} // end namespace allscale
