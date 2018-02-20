#pragma once

#include <array>
#include <type_traits>

namespace allscale {
namespace utils {

namespace {
	template<int I, int N, typename U>
	struct array_builder {
		template<typename Fn, typename... T>
		std::array<U, N> operator()(Fn&& fn, T&&... vals) const {
			return array_builder<I + 1, N, U>{}(std::forward<Fn>(fn), std::forward<T>(vals)..., fn());
		}
	};

	template<int N, typename U>
	struct array_builder<N, N, U> {
		template<typename Fn, typename... T>
		std::array<U, N> operator()(Fn&&, T&&... vals) const {
			return { { std::forward<T>(vals)... } };
		}
	};
}

/*
 * Create an Array of N elements, initialized with the elements returned by fn. Can be used to create an array of elements without default constructor
 *
 */
template<int N, typename Fn, typename U = typename std::result_of<Fn()>::type>
std::array<U, N> build_array(Fn&& fn) {
    return array_builder<0, N, U>()(std::forward<Fn>(fn));
}

} // end namespace utils
} // end namespace allscale
