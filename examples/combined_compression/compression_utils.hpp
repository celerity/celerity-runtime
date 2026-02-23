#pragma once

#include <celerity.h>

namespace celerity {

// Primary template for is_vec
template <typename T>
struct is_vec : std::false_type {};

// Specialization for sycl::vec
template <typename T, int N>
struct is_vec<sycl::vec<T, N>> : std::true_type {};

template <typename T>
struct vec_element_type {
	using type = T;
};

template <typename T, int N>
struct vec_element_type<sycl::vec<T, N>> {
	using type = T;
};

template <typename T>
struct vec_size {
	static constexpr int value = 1;
};

template <typename T, int N>
struct vec_size<sycl::vec<T, N>> {
	static constexpr int value = N;
};
} // namespace celerity
