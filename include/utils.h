#pragma once

#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>

// Certain types such as buffers and accessors could technically be declared const in most
// situations, as they are being captured by-value into the inner lambda, however we don't want
// that as it gives the wrong semantic impression.
// To work around this, we define a copy constructor that receives a non-const argument,
// which tricks clang-tidy into thinking that the declaration cannot be made const.
#define CELERITY_DETAIL_HACK_CLANG_TIDY_ALLOW_NON_CONST(type)                                                                                                  \
	type(type& other) : type(std::as_const(other)) {}

namespace celerity::detail::utils {

template <typename BitMaskT>
constexpr inline uint32_t popcount(const BitMaskT bit_mask) noexcept {
	static_assert(std::is_integral_v<BitMaskT> && std::is_unsigned_v<BitMaskT>, "popcount argument needs to be an unsigned integer type.");

	uint32_t counter = 0;
	for(auto b = bit_mask; b; b >>= 1) {
		counter += b & 1;
	}
	return counter;
}

template <typename... F>
struct overload : F... {
	explicit constexpr overload(F... f) : F(f)... {}
	using F::operator()...;
};

template <typename Variant, typename... Arms>
decltype(auto) match(Variant&& v, Arms&&... arms) {
	return std::visit(overload{std::forward<Arms>(arms)...}, std::forward<Variant>(v));
}

// Implementation from Boost.ContainerHash, licensed under the Boost Software License, Version 1.0.
inline void hash_combine(std::size_t& seed, std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

struct pair_hash {
	template <typename U, typename V>
	std::size_t operator()(const std::pair<U, V>& p) const {
		std::size_t seed = 0;
		hash_combine(seed, std::hash<U>{}(p.first));
		hash_combine(seed, std::hash<V>{}(p.second));
		return seed;
	}
};

} // namespace celerity::detail::utils
