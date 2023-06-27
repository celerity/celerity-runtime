#pragma once

#include <cstdint>
#include <functional>
#include <type_traits>
#include <variant>

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

namespace utils_detail {

	template <typename... Without, typename... ToKeep, typename T, typename... Ts>
	static auto tuple_without_impl(const std::tuple<ToKeep...>& to_keep, const std::tuple<T, Ts...>& to_check) {
		if constexpr((std::is_same_v<T, Without> || ...)) {
			if constexpr(sizeof...(Ts) == 0) {
				return to_keep;
			} else {
				return tuple_without_impl<Without...>(to_keep, std::tuple{std::get<Ts>(to_check)...});
			}
		} else {
			if constexpr(sizeof...(Ts) == 0) {
				return std::tuple_cat(to_keep, to_check);
			} else {
				return tuple_without_impl<Without...>(std::tuple_cat(to_keep, std::tuple{std::get<T>(to_check)}), std::tuple{std::get<Ts>(to_check)...});
			}
		}
	}

} // namespace utils_detail

template <typename... Without, typename... Ts>
static auto tuple_without(const std::tuple<Ts...>& tuple) {
	if constexpr(sizeof...(Ts) > 0) {
		return utils_detail::tuple_without_impl<Without...>({}, tuple);
	} else {
		return tuple;
	}
}


} // namespace celerity::detail::utils
