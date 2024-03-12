#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>

#include <fmt/format.h>

#include "types.h"


namespace celerity::detail::utils {

/// Like std::move, but move-constructs the result so it does not reference the argument after returning.
template <typename T>
T take(T& from) {
	return std::move(from);
}

template <typename T, typename P>
bool isa(const P* p) {
	return dynamic_cast<const T*>(p) != nullptr;
}

template <typename T, typename P>
auto as(P* p) {
	assert(isa<T>(p));
	return static_cast<std::conditional_t<std::is_const_v<P>, const T*, T*>>(p);
}

template <typename BitMaskT>
constexpr inline uint32_t popcount(const BitMaskT bit_mask) noexcept {
	static_assert(std::is_integral_v<BitMaskT> && std::is_unsigned_v<BitMaskT>, "popcount argument needs to be an unsigned integer type.");

	uint32_t counter = 0;
	for(auto b = bit_mask; b; b >>= 1) {
		counter += b & 1;
	}
	return counter;
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

namespace celerity::detail::utils_detail {

template <typename... Without, typename... ToKeep, typename T, typename... Ts>
constexpr auto tuple_without_impl(const std::tuple<ToKeep...>& to_keep, const std::tuple<T, Ts...>& to_check) {
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

template <typename Tuple, typename Callback>
constexpr void tuple_for_each_pair_impl(const Tuple&, Callback&&, std::index_sequence<>) {}

template <typename Tuple, size_t I1, size_t I2, size_t... Is, typename Callback>
constexpr void tuple_for_each_pair_impl(const Tuple& tuple, const Callback& cb, std::index_sequence<I1, I2, Is...>) {
	cb(std::get<I1>(tuple), std::get<I2>(tuple));
	tuple_for_each_pair_impl(tuple, cb, std::index_sequence<Is...>{});
}

} // namespace celerity::detail::utils_detail

namespace celerity::detail::utils {

/// See `utils::type_switch_t`.
template <typename Lookup, typename... KVs>
struct type_switch {};

/// `switch` equivalent of `std::conditional_t`. Use as `utils::type_switch_t<lookup-type, key-type-1(result-type-1), key-type-2(result-type-2), ...>`
template <typename Lookup, typename... KVs>
using type_switch_t = typename type_switch<Lookup, KVs...>::type;

template <typename MatchingKey, typename Value, typename... KVs>
struct type_switch<MatchingKey, MatchingKey(Value), KVs...> {
	using type = Value;
};

template <typename NonMatching, typename Key, typename Value, typename... KVs>
struct type_switch<NonMatching, Key(Value), KVs...> {
	using type = type_switch_t<NonMatching, KVs...>;
};

template <typename... Without, typename... Ts>
constexpr auto tuple_without(const std::tuple<Ts...>& tuple) {
	if constexpr(sizeof...(Ts) > 0) {
		return utils_detail::tuple_without_impl<Without...>({}, tuple);
	} else {
		return tuple;
	}
}

template <typename Tuple, typename Callback>
constexpr void tuple_for_each_pair(const Tuple& tuple, const Callback& cb) {
	static_assert(std::tuple_size_v<Tuple> % 2 == 0, "an even number of entries is required");
	utils_detail::tuple_for_each_pair_impl(tuple, cb, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename Variant, typename Alternative>
Alternative& replace(Variant& variant, Alternative&& alternative) {
	return std::get<Alternative>(variant = std::forward<Alternative>(alternative));
}

/// Fiddles out the base name of a (possibly templated) struct or class from a full (possibly mangled) type name.
/// The input parameter should be `typeid(Struct*)`, i.e. a _pointer_ to the desired struct type.
std::string get_simplified_type_name_from_pointer(const std::type_info& pointer_type_info);

/// Fiddles out the base name of a (possibly templated) struct or class from a full (possibly mangled) type name.
template <typename Struct>
std::string get_simplified_type_name() {
	// Using a pointer will also make this function work types that have no definitions, which commonly happens for kernel name type.
	return get_simplified_type_name_from_pointer(typeid(Struct*));
}

/// Escapes "<", ">", and "&" with their corresponding HTML escape sequences
std::string escape_for_dot_label(std::string str);

/// Print the buffer id as either 'B1' or 'B1 "name"' (if `name` is non-empty)
std::string make_buffer_debug_label(const buffer_id bid, const std::string& name = "");

enum class panic_solution {
	log_and_abort,     ///< default
	throw_logic_error, ///< enabled in unit tests to detect and recover from panics
};

/// Globally and atomically sets the behavior of `utils::panic()`.
void set_panic_solution(panic_solution solution);

/// Either throws or aborts with a message, depending on the global `panic_solution` setting.
[[noreturn]] void panic(const std::string& msg);

/// Either throws or aborts with a message, depending on the global `panic_solution` setting.
template <typename... FmtParams, std::enable_if_t<sizeof...(FmtParams) >= 2, int> = 0>
[[noreturn]] void panic(const FmtParams&... fmt_args) {
	// TODO also receive a std::source_location with C++20.
	panic(fmt::format(fmt_args...));
}

/// Ignores, logs, or panics on an error depending on the `error_policy`.
void report_error(const error_policy policy, const std::string& msg);

/// Ignores, logs, or panics on an error depending on the `error_policy`.
template <typename... FmtParams, std::enable_if_t<sizeof...(FmtParams) >= 1, int> = 0>
void report_error(const error_policy policy, const fmt::format_string<FmtParams...> fmt_string, FmtParams&&... fmt_args) {
	// TODO also receive a std::source_location with C++20.
	if(policy != error_policy::ignore) { report_error(policy, fmt::format(fmt_string, std::forward<FmtParams>(fmt_args)...)); }
}

} // namespace celerity::detail::utils
