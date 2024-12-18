#pragma once

#include "types.h"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <fmt/format.h>


#define CELERITY_DETAIL_UTILS_CAT_2(a, b) a##b
#define CELERITY_DETAIL_UTILS_CAT(a, b) CELERITY_DETAIL_UTILS_CAT_2(a, b)

namespace celerity::detail::utils {

template <typename T, typename P>
bool isa(const P* p) {
	return dynamic_cast<const T*>(p) != nullptr;
}

template <typename T, typename P>
auto as(P* p) {
	assert(isa<T>(p));
	return static_cast<std::conditional_t<std::is_const_v<P>, const T*, T*>>(p);
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

std::string make_task_debug_label(const task_type tt, const task_id tid, const std::string& debug_name, bool title_case = false);


[[noreturn]] void unreachable();

enum class panic_solution {
	log_and_abort,     ///< default
	throw_logic_error, ///< enabled in unit tests to detect and recover from panics
};

/// Globally and atomically sets the behavior of `utils::panic()`.
void set_panic_solution(panic_solution solution);

/// Either throws or aborts with a message, depending on the global `panic_solution` setting.
[[noreturn]] void panic(const std::string& msg);

/// Either throws or aborts with a message, depending on the global `panic_solution` setting.
template <typename... FmtParams>
[[noreturn]] void panic(fmt::format_string<FmtParams...> fmt_string, FmtParams&&... fmt_args) {
	// TODO also receive a std::source_location with C++20.
	panic(fmt::format(fmt_string, std::forward<FmtParams>(fmt_args)...));
}

/// Ignores, logs, or panics on an error depending on the `error_policy`.
void report_error(const error_policy policy, const std::string& msg);

/// Ignores, logs, or panics on an error depending on the `error_policy`.
template <typename... FmtParams, std::enable_if_t<sizeof...(FmtParams) >= 1, int> = 0>
void report_error(const error_policy policy, const fmt::format_string<FmtParams...> fmt_string, FmtParams&&... fmt_args) {
	// TODO also receive a std::source_location with C++20.
	if(policy != error_policy::ignore) { report_error(policy, fmt::format(fmt_string, std::forward<FmtParams>(fmt_args)...)); }
}


template <typename Container>
Container set_intersection(const Container& lhs, const Container& rhs) {
	assert(std::ranges::is_sorted(lhs));
	assert(std::ranges::is_sorted(rhs));
	Container intersection;
	std::ranges::set_intersection(lhs, rhs, std::back_inserter(intersection));
	return intersection;
}

template <typename Container, typename Key>
bool contains(const Container& container, const Key& key) {
	if constexpr(requires(Container c) { c.find(key); }) {
		return container.find(key) != std::end(container);
	} else {
		return std::ranges::find(container, key) != std::end(container);
	}
}

/// Replaces all occurrences of `pattern` in `in` with `with`. If `pattern` is empty, returns the input string unchanged.
std::string replace_all(const std::string_view& input, const std::string_view& pattern, const std::string_view& replacement);

template <std::integral Integral>
[[nodiscard]] constexpr Integral ceil(const Integral quantity, const Integral granularity) {
	return (quantity + granularity - 1) / granularity * granularity;
}

template <typename Void>
    requires(std::is_void_v<Void>)
[[nodiscard]] constexpr Void* offset(Void* const ptr, const size_t offset_bytes) {
	using byte_type = std::conditional_t<std::is_const_v<Void>, const std::byte, std::byte>;
	return static_cast<byte_type*>(ptr) + offset_bytes;
}

} // namespace celerity::detail::utils
