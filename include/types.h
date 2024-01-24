#pragma once

#include <cstdlib>
#include <functional>
#include <type_traits>

namespace celerity::detail {

/// Like `false`, but dependent on one or more template parameters. Use as the condition of always-failing static assertions in overloads, template
/// specializations or `if constexpr` branches.
template <typename...>
constexpr bool constexpr_false = false;

} // namespace celerity::detail

/// Defines a POD type with a single member `value` from which and to which it is implicitly convertible. Since C++ only allows a single implicit conversion to
/// happen when types need to be adjusted, this retains strong type safety between multiple type aliases (e.g. task_id is not implicitly convertible to
/// node_id), but arithmetic operations will automatically work on the value type.
#define CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(ALIAS_NAME, VALUE_TYPE)                                                                                       \
	namespace celerity::detail {                                                                                                                               \
		struct ALIAS_NAME {                                                                                                                                    \
			using value_type = VALUE_TYPE;                                                                                                                     \
			VALUE_TYPE value;                                                                                                                                  \
			ALIAS_NAME() = default;                                                                                                                            \
			constexpr ALIAS_NAME(const value_type value) : value(value) {}                                                                                     \
			constexpr operator value_type&() { return value; }                                                                                                 \
			constexpr operator const value_type&() const { return value; }                                                                                     \
		};                                                                                                                                                     \
	}                                                                                                                                                          \
	template <>                                                                                                                                                \
	struct std::hash<celerity::detail::ALIAS_NAME> {                                                                                                           \
		std::size_t operator()(const celerity::detail::ALIAS_NAME a) const noexcept { return std::hash<VALUE_TYPE>{}(a.value); }                               \
	};

CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(task_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(buffer_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(node_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(command_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(collective_group_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(reduction_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(host_object_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(hydration_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(transfer_id, size_t)

// verify properties of type conversion as documented for CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS
static_assert(std::is_standard_layout_v<celerity::detail::hydration_id> && std::is_trivially_default_constructible_v<celerity::detail::hydration_id>);
static_assert(std::is_convertible_v<celerity::detail::task_id, size_t>);
static_assert(std::is_convertible_v<size_t, celerity::detail::task_id>);
static_assert(!std::is_convertible_v<celerity::detail::task_id, celerity::detail::node_id>);

// declared in this header for include-dependency reasons
namespace celerity::experimental {

enum class side_effect_order { sequential };

}

namespace celerity::detail {

struct reduction_info {
	reduction_id rid = 0;
	buffer_id bid = 0;
	bool init_from_buffer = false;
};

constexpr node_id master_node_id = 0;

enum class error_policy {
	ignore,
	log_warning,
	log_error,
	panic,
};

} // namespace celerity::detail
