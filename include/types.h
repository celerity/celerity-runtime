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

#undef CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS

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

inline constexpr reduction_id no_reduction_id = 0;

/// Uniquely identifies one version of a buffer's (distributed) data at task granularity. The structure is used to tie together the sending and receiving ends
/// of peer-to-peer data transfers.
struct transfer_id {
	/// The first task (by order of task id) to require this version of the buffer.
	task_id consumer_tid = -1;

	/// The buffer's id.
	buffer_id bid = -1;

	/// The reduction the data belongs to. If `!= no_reduction_id`, the transferred data consists of partial results that will be consumed by a subsequent
	/// reduction command to produce the final value.
	///
	/// Since a task cannot require data both as part of a reduction and with its final value at the same time, this field is not necessary to identify the
	/// transfer version, but is used for sanity checks. It might become additionally valuable once we allow the user to specify the buffer subrange each
	/// reduction is targeting.
	reduction_id rid = no_reduction_id;

	transfer_id() = default;
	transfer_id(const task_id consumer_tid, const buffer_id bid, const reduction_id rid = no_reduction_id) : consumer_tid(consumer_tid), bid(bid), rid(rid) {}

	friend bool operator==(const transfer_id& lhs, const transfer_id& rhs) {
		return lhs.consumer_tid == rhs.consumer_tid && lhs.bid == rhs.bid && lhs.rid == rhs.rid;
	}
	friend bool operator!=(const transfer_id& lhs, const transfer_id& rhs) { return !(lhs == rhs); }
};

enum class error_policy {
	ignore,
	log_warning,
	log_error,
	panic,
};

} // namespace celerity::detail

template <>
struct std::hash<celerity::detail::transfer_id> {
	std::size_t operator()(const celerity::detail::transfer_id& t) const noexcept; // defined in utils.cc
};
