#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
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
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(memory_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(device_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(raw_allocation_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(instruction_id, size_t)
CELERITY_DETAIL_DEFINE_STRONG_TYPE_ALIAS(message_id, size_t)

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

inline constexpr node_id master_node_id = 0;

/// Uniquely identifies an allocation across all memories on the local node. This is the instruction-graph equivalent of a USM pointer.
///
/// As allocation_ids are used extensively within the code but its constituents (memory_id and raw_allocation_id) rarely need to be inspected, it is bit-encoded
/// into a single integer member.
class allocation_id {
  public:
	constexpr static size_t memory_id_bits = 8;
	constexpr static size_t max_memory_id = (1 << memory_id_bits) - 1;
	constexpr static size_t raw_allocation_id_bits = sizeof(size_t) * 8 - memory_id_bits;
	constexpr static size_t max_raw_allocation_id = (size_t(1) << raw_allocation_id_bits) - 1;

	/// Constructs an allocation_id that does not point to memory (equivalent to `null_allocation_id`).
	constexpr allocation_id() : m_mid(0), m_raid(0) {}

	constexpr allocation_id(const memory_id mid, const raw_allocation_id raid) : m_mid(mid), m_raid(raid) {
		assert(mid <= max_memory_id);
		assert(raid <= max_raw_allocation_id);
	}

	constexpr memory_id get_memory_id() const { return m_mid; }
	constexpr raw_allocation_id get_raw_allocation_id() const { return m_raid; }

	friend constexpr bool operator==(const allocation_id& lhs, const allocation_id& rhs) { return lhs.m_mid == rhs.m_mid && lhs.m_raid == rhs.m_raid; }
	friend constexpr bool operator!=(const allocation_id& lhs, const allocation_id& rhs) { return !(lhs == rhs); }

  private:
	friend struct std::hash<allocation_id>;
	size_t m_mid : memory_id_bits;
	size_t m_raid : raw_allocation_id_bits;
};

/// allocation_id equivalent of a null pointer.
inline constexpr allocation_id null_allocation_id{};

inline constexpr collective_group_id non_collective_group_id = 0;
inline constexpr collective_group_id root_collective_group_id = 1;

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

/// Instruction-graph equivalent of a USM pointer that permits pointer arithmetic.
/// The offset will be applied by the executor once the allocation pointer is known.
struct allocation_with_offset {
	allocation_id id = null_allocation_id;
	size_t offset_bytes = 0;

	/// Constructs the equivalent of a null pointer.
	allocation_with_offset() = default;

	allocation_with_offset(const detail::allocation_id aid, const size_t offset_bytes = 0) : id(aid), offset_bytes(offset_bytes) {}

	friend bool operator==(const allocation_with_offset& lhs, const allocation_with_offset& rhs) {
		return lhs.id == rhs.id && lhs.offset_bytes == rhs.offset_bytes;
	}
	friend bool operator!=(const allocation_with_offset& lhs, const allocation_with_offset& rhs) {
		return lhs.id == rhs.id && lhs.offset_bytes == rhs.offset_bytes;
	}
};

} // namespace celerity::detail

template <>
struct std::hash<celerity::detail::transfer_id> {
	std::size_t operator()(const celerity::detail::transfer_id& t) const noexcept; // defined in utils.cc
};

template <>
struct std::hash<celerity::detail::allocation_id> {
	std::size_t operator()(const celerity::detail::allocation_id aid) const noexcept {
		static_assert(sizeof(celerity::detail::allocation_id) == sizeof(size_t));
		size_t hash = 0;
		memcpy(&hash, &aid, sizeof(size_t));
		return hash;
	}
};

namespace celerity::detail {

enum class error_policy {
	ignore,
	log_warning,
	log_error,
	panic,
};

enum class task_type {
	epoch,          ///< task epoch (graph-level serialization point)
	host_compute,   ///< host task with explicit global size and celerity-defined split
	device_compute, ///< device compute task
	collective,     ///< host task with implicit 1d global size = #ranks and fixed split
	master_node,    ///< zero-dimensional host task
	horizon,        ///< task horizon
	fence,          ///< promise-side of an async experimental::fence
};

enum class execution_target {
	none,
	host,
	device,
};

enum class epoch_action {
	none,
	barrier,
	shutdown,
};

} // namespace celerity::detail
