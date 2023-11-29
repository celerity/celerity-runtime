#pragma once

#include <cstdlib>
#include <functional>
#include <utility>

namespace celerity {
namespace detail {

	/// Like `false`, but dependent on one or more template parameters. Use as the condition of always-failing static assertions in overloads, template
	/// specializations or `if constexpr` branches.
	template <typename...>
	constexpr bool constexpr_false = false;

	template <typename T, typename UniqueName>
	class PhantomType {
	  public:
		using underlying_t = T;

		constexpr PhantomType() = default;
		constexpr PhantomType(const T& value) : m_value(value) {}
		constexpr PhantomType(T&& value) : m_value(std::move(value)) {}

		// Allow implicit conversion to underlying type, otherwise it becomes too annoying to use.
		// Luckily compilers won't do more than one user-defined conversion, so something like
		// PhantomType1<T> -> T -> PhantomType2<T>, can't happen. Therefore we still retain
		// strong typesafety between phantom types with the same underlying type.
		constexpr operator T&() { return m_value; }
		constexpr operator const T&() const { return m_value; }

	  private:
		T m_value;
	};

} // namespace detail
} // namespace celerity

#define MAKE_PHANTOM_TYPE(TypeName, UnderlyingT)                                                                                                               \
	namespace celerity {                                                                                                                                       \
		namespace detail {                                                                                                                                     \
			using TypeName = PhantomType<UnderlyingT, class TypeName##_PhantomType>;                                                                           \
		}                                                                                                                                                      \
	}                                                                                                                                                          \
	namespace std {                                                                                                                                            \
		template <>                                                                                                                                            \
		struct hash<celerity::detail::TypeName> {                                                                                                              \
			std::size_t operator()(const celerity::detail::TypeName& t) const noexcept { return std::hash<UnderlyingT>{}(static_cast<UnderlyingT>(t)); }       \
		};                                                                                                                                                     \
	}

MAKE_PHANTOM_TYPE(task_id, size_t)
MAKE_PHANTOM_TYPE(buffer_id, size_t)
MAKE_PHANTOM_TYPE(node_id, size_t)
MAKE_PHANTOM_TYPE(command_id, size_t)
MAKE_PHANTOM_TYPE(collective_group_id, size_t)
MAKE_PHANTOM_TYPE(reduction_id, size_t)
MAKE_PHANTOM_TYPE(host_object_id, size_t)
MAKE_PHANTOM_TYPE(hydration_id, size_t);
MAKE_PHANTOM_TYPE(transfer_id, size_t)


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
	throw_exception,
};

} // namespace celerity::detail
