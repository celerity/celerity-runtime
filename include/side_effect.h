#pragma once

#include <type_traits>

#include <spdlog/fmt/fmt.h>

#include "handler.h"
#include "host_object.h"


namespace celerity::detail {

template <experimental::side_effect_order Order>
struct side_effect_order_tag {
	inline static constexpr experimental::side_effect_order order = Order;
};

} // namespace celerity::detail

namespace celerity::experimental {

enum class side_effect_order { relaxed, exclusive, sequential };

inline constexpr detail::side_effect_order_tag<side_effect_order::relaxed> relaxed_order;
inline constexpr detail::side_effect_order_tag<side_effect_order::exclusive> exclusive_order;
inline constexpr detail::side_effect_order_tag<side_effect_order::sequential> sequential_order;

/**
 * Provides access to a `host_object` through capture in a `host_task`. Inside the host task kernel, the internal state of the host object can be accessed
 * through the `*` or `->` operators. This behavior is similar to accessors on buffers.
 */
template <typename T, side_effect_order Order = side_effect_order::sequential>
class side_effect {
  public:
	using object_type = typename host_object<T>::object_type;
	constexpr static inline side_effect_order order = Order;

	explicit side_effect(const host_object<T>& object, handler& cgh, detail::side_effect_order_tag<Order> = {}) : m_object{object} {
		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = static_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(object.get_id(), order);
		}
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, object_type>& operator*() const {
		return *m_object.get_object();
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, object_type>* operator->() const {
		return m_object.get_object();
	}

  private:
	host_object<T> m_object;
};

template <typename T>
side_effect(const host_object<T>&, handler&) -> side_effect<T>;

template <typename T, side_effect_order Order>
side_effect(const host_object<T>&, handler&, detail::side_effect_order_tag<Order>) -> side_effect<T, Order>;

} // namespace celerity::experimental


template <>
struct fmt::formatter<celerity::experimental::side_effect_order> {
	using side_effect_order = celerity::experimental::side_effect_order;

	constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

	template <typename FormatContext>
	auto format(const side_effect_order seo, FormatContext& ctx) {
		switch(seo) {
		case side_effect_order::relaxed: fmt::format_to(ctx.out(), "relaxed"); break;
		case side_effect_order::exclusive: fmt::format_to(ctx.out(), "exclusive"); break;
		case side_effect_order::sequential: fmt::format_to(ctx.out(), "sequential"); break;
		}
		return ctx.out();
	}
};
