#pragma once

#include <type_traits>

#include "handler.h"
#include "host_object.h"


namespace celerity::experimental {

/**
 * Provides access to a `host_object` through capture in a `host_task`. Inside the host task kernel, the internal state of the host object can be accessed
 * through the `*` or `->` operators. This behavior is similar to accessors on buffers.
 */
template <typename T, side_effect_order Order = side_effect_order::sequential>
class side_effect {
  public:
	using object_type = typename host_object<T>::object_type;
	constexpr static inline side_effect_order order = Order;

	explicit side_effect(const host_object<T>& object, handler& cgh) : m_object{object.get_object()} {
		detail::add_requirement(cgh, object.get_id(), order);
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(object));
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, object_type>& operator*() const {
		return *m_object;
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, object_type>* operator->() const {
		return m_object;
	}

  private:
	object_type* m_object;
};

template <side_effect_order Order>
class side_effect<void, Order> {
  public:
	using object_type = typename host_object<void>::object_type;
	constexpr static inline side_effect_order order = Order;

	explicit side_effect(const host_object<void>& object, handler& cgh) {
		detail::add_requirement(cgh, object.get_id(), order);
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(object));
	}
};

template <typename T>
side_effect(const host_object<T>&, handler&) -> side_effect<T>;

} // namespace celerity::experimental