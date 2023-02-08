#pragma once

#include <type_traits>

#include "cgf_diagnostics.h"
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
		detail::add_requirement(cgh, object.get_id(), order, false);
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(object));
	}

	side_effect(const side_effect& other) {
		m_object = other.m_object;
		if(detail::cgf_diagnostics::is_available()) { detail::cgf_diagnostics::get_instance().register_side_effect(); }
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
		detail::add_requirement(cgh, object.get_id(), order, true);
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(object));
	}

	// Note: We don't register the side effect with CGF diagnostics b/c it makes little sense to capture void side effects.
};

template <typename T>
side_effect(const host_object<T>&, handler&) -> side_effect<T>;

} // namespace celerity::experimental