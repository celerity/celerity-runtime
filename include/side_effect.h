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
	struct ctor_internal_tag {};

  public:
	using instance_type = typename host_object<T>::instance_type;
	constexpr static inline side_effect_order order = Order;

	explicit side_effect(host_object<T>& object, handler& cgh) : side_effect(ctor_internal_tag{}, object, cgh) {}

	[[deprecated("Creating side_effect from const host_object is deprecated, capture host_object by reference instead")]] explicit side_effect(
	    const host_object<T>& object, handler& cgh)
	    : side_effect(ctor_internal_tag{}, object, cgh) {}

	side_effect(const side_effect& other) : m_instance(other.m_instance) {
		if(detail::cgf_diagnostics::is_available()) { detail::cgf_diagnostics::get_instance().register_side_effect(); }
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, instance_type>& operator*() const {
		return *m_instance;
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, instance_type>* operator->() const {
		return m_instance;
	}

  private:
	instance_type* m_instance;

	side_effect(ctor_internal_tag /* tag */, const host_object<T>& object, handler& cgh) : m_instance{detail::get_host_object_instance(object)} {
		detail::add_requirement(cgh, detail::get_host_object_id(object), order, false);
	}
};

template <side_effect_order Order>
class side_effect<void, Order> {
  public:
	using instance_type = typename host_object<void>::instance_type;
	constexpr static inline side_effect_order order = Order;

	explicit side_effect(const host_object<void>& object, handler& cgh) { detail::add_requirement(cgh, detail::get_host_object_id(object), order, true); }

	// Note: We don't register the side effect with CGF diagnostics b/c it makes little sense to capture void side effects.
};

template <typename T>
side_effect(const host_object<T>&, handler&) -> side_effect<T>;

} // namespace celerity::experimental
