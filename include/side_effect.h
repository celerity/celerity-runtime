#pragma once

#include "handler.h"
#include "host_object.h"


namespace celerity::experimental {

template <typename T, side_effect_order Order = side_effect_order::sequential>
class side_effect;

template <typename T, side_effect_order Order>
class side_effect {
  public:
	using object_type = std::remove_reference_t<T>;
	constexpr static inline side_effect_order order = Order;

	explicit side_effect(const host_object<T>& object, handler& cgh) : object{object} {
		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = static_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(object.get_id(), order);
		}
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, object_type>& operator*() const {
		return *object.get_object();
	}

	template <typename U = T>
	std::enable_if_t<!std::is_void_v<U>, object_type>* operator->() const {
		return object.get_object();
	}

  private:
	host_object<T> object;
};

template <typename T>
side_effect(const host_object<T>&, handler&) -> side_effect<T>;

} // namespace celerity::experimental