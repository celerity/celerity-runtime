#pragma once

#include <type_traits>

#include "legacy_backend/type.h"

namespace celerity::detail::legacy_backend_detail {

template <legacy_backend::type Type>
struct is_enabled : public std::false_type {};

template <legacy_backend::type Type>
constexpr bool is_enabled_v = is_enabled<Type>::value;

template <legacy_backend::type Type>
struct name {
	static constexpr const char* value = "(unknown)";
};

template <legacy_backend::type Type>
constexpr const char* name_v = name<Type>::value;

template <>
struct is_enabled<legacy_backend::type::generic> : public std::true_type {};

template <>
struct name<legacy_backend::type::generic> {
	static constexpr const char* value = "generic";
};

#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
template <>
struct is_enabled<legacy_backend::type::cuda> : public std::true_type {};
#endif

template <>
struct name<legacy_backend::type::cuda> {
	static constexpr const char* value = "CUDA";
};

} // namespace celerity::detail::legacy_backend_detail
