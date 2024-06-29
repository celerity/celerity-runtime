#pragma once

#include <stdexcept>

#include "legacy_backend/type.h"

namespace celerity::detail::legacy_backend_detail {

template <legacy_backend::type Type>
struct backend_operations {
	template <typename... Args>
	static void memcpy_strided_device(Args&&... args) {
		throw std::runtime_error{"Invalid backend"};
	}
};

} // namespace celerity::detail::legacy_backend_detail
