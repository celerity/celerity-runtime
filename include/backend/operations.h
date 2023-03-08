#pragma once

#include <stdexcept>

#include "backend/type.h"

namespace celerity::detail::backend_detail {

template <backend::type Type>
struct backend_operations {
	template <typename... Args>
	static void memcpy_strided_device(Args&&... args) {
		throw std::runtime_error{"Invalid backend"};
	}
};

} // namespace celerity::detail::backend_detail