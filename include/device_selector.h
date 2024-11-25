#pragma once

#include <functional>
#include <variant>
#include <vector>

#include <sycl/sycl.hpp>

namespace celerity::detail {

struct auto_select_devices {};
using device_selector = std::function<int(const sycl::device&)>;
using devices_or_selector = std::variant<auto_select_devices, std::vector<sycl::device>, device_selector>;

} // namespace celerity::detail
