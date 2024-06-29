#pragma once

#include "grid.h"

#include <functional>
#include <variant>
#include <vector>

#include <sycl/sycl.hpp>

namespace celerity::detail {

class communicator;
class host_queue;

using device_kernel_launcher = std::function<void(sycl::handler& sycl_cgh, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs)>;
using host_task_launcher = std::function<void(const box<3>& execution_range, const communicator* collective_comm)>;
using command_group_launcher = std::variant<device_kernel_launcher, host_task_launcher>;

} // namespace celerity::detail
