#pragma once

#include "grid.h"
#include "host_queue.h"

#include <functional>
#include <variant>
#include <vector>

#include <mpi.h>

namespace celerity::detail {

struct async_event {}; // [IDAG placeholder]

using device_kernel_launcher = std::function<void(sycl::handler& sycl_cgh, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs)>;
using host_task_launcher = std::function<async_event(host_queue& q, const box<3>& execution_range, MPI_Comm mpi_comm)>;
using command_group_launcher = std::variant<device_kernel_launcher, host_task_launcher>;

} // namespace celerity::detail
