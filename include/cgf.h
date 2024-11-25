#pragma once

#include "grid.h"
#include "hint.h"
#include "range_mapper.h"
#include "ranges.h"
#include "reduction.h"
#include "sycl_wrappers.h"
#include "types.h"

#include <functional>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <sycl/sycl.hpp>

namespace celerity::detail {

class communicator;

struct task_geometry {
	int dimensions = 0;
	range<3> global_size{1, 1, 1};
	id<3> global_offset;
	range<3> granularity{1, 1, 1};
};

struct buffer_access {
	buffer_id bid = -1;
	access_mode mode = access_mode::atomic;
	std::unique_ptr<range_mapper_base> range_mapper;
};

struct host_object_effect {
	host_object_id hoid = -1;
	experimental::side_effect_order order = experimental::side_effect_order::sequential;
};

using device_kernel_launcher = std::function<void(sycl::handler& sycl_cgh, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs)>;
using host_task_launcher = std::function<void(const range<3>& global_range, const box<3>& execution_range, const communicator* collective_comm)>;
using command_group_launcher = std::variant<device_kernel_launcher, host_task_launcher>;

/// Captures the raw contents of a command group function (CGF) invocation as recorded by `celerity::handler`.
/// This is passed on to `task_manager`, which validates the structure and turns it into a TDAG node.
struct raw_command_group {
	std::optional<detail::task_type> task_type; ///< nullopt until a kernel or host task has been submitted
	std::vector<buffer_access> buffer_accesses;
	std::vector<host_object_effect> side_effects;
	std::vector<reduction_info> reductions;
	std::optional<detail::collective_group_id> collective_group_id;
	std::optional<task_geometry> geometry;
	std::optional<detail::command_group_launcher> launcher;
	std::optional<std::string> task_name;
	std::vector<std::unique_ptr<detail::hint_base>> hints;
};

class task_promise {
  public:
	task_promise() = default;
	task_promise(const task_promise&) = delete;
	task_promise(task_promise&&) = delete;
	task_promise& operator=(const task_promise&) = delete;
	task_promise& operator=(task_promise&&) = delete;
	virtual ~task_promise() = default;

	virtual void fulfill() = 0;
	virtual allocation_id get_user_allocation_id() = 0; // TODO move to struct task instead
};

} // namespace celerity::detail
