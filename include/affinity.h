#pragma once

#include <cstdint>

namespace celerity {
namespace detail {

	uint32_t affinity_cores_available();

	/* a priori we need 3 threads, plus 1 for parallel-task workers and at least one more for host-task.
	 This depends on the application invoking celerity. */
	constexpr static uint64_t min_cores_needed = 5;

} // namespace detail
} // namespace celerity
