#include "named_threads.h"

#include <cassert>

#include <fmt/format.h>

#include "affinity.h"
#include "tracy.h"

namespace celerity::detail::named_threads {

thread_type task_type_device_submitter(const uint32_t n) {
	assert(n < thread_type_step);
	return thread_type(static_cast<uint32_t>(thread_type::first_device_submitter) + n); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}
thread_type task_type_host_queue(const uint32_t n) {
	assert(n < thread_type_step);
	return thread_type(static_cast<uint32_t>(thread_type::first_host_queue) + n); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}
thread_type task_type_test(const uint32_t n) {
	assert(n < thread_type_step);
	return thread_type(static_cast<uint32_t>(thread_type::first_test) + n); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}

std::string thread_type_to_string(const thread_type t_type) {
	switch(t_type) {
	case thread_type::application: return "cy-application";
	case thread_type::scheduler: return "cy-scheduler";
	case thread_type::executor: return "cy-executor";
	case thread_type::alloc: return "cy-alloc";
	default: break;
	}
	if(t_type >= thread_type::first_device_submitter && t_type < thread_type::first_host_queue) { //
		return fmt::format("cy-dev-sub-{}", static_cast<uint32_t>(t_type) - static_cast<uint32_t>(thread_type::first_device_submitter));
	}
	if(t_type >= thread_type::first_host_queue && t_type < thread_type::first_test) { //
		return fmt::format("cy-host-{}", static_cast<uint32_t>(t_type) - static_cast<uint32_t>(thread_type::first_host_queue));
	}
	if(t_type >= thread_type::first_test && t_type <= thread_type::max) { //
		return fmt::format("cy-test-{}", static_cast<uint32_t>(t_type) - static_cast<uint32_t>(thread_type::first_test));
	}
	return fmt::format("unknown({})", static_cast<uint32_t>(t_type));
}

// Sets the name for the invoking thread to its canonical string representation using OS-specific functions, if available
// Has a per-platform implementation in the platform-specific files
void set_current_thread_name(const thread_type t_type);

void name_and_pin_and_order_this_thread(const thread_type t_type) {
	set_current_thread_name(t_type);
	thread_pinning::pin_this_thread(t_type);
	CELERITY_DETAIL_TRACY_SET_THREAD_NAME_AND_ORDER(t_type);
}

} // namespace celerity::detail::named_threads
