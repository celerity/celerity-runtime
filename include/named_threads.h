#pragma once

#include <cstdint>
#include <string>


namespace celerity::detail::named_threads {

constexpr uint32_t thread_type_step = 10000;

// The threads Celerity interacts with ("application") and creates (everything else), identified for the purpose of naming and pinning.
enum class thread_type : uint32_t {
	application = 0 * thread_type_step,            // pinned
	scheduler = 1 * thread_type_step,              // pinned
	executor = 2 * thread_type_step,               // pinned
	alloc = 3 * thread_type_step,                  //
	first_device_submitter = 4 * thread_type_step, // pinned
	first_host_queue = 5 * thread_type_step,       //
	first_test = 6 * thread_type_step,             //
	max = 7 * thread_type_step,                    //
};
// Builds the n-th thread types of various kinds
thread_type task_type_device_submitter(const uint32_t n);
thread_type task_type_host_queue(const uint32_t n);
thread_type task_type_test(const uint32_t n);

// Converts a thread type to a canoncial string representation
std::string thread_type_to_string(const thread_type t_type);

// Performs naming, pinning and tracy ordering (if enabled for this thread) of the invoking thread
// This should be the first thing called in any thread that is part of the Celerity runtime
void name_and_pin_and_order_this_thread(const thread_type t_type);

} // namespace celerity::detail::named_threads
