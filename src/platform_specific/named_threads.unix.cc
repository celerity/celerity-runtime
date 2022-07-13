#include "named_threads.h"

#include <cassert>
#include <type_traits>

#include <pthread.h>

namespace celerity::detail {

static_assert(std::is_same_v<std::thread::native_handle_type, pthread_t>, "Unexpected native thread handle type");

constexpr auto PTHREAD_MAX_THREAD_NAME_LEN = 16;

std::thread::native_handle_type get_current_thread_handle() { return pthread_self(); }

void set_thread_name(const std::thread::native_handle_type thread_handle, const std::string& name) {
	auto truncated_name = name;
	truncated_name.resize(PTHREAD_MAX_THREAD_NAME_LEN - 1); // -1 because of null terminator
	[[maybe_unused]] const auto res = pthread_setname_np(thread_handle, truncated_name.c_str());
	assert(res == 0 && "Failed to set thread name");
}

std::string get_thread_name(const std::thread::native_handle_type thread_handle) {
	char name[PTHREAD_MAX_THREAD_NAME_LEN] = {};
	[[maybe_unused]] const auto res = pthread_getname_np(thread_handle, name, PTHREAD_MAX_THREAD_NAME_LEN);
	assert(res == 0 && "Failed to get thread name");
	return name; // Automatically strips null terminator
}

} // namespace celerity::detail
