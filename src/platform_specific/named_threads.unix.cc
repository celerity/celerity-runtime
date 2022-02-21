#include "named_threads.h"

#include <cassert>
#include <type_traits>

#include <pthread.h>

namespace celerity::detail {

static_assert(std::is_same_v<std::thread::native_handle_type, pthread_t>, "Unexpected native thread handle type");

constexpr auto PTHREAD_MAX_THREAD_NAME_LEN = 16;

static inline void set_thread_name_unix(const pthread_t thread_handle, const std::string& name) {
	auto truncated_name = name;
	truncated_name.resize(PTHREAD_MAX_THREAD_NAME_LEN - 1); // -1 because of null terminator
	[[maybe_unused]] const auto res = pthread_setname_np(thread_handle, truncated_name.c_str());
	assert(res == 0 && "Failed to set thread name");
}

static inline std::string get_thread_name_unix(const pthread_t thread_handle) {
	auto name = std::string(PTHREAD_MAX_THREAD_NAME_LEN, '\0');
	[[maybe_unused]] const auto res = pthread_getname_np(thread_handle, name.data(), name.size());
	assert(res == 0 && "Failed to get thread name");
	return name.c_str(); // Strip null terminator
}

void set_thread_name(std::thread& thread, const std::string& name) {
	set_thread_name_unix(thread.native_handle(), name);
}

void set_current_thread_name(const std::string& name) {
	set_thread_name_unix(pthread_self(), name);
}

std::string get_thread_name(std::thread& thread) {
	return get_thread_name_unix(thread.native_handle());
}

std::string get_current_thread_name() {
	return get_thread_name_unix(pthread_self());
}

} // namespace celerity::detail
