#include "named_threads.h"
#include "version.h"

#include <cassert>
#include <type_traits>

#include <pthread.h>

namespace celerity::detail {

static_assert(std::is_same_v<std::thread::native_handle_type, pthread_t>, "Unexpected native thread handle type");

constexpr auto PTHREAD_MAX_THREAD_NAME_LEN = 16;

std::thread::native_handle_type get_current_thread_handle() { return pthread_self(); }

void set_thread_name([[maybe_unused]] const std::thread::native_handle_type thread_handle, [[maybe_unused]] const std::string& name) {
#if CELERITY_DETAIL_HAS_NAMED_THREADS
	auto truncated_name = name;
	truncated_name.resize(PTHREAD_MAX_THREAD_NAME_LEN - 1); // -1 because of null terminator
	[[maybe_unused]] const auto res = pthread_setname_np(thread_handle, truncated_name.c_str());
	assert(res == 0 && "Failed to set thread name");
#endif
}

std::string get_thread_name([[maybe_unused]] const std::thread::native_handle_type thread_handle) {
#if CELERITY_DETAIL_HAS_NAMED_THREADS
	char name[PTHREAD_MAX_THREAD_NAME_LEN] = {};
	[[maybe_unused]] const auto res = pthread_getname_np(thread_handle, name, PTHREAD_MAX_THREAD_NAME_LEN);
	assert(res == 0 && "Failed to get thread name");
	return name; // Automatically strips null terminator
#else
	return {};
#endif
}

} // namespace celerity::detail
