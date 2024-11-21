#include "named_threads.h"

#include "version.h"

#include <cassert>
#include <string>
#include <thread>
#include <type_traits>

#include <pthread.h>


namespace celerity::detail::named_threads {

static_assert(std::is_same_v<std::thread::native_handle_type, pthread_t>, "Unexpected native thread handle type");

constexpr auto PTHREAD_MAX_THREAD_NAME_LEN = 16;

void set_current_thread_name([[maybe_unused]] const thread_type t_type) {
#if CELERITY_DETAIL_HAS_NAMED_THREADS
	auto name = thread_type_to_string(t_type);
	assert(name.size() < PTHREAD_MAX_THREAD_NAME_LEN && "Thread name too long");
	name.resize(PTHREAD_MAX_THREAD_NAME_LEN - 1); // -1 because of null terminator
	[[maybe_unused]] const auto res = pthread_setname_np(pthread_self(), name.c_str());
	assert(res == 0 && "Failed to set thread name");
#endif
}

std::string get_thread_name([[maybe_unused]] const std::thread::native_handle_type thread_handle) {
#if CELERITY_DETAIL_HAS_NAMED_THREADS
	char name[PTHREAD_MAX_THREAD_NAME_LEN] = {};
	[[maybe_unused]] const auto res = pthread_getname_np(thread_handle, static_cast<char*>(name), PTHREAD_MAX_THREAD_NAME_LEN);
	assert(res == 0 && "Failed to get thread name");
	return name; // Automatically strips null terminator
#else
	return {};
#endif
}

std::string get_current_thread_name() { return get_thread_name(pthread_self()); }

} // namespace celerity::detail::named_threads
