#include "named_threads.h"

#include <cassert>

#include <windows.h>

namespace celerity::detail {

static inline void set_thread_name_windows(const HANDLE thread_handle, const std::string& name) {
	const auto wname = std::wstring(name.begin(), name.end());
	[[maybe_unused]] const auto res = SetThreadDescription(thread_handle, wname.c_str());
	assert(SUCCEEDED(res) && "Failed to set thread name");
}

static inline std::string get_thread_name_windows(const HANDLE thread_handle) {
	PWSTR name = nullptr;
	const auto res = GetThreadDescription(thread_handle, &name);
	assert(SUCCEEDED(res) && "Failed to get thread name");
	std::string name_str;
	if(SUCCEEDED(res)) {
		name_str = std::string(name, name + wcslen(name));
		LocalFree(name);
	}
	return name_str;
}

void set_thread_name(std::thread& thread, const std::string& name) {
	set_thread_name_windows(thread.native_handle(), name);
}

void set_current_thread_name(const std::string& name) {
	set_thread_name_windows(GetCurrentThread(), name);
}

std::string get_thread_name(std::thread& thread) {
	return get_thread_name_windows(thread.native_handle());
}

std::string get_current_thread_name() {
	return get_thread_name_windows(GetCurrentThread());
}

} // namespace celerity::detail
