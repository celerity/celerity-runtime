#include "named_threads.h"

#include <cassert>
#include <cwchar>
#include <thread>
#include <type_traits>

#include <windows.h>


namespace celerity::detail::named_threads {

static_assert(std::is_same_v<std::thread::native_handle_type, HANDLE>, "Unexpected native thread handle type");

static inline std::string convert_string(const std::wstring& str) {
	const auto* src = str.c_str();
	auto mbstate = std::mbstate_t{};
	const auto len = std::wcsrtombs(nullptr, &src, 0, &mbstate);
	auto dst = std::string(len, '\0'); // Automatically includes space for the null terminator
	std::wcsrtombs(dst.data(), &src, dst.size(), &mbstate);
	return dst;
}

static inline std::wstring convert_string(const std::string& str) {
	const auto* src = str.c_str();
	auto mbstate = std::mbstate_t{};
	const auto len = std::mbsrtowcs(nullptr, &src, 0, &mbstate);
	auto dst = std::wstring(len, L'\0'); // Automatically includes space for the null terminator
	std::mbsrtowcs(dst.data(), &src, dst.size(), &mbstate);
	return dst;
}

void set_current_thread_name([[maybe_unused]] const thread_type t_type) {
	const auto wname = convert_string(thread_type_to_string(t_type));
	[[maybe_unused]] const auto res = SetThreadDescription(GetCurrentThread(), wname.c_str());
	assert(SUCCEEDED(res) && "Failed to set thread name");
}

std::string get_thread_name(const std::thread::native_handle_type thread_handle) {
	PWSTR wname = nullptr;
	const auto res = GetThreadDescription(thread_handle, &wname);
	assert(SUCCEEDED(res) && "Failed to get thread name");
	std::string name;
	if(SUCCEEDED(res)) {
		name = convert_string(wname);
		LocalFree(wname); // Will leak if convert_string throws
	}
	return name;
}

std::string get_current_thread_name() { return get_thread_name(GetCurrentThread()); }

} // namespace celerity::detail::named_threads
