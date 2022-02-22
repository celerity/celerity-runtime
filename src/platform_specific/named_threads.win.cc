#include "named_threads.h"

#include <cassert>
#include <cwchar>
#include <type_traits>

#include <windows.h>

namespace celerity::detail {

static_assert(std::is_same_v<std::thread::native_handle_type, HANDLE>, "Unexpected native thread handle type");

template <typename SrcT, typename DstT>
static inline DstT convert_string(const SrcT& str) {
	static_assert(
	    (std::is_same_v<SrcT, std::string> && std::is_same_v<DstT, std::wstring>) || (std::is_same_v<SrcT, std::wstring> && std::is_same_v<DstT, std::string>),
	    "Unsupported string type");

	constexpr auto converter = [](typename DstT::value_type* dst, const typename SrcT::value_type** src, const std::size_t len, std::mbstate_t* ps) {
		if constexpr(std::is_same_v<SrcT, std::string>) {
			return std::mbsrtowcs(dst, src, len, ps);
		} else {
			return std::wcsrtombs(dst, src, len, ps);
		}
	};

	const auto* src = str.c_str();
	auto mbstate = std::mbstate_t{};
	const auto len = converter(nullptr, &src, 0, &mbstate);
	auto dst = DstT(len, L'\0'); // Automatically includes space for the null terminator
	converter(dst.data(), &src, dst.size(), &mbstate);
	return dst;
}

static inline void set_thread_name_windows(const HANDLE thread_handle, const std::string& name) {
	const auto wname = convert_string<std::string, std::wstring>(name);
	[[maybe_unused]] const auto res = SetThreadDescription(thread_handle, wname.c_str());
	assert(SUCCEEDED(res) && "Failed to set thread name");
}

static inline std::string get_thread_name_windows(const HANDLE thread_handle) {
	PWSTR wname = nullptr;
	const auto res = GetThreadDescription(thread_handle, &wname);
	assert(SUCCEEDED(res) && "Failed to get thread name");
	std::string name;
	if(SUCCEEDED(res)) {
		name = convert_string<std::wstring, std::string>(wname);
		LocalFree(wname);
	}
	return name;
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
