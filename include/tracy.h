#pragma once

#if CELERITY_TRACY_SUPPORT

#include "config.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

namespace celerity::detail::tracy_detail {

// This is intentionally not an atomic, as parts of Celerity (= live_executor) expect it not to change after runtime startup.
// We start with `full` tracing to see the runtime startup trigger (i.e. buffer / queue construction), and adjust the setting in runtime::runtime() immediately
// after parsing the config.
inline tracy_mode g_tracy_mode = tracy_mode::full;

/// Tracy is enabled via environment variable, either in fast or full mode.
inline bool is_enabled() { return g_tracy_mode != tracy_mode::off; }

/// Tracy is enabled via environment variable, in full mode.
inline bool is_enabled_full() { return g_tracy_mode == tracy_mode::full; }

template <typename Value>
struct plot {
	const char* identifier = nullptr;
	Value last_value = 0;

	explicit plot(const char* const identifier) : identifier(identifier) {
		TracyPlot(identifier, static_cast<Value>(0));
		TracyPlotConfig(identifier, tracy::PlotFormatType::Number, true /* step */, true /* fill*/, 0);
	}

	void update(const Value value_in) {
		const auto value = static_cast<Value>(value_in);
		if(value != last_value) {
			TracyPlot(identifier, value);
			last_value = value;
		}
	}
};

/// Helper to pass fmt::formatted strings to Tracy's (pointer, size) functions.
template <typename ApplyFn, typename... FmtParams, std::enable_if_t<(sizeof...(FmtParams) > 0), int> = 0>
void apply_string(const ApplyFn& apply, fmt::format_string<FmtParams...> fmt_string, FmtParams&&... fmt_args) {
	apply(fmt::format(fmt_string, std::forward<FmtParams>(fmt_args)...));
}

template <typename ApplyFn, typename... FmtParams>
void apply_string(const ApplyFn& apply, const std::string_view& string) {
	apply(string);
}

/// Base for sorting keys for the visual order of threads in Tracy. Higher = further down. Order between duplicate keys is automatic per Tracy terms.
enum thread_order : int32_t {
	scheduler = 0x10000,               // scheduler thread (1)
	executor = 0x10001,                // executor thread (1)
	thread_queue = 0x10002,            // thread_queue instance (any number, no internal order)
	immediate_lane = 0x10003,          // immediate pseudo-lane (1)
	alloc_lane = 0x10004,              // alloc fiber (1)
	host_first_lane = 0x10100,         // host lane (0..256)
	first_device_first_lane = 0x10200, // first lane (0.256) of first device (0..256)
	num_lanes_per_device = 0x100,
	send_receive_first_lane = 0x21000, // first send/receive pseudo-lane (any number)
};

/// Tracy requires thread and fiber names to be live for the duration of the program, so if they are formatted dynamically, we need to leak them.
inline const char* leak_name(const std::string& name) {
	auto* leaked = malloc(name.size() + 1);
	memcpy(leaked, name.data(), name.size() + 1);
	return static_cast<const char*>(leaked);
}

} // namespace celerity::detail::tracy_detail

#define CELERITY_DETAIL_IF_TRACY_SUPPORTED(...) __VA_ARGS__

#else

#define CELERITY_DETAIL_IF_TRACY_SUPPORTED(...)

#endif


#define CELERITY_DETAIL_IF_TRACY_ENABLED(...) CELERITY_DETAIL_IF_TRACY_SUPPORTED(if(::celerity::detail::tracy_detail::is_enabled()) { __VA_ARGS__; })
#define CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(...) CELERITY_DETAIL_IF_TRACY_SUPPORTED(if(::celerity::detail::tracy_detail::is_enabled_full()) { __VA_ARGS__; })

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED(TAG, COLOR_NAME)                                                                                                     \
	CELERITY_DETAIL_IF_TRACY_SUPPORTED(ZoneNamedNC(___tracy_scoped_zone, TAG, ::tracy::Color::COLOR_NAME, ::celerity::detail::tracy_detail::is_enabled()))

#define CELERITY_DETAIL_TRACY_ZONE_NAME(...)                                                                                                                   \
	CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(::celerity::detail::tracy_detail::apply_string([&](const auto& n) { ZoneName(n.data(), n.size()); }, __VA_ARGS__))
#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)                                                                                                                   \
	CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(::celerity::detail::tracy_detail::apply_string([&](const auto& t) { ZoneText(t.data(), t.size()); }, __VA_ARGS__))

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED_V(TAG, COLOR_NAME, ...)                                                                                              \
	CELERITY_DETAIL_TRACY_ZONE_SCOPED(TAG, COLOR_NAME);                                                                                                        \
	CELERITY_DETAIL_TRACY_ZONE_NAME(__VA_ARGS__);

#define CELERITY_DETAIL_TRACY_SET_THREAD_NAME_AND_ORDER(NAME, ORDER) CELERITY_DETAIL_IF_TRACY_ENABLED(::tracy::SetThreadNameWithHint(NAME, ORDER))
