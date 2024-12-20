#pragma once

#include "version.h" // required for CELERITY_TRACY_SUPPORT

#if CELERITY_TRACY_SUPPORT

#include "types.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>


namespace celerity::detail::tracy_detail {

// This is intentionally not an atomic, as parts of Celerity (= live_executor) expect it not to change after runtime startup.
// We start with `full` tracing to see the runtime startup trigger (i.e. buffer / queue construction), and adjust the setting in runtime::runtime() immediately
// after parsing the config.
inline tracy_mode g_tracy_mode = tracy_mode::full; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

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
enum lane_order : int32_t {
	thread = 0,                        // offset by named_threads::thread_type
	thread_max = 100'000 - 1,          //
	immediate_lane = 100'000,          // immediate pseudo-lane (1)
	alloc_lane = 100'001,              // alloc fiber (1)
	host_first_lane = 110'000,         // host lane (0..100)
	first_device_first_lane = 120'000, // first lane (0.100) of first device (0..100)
	num_lanes_per_device = 100,        //
	send_receive_first_lane = 130'000, // first send/receive pseudo-lane (any number)
};

/// Tracy requires thread and fiber names to be live for the duration of the program, so if they are formatted dynamically, we need to leak them.
inline const char* leak_name(const std::string& name) {
	auto* leaked = malloc(name.size() + 1); // NOLINT
	memcpy(leaked, name.data(), name.size() + 1);
	return static_cast<const char*>(leaked);
}

inline void set_thread_name_and_order(const std::string& name, const int32_t index) {
	const int32_t order = tracy_detail::lane_order::thread + index;
	assert(order <= static_cast<int32_t>(tracy_detail::lane_order::thread_max));
	tracy::SetThreadNameWithHint(leak_name(name), order);
}

} // namespace celerity::detail::tracy_detail

namespace celerity::detail {

enum class trace_color : std::underlying_type_t<tracy::Color::ColorType> {
	generic_red = tracy::Color::Red,
	generic_green = tracy::Color::Green,
	generic_blue = tracy::Color::Blue,
	generic_yellow = tracy::Color::Yellow,

	buffer_ctor = tracy::Color::DarkSlateBlue,
	buffer_dtor = tracy::Color::DarkCyan,

	cuda_memcpy = tracy::Color::ForestGreen,
	cuda_memcpy_1d = cuda_memcpy,
	cuda_memcpy_2d = cuda_memcpy,
	cuda_memcpy_3d = cuda_memcpy,
	cuda_record_event = tracy::Color::ForestGreen,

	distr_queue_ctor = tracy::Color::DarkSlateBlue,
	distr_queue_dtor = tracy::Color::DarkCyan,
	distr_queue_slow_full_sync = tracy::Color::Red2,
	distr_queue_submit = tracy::Color::Orange3,

	executor_fetch = tracy::Color::Gray,
	executor_idle = tracy::Color::SlateGray,
	executor_issue = tracy::Color::Blue,
	executor_issue_copy = tracy::Color::Green4,
	executor_issue_device_kernel = tracy::Color::Yellow2,
	executor_make_accessor_info = tracy::Color::Magenta3,
	executor_oob_check = tracy::Color::Red,
	executor_oob_init = executor_oob_check,
	executor_retire = tracy::Color::Brown,
	executor_starve = tracy::Color::DarkSlateGray,

	host_object_ctor = tracy::Color::DarkSlateBlue,
	host_object_dtor = tracy::Color::DarkCyan,

	iggen_allocate = tracy::Color::Teal,
	iggen_anticipate = iggen_allocate,
	iggen_coherence = tracy::Color::Red2,
	iggen_launch_kernel = tracy::Color::Blue2,
	iggen_perform_buffer_access = tracy::Color::Red3,
	iggen_satisfy_buffer_requirements = tracy::Color::ForestGreen,
	iggen_split_task = tracy::Color::Maroon,

	mpi_finalize = tracy::Color::LightSkyBlue,
	mpi_init = tracy::Color::LightSkyBlue,

	out_of_order_engine_assign = tracy::Color::Blue3,
	out_of_order_engine_complete = tracy::Color::Blue3,
	out_of_order_engine_submit = tracy::Color::Blue3,

	queue_ctor = distr_queue_ctor,
	queue_dtor = distr_queue_dtor,
	queue_fence = tracy::Color::Green2,
	queue_submit = distr_queue_submit,
	queue_wait = distr_queue_slow_full_sync,

	runtime_select_devices = tracy::Color::PaleVioletRed,
	runtime_shutdown = tracy::Color::DimGray,
	runtime_startup = tracy::Color::DarkGray,

	scheduler_buffer_created = tracy::Color::DarkGreen,
	scheduler_buffer_destroyed = scheduler_buffer_created,
	scheduler_buffer_name_changed = tracy::Color::DarkGreen,
	scheduler_build_task = tracy::Color::WebMaroon,
	scheduler_compile_command = tracy::Color::MidnightBlue,
	scheduler_host_object_created = tracy::Color::DarkGreen,
	scheduler_host_object_destroyed = scheduler_host_object_created,
	scheduler_prune = tracy::Color::Gray,

	sycl_init = tracy::Color::Orange2,
	sycl_submit = tracy::Color::Orange2,
};

}

#define CELERITY_DETAIL_IF_TRACY_SUPPORTED(...) __VA_ARGS__

#else

#define CELERITY_DETAIL_IF_TRACY_SUPPORTED(...)

#endif


#define CELERITY_DETAIL_IF_TRACY_ENABLED(...) CELERITY_DETAIL_IF_TRACY_SUPPORTED(if(::celerity::detail::tracy_detail::is_enabled()) { __VA_ARGS__; })
#define CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(...) CELERITY_DETAIL_IF_TRACY_SUPPORTED(if(::celerity::detail::tracy_detail::is_enabled_full()) { __VA_ARGS__; })

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED(TAG, COLOR_NAME)                                                                                                     \
	CELERITY_DETAIL_IF_TRACY_SUPPORTED(ZoneNamedNC(___tracy_scoped_zone, TAG,                                                                                  \
	    static_cast<std::underlying_type_t<::celerity::detail::trace_color>>(::celerity::detail::trace_color::COLOR_NAME),                                     \
	    ::celerity::detail::tracy_detail::is_enabled()))

#define CELERITY_DETAIL_TRACY_ZONE_NAME(...)                                                                                                                   \
	CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(::celerity::detail::tracy_detail::apply_string([&](const auto& n) { ZoneName(n.data(), n.size()); }, __VA_ARGS__))
#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)                                                                                                                   \
	CELERITY_DETAIL_IF_TRACY_ENABLED_FULL(::celerity::detail::tracy_detail::apply_string([&](const auto& t) { ZoneText(t.data(), t.size()); }, __VA_ARGS__))

#define CELERITY_DETAIL_TRACY_ZONE_SCOPED_V(TAG, COLOR_NAME, ...)                                                                                              \
	CELERITY_DETAIL_TRACY_ZONE_SCOPED(TAG, COLOR_NAME);                                                                                                        \
	CELERITY_DETAIL_TRACY_ZONE_NAME(__VA_ARGS__);

#define CELERITY_DETAIL_SET_CURRENT_THREAD_NAME_AND_ORDER(NAME, ORDER)                                                                                         \
	CELERITY_DETAIL_IF_TRACY_ENABLED(::celerity::detail::tracy_detail::set_thread_name_and_order(NAME, ORDER))
