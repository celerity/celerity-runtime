#pragma once

#if CELERITY_ENABLE_TRACY

#include <chrono>

#include "print_utils.h"
#include "types.h"

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#define CELERITY_DETAIL_TRACY_CAT_2(a, b) a##b
#define CELERITY_DETAIL_TRACY_CAT(a, b) CELERITY_DETAIL_TRACY_CAT_2(a, b)
#define CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tag) CELERITY_DETAIL_TRACY_CAT(tag, __COUNTER__)

#define CELERITY_DETAIL_TRACY_ZONE_BEGIN(scoped_ctx, scoped_name, tag, color_name, ...)                                                                        \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	TracyCZoneN(scoped_ctx, tag, true);                                                                                                                        \
	TracyCZoneName(scoped_ctx, name.c_str(), name.size());                                                                                                     \
	TracyCZoneColor(scoped_ctx, tracy::Color::color_name);

#define CELERITY_DETAIL_TRACY_ZONE_END(scoped_ctx) TracyCZoneEnd(scoped_ctx);

#define CELERITY_DETAIL_TRACY_SCOPED_ZONE_2(scoped_name, tag, color_name, ...)                                                                                 \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	ZoneScopedNC((tag), tracy::Color::color_name);                                                                                                             \
	ZoneName(scoped_name.data(), scoped_name.size());

#define CELERITY_DETAIL_TRACY_SCOPED_ZONE(tag, color_name, ...)                                                                                                \
	CELERITY_DETAIL_TRACY_SCOPED_ZONE_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(name_), tag, color_name, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ZONE_TEXT_2(scoped_name, ...)                                                                                                    \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	ZoneText(scoped_name.data(), scoped_name.size());

#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...) CELERITY_DETAIL_TRACY_ZONE_TEXT_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_name_), __VA_ARGS__)

namespace celerity::detail {

struct tracy_async_fiber {
	const char* thread_name; // to allow tracy_context::get_thread from a tracy_fiber
	size_t index;
	std::string fiber_name;
	std::optional<TracyCZoneCtx> current_zone;
	std::chrono::steady_clock::time_point current_zone_begin;
	tracy_async_fiber(const char* const thread_name, const size_t index)
	    : thread_name(thread_name), index(index), fiber_name(fmt::format("{} async ({})", thread_name, index)) {}
};

using tracy_async_lane = tracy_async_fiber*;

tracy_async_lane tracy_acquire_lane(const char* thread_name);
void tracy_release_lane(tracy_async_lane lane);

struct tracy_fiber_scope_guard {
	tracy_fiber_scope_guard() = default;
	tracy_fiber_scope_guard(const tracy_fiber_scope_guard&) = delete;
	tracy_fiber_scope_guard(tracy_fiber_scope_guard&&) = delete;
	tracy_fiber_scope_guard& operator=(const tracy_fiber_scope_guard&) = delete;
	tracy_fiber_scope_guard& operator=(tracy_fiber_scope_guard&&) = delete;
	~tracy_fiber_scope_guard() { TracyFiberLeave; }
};

#define CELERITY_DETAIL_TRACY_DECLARE_ASYNC_LANE(lane) ::celerity::detail::tracy_async_lane lane = nullptr;

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_2(scoped_ctx, scoped_name, thread_name, lane, tag, color_name, ...)                                             \
	(lane) = celerity::detail::tracy_acquire_lane((thread_name));                                                                                              \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	TracyFiberEnter((lane)->fiber_name.c_str());                                                                                                               \
	if((lane)->current_zone.has_value()) {                                                                                                                     \
		TracyCZoneEnd(*(lane)->current_zone);                                                                                                                  \
		(lane)->current_zone = std::nullopt;                                                                                                                   \
	}                                                                                                                                                          \
	TracyCZoneNC(scoped_ctx, (tag), tracy::Color::color_name, true);                                                                                           \
	TracyCZoneName(scoped_ctx, scoped_name.c_str(), scoped_name.size());                                                                                       \
	(lane)->current_zone = scoped_ctx;                                                                                                                         \
	(lane)->current_zone_begin = std::chrono::steady_clock::now();

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN(out_lane, thread_name, tag, color_name, ...)                                                                    \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_ctx_),                                                         \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_name_), thread_name, out_lane, tag, color_name, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED_2(scoped_ctx, scoped_name, scoped_guard, thread_name, lane, tag, color_name, ...)                        \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_2(scoped_ctx, scoped_name, thread_name, lane, tag, color_name, __VA_ARGS__)                                         \
	tracy_fiber_scope_guard scoped_guard;

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(out_lane, thread_name, tag, color_name, ...)                                                             \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_ctx_),                                                  \
	    CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_name_), CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_guard_), thread_name, out_lane, tag,     \
	    color_name, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_SUSPEND() TracyFiberLeave

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME(lane) TracyFiberEnter((lane)->fiber_name.c_str())

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED_2(lane, scoped_guard)                                                                                   \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME(lane)                                                                                                              \
	tracy_fiber_scope_guard scoped_guard;

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED(lane)                                                                                                   \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED_2(lane, CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_guard_))

#define CELERITY_DETAIL_TRACY_ASYNC_ELAPSED_TIME_SECONDS(lane)                                                                                                 \
	(std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - (lane)->current_zone_begin).count())

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT_2(scoped_name, lane, ...)                                                                                        \
	const auto scoped_name = fmt::format(__VA_ARGS__);                                                                                                         \
	TracyCZoneText((lane)->current_zone.value(), scoped_name.data(), scoped_name.size());

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(lane, ...)                                                                                                       \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT_2(CELERITY_DETAIL_TRACY_MAKE_SCOPED_IDENTIFIER(tracy_name_), lane, __VA_ARGS__)

#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_END(lane)                                                                                                             \
	if(lane != nullptr) {                                                                                                                                      \
		TracyFiberEnter(lane->fiber_name.c_str());                                                                                                             \
		if(lane->current_zone.has_value()) {                                                                                                                   \
			TracyCZoneEnd(*lane->current_zone);                                                                                                                \
			lane->current_zone = std::nullopt;                                                                                                                 \
		}                                                                                                                                                      \
		TracyFiberLeave;                                                                                                                                       \
		celerity::detail::tracy_release_lane(lane);                                                                                                            \
	}

#define CELERITY_DETAIL_TRACY_SET_CURRENT_THREAD_NAME(name) tracy::SetThreadName(name);

} // namespace celerity::detail

#else

#define CELERITY_DETAIL_TRACY_DECLARE_ASYNC_LANE(...)
#define CELERITY_DETAIL_TRACY_SCOPED_ZONE(...)
#define CELERITY_DETAIL_TRACY_ZONE_TEXT(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_SUSPEND(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME_SCOPED(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(...)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_ELAPSED_TIME_SECONDS(...) (0.0)
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_END(...)
#define CELERITY_DETAIL_TRACY_SET_CURRENT_THREAD_NAME(...)

#endif
