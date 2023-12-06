#pragma once

#include <string>

#include <spdlog/spdlog.h>

#include "print_utils.h" // any translation unit that needs logging probably also wants pretty-printing
#include "utils.h"


#define CELERITY_LOG(level, ...)                                                                                                                               \
	(::spdlog::should_log(level)                                                                                                                               \
	        ? SPDLOG_LOGGER_CALL(::spdlog::default_logger_raw(), level, "{}{}", ::celerity::detail::active_log_ctx->repr, ::fmt::format(__VA_ARGS__))          \
	        : (void)0)

// TODO Add a macro similar to SPDLOG_ACTIVE_LEVEL, configurable through CMake
#define CELERITY_TRACE(...) CELERITY_LOG(::celerity::detail::log_level::trace, __VA_ARGS__)
#define CELERITY_DEBUG(...) CELERITY_LOG(::celerity::detail::log_level::debug, __VA_ARGS__)
#define CELERITY_INFO(...) CELERITY_LOG(::celerity::detail::log_level::info, __VA_ARGS__)
#define CELERITY_WARN(...) CELERITY_LOG(::celerity::detail::log_level::warn, __VA_ARGS__)
#define CELERITY_ERROR(...) CELERITY_LOG(::celerity::detail::log_level::err, __VA_ARGS__)
#define CELERITY_CRITICAL(...) CELERITY_LOG(::celerity::detail::log_level::critical, __VA_ARGS__)

namespace celerity::detail {

using log_level = spdlog::level::level_enum;

struct log_context {
	std::string repr;

	log_context() = default;

	template <typename... Es>
	explicit log_context(const std::tuple<Es...>& entries) {
		static_assert(sizeof...(Es) % 2 == 0, "log_context requires key/value pairs");
		if constexpr(sizeof...(Es) > 0) {
			repr += "[";
			int i = 0;
			celerity::detail::utils::tuple_for_each_pair(entries, [&](const auto& a, const auto& b) {
				if(i++ > 0) { repr += ", "; }
				fmt::format_to(std::back_inserter(repr), "{}={}", a, b);
			});
			repr += "] ";
		}
	}
};

inline const log_context null_log_ctx;
inline thread_local const log_context* active_log_ctx = &null_log_ctx;

class set_log_context_guard {
  public:
	set_log_context_guard(const log_context& ctx) : m_ctx_before(celerity::detail::active_log_ctx) { celerity::detail::active_log_ctx = &ctx; }
	set_log_context_guard(const set_log_context_guard&) = delete;
	set_log_context_guard(set_log_context_guard&&) = delete;
	set_log_context_guard& operator=(const set_log_context_guard&) = delete;
	set_log_context_guard& operator=(set_log_context_guard&&) = delete;
	~set_log_context_guard() { celerity::detail::active_log_ctx = m_ctx_before; }

  private:
	const log_context* m_ctx_before;
};

} // namespace celerity::detail
