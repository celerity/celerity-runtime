#pragma once

#include <spdlog/spdlog.h>

#include "print_utils.h" // any translation unit that needs logging probably also wants pretty-printing


#define CELERITY_LOG(level, ...) (::spdlog::should_log(level) ? SPDLOG_LOGGER_CALL(::spdlog::default_logger_raw(), level, __VA_ARGS__) : (void)0)

// TODO Add a macro similar to SPDLOG_ACTIVE_LEVEL, configurable through CMake
#define CELERITY_TRACE(...) CELERITY_LOG(::celerity::detail::log_level::trace, __VA_ARGS__)
#define CELERITY_DEBUG(...) CELERITY_LOG(::celerity::detail::log_level::debug, __VA_ARGS__)
#define CELERITY_INFO(...) CELERITY_LOG(::celerity::detail::log_level::info, __VA_ARGS__)
#define CELERITY_WARN(...) CELERITY_LOG(::celerity::detail::log_level::warn, __VA_ARGS__)
#define CELERITY_ERROR(...) CELERITY_LOG(::celerity::detail::log_level::err, __VA_ARGS__)
#define CELERITY_CRITICAL(...) CELERITY_LOG(::celerity::detail::log_level::critical, __VA_ARGS__)

namespace celerity::detail {

using log_level = spdlog::level::level_enum;

} // namespace celerity::detail
