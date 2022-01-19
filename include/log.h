#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

// TODO: Make this configurable through CMake?
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/spdlog.h>

// Enable formatting of types that support operator<<(std::ostream&, T)
#include <spdlog/fmt/ostr.h>

#include "print_utils.h"

#define CELERITY_LOG_SET_SCOPED_CTX(ctx) CELERITY_DETAIL_LOG_SET_SCOPED_CTX(ctx)

#define CELERITY_TRACE(...) SPDLOG_TRACE("{}{}", *celerity::detail::active_log_ctx, fmt::format(__VA_ARGS__))
#define CELERITY_DEBUG(...) SPDLOG_DEBUG("{}{}", *celerity::detail::active_log_ctx, fmt::format(__VA_ARGS__))
#define CELERITY_INFO(...) SPDLOG_INFO("{}{}", *celerity::detail::active_log_ctx, fmt::format(__VA_ARGS__))
#define CELERITY_WARN(...) SPDLOG_WARN("{}{}", *celerity::detail::active_log_ctx, fmt::format(__VA_ARGS__))
#define CELERITY_ERROR(...) SPDLOG_ERROR("{}{}", *celerity::detail::active_log_ctx, fmt::format(__VA_ARGS__))
#define CELERITY_CRITICAL(...) SPDLOG_CRITICAL("{}{}", *celerity::detail::active_log_ctx, fmt::format(__VA_ARGS__))

namespace celerity {
namespace detail {

	using log_map = std::unordered_map<std::string, std::variant<std::string_view, size_t>>;
	using log_level = spdlog::level::level_enum;

	struct log_context {
		std::string value;
		log_context() = default;
		explicit log_context(const log_map& values) { value = fmt::format("[{}] ", values); }
	};

	inline const std::string null_log_ctx;
	inline thread_local const std::string* active_log_ctx = &null_log_ctx;

	struct log_ctx_setter {
		log_ctx_setter(log_context& ctx) { celerity::detail::active_log_ctx = &ctx.value; }
		~log_ctx_setter() { celerity::detail::active_log_ctx = &celerity::detail::null_log_ctx; }
	};

#define CELERITY_DETAIL_LOG_SET_SCOPED_CTX(ctx)                                                                                                                \
	log_ctx_setter _set_log_ctx_##__COUNTER__ { ctx }

} // namespace detail
} // namespace celerity

template <>
struct fmt::formatter<celerity::detail::log_map> {
	constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

	template <typename FormatContext>
	auto format(const celerity::detail::log_map& map, FormatContext& ctx) {
		auto&& out = ctx.out();
		int i = 0;
		for(const auto& [k, v] : map) {
			if(i++ > 0) { fmt::format_to(out, ", "); }
			fmt::format_to(out, "{}=", k);
			if(std::holds_alternative<std::string_view>(v)) {
				fmt::format_to(out, "{}", std::get<std::string_view>(v));
			} else if(std::holds_alternative<size_t>(v)) {
				fmt::format_to(out, "{}", std::get<size_t>(v));
			} else {
				assert(false);
			}
		}
		return out;
	}
};