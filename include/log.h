#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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

	using log_level = spdlog::level::level_enum;

	template <typename... Es>
	struct log_map {
		const std::tuple<Es...>& entries;
		log_map(const std::tuple<Es...>& entries) : entries(entries) {}
	};

	struct log_context {
		std::string value;
		log_context() = default;
		template <typename... Es>
		explicit log_context(const std::tuple<Es...>& entries) {
			static_assert(sizeof...(Es) % 2 == 0, "log_context requires key/value pairs");
			value = fmt::format("[{}] ", log_map{entries});
		}
	};

	inline const std::string null_log_ctx;
	inline thread_local const std::string* active_log_ctx = &null_log_ctx;

	struct log_ctx_setter {
		log_ctx_setter(log_context& ctx) { celerity::detail::active_log_ctx = &ctx.value; }
		~log_ctx_setter() { celerity::detail::active_log_ctx = &celerity::detail::null_log_ctx; }
	};

#define CELERITY_DETAIL_LOG_SET_SCOPED_CTX(ctx)                                                                                                                \
	log_ctx_setter _set_log_ctx_##__COUNTER__ { ctx }

	template <typename Tuple, typename Callback>
	constexpr void tuple_for_each_pair_impl(const Tuple&, Callback&&, std::index_sequence<>) {}

	template <typename Tuple, size_t I1, size_t I2, size_t... Is, typename Callback>
	constexpr void tuple_for_each_pair_impl(const Tuple& tuple, Callback&& cb, std::index_sequence<I1, I2, Is...>) {
		cb(std::get<I1>(tuple), std::get<I2>(tuple));
		tuple_for_each_pair_impl(tuple, cb, std::index_sequence<Is...>{});
	}

	template <typename... Es, typename Callback>
	constexpr void tuple_for_each_pair(const std::tuple<Es...>& tuple, Callback&& cb) {
		static_assert(sizeof...(Es) % 2 == 0, "an even number of entries is required");
		tuple_for_each_pair_impl(tuple, std::forward<Callback>(cb), std::make_index_sequence<sizeof...(Es)>{});
	}

} // namespace detail
} // namespace celerity

template <typename... Es>
struct fmt::formatter<celerity::detail::log_map<Es...>> {
	constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

	template <typename FormatContext>
	auto format(const celerity::detail::log_map<Es...>& map, FormatContext& ctx) {
		auto&& out = ctx.out();
		int i = 0;
		tuple_for_each_pair(map.entries, [&i, &out](auto& a, auto& b) {
			if(i++ > 0) { fmt::format_to(out, ", "); }
			fmt::format_to(out, "{}={}", a, b);
		});
		return out;
	}
};