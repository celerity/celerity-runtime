#pragma once

#include "backend/sycl_backend.h"
#include "command_graph.h"
#include "intrusive_graph.h"
#include "nd_memory.h"
#include "print_utils.h"
#include "utils.h"

#include <chrono>
#include <type_traits>

#include <fmt/format.h>
#include <matchbox.hh>


template <>
struct fmt::formatter<celerity::detail::dependency_kind> : fmt::formatter<std::string_view> {
	format_context::iterator format(const celerity::detail::dependency_kind kind, format_context& ctx) const {
		const auto repr = [=]() -> std::string_view {
			switch(kind) {
			case celerity::detail::dependency_kind::anti_dep: return "anti_dep";
			case celerity::detail::dependency_kind::true_dep: return "true_dep";
			default: return "???";
			}
		}();
		return std::copy(repr.begin(), repr.end(), ctx.out());
	}
};

template <>
struct fmt::formatter<celerity::detail::dependency_origin> : fmt::formatter<std::string_view> {
	format_context::iterator format(const celerity::detail::dependency_origin origin, format_context& ctx) const {
		const auto repr = [=]() -> std::string_view {
			switch(origin) {
			case celerity::detail::dependency_origin::dataflow: return "dataflow";
			case celerity::detail::dependency_origin::collective_group_serialization: return "collective_group_serialization";
			case celerity::detail::dependency_origin::execution_front: return "execution_front";
			case celerity::detail::dependency_origin::last_epoch: return "last_epoch";
			default: return "???";
			}
		}();
		return std::copy(repr.begin(), repr.end(), ctx.out());
	}
};


template <>
struct fmt::formatter<celerity::detail::sycl_backend_type> : fmt::formatter<std::string_view> {
	format_context::iterator format(const celerity::detail::sycl_backend_type type, format_context& ctx) const {
		const auto repr = [=]() -> std::string_view {
			switch(type) {
			case celerity::detail::sycl_backend_type::generic: return "generic";
			case celerity::detail::sycl_backend_type::cuda: return "CUDA";
			default: celerity::detail::utils::unreachable(); // LCOV_EXCL_LINE
			}
		}();
		return std::copy(repr.begin(), repr.end(), ctx.out());
	}
};

namespace celerity::detail {

inline const char* print_command_type(const command& cmd) {
	return matchbox::match(
	    cmd,                                                    //
	    [](const epoch_command&) { return "epoch"; },           //
	    [](const horizon_command&) { return "horizon"; },       //
	    [](const execution_command&) { return "execution"; },   //
	    [](const push_command&) { return "push"; },             //
	    [](const await_push_command&) { return "await push"; }, //
	    [](const reduction_command&) { return "reduction"; },   //
	    [](const fence_command&) { return "fence"; });
}

/// Wrap a `std::chrono::duration` in this to auto-format it as seconds, milliseconds, microseconds, or nanoseconds.
struct as_sub_second {
	template <typename Rep, typename Period>
	as_sub_second(const std::chrono::duration<Rep, Period>& duration) : seconds(std::chrono::duration_cast<std::chrono::duration<double>>(duration)) {}
	std::chrono::duration<double> seconds;
};

/// Wrap a byte count in this to auto-format it as KB / MB / etc.
struct as_decimal_size {
	template <typename Number, std::enable_if_t<std::is_arithmetic_v<Number>, int> = 0>
	as_decimal_size(const Number bytes) : bytes(static_cast<double>(bytes)) {}
	double bytes;
};

/// Wrap a byte-per-second ratio in this to auto-format it as KB/s, MB/s, ...
struct as_decimal_throughput {
	template <typename Rep, typename Period>
	as_decimal_throughput(size_t bytes, const std::chrono::duration<Rep, Period>& duration)
	    : bytes_per_sec(static_cast<double>(bytes) / std::chrono::duration_cast<std::chrono::duration<double>>(duration).count()) {}
	double bytes_per_sec;
};

} // namespace celerity::detail

template <>
struct fmt::formatter<celerity::detail::as_sub_second> : fmt::formatter<double> {
	format_context::iterator format(const celerity::detail::as_sub_second ss, format_context& ctx) const {
		std::string_view unit = " s";
		double unit_time = ss.seconds.count();
		if(unit_time != 0.0) {
			if(std::abs(unit_time) < 1.0) { unit_time *= 1000.0, unit = " ms"; }
			if(std::abs(unit_time) < 1.0) { unit_time *= 1000.0, unit = " µs"; }
			if(std::abs(unit_time) < 1.0) { unit_time *= 1000.0, unit = " ns"; }
		}
		auto out = fmt::formatter<double>::format(unit_time, ctx);
		return std::copy(unit.begin(), unit.end(), out);
	}
};

template <>
struct fmt::formatter<celerity::detail::as_decimal_size> : fmt::formatter<double> {
	format_context::iterator format(const celerity::detail::as_decimal_size bs, format_context& ctx) const {
		std::string_view unit = " bytes";
		double unit_size = static_cast<double>(bs.bytes);
		if(unit_size >= 1000) { unit_size /= 1000, unit = " kB"; }
		if(unit_size >= 1000) { unit_size /= 1000, unit = " MB"; }
		if(unit_size >= 1000) { unit_size /= 1000, unit = " GB"; }
		if(unit_size >= 1000) { unit_size /= 1000, unit = " TB"; }
		auto out = fmt::formatter<double>::format(unit_size, ctx);
		return std::copy(unit.begin(), unit.end(), out);
	}
};

template <>
struct fmt::formatter<celerity::detail::as_decimal_throughput> : fmt::formatter<celerity::detail::as_decimal_size> {
	format_context::iterator format(const celerity::detail::as_decimal_throughput bt, format_context& ctx) const {
		auto out = fmt::formatter<celerity::detail::as_decimal_size>::format(celerity::detail::as_decimal_size(bt.bytes_per_sec), ctx);
		const std::string_view unit = "/s";
		return std::copy(unit.begin(), unit.end(), out);
	}
};

template <>
struct fmt::formatter<celerity::detail::region_layout> {
	constexpr format_parse_context::iterator parse(format_parse_context& ctx) { return ctx.begin(); }

	format_context::iterator format(const celerity::detail::region_layout& layout, format_context& ctx) const {
		return matchbox::match(
		    layout, [&](const celerity::detail::strided_layout& layout) { return fmt::format_to(ctx.out(), "({})", layout.allocation); },
		    [&](const celerity::detail::linearized_layout& layout) { return fmt::format_to(ctx.out(), "+{} bytes", layout.offset_bytes); });
	}
};
