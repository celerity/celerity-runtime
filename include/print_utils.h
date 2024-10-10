#pragma once

#include "backend/sycl_backend.h"
#include "command.h"
#include "grid.h"
#include "intrusive_graph.h"
#include "nd_memory.h"
#include "ranges.h"
#include "types.h"
#include "utils.h"

#include <chrono>
#include <type_traits>

#include <fmt/format.h>


template <typename Interface, int Dims>
struct fmt::formatter<celerity::detail::coordinate<Interface, Dims>> : fmt::formatter<size_t> {
	format_context::iterator format(const Interface& coord, format_context& ctx) const {
		auto out = ctx.out();
		*out++ = '[';
		for(int d = 0; d < Dims; ++d) {
			if(d != 0) *out++ = ',';
			out = formatter<size_t>::format(coord[d], ctx);
		}
		*out++ = ']';
		return out;
	}
};

template <int Dims>
struct fmt::formatter<celerity::id<Dims>> : fmt::formatter<celerity::detail::coordinate<celerity::id<Dims>, Dims>> {};

template <int Dims>
struct fmt::formatter<celerity::range<Dims>> : fmt::formatter<celerity::detail::coordinate<celerity::range<Dims>, Dims>> {};

template <int Dims>
struct fmt::formatter<celerity::detail::box<Dims>> : fmt::formatter<celerity::id<Dims>> {
	format_context::iterator format(const celerity::detail::box<Dims>& box, format_context& ctx) const {
		auto out = ctx.out();
		out = formatter<celerity::id<Dims>>::format(box.get_min(), ctx);
		out = std::copy_n(" - ", 3, out);
		out = formatter<celerity::id<Dims>>::format(box.get_max(), ctx);
		return out;
	}
};

template <int Dims>
struct fmt::formatter<celerity::detail::region<Dims>> : fmt::formatter<celerity::detail::box<Dims>> {
	format_context::iterator format(const celerity::detail::region<Dims>& region, format_context& ctx) const {
		auto out = ctx.out();
		*out++ = '{';
		for(size_t i = 0; i < region.get_boxes().size(); ++i) {
			if(i != 0) out = std::copy_n(", ", 2, out);
			out = formatter<celerity::detail::box<Dims>>::format(region.get_boxes()[i], ctx);
		}
		*out++ = '}';
		return out;
	}
};

template <int Dims>
struct fmt::formatter<celerity::subrange<Dims>> : fmt::formatter<celerity::id<Dims>> {
	format_context::iterator format(const celerity::subrange<Dims>& sr, format_context& ctx) const {
		auto out = ctx.out();
		out = formatter<celerity::id<Dims>>::format(sr.offset, ctx);
		out = std::copy_n(" + ", 3, out);
		out = formatter<celerity::id<Dims>>::format(celerity::id(sr.range), ctx); // cast to id to avoid multiple inheritance
		return out;
	}
};

template <int Dims>
struct fmt::formatter<celerity::chunk<Dims>> : fmt::formatter<celerity::subrange<Dims>> {
	format_context::iterator format(const celerity::chunk<Dims>& chunk, format_context& ctx) const {
		auto out = ctx.out();
		out = fmt::formatter<celerity::subrange<Dims>>::format(celerity::subrange(chunk.offset, chunk.range), ctx);
		out = std::copy_n(" : ", 3, out);
		out = formatter<celerity::id<Dims>>::format(celerity::id(chunk.global_size), ctx); // cast to id to avoid multiple inheritance
		return out;
	}
};


// TODO prefix type aliases like in GDB pretty_printers (requires removing explicit prefixes elsewhere in the code)
#define CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(TYPE_ALIAS)                                                                              \
	template <>                                                                                                                                                \
	struct fmt::formatter<celerity::detail::TYPE_ALIAS> : fmt::formatter<celerity::detail::TYPE_ALIAS::value_type> {};

CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(task_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(buffer_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(node_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(command_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(collective_group_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(reduction_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(host_object_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(hydration_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(memory_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(device_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(raw_allocation_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(instruction_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS(message_id)

#undef CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_STRONG_TYPE_ALIAS


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
struct fmt::formatter<celerity::detail::allocation_id> {
	constexpr format_parse_context::iterator parse(format_parse_context& ctx) { return ctx.begin(); }

	format_context::iterator format(const celerity::detail::allocation_id aid, format_context& ctx) const {
		if(aid == celerity::detail::null_allocation_id) { return fmt::format_to(ctx.out(), "null"); }
		return fmt::format_to(ctx.out(), "M{}.A{}", aid.get_memory_id(), aid.get_raw_allocation_id());
	}
};

template <>
struct fmt::formatter<celerity::detail::transfer_id> {
	constexpr format_parse_context::iterator parse(format_parse_context& ctx) { return ctx.begin(); }

	format_context::iterator format(const celerity::detail::transfer_id& trid, format_context& ctx) const {
		const auto [tid, bid, rid] = trid;
		auto out = ctx.out();
		if(rid != celerity::detail::no_reduction_id) {
			out = fmt::format_to(out, "T{}.B{}.R{}", tid, bid, rid);
		} else {
			out = fmt::format_to(out, "T{}.B{}", tid, bid);
		}
		return ctx.out();
	}
};

template <>
struct fmt::formatter<celerity::detail::command_type> : fmt::formatter<std::string_view> {
	format_context::iterator format(const celerity::detail::command_type type, format_context& ctx) const {
		const auto repr = [=]() -> std::string_view {
			switch(type) {
			case celerity::detail::command_type::epoch: return "epoch";
			case celerity::detail::command_type::horizon: return "horizon";
			case celerity::detail::command_type::execution: return "execution";
			case celerity::detail::command_type::push: return "push";
			case celerity::detail::command_type::await_push: return "await push";
			case celerity::detail::command_type::reduction: return "reduction";
			case celerity::detail::command_type::fence: return "fence";
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
