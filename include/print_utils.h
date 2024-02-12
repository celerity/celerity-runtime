#pragma once

#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "types.h"

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
struct fmt::formatter<celerity::detail::allocation_with_offset> {
	constexpr format_parse_context::iterator parse(format_parse_context& ctx) { return ctx.begin(); }

	format_context::iterator format(const celerity::detail::allocation_with_offset& aid, format_context& ctx) const {
		if(aid.offset_bytes > 0) {
			return fmt::format_to(ctx.out(), "{} + {} bytes", aid.id, aid.offset_bytes);
		} else {
			return fmt::format_to(ctx.out(), "{}", aid.id);
		}
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
