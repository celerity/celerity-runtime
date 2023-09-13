#pragma once

#include "grid.h"
#include "ranges.h"

#include <spdlog/fmt/fmt.h>

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
