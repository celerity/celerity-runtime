#include "grid_test_utils.h"

#if CELERITY_DETAIL_HAVE_CAIRO
#include <cairo/cairo.h>
#endif

using namespace celerity;
using namespace celerity::detail;

// forward declarations for functions not exposed in grid.h
namespace celerity::test_utils {

// input: h as an angle in [0,360] and s,l in [0,1] - output: r,g,b in [0,1]
std::array<float, 3> hsl2rgb(const float h, const float s, const float l) {
	constexpr auto hue2rgb = [](const float p, const float q, float t) {
		if(t < 0) t += 1;
		if(t > 1) t -= 1;
		if(t < 1.f / 6) return p + (q - p) * 6 * t;
		if(t < 1.f / 2) return q;
		if(t < 2.f / 3) return p + (q - p) * (2.f / 3 - t) * 6;
		return p;
	};

	if(s == 0) return {l, l, l}; // achromatic

	const auto q = l < 0.5 ? l * (1 + s) : l + s - l * s;
	const auto p = 2 * l - q;
	const auto r = hue2rgb(p, q, h + 1.f / 3);
	const auto g = hue2rgb(p, q, h);
	const auto b = hue2rgb(p, q, h - 1.f / 3);
	return {r, g, b};
}

void render_boxes(const box_vector<2>& boxes, const std::string_view suffix) {
#if CELERITY_DETAIL_HAVE_CAIRO
	const auto env = std::getenv("CELERITY_RENDER_REGIONS");
	if(env == nullptr || env[0] == 0) return;

	constexpr int ruler_width = 30;
	constexpr int ruler_space = 4;
	constexpr int text_margin = 2;
	constexpr int border_start = ruler_width + ruler_space;
	constexpr int cell_size = 20;
	constexpr int border_end = 30;
	constexpr int inset = 1;

	const auto bounds = bounding_box(boxes);
	const auto canvas_width = border_start + static_cast<int>(bounds.get_max()[1]) * cell_size + border_end;
	const auto canvas_height = border_start + static_cast<int>(bounds.get_max()[0]) * cell_size + border_end;

	cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, canvas_width, canvas_height);
	cairo_t* cr = cairo_create(surface);

	cairo_select_font_face(cr, "sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
	cairo_set_font_size(cr, 12);

	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_set_line_width(cr, 1);
	for(int i = 0; i < static_cast<int>(bounds.get_max()[1]) + 1; ++i) {
		const auto x = border_start + 2 * inset + i * cell_size;
		cairo_move_to(cr, static_cast<float>(x) - 0.5f, text_margin);
		cairo_line_to(cr, static_cast<float>(x) - 0.5f, ruler_width);
		cairo_stroke(cr);
		const auto label = fmt::format("{}", i);
		cairo_text_extents_t te;
		cairo_text_extents(cr, label.c_str(), &te);
		cairo_move_to(cr, x + text_margin, text_margin + te.height);
		cairo_show_text(cr, label.c_str());
	}
	for(int i = 0; i < static_cast<int>(bounds.get_max()[0]) + 1; ++i) {
		const auto y = border_start + 2 * inset + i * cell_size;
		cairo_move_to(cr, text_margin, static_cast<float>(y) - 0.5f);
		cairo_line_to(cr, ruler_width, static_cast<float>(y) - 0.5f);
		cairo_stroke(cr);
		const auto label = fmt::format("{}", i);
		cairo_text_extents_t te;
		cairo_text_extents(cr, label.c_str(), &te);
		cairo_move_to(cr, text_margin, y + te.height + text_margin);
		cairo_show_text(cr, label.c_str());
	}

	cairo_set_operator(cr, CAIRO_OPERATOR_HSL_HUE);
	for(size_t i = 0; i < boxes.size(); ++i) {
		const auto hue = static_cast<float>(i) / static_cast<float>(boxes.size());
		const auto [r, g, b] = hsl2rgb(hue, 0.8f, 0.6f);
		cairo_set_source_rgb(cr, r, g, b);
		const auto sr = static_cast<subrange<2>>(boxes[i]);
		const auto x = border_start + 2 * inset + static_cast<int>(sr.offset[1]) * cell_size;
		const auto y = border_start + 2 * inset + static_cast<int>(sr.offset[0]) * cell_size;
		const auto w = static_cast<int>(sr.range[1]) * cell_size - 2 * inset;
		const auto h = static_cast<int>(sr.range[0]) * cell_size - 2 * inset;
		cairo_rectangle(cr, x, y, w, h);
		cairo_fill(cr);
	}

	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	cairo_rectangle(cr, 0, 0, canvas_width, canvas_height);
	cairo_set_operator(cr, CAIRO_OPERATOR_DEST_OVER);
	cairo_fill(cr);

	cairo_destroy(cr);

	const auto test_name = Catch::getResultCapture().getCurrentTestName();
	const auto image_name = fmt::format("{}-{}.png", std::regex_replace(test_name, std::regex("[^a-zA-Z0-9]+"), "-"), suffix);
	cairo_surface_write_to_png(surface, image_name.c_str());
	cairo_surface_destroy(surface);
#else
	(void)boxes;
#endif
}

} // namespace celerity::test_utils
