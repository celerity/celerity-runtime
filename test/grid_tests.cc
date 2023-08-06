#include "grid.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <random>
#include <regex>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#if CELERITY_DETAIL_HAVE_CAIRO
#include <cairo/cairo.h>
#endif

using namespace celerity;
using namespace celerity::detail;

// forward declarations for functions not exposed in grid.h
namespace celerity::detail::grid_detail {

} // namespace celerity::detail::grid_detail

struct partition_vector_order {
	template <int Dims>
	bool operator()(const std::vector<box<Dims>>& lhs, const std::vector<box<Dims>>& rhs) {
		if(lhs.size() < rhs.size()) return true;
		if(lhs.size() > rhs.size()) return false;
		constexpr box_coordinate_order box_order;
		for(size_t i = 0; i < lhs.size(); ++i) {
			if(box_order(lhs[i], rhs[i])) return true;
			if(box_order(rhs[i], lhs[i])) return false;
		}
		return false;
	}
};

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

void render_boxes(const std::vector<box<2>>& boxes, const std::string_view suffix = "region") {
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


TEST_CASE("split_box dissects boxes as expected - 3d", "[grid]") {
	const box<3> input_box{{0, 0, 0}, {7, 9, 5}};
	const std::vector<std::vector<size_t>> cuts{
	    {0, 4, 8, 12},
	    {8, 9},
	};
	std::vector<box<3>> expected{
	    {{0, 0, 0}, {4, 8, 5}},
	    {{0, 8, 0}, {4, 9, 5}},
	    {{4, 0, 0}, {7, 8, 5}},
	    {{4, 8, 0}, {7, 9, 5}},
	};

	std::vector<box<3>> split;
	grid_detail::dissect_box(input_box, cuts, split, 0);

	std::sort(split.begin(), split.end(), box_coordinate_order());
	std::sort(expected.begin(), expected.end(), box_coordinate_order());
	CHECK(split == expected);
}

template <int MergeDim, int Dims>
void test_directional_merge(std::vector<box<Dims>> unmerged, std::vector<box<Dims>> merged) {
	CAPTURE(MergeDim);
	std::minstd_rand rng(42);
	std::shuffle(unmerged.begin(), unmerged.end(), rng);
	CAPTURE(unmerged);
	auto test = unmerged;
	test.erase(grid_detail::merge_connected_boxes_along_dim<MergeDim, Dims>(test.begin(), test.end()), test.end());
	std::sort(test.begin(), test.end(), box_coordinate_order());
	std::sort(merged.begin(), merged.end(), box_coordinate_order());
	CHECK(test == merged);
}

TEST_CASE("directional merge of non-overlapping boxes - 1d", "[grid]") {
	const std::vector<box<1>> unmerged{
	    {{0}, {2}},
	    {{2}, {4}},
	    {{4}, {8}},
	    {{10}, {12}},
	};
	const std::vector<box<1>> merged{
	    {{0}, {8}},
	    {{10}, {12}},
	};
	test_directional_merge<0>(unmerged, merged);
}

TEST_CASE("directional merge of overlapping boxes - 1d", "[grid]") {
	const std::vector<box<1>> unmerged{
	    {{0}, {6}},
	    {{2}, {4}},
	    {{8}, {12}},
	    {{10}, {16}},
	    {{16}, {18}},
	};
	const std::vector<box<1>> merged{
	    {{0}, {6}},
	    {{8}, {18}},
	};
	test_directional_merge<0>(unmerged, merged);
}

TEST_CASE("directional merge of non-overlapping boxes - 2d", "[grid]") {
	const std::vector<box<2>> unmerged{
	    {{0, 0}, {2, 2}},
	    {{0, 2}, {2, 4}},
	    {{0, 4}, {2, 6}},
	    {{2, 2}, {4, 4}},
	    {{2, 4}, {4, 6}},
	    {{2, 6}, {4, 8}},
	    {{4, 4}, {6, 6}},
	    {{4, 6}, {6, 8}},
	    {{4, 8}, {6, 10}},
	};

	const std::vector<box<2>> merged_dim0{
	    {{0, 0}, {2, 2}},
	    {{0, 2}, {4, 4}},
	    {{0, 4}, {6, 6}},
	    {{2, 6}, {6, 8}},
	    {{4, 8}, {6, 10}},
	};
	test_directional_merge<0>(unmerged, merged_dim0);

	const std::vector<box<2>> merged_dim1{
	    {{0, 0}, {2, 6}},
	    {{2, 2}, {4, 8}},
	    {{4, 4}, {6, 10}},
	};
	test_directional_merge<1>(unmerged, merged_dim1);

	render_boxes(unmerged, "unmerged");
	render_boxes(merged_dim0, "merged-dim0");
	render_boxes(merged_dim1, "merged-dim1");
}

TEST_CASE("directional merge of overlapping boxes - 2d", "[grid]") {
	const std::vector<box<2>> unmerged{
	    {{0, 0}, {12, 3}},
	    {{0, 1}, {12, 4}},
	    {{0, 4}, {12, 6}},
	    {{0, 8}, {12, 10}},
	    {{0, 0}, {3, 12}},
	    {{1, 0}, {4, 12}},
	    {{4, 0}, {6, 12}},
	    {{8, 0}, {10, 12}},
	};

	const std::vector<box<2>> merged_dim0{
	    {{0, 0}, {12, 3}},
	    {{0, 1}, {12, 4}},
	    {{0, 4}, {12, 6}},
	    {{0, 8}, {12, 10}},
	    {{0, 0}, {6, 12}},
	    {{8, 0}, {10, 12}},
	};
	test_directional_merge<0>(unmerged, merged_dim0);

	const std::vector<box<2>> merged_dim1{
	    {{0, 0}, {12, 6}},
	    {{0, 8}, {12, 10}},
	    {{0, 0}, {3, 12}},
	    {{1, 0}, {4, 12}},
	    {{4, 0}, {6, 12}},
	    {{8, 0}, {10, 12}},
	};
	test_directional_merge<1>(unmerged, merged_dim1);

	render_boxes(unmerged, "unmerged");
	render_boxes(merged_dim0, "merged-dim0");
	render_boxes(merged_dim1, "merged-dim1");
}

TEST_CASE("directional merge of non-overlapping 3d boxes", "[grid]") {
	const std::vector<box<3>> unmerged{
	    {{0, 0, 2}, {2, 2, 4}},
	    {{0, 2, 0}, {2, 4, 2}},
	    {{0, 2, 2}, {2, 4, 4}},
	    {{2, 0, 0}, {4, 2, 2}},
	    {{2, 0, 2}, {4, 2, 4}},
	    {{2, 2, 0}, {4, 4, 2}},
	    {{2, 2, 2}, {4, 4, 4}},
	};

	const std::vector<box<3>> merged_dim0{
	    {{0, 0, 2}, {4, 2, 4}},
	    {{0, 2, 0}, {4, 4, 2}},
	    {{0, 2, 2}, {4, 4, 4}},
	    {{2, 0, 0}, {4, 2, 2}},
	};
	test_directional_merge<0>(unmerged, merged_dim0);

	const std::vector<box<3>> merged_dim1{
	    {{0, 2, 0}, {2, 4, 2}},
	    {{0, 0, 2}, {2, 4, 4}},
	    {{2, 0, 0}, {4, 4, 2}},
	    {{2, 0, 2}, {4, 4, 4}},
	};
	test_directional_merge<1>(unmerged, merged_dim1);

	const std::vector<box<3>> merged_dim2{
	    {{0, 0, 2}, {2, 2, 4}},
	    {{0, 2, 0}, {2, 4, 4}},
	    {{2, 0, 0}, {4, 2, 4}},
	    {{2, 2, 0}, {4, 4, 4}},
	};
	test_directional_merge<2>(unmerged, merged_dim2);
}

TEST_CASE("region normalization removes overlaps - 2d", "[grid]") {
	const std::vector<box<2>> overlapping{
	    {{0, 0}, {4, 4}},
	    {{2, 2}, {6, 6}},
	    {{4, 8}, {5, 9}},
	};
	std::vector<box<2>> normalized{
	    {{0, 0}, {2, 4}},
	    {{2, 0}, {4, 6}},
	    {{4, 2}, {6, 6}},
	    {{4, 8}, {5, 9}},
	};

	const auto result = grid_detail::normalize(std::vector(overlapping));
	std::sort(normalized.begin(), normalized.end(), box_coordinate_order());
	CHECK(result == normalized);

	render_boxes(overlapping, "input");
	render_boxes(result, "result");
	render_boxes(normalized, "normalized");
}

TEST_CASE("region normalization maximizes extent of fast dimensions - 2d", "[grid]") {
	const std::vector<box<2>> input{
	    {{0, 0}, {8, 2}},
	    {{0, 2}, {2, 4}},
	    {{6, 2}, {8, 4}},
	    {{0, 4}, {8, 6}},
	};
	std::vector<box<2>> normalized{
	    {{0, 0}, {2, 6}},
	    {{2, 0}, {6, 2}},
	    {{2, 4}, {6, 6}},
	    {{6, 0}, {8, 6}},
	};

	const auto result = grid_detail::normalize(std::vector(input));
	std::sort(normalized.begin(), normalized.end(), box_coordinate_order());
	CHECK(result == normalized);

	render_boxes(input, "input");
	render_boxes(result, "result");
	render_boxes(normalized, "normalized");
}

template <int Dims>
std::vector<box<Dims>> create_random_boxes(const size_t grid_size, const size_t max_box_size, const size_t num_boxes, const uint32_t seed) {
	std::minstd_rand rng(seed);
	std::uniform_int_distribution<size_t> offset_dist(0, grid_size - 1);
	std::binomial_distribution<size_t> range_dist(max_box_size - 1, 0.5);
	std::vector<box<Dims>> boxes;
	while(boxes.size() < num_boxes) {
		subrange<Dims> sr;
		bool inbounds = true;
		for(int d = 0; d < Dims; ++d) {
			sr.offset[d] = offset_dist(rng);
			sr.range[d] = 1 + range_dist(rng);
			inbounds &= sr.offset[d] + sr.range[d] <= grid_size;
		}
		if(inbounds) { boxes.emplace_back(sr); }
	}
	return boxes;
}

TEST_CASE("normalizing randomized box sets - 2d", "[grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 200},
	}));

	const auto input_2d = create_random_boxes<2>(grid_size, max_box_size, num_boxes, 42);
	BENCHMARK(fmt::format("{}, native", label)) { return grid_detail::normalize(std::vector(input_2d)); };

	const auto input_3d = grid_detail::boxes_cast<3>(input_2d);
	BENCHMARK(fmt::format("{}, embedded in 3d", label)) { return grid_detail::normalize(std::vector(input_3d)); };

	const auto normalized_2d = grid_detail::normalize(std::vector(input_2d));
	const auto normalized_3d = grid_detail::normalize(std::vector(input_3d));
	CHECK(normalized_3d == grid_detail::boxes_cast<3>(normalized_2d));

	render_boxes(input_2d, fmt::format("{}-input", label));
	render_boxes(normalized_2d, fmt::format("{}-normalized", label));
}

TEST_CASE("normalizing randomized box sets - 3d", "[grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 200},
	}));

	const auto input_3d = create_random_boxes<3>(grid_size, max_box_size, num_boxes, 42);
	BENCHMARK(fmt::format("{} - native", label)) { return grid_detail::normalize(std::vector(input_3d)); };
	test_utils::black_hole(grid_detail::normalize(std::vector(input_3d))); // to attach a profiler
}

template <int Dims>
std::vector<box<Dims>> create_box_tiling(const size_t n_per_side) {
	const size_t length = 5;
	size_t n_linear = 1;
	for(int d = 0; d < Dims; ++d) {
		n_linear *= n_per_side;
	}
	std::vector<box<Dims>> boxes(n_linear);
	for(size_t i = 0; i < n_linear; ++i) {
		subrange<Dims> sr;
		auto dist_i = i;
		for(int d = 0; d < Dims; ++d) {
			sr.offset[d] = length * (dist_i % n_per_side);
			sr.range[d] = length;
			dist_i /= n_per_side;
		}
		boxes[i] = sr;
	}
	return boxes;
}

TEMPLATE_TEST_CASE_SIG("normalizing a fully mergeable tiling of boxes", "[grid]", ((int Dims), Dims), 1, 2, 3) {
	const auto [label, n] = GENERATE(values<std::tuple<const char*, size_t>>({
	    {"small", 4},
	    {"medium", 50},
	    {"large", 1000},
	}));

	const size_t n_per_side = llrint(pow(n, 1.0 / Dims));

	const auto boxes_nd = create_box_tiling<Dims>(n_per_side);
	const auto normalized_nd = grid_detail::normalize(std::vector(boxes_nd));
	CHECK(normalized_nd.size() == 1);

	BENCHMARK(fmt::format("{}, native", label)) { return grid_detail::normalize(std::vector(boxes_nd)); };

	if constexpr(Dims < 3) {
		const auto boxes_3d = grid_detail::boxes_cast<3>(boxes_nd);
		BENCHMARK(fmt::format("{}, embedded in 3d", label)) { return grid_detail::normalize(std::vector(boxes_3d)); };
	}

	if constexpr(Dims == 2) {
		render_boxes(boxes_nd, fmt::format("{}-input", label));
		render_boxes(normalized_nd, fmt::format("{}-normalized", label));
	}
}

// TODO: benchmark small box sets - we want low constant overhead for the common case

TEST_CASE("region union - 2d", "[grid]") {
	const region<2> ra{{
	    {{0, 0}, {3, 3}},
	    {{4, 0}, {7, 3}},
	    {{0, 7}, {1, 9}},
	    {{4, 7}, {6, 9}},
	}};
	const region<2> rb{{
	    {{2, 3}, {5, 6}},
	    {{6, 3}, {9, 6}},
	    {{1, 7}, {2, 9}},
	    {{4, 7}, {6, 9}},
	}};

	std::vector<box<2>> expected{
	    {{0, 0}, {2, 3}},
	    {{2, 0}, {3, 6}},
	    {{3, 3}, {4, 6}},
	    {{4, 0}, {5, 6}},
	    {{5, 0}, {6, 3}},
	    {{6, 0}, {7, 6}},
	    {{7, 3}, {9, 6}},
	    {{0, 7}, {2, 9}},
	    {{4, 7}, {6, 9}},
	};
	std::sort(expected.begin(), expected.end(), box_coordinate_order());

	const auto result = region_union(ra, rb);
	CHECK(result.get_boxes() == expected);

	render_boxes(ra.get_boxes(), "ra");
	render_boxes(rb.get_boxes(), "rb");
	render_boxes(expected, "expected");
	render_boxes(result.get_boxes(), "result");
}

TEST_CASE("region intersection - 2d", "[grid]") {
	const region<2> ra{{
	    {{2, 2}, {6, 6}},
	    {{6, 2}, {8, 4}},
	    {{8, 0}, {9, 4}},
	    {{0, 12}, {3, 14}},
	    {{2, 9}, {4, 11}},
	}};
	const region<2> rb{{
	    {{3, 4}, {7, 8}},
	    {{7, 1}, {8, 4}},
	    {{8, 2}, {9, 5}},
	    {{2, 9}, {3, 14}},
	}};

	std::vector<box<2>> expected{
	    {{3, 4}, {6, 6}},
	    {{7, 2}, {9, 4}},
	    {{2, 9}, {3, 11}},
	    {{2, 12}, {3, 14}},
	};
	std::sort(expected.begin(), expected.end(), box_coordinate_order());

	const auto result = region_intersection(ra, rb);
	CHECK(result.get_boxes() == expected);

	render_boxes(ra.get_boxes(), "ra");
	render_boxes(rb.get_boxes(), "rb");
	render_boxes(expected, "expected");
	render_boxes(result.get_boxes(), "result");
}

TEST_CASE("region difference - 2d", "[grid]") {
	const region<2> ra{{
	    {{0, 0}, {6, 6}},
	    {{1, 8}, {4, 11}},
	    {{8, 2}, {10, 4}},
	}};
	const region<2> rb{{
	    {{1, 1}, {3, 3}},
	    {{2, 2}, {4, 4}},
	    {{0, 9}, {2, 12}},
	    {{4, 11}, {6, 13}},
	    {{7, 1}, {11, 5}},
	}};

	std::vector<box<2>> expected{
	    {{0, 0}, {1, 6}},
	    {{1, 0}, {3, 1}},
	    {{3, 0}, {4, 2}},
	    {{1, 3}, {2, 6}},
	    {{2, 4}, {4, 6}},
	    {{4, 0}, {6, 6}},
	    {{1, 8}, {2, 9}},
	    {{2, 8}, {4, 11}},
	};
	std::sort(expected.begin(), expected.end(), box_coordinate_order());

	const auto result = region_difference(ra, rb);
	CHECK(result.get_boxes() == expected);

	render_boxes(ra.get_boxes(), "ra");
	render_boxes(rb.get_boxes(), "rb");
	render_boxes(expected, "expected");
	render_boxes(result.get_boxes(), "result");
}

TEST_CASE("region normalization - 0d", "[grid]") {
	std::vector<box<0>> r;
	auto n = r;
	CHECK(grid_detail::normalize(std::vector(r)).empty());
	r.emplace_back();
	CHECK(grid_detail::normalize(std::vector(r)) == std::vector{{box<0>()}});
	r.emplace_back();
	CHECK(grid_detail::normalize(std::vector(r)) == std::vector{{box<0>()}});
}

TEST_CASE("region union - 0d", "[grid]") {
	region<0> empty;
	CHECK(empty.empty());
	region<0> unit{{box<0>{}}};
	CHECK(!unit.empty());
	CHECK(region_union(empty, empty).empty());
	CHECK(!region_union(empty, unit).empty());
	CHECK(!region_union(unit, empty).empty());
	CHECK(!region_union(unit, unit).empty());
}

TEST_CASE("region intersection - 0d", "[grid]") {
	region<0> empty;
	CHECK(empty.empty());
	region<0> unit{{box<0>{}}};
	CHECK(!unit.empty());
	CHECK(region_intersection(empty, empty).empty());
	CHECK(region_intersection(empty, unit).empty());
	CHECK(region_intersection(unit, empty).empty());
	CHECK(!region_intersection(unit, unit).empty());
}

TEST_CASE("region difference - 0d", "[grid]") {
	region<0> empty;
	CHECK(empty.empty());
	region<0> unit{{box<0>{}}};
	CHECK(!unit.empty());
	CHECK(region_difference(empty, empty).empty());
	CHECK(region_difference(empty, unit).empty());
	CHECK(!region_difference(unit, empty).empty());
	CHECK(region_difference(unit, unit).empty());
}

TEST_CASE("performing set operations between randomized regions - 2d", "[grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 100},
	}));

	const std::vector inputs_2d{
	    region(create_random_boxes<2>(grid_size, max_box_size, num_boxes, 13)), region(create_random_boxes<2>(grid_size, max_box_size, num_boxes, 37))};
	const std::vector inputs_3d{region_cast<3>(inputs_2d[0]), region_cast<3>(inputs_2d[1])};

	render_boxes(inputs_2d[0].get_boxes(), fmt::format("{}-input-a", label));
	render_boxes(inputs_2d[1].get_boxes(), fmt::format("{}-input-b", label));

	BENCHMARK(fmt::format("union, {}, native", label)) { return region_union(inputs_2d[0], inputs_2d[1]); };
	BENCHMARK(fmt::format("union, {}, embedded in 3d", label)) { return region_union(inputs_3d[0], inputs_3d[1]); };
	BENCHMARK(fmt::format("intersection, {}, native", label)) { return region_intersection(inputs_2d[0], inputs_2d[1]); };
	BENCHMARK(fmt::format("intersection, {}, embedded in 3d", label)) { return region_intersection(inputs_3d[0], inputs_3d[1]); };
	BENCHMARK(fmt::format("difference, {}, native", label)) { return region_difference(inputs_2d[0], inputs_2d[1]); };
	BENCHMARK(fmt::format("difference, {}, embedded in 3d", label)) { return region_difference(inputs_3d[0], inputs_3d[1]); };

	const auto union_2d = region_union(inputs_2d[0], inputs_2d[1]);
	const auto union_3d = region_union(inputs_3d[0], inputs_3d[1]);
	const auto intersection_2d = region_intersection(inputs_2d[0], inputs_2d[1]);
	const auto intersection_3d = region_intersection(inputs_3d[0], inputs_3d[1]);
	const auto difference_2d = region_difference(inputs_2d[0], inputs_2d[1]);
	const auto difference_3d = region_difference(inputs_3d[0], inputs_3d[1]);

	CHECK(union_3d == region_cast<3>(union_2d));
	CHECK(intersection_3d == region_cast<3>(intersection_2d));
	CHECK(difference_3d == region_cast<3>(difference_2d));

	render_boxes(union_2d.get_boxes(), fmt::format("union-{}", label));
	render_boxes(intersection_2d.get_boxes(), fmt::format("intersection-{}", label));
	render_boxes(difference_2d.get_boxes(), fmt::format("difference-{}", label));
}

TEST_CASE("performing set operations between randomized regions - 3d", "[grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 100},
	}));

	const std::vector inputs_3d{
	    region(create_random_boxes<3>(grid_size, max_box_size, num_boxes, 13)), region(create_random_boxes<3>(grid_size, max_box_size, num_boxes, 37))};

	BENCHMARK(fmt::format("union, {}, native", label)) { return region_union(inputs_3d[0], inputs_3d[1]); };
	BENCHMARK(fmt::format("intersection, {}, native", label)) { return region_intersection(inputs_3d[0], inputs_3d[1]); };
	BENCHMARK(fmt::format("difference, {}, native", label)) { return region_difference(inputs_3d[0], inputs_3d[1]); };

	// to attach a profiler
	test_utils::black_hole(region_union(inputs_3d[0], inputs_3d[1]));
	test_utils::black_hole(region_intersection(inputs_3d[0], inputs_3d[1]));
	test_utils::black_hole(region_difference(inputs_3d[0], inputs_3d[1]));
}

std::vector<box<2>> create_interlocking_boxes(const size_t num_boxes_per_side) {
	std::vector<box<2>> boxes;
	for(size_t i = 0; i < num_boxes_per_side; ++i) {
		boxes.emplace_back(id<2>(i, i), id<2>(i + 1, num_boxes_per_side));
		boxes.emplace_back(id<2>(i + 1, i), id<2>(num_boxes_per_side, i + 1));
	}
	return boxes;
}

TEST_CASE("normalizing a fully mergeable, complex tiling of boxes - 2d", "[grid]") {
	const auto [label, n] = GENERATE(values<std::tuple<const char*, size_t>>({
	    {"small", 10},
	    {"large", 200},
	}));

	const auto boxes_2d = create_interlocking_boxes(n);
	const auto boxes_3d = grid_detail::boxes_cast<3>(boxes_2d);

	BENCHMARK(fmt::format("{}, native", label)) { return grid_detail::normalize(std::vector(boxes_2d)); };
	BENCHMARK(fmt::format("{}, embedded in 3d", label)) { return grid_detail::normalize(std::vector(boxes_3d)); };

	render_boxes(boxes_2d, fmt::format("{}-input", label));
}
