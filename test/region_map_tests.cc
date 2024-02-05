#include <random>

#if CELERITY_DETAIL_HAVE_CAIRO
#include <cairo/cairo.h>
#endif

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <celerity.h>

#include "ranges.h"
#include "region_map.h"

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

template <typename ValueType, int Dims>
using region_map_impl = region_map_detail::region_map_impl<ValueType, Dims>;

namespace celerity::detail {
struct region_map_testspy {
	template <typename ValueType, int Dims, typename Callback>
	static void traverse(const region_map_impl<ValueType, Dims>& rm, const Callback& cb) {
		auto recurse = [&cb](auto& node, const size_t level, auto& r) -> void {
			for(size_t i = 0; i < node.m_child_boxes.size(); ++i) {
				if(node.contains_leaves()) {
					cb(level, node.m_child_boxes[i], node.get_child_value(i), 0);
				} else {
					cb(level, node.m_child_boxes[i], std::nullopt, node.get_child_node(i).m_child_boxes.size());
					r(node.get_child_node(i), level + 1, r);
				}
			}
		};
		recurse(*rm.m_root, 0, recurse);
	}

	template <typename ValueType, int Dims>
	static size_t get_num_leaf_nodes(const region_map_impl<ValueType, Dims>& rm) {
		size_t num_leaf_nodes = 0;
		traverse(rm,
		    [&num_leaf_nodes](const size_t /* level */, const box<Dims>& /* box */, const std::optional<ValueType>& value, const size_t /* num_children */) {
			    if(value.has_value()) { num_leaf_nodes++; }
		    });
		return num_leaf_nodes;
	}

	template <typename ValueType, int Dims>
	static size_t get_depth(const region_map_impl<ValueType, Dims>& rm) {
		size_t depth = 1;
		traverse(rm, [&depth](const size_t level, const box<Dims>& /* box */, const std::optional<ValueType>& /* value */, const size_t /* num_children */) {
			depth = std::max(depth, level + 1);
		});
		return depth;
	}

	template <typename ValueType, int Dims>
	static double compute_overlap(const region_map_impl<ValueType, Dims>& rm) {
		std::vector<std::vector<box<Dims>>> boxes_by_level;
		traverse(rm, [&boxes_by_level](const size_t level, const box<Dims>& box, const std::optional<ValueType>& /* value */, const size_t /* num_children */) {
			while(boxes_by_level.size() < level + 1) {
				boxes_by_level.push_back({});
			}
			boxes_by_level[level].push_back(box);
		});

		const size_t num_levels = boxes_by_level.size();
		std::vector<region<Dims>> box_union_by_level(num_levels, region<Dims>{});
		size_t total_overlap_area = 0;

		for(size_t l = 0; l < num_levels; ++l) {
			size_t overlap = 0;
			for(auto& b : boxes_by_level[l]) {
				overlap += region_intersection(box_union_by_level[l], b).get_area();
				box_union_by_level[l] = region_union(box_union_by_level[l], b);
			}
			total_overlap_area += overlap;

			// Do some quick sanity checks:
			// - Every level has to cover the whole extent
			CHECK(box_union_by_level[l] == rm.m_extent);
			// - There is no overlap in leaf nodes
			if(l == num_levels - 1) { CHECK(overlap == 0); }
		}

		// We return a percentage value of how much area in the entire rm is overlapping (this may exceed 1)
		return static_cast<double>(total_overlap_area) / (rm.m_extent.get_area() * num_levels);
	}

	template <typename ValueType, int Dims>
	static void erase(region_map_impl<ValueType, Dims>& rm, const box<Dims>& box) {
		rm.erase(box);
	}

	template <typename ValueType, int Dims>
	static void insert(region_map_impl<ValueType, Dims>& rm, const box<Dims>& box, const ValueType& value) {
		rm.insert(box, value);
	}

	template <typename ValueType, int Dims>
	static void try_merge(region_map_impl<ValueType, Dims>& rm, std::vector<typename region_map_impl<ValueType, Dims>::types::entry> candidates) {
		rm.try_merge(std::move(candidates));
	}
};
} // namespace celerity::detail

// FIXME: We should only call this if we are running a single test case (otherwise we just overwrite the image several times)
template <typename ValueType>
void draw(const region_map_impl<ValueType, 2>& rm) {
#if CELERITY_DETAIL_HAVE_CAIRO
	const auto extent = rm.get_extent();
	const float region_map_height = extent[0];
	const float region_map_width = extent[1];
	const float region_map_aspect_ratio = region_map_width / region_map_height;

	const int canvas_size = 1024;
	const int canvas_width = region_map_aspect_ratio < 1 ? static_cast<int>(canvas_size * region_map_aspect_ratio) : canvas_size;
	const int canvas_height = region_map_aspect_ratio < 1 ? canvas_size : static_cast<int>(canvas_size / region_map_aspect_ratio);
	cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, canvas_width, canvas_height);
	cairo_t* cr = cairo_create(surface);

	cairo_select_font_face(cr, "sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
	cairo_set_font_size(cr, 10.0);

	region_map_testspy::traverse(rm, [&](const size_t level, const box<2>& box, const std::optional<int>& value, const size_t num_children) {
		const auto min = box.get_min();
		const auto max = box.get_max();
		const float inset = 3.f;
		const auto y = (static_cast<float>(min[0]) / region_map_height) * static_cast<float>(canvas_height) + static_cast<float>(level) * inset;
		const auto x = (static_cast<float>(min[1]) / region_map_width) * static_cast<float>(canvas_width) + static_cast<float>(level) * inset;
		const auto h = (static_cast<float>(max[0] - min[0]) / region_map_height) * static_cast<float>(canvas_height) - static_cast<float>(level) * 2 * inset;
		const auto w = (static_cast<float>(max[1] - min[1]) / region_map_width) * static_cast<float>(canvas_width) - static_cast<float>(level) * 2 * inset;

		if(value.has_value()) {
			cairo_set_source_rgb(cr, 0.9, 0.0, 0.5);
			cairo_set_dash(cr, nullptr, 0, 0);

			cairo_text_extents_t extents;
			const auto label = fmt::format("{}", *value);
			cairo_text_extents(cr, label.c_str(), &extents);
			cairo_move_to(cr, (x + w / 2) - (extents.width / 2 + extents.x_bearing), (y + h / 2) + (extents.height / 2));
			cairo_show_text(cr, label.c_str());

		} else {
			cairo_set_source_rgb(cr, 0.2, 0.5, 0.8);

			cairo_text_extents_t extents;
			const auto label = fmt::format("{}: {}", box, num_children);
			cairo_text_extents(cr, label.c_str(), &extents);
			cairo_move_to(cr, x + 10, y + (extents.height / 2) + 10);
			cairo_show_text(cr, label.c_str());

			static const double dashed3[] = {1.0};
			cairo_set_dash(cr, &dashed3[0], 1, 0);
		}

		cairo_rectangle(cr, x, y, w, h);
		if(value.has_value()) {
			cairo_stroke(cr);
		} else {
			cairo_stroke_preserve(cr);
			cairo_set_source_rgba(cr, 0.2, 0.5, 0.8, 0.1);
			cairo_fill(cr);
		}
	});

	cairo_destroy(cr);
	cairo_surface_write_to_png(surface, "region_map.png");
	cairo_surface_destroy(surface);
#endif
}

// TODO: We currently mostly do black-box testing of the region map API. Consider adding some tests for inner_node as well.
// Black-box tests should suffice to ensure correctness, but not for testing certain optimizations, for example in-place and localized updates.

// Regression test: An earlier implementation of try_merge kept intermediate merge results around as merge candidates
//                  for the next round. This resulted in attempted merges with boxes that no longer existed.
// NOTE: This test makes assumptions about the order of dimensions in which we probe for merges (dim 0 before dim 1)
TEST_CASE("region_map::try_merge does not attempt to merge intermediate results that no longer exist", "[region_map]") {
	region_map_impl<int, 2> rm(box<2>::full_range({99, 99}), -1);

	std::vector<std::pair<box<2>, int>> entries = {
	    // These first three entries will be merged
	    {{{0, 0}, {33, 66}}, 1},
	    {{{33, 0}, {66, 66}}, 1},
	    {{{66, 0}, {99, 66}}, 1},

	    // The result of the first merge could have been merged with this one,
	    // but once the second merge is done they no longer match
	    {{{0, 66}, {66, 99}}, 1},

	    // Remainder, will not be merged
	    {{{66, 66}, {99, 99}}, 2},
	};

	region_map_testspy::erase(rm, {{0, 0}, {99, 99}}); // Remove default value entry
	for(auto& [box, value] : entries) {
		region_map_testspy::insert(rm, box, value);
	}

	region_map_testspy::try_merge(rm, entries);
}

#define CHECK_RESULTS(results, ...)                                                                                                                            \
	do {                                                                                                                                                       \
		const decltype(results) expected = {__VA_ARGS__};                                                                                                      \
		CHECK(results.size() == expected.size());                                                                                                              \
		for(auto& exp : expected) {                                                                                                                            \
			REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [&exp](auto& r) { return r == exp; }));                                                   \
		}                                                                                                                                                      \
	} while(0)

TEST_CASE("region_map can be moved", "[region_map]") {
	constexpr size_t size = 128;
	const int default_value = -1;
	region_map_impl<int, 1> rm1{box<1>::full_range({size}), default_value};
	rm1.update_box({0, size}, 1337);
	auto results1 = rm1.get_region_values({0, size});
	CHECK_RESULTS(results1, {{0, size}, 1337});

	region_map_impl<int, 1> rm2{std::move(rm1)};
	auto results2 = rm2.get_region_values({0, size});
	CHECK_RESULTS(results2, {{0, size}, 1337});

	region_map_impl<int, 1> rm3{box<1>::full_range({size}), default_value};
	rm3 = std::move(rm2);
	auto results3 = rm3.get_region_values({0, size});
	CHECK_RESULTS(results3, {{0, size}, 1337});
}

TEST_CASE("region_map handles basic operations in 0D", "[region_map]") {
	const int default_value = -1;
	region_map_impl<int, 0> rm{{}, default_value};

	SECTION("query default value") {
		const auto results = rm.get_region_values({0, 1});
		CHECK_RESULTS(results, {{0, 1}, default_value});
	}
}

TEST_CASE("region_map handles basic operations in 1D", "[region_map]") {
	constexpr size_t size = 128;
	const size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 1> rm{box<1>::full_range({size}), default_value};

	SECTION("query default value") {
		const auto results = rm.get_region_values({0, size});
		CHECK_RESULTS(results, {{0, size}, default_value});
	}

	SECTION("query partial default value") {
		const auto results = rm.get_region_values({64, 96});
		CHECK_RESULTS(results, {{64, 96}, default_value});
	}

	SECTION("update full extent") {
		rm.update_box({0, 128}, 1337);
		const auto results = rm.get_region_values({64, 96});
		CHECK_RESULTS(results, {{64, 96}, 1337});
	}

	SECTION("update simple") {
		rm.update_box({0, 64}, 1337);
		const auto results = rm.get_region_values({0, size});
		CHECK_RESULTS(results, {{0, 64}, 1337}, {{64, size}, default_value});
	}

	SECTION("update with split") {
		rm.update_box({32, 96}, 1337);
		const auto results = rm.get_region_values({0, size});
		CHECK_RESULTS(results, {{0, 32}, default_value}, {{32, 96}, 1337}, {{96, size}, default_value});
	}

	SECTION("update multiple") {
		constexpr size_t num_parts = 16;
		constexpr size_t slice = size / num_parts;
		// Iteratively split line into multiple parts
		for(size_t i = 0; i < num_parts; ++i) {
			rm.update_box(box<1>{i * slice, i * slice + slice}, i);
			const auto results = rm.get_region_values({0, size});
			REQUIRE_LOOP(results.size() == static_cast<size_t>(i + (i < (num_parts - 1) ? 2 : 1)));
			for(size_t j = 0; j < i + 1; ++j) {
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [j, slice](auto& r) {
					return r == std::pair{box<1>{j * slice, j * slice + slice}, j};
				}));
			}
			if(i < num_parts - 1) {
				// Check that original value still exists
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [i, slice](auto& r) {
					return r == std::pair{box<1>{(i + 1) * slice, size}, std::numeric_limits<size_t>::max()};
				}));
			}
		}
	}
}

TEST_CASE("region_map handles basic operations in 2D", "[region_map]") {
	constexpr size_t height = 128;
	constexpr size_t width = 192;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 2> rm{box<2>::full_range({height, width}), default_value};

	SECTION("query default value") {
		const auto results = rm.get_region_values({{0, 0}, {height, width}});
		CHECK_RESULTS(results, {{{0, 0}, {height, width}}, default_value});
	}

	SECTION("query partial default value") {
		const auto results = rm.get_region_values({{64, 64}, {96, 96}});
		CHECK_RESULTS(results, {{{64, 64}, {96, 96}}, default_value});
	}

	SECTION("update full extent") {
		rm.update_box({{0, 0}, {height, width}}, 1337);
		const auto results = rm.get_region_values({{64, 64}, {96, 96}});
		CHECK_RESULTS(results, {{{64, 64}, {96, 96}}, 1337});
	}

	SECTION("update simple") {
		rm.update_box({{0, 0}, {64, width}}, 1337);
		const auto results = rm.get_region_values({{0, 0}, {height, width}});
		CHECK_RESULTS(results, {{{0, 0}, {64, width}}, 1337}, {{{64, 0}, {height, width}}, default_value});
	}

	SECTION("update with split, quasi 1D") {
		rm.update_box({{32, 0}, {96, width}}, 1337);
		const auto results = rm.get_region_values({{0, 0}, {height, width}});
		CHECK_RESULTS(results, {{{0, 0}, {32, width}}, default_value}, {{{32, 0}, {96, width}}, 1337}, {{{96, 0}, {height, width}}, default_value});
	}

	SECTION("update with split, 2D") {
		rm.update_box({{32, 32}, {96, 96}}, 1337);
		const auto results = rm.get_region_values({{0, 0}, {height, width}});

		CHECK_RESULTS(results,
		    // Original values
		    {{{0, 0}, {height, 32}}, default_value}, {{{0, 96}, {height, width}}, default_value}, {{{0, 32}, {32, 96}}, default_value},
		    {{{96, 32}, {height, 96}}, default_value},

		    // Updated value
		    {{{32, 32}, {96, 96}}, 1337});
	}

	SECTION("update multiple") {
		constexpr size_t num_rows = 16;
		constexpr size_t row_height = height / num_rows;
		// Iteratively split domain into multiple rows
		for(size_t i = 0; i < num_rows; ++i) {
			rm.update_box(box<2>{{i * row_height, 0}, {i * row_height + row_height, width}}, i);
			const auto results = rm.get_region_values({{0, 0}, {height, width}});
			// Until the last iteration we have to account for the original value.
			REQUIRE_LOOP(results.size() == static_cast<size_t>(i + (i < (num_rows - 1) ? 2 : 1)));
			for(size_t j = 0; j < i + 1; ++j) {
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [j, row_height](auto& r) {
					return r == std::pair{box<2>{{j * row_height, 0}, {j * row_height + row_height, width}}, j};
				}));
			}
			if(i < num_rows - 1) {
				// Check that original value still exists
				CHECK(std::any_of(results.begin(), results.end(), [i, row_height, default_value](auto& r) {
					return r == std::pair{box<2>{{(i + 1) * row_height, 0}, {height, width}}, default_value};
				}));
			}
		}

		// Now drive a center column through all of them
		rm.update_box(box<2>{{0, 48}, {height, 80}}, std::numeric_limits<size_t>::max() - 2);
		const auto results = rm.get_region_values({{0, 0}, {height, width}});
		CHECK(std::any_of(results.begin(), results.end(), [](auto& r) {
			return r == std::pair{box<2>{{0, 48}, {height, 80}}, std::numeric_limits<size_t>::max() - 2};
		}));

		for(size_t i = 0; i < num_rows; ++i) {
			REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [i, row_height](auto& r) {
				return r == std::pair{box<2>{{i * row_height, 0}, {i * row_height + row_height, 48}}, i};
			}));
			REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [i, row_height](auto& r) {
				return r == std::pair{box<2>{{i * row_height, 80}, {i * row_height + row_height, width}}, i};
			}));
		}
	}

	SECTION("update growing from two sides") {
		constexpr size_t num_rows = 16;
		constexpr size_t row_height = height / num_rows;
		// Iteratively split domain into multiple rows, working inwards from two sides
		for(size_t i = 0; i < num_rows / 2; ++i) {
			rm.update_box(box<2>{{i * row_height, 0}, {i * row_height + row_height, width}}, i);
			rm.update_box(box<2>{{(num_rows - 1 - i) * row_height, 0}, {(num_rows - 1 - i) * row_height + row_height, width}}, num_rows + i);

			const auto results = rm.get_region_values({{0, 0}, {height, width}});

			// Until the last iteration we have to account for the original value.
			REQUIRE_LOOP(results.size() == 2 * (i + 1) + (i < (num_rows / 2 - 1) ? 1 : 0));

			for(size_t j = 0; j < i + 1; ++j) {
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [j, row_height](auto& r) {
					return r == std::pair{box<2>{{j * row_height, 0}, {j * row_height + row_height, width}}, j};
				}));
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [j, row_height, num_rows](auto& r) {
					return r == std::pair{box<2>{{(num_rows - 1 - j) * row_height, 0}, {(num_rows - 1 - j) * row_height + row_height, width}}, num_rows + j};
				}));
			}

			if(i < num_rows / 2 - 1) {
				// Check that original value still exists
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [i, row_height, num_rows, default_value](auto& r) {
					return r == std::pair{box<2>{{(i + 1) * row_height, 0}, {(num_rows - 1 - i) * row_height, width}}, default_value};
				}));
			}
		}
	}

	// TODO: Also in 1D/3D?
	SECTION("update boxes random order") {
		std::vector<std::pair<box<2>, size_t>> update_boxes;
		size_t x = 100;
		constexpr size_t box_height = height / 16;
		constexpr size_t box_width = width / 16;
		for(size_t i = 0; i < 16; ++i) {
			for(size_t j = 0; j < 16; ++j) {
				const id<2> min = {i * box_height, j * box_width};
				const id<2> max = min + id<2>{box_height, box_width};
				update_boxes.push_back(std::pair{box<2>{min, max}, x++});
			}
		}
		std::mt19937 g(123);
		std::shuffle(update_boxes.begin(), update_boxes.end(), g);
		for(size_t i = 0; i < update_boxes.size(); ++i) {
			const auto& [box, x] = update_boxes[i];
			rm.update_box(box, x);

			const auto results = rm.get_region_values({{0, 0}, {height, width}});
			// We don't bother with checking for the original value, but verify that all boxes updated so far are present.
			for(size_t j = 0; j < i + 1; ++j) {
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [j, update_boxes](auto& r) { return r == update_boxes[j]; }));
			}
		}
	}

	draw(rm);
}

TEST_CASE("region_map handles basic operations in 3D", "[region_map]") {
	constexpr size_t depth = 128;
	constexpr size_t height = 192;
	constexpr size_t width = 256;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 3> rm{box<3>::full_range({depth, height, width}), default_value};

	SECTION("query default value") {
		const auto results = rm.get_region_values({{0, 0, 0}, {depth, height, width}});
		CHECK_RESULTS(results, {{{0, 0, 0}, {depth, height, width}}, default_value});
	}

	SECTION("query partial default value") {
		const auto results = rm.get_region_values({{62, 64, 66}, {92, 94, 96}});
		CHECK_RESULTS(results, {{{62, 64, 66}, {92, 94, 96}}, default_value});
	}

	SECTION("update full extent") {
		rm.update_box({{0, 0, 0}, {depth, height, width}}, 1337);
		const auto results = rm.get_region_values({{62, 64, 66}, {92, 94, 96}});
		CHECK_RESULTS(results, {{{62, 64, 66}, {92, 94, 96}}, 1337});
	}

	SECTION("update simple") {
		rm.update_box({{0, 0, 0}, {64, height, width}}, 1337);
		const auto results = rm.get_region_values({{0, 0, 0}, {depth, height, width}});
		CHECK_RESULTS(results, {{{0, 0, 0}, {64, height, width}}, 1337}, {{{64, 0, 0}, {depth, height, width}}, default_value}, );
	}

	SECTION("update with split, quasi 1D") {
		rm.update_box({{32, 0, 0}, {96, height, width}}, 1337);
		const auto results = rm.get_region_values({{0, 0, 0}, {depth, height, width}});
		CHECK_RESULTS(results, {{{0, 0, 0}, {32, height, width}}, default_value}, {{{32, 0, 0}, {96, height, width}}, 1337},
		    {{{96, 0, 0}, {depth, height, width}}, default_value}, );
	}

	SECTION("update with split, quasi 2D") {
		rm.update_box({{32, 32, 0}, {96, 96, width}}, 1337);
		const auto results = rm.get_region_values({{0, 0, 0}, {depth, height, width}});

		CHECK_RESULTS(results,
		    // Original values
		    {{{0, 0, 0}, {depth, 32, width}}, default_value}, {{{0, 96, 0}, {depth, height, width}}, default_value},
		    {{{0, 32, 0}, {32, 96, width}}, default_value}, {{{96, 32, 0}, {depth, 96, width}}, default_value},

		    // Updated value
		    {{{32, 32, 0}, {96, 96, width}}, 1337});
	}

	SECTION("update with split, 3D") {
		rm.update_box({{32, 32, 32}, {96, 96, 96}}, 1337);
		const auto results = rm.get_region_values({{0, 0, 0}, {depth, height, width}});

		CHECK_RESULTS(results,
		    // Original values
		    {{{0, 0, 0}, {depth, height, 32}}, default_value}, {{{0, 0, 96}, {depth, height, width}}, default_value},
		    {{{0, 0, 32}, {depth, 32, 96}}, default_value}, {{{0, 96, 32}, {depth, height, 96}}, default_value}, {{{0, 32, 32}, {32, 96, 96}}, default_value},
		    {{{96, 32, 32}, {depth, 96, 96}}, default_value},

		    // Updated value
		    {{{32, 32, 32}, {96, 96, 96}}, 1337});
	}
}

TEMPLATE_TEST_CASE_SIG("region_map updates get clamped to extent", "[region_map]", ((int Dims), Dims), 1, 2, 3) {
	const auto full_box = test_utils::truncate_box<Dims>({{0, 0, 0}, {64, 96, 128}});
	region_map_impl<size_t, Dims> rm{full_box, 0};

	// TODO boxes based on ids cannot be negative, so we cannot test clamping of the minimum at the moment
	const auto exceeding_box = box<Dims>({}, test_utils::truncate_range<Dims>({72, 102, 136}));

	rm.update_box(exceeding_box, 1337);
	const auto results = rm.get_region_values(exceeding_box);
	CHECK_RESULTS(results, {full_box, 1337});
}

// This doesn't test anything in paticular, more of a smoke test.
TEST_CASE("region_map correctly handles complex queries", "[region_map]") {
	region_map_impl<size_t, 2> rm{box<2>::full_range({5, 9}), 99999};

	const std::initializer_list<box<2>> data = {{{0, 0}, {2, 3}}, {{2, 0}, {5, 2}}, {{2, 2}, {5, 3}}, {{0, 3}, {3, 4}}, {{3, 3}, {4, 4}}, {{4, 3}, {5, 4}},
	    {{0, 4}, {1, 9}}, {{1, 4}, {3, 9}}, {{3, 4}, {5, 6}}, {{3, 6}, {5, 7}}, {{3, 7}, {4, 9}}, {{4, 7}, {5, 9}}};

	for(size_t i = 0; i < data.size(); ++i) {
		rm.update_box(*(data.begin() + i), i);
	}

	SECTION("query single boxes") {
		const auto query_and_check = [&](const box<2>& box, size_t expected) {
			const auto results = rm.get_region_values(box);
			REQUIRE(results.size() == 1);
			CHECK(results[0] == std::pair{box, expected});
		};

		// Query one on each side
		query_and_check({{0, 0}, {2, 2}}, 0);
		query_and_check({{0, 8}, {1, 9}}, 6);
		query_and_check({{4, 0}, {5, 2}}, 1);
		query_and_check({{4, 7}, {5, 9}}, 11);

		// And some in the middle
		query_and_check({{3, 3}, {4, 4}}, 4);
		query_and_check({{1, 4}, {3, 6}}, 7);
	}

	SECTION("query overlapping") {
		const auto query_and_check = [&](const box<2>& box, const std::vector<std::pair<celerity::detail::box<2>, size_t>>& expected) {
			const auto results = rm.get_region_values(box);
			CHECK(results.size() == expected.size());
			for(const auto& e : expected) {
				REQUIRE_LOOP(std::any_of(results.begin(), results.end(), [&](auto& r) { return r == e; }));
			}
		};

		query_and_check({{1, 2}, {3, 4}}, {
		                                      {{{1, 2}, {2, 3}}, 0},
		                                      {{{2, 2}, {3, 3}}, 2},
		                                      {{{1, 3}, {3, 4}}, 3},
		                                  });

		query_and_check({{1, 4}, {5, 7}}, {
		                                      {{{1, 4}, {3, 7}}, 7},
		                                      {{{3, 4}, {5, 6}}, 8},
		                                      {{{3, 6}, {5, 7}}, 9},
		                                  });
	}

	draw(rm);
}

TEST_CASE("region map merges entries with the same value upon update in 1D", "[region_map]") {
	constexpr size_t size = 128;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 1> rm{box<1>::full_range({size}), default_value};

	SECTION("simple merge") {
		rm.update_box({0, 64}, 3);
		rm.update_box({64, size}, 3);
		REQUIRE(region_map_testspy::get_num_leaf_nodes(rm) == 1);
	}

	SECTION("multi-merge") {
		rm.update_box({0, 32}, 3);
		rm.update_box({64, size}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 3); // Includes the default initialized slice
		// Now insert the missing slice that should allow all boxes to be merged into one
		rm.update_box({32, 64}, 3);
		REQUIRE(region_map_testspy::get_num_leaf_nodes(rm) == 1);
	}
}

TEST_CASE("region map merges entries with the same value upon update in 2D", "[region_map]") {
	constexpr size_t height = 64;
	constexpr size_t width = 128;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 2> rm{box<2>::full_range({height, width}), default_value};

	SECTION("simple merge") {
		rm.update_box({{0, 0}, {height, 64}}, 3);
		rm.update_box({{0, 64}, {height, width}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 1);
	}

	SECTION("multi-merge") {
		rm.update_box({{0, 0}, {height, 64}}, 3);
		// This box has an offset in dimension 0, which prevents merging
		rm.update_box({{32, 64}, {height, width}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 3); // Includes the default initialized slice
		// Now insert the missing slice that should allow all boxes to be merged into one
		rm.update_box({{0, 64}, {32, width}}, 3);
		REQUIRE(region_map_testspy::get_num_leaf_nodes(rm) == 1);
	}

	SECTION("merge cascade") {
		// Same as before, but ensure that the tree is several levels deep
		// Start by filling the tree with "horizontal bars" of decreasing length, preventing any merges between them
		for(size_t i = 0; i < height / 2; ++i) {
			rm.update_box({{i * 2, 0}, {i * 2 + 2, width - 2 - i * 2}}, 3);
		}
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 2 * (height / 2)); // Every bar creates two entries (old and new value)
		CHECK(region_map_testspy::get_depth(rm) > 2);                          // Tree should be several levels deep by now
		// Now update the values of the vertical bars, skip last one to prevent merge
		for(size_t i = 0; i < (height / 2) - 1; ++i) {
			rm.update_box({{i * 2, width - 2 - i * 2}, {height, width - 2 - i * 2 + 2}}, 3);
		}
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 2 * (height / 2)); // No merges so far
		// Update the final column, causing a merge cascade
		rm.update_box({{62, width / 2}, {height, (width / 2) + 2}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 1);
		CHECK(region_map_testspy::get_depth(rm) == 1);
	}

	draw(rm);
}

TEST_CASE("region map merges entries with the same value upon update in 3D", "[region_map]") {
	constexpr size_t depth = 64;
	constexpr size_t height = 96;
	constexpr size_t width = 128;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 3> rm{box<3>::full_range({depth, height, width}), default_value};

	SECTION("simple merge, quasi 1D") {
		rm.update_box({{0, 0, 0}, {depth, 64, width}}, 3);
		rm.update_box({{0, 64, 0}, {depth, height, width}}, 3);
		REQUIRE(region_map_testspy::get_num_leaf_nodes(rm) == 1);
	}

	SECTION("multi-merge") {
		rm.update_box({{0, 0, 0}, {depth, 64, width}}, 3);
		rm.update_box({{32, 64, 0}, {depth, height, width}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 3);
		rm.update_box({{0, 64, 0}, {32, height, width}}, 3);
		REQUIRE(region_map_testspy::get_num_leaf_nodes(rm) == 1);
	}

	// TODO: Come up with merge cascade in 3D as well
}

// NOTE: Merging on query is not required (or possible) in 1D: All merges will be done on update.

TEST_CASE("region_map merges truncated result boxes with the same value upon querying in 2D", "[region_map]") {
	constexpr size_t height = 5;
	constexpr size_t width = 9;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 2> rm{box<2>::full_range({height, width}), default_value};

	SECTION("simple merge") {
		// Set up in such a way that values cannot be merged upon update
		rm.update_box({{0, 0}, {height, 3}}, 3);
		rm.update_box({{2, 3}, {height, 6}}, 3);
		rm.update_box({{0, 6}, {height, width}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 4);

		// Even though our query spans three different rects, a single value should be returned
		const auto results = rm.get_region_values({{2, 0}, {5, 9}});
		CHECK_RESULTS(results, {{{2, 0}, {height, width}}, 3});
	}

	// TODO: Add template option to control merge order, making this non-ambiguous?
	SECTION("ambiguous merge") {
		rm.update_box({{0, 0}, {3, 3}}, 3);
		rm.update_box({{3, 1}, {height, 3}}, 3);
		rm.update_box({{1, 3}, {3, width}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 6);

		const auto results = rm.get_region_values({{1, 1}, {5, 9}});
		// The exact result is ambiguous depending on how boxes were merged. However there should always be 3
		CHECK(results.size() == 3);
		// One is the non-mergeable default-initialized section
		CHECK(std::any_of(results.begin(), results.end(), [default_value](auto& r) { return r == std::pair{box<2>{{3, 3}, {height, width}}, default_value}; }));
		// The other two are either of these two variants
		const bool variant_1 = std::any_of(results.begin(), results.end(), [](auto& r) {
			return r == std::pair{box<2>{{1, 1}, {height, 3}}, size_t(3)};
		}) && std::any_of(results.begin(), results.end(), [](auto& r) {
			return r == std::pair{box<2>{{1, 3}, {3, width}}, size_t(3)};
		});
		const bool variant_2 = std::any_of(results.begin(), results.end(), [](auto& r) {
			return r == std::pair{box<2>{{1, 1}, {3, width}}, size_t(3)};
		}) && std::any_of(results.begin(), results.end(), [](auto& r) {
			return r == std::pair{box<2>{{3, 1}, {height, 3}}, size_t(3)};
		});
		CHECK(variant_1 != variant_2);
	}

	draw(rm);
}

TEST_CASE("region_map merges truncated result boxes with the same value upon querying in 3D", "[region_map]") {
	constexpr size_t depth = 32;
	constexpr size_t height = 64;
	constexpr size_t width = 96;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 3> rm{box<3>::full_range({depth, height, width}), default_value};

	SECTION("simple merge") {
		// Setup in such a way that values cannot be merged upon update
		rm.update_box({{0, 0, 0}, {depth, 24, 24}}, 3);
		rm.update_box({{0, 24, 0}, {depth, 48, 48}}, 3);
		rm.update_box({{0, 48, 0}, {depth, height, width}}, 3);
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 5);

		// Even though our query spans three different boxes, a single value should be returned
		const auto results = rm.get_region_values({{0, 0, 0}, {depth, height, 24}});
		CHECK_RESULTS(results, {{{0, 0, 0}, {depth, height, 24}}, 3});
	}

	// TODO: Come up with ambiguous merge example in 3D
}

TEST_CASE("region_map supports apply_to_values", "[region_map]") {
	constexpr size_t size = 128;
	constexpr size_t default_value = std::numeric_limits<size_t>::max();
	region_map_impl<size_t, 1> rm{box<1>::full_range({size}), default_value};

	const auto query_and_check = [&](const box<1>& box, size_t expected) {
		const auto results = rm.get_region_values(box);
		CHECK(results.size() == 1);
		CHECK(results[0] == std::pair{box, expected});
	};

	rm.update_box({0, 32}, 1);
	rm.update_box({32, 64}, 2);
	rm.update_box({64, 96}, 3);
	rm.update_box({96, size}, 4);

	SECTION("basic value update") {
		rm.apply_to_values([](size_t v) { return v * v; });
		query_and_check({0, 32}, 1);
		query_and_check({32, 64}, 4);
		query_and_check({64, 96}, 9);
		query_and_check({96, size}, 16);
	}

	SECTION("same values are merged after update") {
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 4);
		rm.apply_to_values([](size_t v) -> size_t { return v != 2 ? 42 : 1337; });
		CHECK(region_map_testspy::get_num_leaf_nodes(rm) == 3);
		query_and_check({0, 32}, 42);
		query_and_check({32, 64}, 1337);
		query_and_check({64, size}, 42);
	}
}

// TODO: This only works until count_sqrt exceeds per-node value limit.
TEST_CASE("inserting consecutive boxes results in zero overlap", "[region_map][performance]") {
	const bool row_wise_insert = GENERATE(true, false);

	const size_t height = 64;
	const size_t width = 128;
	region_map_impl<size_t, 2> rm{box<2>::full_range({height, width}), std::numeric_limits<size_t>::max()};

	const size_t count_sqrt = 4;
	REQUIRE(height % count_sqrt == 0);
	REQUIRE(width % count_sqrt == 0);

	const auto insert_box = [&](const size_t i, const size_t j) {
		const id<2> min = {i * (height / count_sqrt), j * (width / count_sqrt)};
		const id<2> max = min + id<2>{height / count_sqrt, width / count_sqrt};
		rm.update_box({min, max}, i * count_sqrt + j);
	};

	for(size_t i = 0; i < count_sqrt; ++i) {
		for(size_t j = 0; j < count_sqrt; ++j) {
			if(row_wise_insert) {
				insert_box(i, j);
			} else {
				insert_box(j, i);
			}
		}
	}

	CHECK(region_map_testspy::get_num_leaf_nodes(rm) == count_sqrt * count_sqrt);
	CHECK(region_map_testspy::compute_overlap(rm) == 0);
	draw(rm);
}

TEST_CASE("query regions are clamped from both sides in region maps with non-zero offset", "[region_map]") {
	const auto region_box = box<3>({1, 2, 3}, {7, 9, 11});
	region_map<int> rm(region_box, 42);
	CHECK(rm.get_region_values(box<3>::full_range({20, 19, 18})) == std::vector{std::pair{region_box, 42}});
}

TEMPLATE_TEST_CASE_SIG("get_region_values(<empty-region>) returns no boxes", "[region_map]", ((int Dims), Dims), 0, 1, 2, 3) {
	region_map<int> rm(range_cast<3>(test_utils::truncate_range<Dims>({2, 3, 4})), -1);
	CHECK(rm.get_region_values(box<3>()).empty());
	CHECK(rm.get_region_values(region<3>()).empty());
}

TEMPLATE_TEST_CASE_SIG("update(<empty-box>) has no effect", "[region_map]", ((int Dims), Dims), 0, 1, 2, 3) {
	region_map<int> rm(range_cast<3>(test_utils::truncate_range<Dims>({2, 3, 4})), 0);
	rm.update_box(box<3>(), 1);
	rm.update_region(box<3>(), 2);
	const auto unit_box = box_cast<3>(box<0>());
	CHECK(rm.get_region_values(unit_box) == std::vector{std::pair{unit_box, 0}});
}
