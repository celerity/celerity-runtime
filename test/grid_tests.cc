#include "grid_test_utils.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <random>
#include <regex>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;

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

	test_utils::render_boxes(unmerged, "unmerged");
	test_utils::render_boxes(merged_dim0, "merged-dim0");
	test_utils::render_boxes(merged_dim1, "merged-dim1");
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

	test_utils::render_boxes(unmerged, "unmerged");
	test_utils::render_boxes(merged_dim0, "merged-dim0");
	test_utils::render_boxes(merged_dim1, "merged-dim1");
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

	test_utils::render_boxes(overlapping, "input");
	test_utils::render_boxes(result, "result");
	test_utils::render_boxes(normalized, "normalized");
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

	test_utils::render_boxes(input, "input");
	test_utils::render_boxes(result, "result");
	test_utils::render_boxes(normalized, "normalized");
}

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

	test_utils::render_boxes(ra.get_boxes(), "ra");
	test_utils::render_boxes(rb.get_boxes(), "rb");
	test_utils::render_boxes(expected, "expected");
	test_utils::render_boxes(result.get_boxes(), "result");
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

	test_utils::render_boxes(ra.get_boxes(), "ra");
	test_utils::render_boxes(rb.get_boxes(), "rb");
	test_utils::render_boxes(expected, "expected");
	test_utils::render_boxes(result.get_boxes(), "result");
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

	test_utils::render_boxes(ra.get_boxes(), "ra");
	test_utils::render_boxes(rb.get_boxes(), "rb");
	test_utils::render_boxes(expected, "expected");
	test_utils::render_boxes(result.get_boxes(), "result");
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
