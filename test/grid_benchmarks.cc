#include "grid_test_utils.h"

#include <algorithm>
#include <iterator>
#include <random>
#include <regex>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;

template <int Dims>
box_vector<Dims> create_random_boxes(const size_t grid_size, const size_t max_box_size, const size_t num_boxes, const uint32_t seed) {
	std::minstd_rand rng(seed);
	std::uniform_int_distribution<size_t> offset_dist(0, grid_size - 1);
	std::binomial_distribution<size_t> range_dist(max_box_size - 1, 0.5);
	box_vector<Dims> boxes;
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

TEST_CASE("normalizing randomized box sets - 2d", "[benchmark][grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 200},
	}));

	const auto input_2d = create_random_boxes<2>(grid_size, max_box_size, num_boxes, 42);
	BENCHMARK(fmt::format("{}, native", label)) { return grid_detail::normalize(test_utils::copy(input_2d)); };

	const auto input_3d = boxes_cast<3>(input_2d);
	BENCHMARK(fmt::format("{}, embedded in 3d", label)) { return grid_detail::normalize(test_utils::copy(input_3d)); };

	const auto normalized_2d = grid_detail::normalize(test_utils::copy(input_2d));
	const auto normalized_3d = grid_detail::normalize(test_utils::copy(input_3d));
	CHECK(normalized_3d == boxes_cast<3>(normalized_2d));

	test_utils::render_boxes(input_2d, fmt::format("{}-input", label));
	test_utils::render_boxes(normalized_2d, fmt::format("{}-normalized", label));
}

TEST_CASE("normalizing randomized box sets - 3d", "[benchmark][grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 200},
	}));

	const auto input_3d = create_random_boxes<3>(grid_size, max_box_size, num_boxes, 42);
	BENCHMARK(fmt::format("{} - native", label)) { return grid_detail::normalize(test_utils::copy(input_3d)); };
	test_utils::black_hole(grid_detail::normalize(test_utils::copy(input_3d))); // to attach a profiler
}

template <int Dims>
box_vector<Dims> create_box_tiling(const size_t n_per_side) {
	const size_t length = 5;
	size_t n_linear = 1;
	for(int d = 0; d < Dims; ++d) {
		n_linear *= n_per_side;
	}
	box_vector<Dims> boxes(n_linear, box<Dims>());
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

TEMPLATE_TEST_CASE_SIG("normalizing a fully mergeable tiling of boxes", "[benchmark][grid]", ((int Dims), Dims), 1, 2, 3) {
	const auto [label, n] = GENERATE(values<std::tuple<const char*, size_t>>({
	    {"small", 4},
	    {"medium", 50},
	    {"large", 1000},
	}));

	const size_t n_per_side = llrint(pow(n, 1.0 / Dims));

	const auto boxes_nd = create_box_tiling<Dims>(n_per_side);
	const auto normalized_nd = grid_detail::normalize(test_utils::copy(boxes_nd));
	CHECK(normalized_nd.size() == 1);

	BENCHMARK(fmt::format("{}, native", label)) { return grid_detail::normalize(test_utils::copy(boxes_nd)); };

	if constexpr(Dims < 3) {
		const auto boxes_3d = boxes_cast<3>(boxes_nd);
		BENCHMARK(fmt::format("{}, embedded in 3d", label)) { return grid_detail::normalize(test_utils::copy(boxes_3d)); };
	}

	if constexpr(Dims == 2) {
		test_utils::render_boxes(boxes_nd, fmt::format("{}-input", label));
		test_utils::render_boxes(normalized_nd, fmt::format("{}-normalized", label));
	}
}

TEST_CASE("performing set operations between randomized regions - 2d", "[benchmark][grid]") {
	const auto [label, grid_size, max_box_size, num_boxes] = GENERATE(values<std::tuple<const char*, size_t, size_t, size_t>>({
	    {"small", 10, 5, 4},
	    {"medium", 50, 1, 50},
	    {"large", 200, 20, 100},
	}));

	const std::vector inputs_2d{
	    region(create_random_boxes<2>(grid_size, max_box_size, num_boxes, 13)), region(create_random_boxes<2>(grid_size, max_box_size, num_boxes, 37))};
	const std::vector inputs_3d{region_cast<3>(inputs_2d[0]), region_cast<3>(inputs_2d[1])};

	test_utils::render_boxes(inputs_2d[0].get_boxes(), fmt::format("{}-input-a", label));
	test_utils::render_boxes(inputs_2d[1].get_boxes(), fmt::format("{}-input-b", label));

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

	test_utils::render_boxes(union_2d.get_boxes(), fmt::format("union-{}", label));
	test_utils::render_boxes(intersection_2d.get_boxes(), fmt::format("intersection-{}", label));
	test_utils::render_boxes(difference_2d.get_boxes(), fmt::format("difference-{}", label));
}

TEST_CASE("performing set operations between randomized regions - 3d", "[benchmark][grid]") {
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

box_vector<2> create_interlocking_boxes(const size_t num_boxes_per_side) {
	box_vector<2> boxes;
	for(size_t i = 0; i < num_boxes_per_side; ++i) {
		boxes.emplace_back(id<2>(i, i), id<2>(i + 1, num_boxes_per_side));
		boxes.emplace_back(id<2>(i + 1, i), id<2>(num_boxes_per_side, i + 1));
	}
	return boxes;
}

TEST_CASE("normalizing a fully mergeable, complex tiling of boxes - 2d", "[benchmark][grid]") {
	const auto [label, n] = GENERATE(values<std::tuple<const char*, size_t>>({
	    {"small", 10},
	    {"large", 200},
	}));

	const auto boxes_2d = create_interlocking_boxes(n);
	const auto boxes_3d = boxes_cast<3>(boxes_2d);

	BENCHMARK(fmt::format("{}, native", label)) { return grid_detail::normalize(test_utils::copy(boxes_2d)); };
	BENCHMARK(fmt::format("{}, embedded in 3d", label)) { return grid_detail::normalize(test_utils::copy(boxes_3d)); };

	test_utils::render_boxes(boxes_2d, fmt::format("{}-input", label));
}
