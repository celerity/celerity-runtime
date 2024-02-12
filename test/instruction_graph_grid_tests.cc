#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "grid_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::detail::instruction_graph_generator_detail;


TEST_CASE("boxes_edge_connected_correctly detects 2/4/6-connectivity", "[instruction_graph_generator][grid]") {
	using instruction_graph_generator_detail::boxes_edge_connected;

	SECTION("0D") { CHECK(boxes_edge_connected<0>({}, {})); }

	SECTION("1D") {
		CHECK(boxes_edge_connected<1>({10, 20}, {0, 10}));
		CHECK(boxes_edge_connected<1>({10, 20}, {9, 10}));
		CHECK(boxes_edge_connected<1>({10, 20}, {10, 11}));
		CHECK(boxes_edge_connected<1>({10, 20}, {10, 20}));
		CHECK(boxes_edge_connected<1>({10, 20}, {8, 12}));
		CHECK(boxes_edge_connected<1>({10, 20}, {19, 20}));
		CHECK(boxes_edge_connected<1>({10, 20}, {20, 21}));
		CHECK(boxes_edge_connected<1>({10, 20}, {20, 30}));

		CHECK_FALSE(boxes_edge_connected<1>({0, 0}, {0, 0}));
		CHECK_FALSE(boxes_edge_connected<1>({10, 20}, {0, 0}));
		CHECK_FALSE(boxes_edge_connected<1>({10, 20}, {0, 9}));
		CHECK_FALSE(boxes_edge_connected<1>({10, 20}, {21, 22}));
		CHECK_FALSE(boxes_edge_connected<1>({10, 20}, {21, 30}));
	}

	SECTION("2D") {
		const box<2> q1{{10, 10}, {20, 20}};
		const box<2> q2{{20, 10}, {30, 20}};
		const box<2> q3{{10, 20}, {20, 30}};
		const box<2> q4{{20, 20}, {30, 30}};

		CHECK(boxes_edge_connected(q1, q2));
		CHECK(boxes_edge_connected(q1, q3));
		CHECK(boxes_edge_connected(q2, q1));
		CHECK(boxes_edge_connected(q2, q4));
		CHECK(boxes_edge_connected(q3, q1));
		CHECK(boxes_edge_connected(q3, q4));
		CHECK(boxes_edge_connected(q4, q2));
		CHECK(boxes_edge_connected(q4, q3));

		CHECK_FALSE(boxes_edge_connected(q1, q4));
		CHECK_FALSE(boxes_edge_connected(q2, q3));
		CHECK_FALSE(boxes_edge_connected(q3, q2));
		CHECK_FALSE(boxes_edge_connected(q4, q1));

		const box<2> left{{10, 0}, {30, 9}};
		const box<2> right{{10, 31}, {30, 40}};

		CHECK_FALSE(boxes_edge_connected(q1, left));
		CHECK_FALSE(boxes_edge_connected(q2, left));
		CHECK_FALSE(boxes_edge_connected(q3, left));
		CHECK_FALSE(boxes_edge_connected(q4, left));

		CHECK_FALSE(boxes_edge_connected(q1, right));
		CHECK_FALSE(boxes_edge_connected(q2, right));
		CHECK_FALSE(boxes_edge_connected(q3, right));
		CHECK_FALSE(boxes_edge_connected(q4, right));
	}

	SECTION("3D") {
		const box<3> q1{{10, 10, 10}, {20, 20, 20}};
		const box<3> q2{{20, 10, 10}, {30, 20, 20}};
		const box<3> q3{{10, 20, 10}, {20, 30, 20}};
		const box<3> q4{{20, 20, 10}, {30, 30, 20}};
		const box<3> q5{{10, 10, 20}, {20, 20, 30}};
		const box<3> q6{{20, 10, 20}, {30, 20, 30}};
		const box<3> q7{{10, 20, 20}, {20, 30, 30}};
		const box<3> q8{{20, 20, 20}, {30, 30, 30}};

		CHECK(boxes_edge_connected(q1, q1));
		CHECK(boxes_edge_connected(q1, q2));
		CHECK(boxes_edge_connected(q1, q3));
		CHECK(boxes_edge_connected(q1, q5));
		CHECK_FALSE(boxes_edge_connected(q1, q4));
		CHECK_FALSE(boxes_edge_connected(q1, q6));
		CHECK_FALSE(boxes_edge_connected(q1, q7));
		CHECK_FALSE(boxes_edge_connected(q1, q8));

		CHECK(boxes_edge_connected(q2, q2));
		CHECK(boxes_edge_connected(q2, q1));
		CHECK(boxes_edge_connected(q2, q4));
		CHECK(boxes_edge_connected(q2, q6));
		CHECK_FALSE(boxes_edge_connected(q2, q3));
		CHECK_FALSE(boxes_edge_connected(q2, q5));
		CHECK_FALSE(boxes_edge_connected(q2, q7));
		CHECK_FALSE(boxes_edge_connected(q2, q8));

		CHECK(boxes_edge_connected(q7, q7));
		CHECK(boxes_edge_connected(q7, q3));
		CHECK(boxes_edge_connected(q7, q5));
		CHECK(boxes_edge_connected(q7, q8));
		CHECK_FALSE(boxes_edge_connected(q7, q1));
		CHECK_FALSE(boxes_edge_connected(q7, q2));
		CHECK_FALSE(boxes_edge_connected(q7, q4));
		CHECK_FALSE(boxes_edge_connected(q7, q6));
	}
}

TEST_CASE("connected_subregion_bounding_boxes correctly merges bounding boxes", "[instruction_graph_generator][grid]") {
	using instruction_graph_generator_detail::connected_subregion_bounding_boxes;

	// for 0D and 1D this is somewhat pointless as region normalization will merge any connected boxes.

	SECTION("0D") {
		CHECK(connected_subregion_bounding_boxes(region<0>()).empty());
		CHECK(connected_subregion_bounding_boxes(region<0>(box<0>())) == box_vector<0>{box<0>()});
	}

	SECTION("1D") {
		const region<1> input_region({{0, 5}, {6, 10}, {10, 15}, {19, 25}, {12, 21}, {30, 39}, {35, 40}});
		const box_vector<1> expected_boxes{{{0, 5}, {6, 25}, {30, 40}}};

		auto result_boxes = connected_subregion_bounding_boxes(input_region);
		std::sort(result_boxes.begin(), result_boxes.end(), box_coordinate_order());
		CHECK(result_boxes == expected_boxes);
	}

	SECTION("2D") {
		const region<2> input_region({//
		    /* (1) connected part */ box<2>({5, 0}, {10, 15}), box<2>({10, 10}, {20, 30}), box<2>({20, 10}, {30, 20}),
		    /* (2) inside bounding box, but disconnected */ box<2>({25, 25}, {40, 30}),
		    /* (3) touching both (1) and (2) */ box<2>({20, 30}, {25, 35})});
		test_utils::render_boxes(input_region.get_boxes(), "input");

		const box_vector<2> expected_boxes{box<2>({5, 0}, {30, 30}), box<2>({20, 30}, {25, 35}), box<2>({25, 25}, {40, 30})};
		test_utils::render_boxes(expected_boxes, "expected");

		auto result_boxes = connected_subregion_bounding_boxes(input_region);
		std::sort(result_boxes.begin(), result_boxes.end(), box_coordinate_order());
		test_utils::render_boxes(result_boxes, "result");

		CHECK(result_boxes == expected_boxes);
	}

	SECTION("3D") {
		const region<3> input_region({
		    // (1) connected part
		    box<3>({10, 10, 10}, {20, 20, 30}),
		    box<3>({20, 10, 10}, {30, 20, 20}),
		    box<3>({10, 20, 10}, {20, 30, 20}),
		    // (2) inside bounding box, but disconnected
		    box<3>({21, 21, 21}, {29, 29, 29}),
		    // (3) touching (1) on a 1D edge
		    box<3>({0, 0, 10}, {10, 10, 30}),
		    // (4) touching (2) on a 0D corner
		    box<3>({20, 20, 30}, {25, 25, 35}),
		});

		const box_vector<3> expected_boxes = {
		    box<3>({0, 0, 10}, {10, 10, 30}),
		    box<3>({10, 10, 10}, {30, 30, 30}),
		    box<3>({20, 20, 30}, {25, 25, 35}),
		    box<3>({21, 21, 21}, {29, 29, 29}),
		};

		auto result_boxes = connected_subregion_bounding_boxes(input_region);
		std::sort(result_boxes.begin(), result_boxes.end(), box_coordinate_order());

		CHECK(result_boxes == expected_boxes);
	}
}

TEST_CASE("instruction_graph_generator computes communicator-compatible boxes from send-regions", "[instruction_graph_generator][grid]") {
	const auto max = communicator_max_coordinate;
	const auto huge = 3 * max + 100;

	SECTION("for a send-box with huge 0d extent") {
		const auto full_box = box<3>({0, 0, 0}, {huge, 2, 3});
		const auto split_boxes = split_into_communicator_compatible_boxes(full_box.get_range(), full_box);
		const box_vector<3> expected{
		    box<3>({0, 0, 0}, {max, 2, 3}),
		    box<3>({max, 0, 0}, {2 * max, 2, 3}),
		    box<3>({2 * max, 0, 0}, {3 * max, 2, 3}),
		    box<3>({3 * max, 0, 0}, {huge, 2, 3}),
		};
		CHECK(split_boxes == expected);
	}

	SECTION("for a send-box in a buffer with huge d0 extent") {
		const auto buffer_range = range<3>({huge, 2, 3});
		const auto access_box = box<3>({999, 0, 0}, {999 + max, 2, 3});
		const auto split_access_boxes = split_into_communicator_compatible_boxes(buffer_range, access_box);
		const box_vector<3> expected{box<3>({999, 0, 0}, {999 + max, 2, 3})};
		CHECK(split_access_boxes == expected);
	}

	SECTION("for a send-box with huge 0d extent and non-zero d0 offset") {
		const auto buffer_range = range<3>({2 * huge, 2, 3});
		const auto access_box = box<3>({999, 0, 0}, {999 + huge, 2, 3});
		const auto split_access_boxes = split_into_communicator_compatible_boxes(buffer_range, access_box);
		const box_vector<3> expected{
		    box<3>({999, 0, 0}, {999 + max, 2, 3}),
		    box<3>({999 + max, 0, 0}, {999 + 2 * max, 2, 3}),
		    box<3>({999 + 2 * max, 0, 0}, {999 + 3 * max, 2, 3}),
		    box<3>({999 + 3 * max, 0, 0}, {999 + huge, 2, 3}),
		};
		CHECK(split_access_boxes == expected);
	}

	SECTION("for a small send-box in a buffer box with huge d1 extent") {
		const auto d2_extent = GENERATE(values<size_t>({1, 2}));

		const range buffer_range(10, huge, d2_extent);
		const box<3> access_box({3, 10, 0}, {8, 13, d2_extent});
		const auto split_access_boxes = split_into_communicator_compatible_boxes(buffer_range, access_box);
		const box_vector<3> expected{
		    box<3>({3, 10, 0}, {4, 13, d2_extent}),
		    box<3>({4, 10, 0}, {5, 13, d2_extent}),
		    box<3>({5, 10, 0}, {6, 13, d2_extent}),
		    box<3>({6, 10, 0}, {7, 13, d2_extent}),
		    box<3>({7, 10, 0}, {8, 13, d2_extent}),
		};
		CHECK(split_access_boxes == expected);
	}
}

TEST_CASE("instruction_graph_generator warns when forced to issue an excessive amount of small transfers for MPI compatibility",
    "[instruction_graph_generator][grid]") //
{
	test_utils::allow_max_log_level(log_level::warn);

	const range buffer_range(4096, communicator_max_coordinate + 1, 1);
	const box<3> column({0, 0, 0}, {4096, 1, 1});
	const auto split_boxes = split_into_communicator_compatible_boxes(buffer_range, column);

	for(size_t y = 0; y < 4096; ++y) {
		REQUIRE_LOOP(split_boxes[y] == box<3>({y, 0, 0}, {y + 1, 1, 1}));
	}

	CHECK(test_utils::log_contains_exact(log_level::warn,
	    "Celerity is generating an excessive amount of small transfers to keep strides representable as 32-bit integers for MPI compatibility. "
	    "This might be very slow and / or exhaust system memory. Consider transposing your buffer layout to remedy this."));
}

template <int Dims>
box_vector<Dims> flatten_regions(const std::vector<region<Dims>>& regions) {
	box_vector<Dims> boxes;
	for(const auto& r : regions) {
		boxes.insert(boxes.end(), r.get_boxes().begin(), r.get_boxes().end());
	}
	return boxes;
}

TEST_CASE(
    "instruction_graph_generator computes concurrent splits for potentially-overlapping receive- and copy regions", "[instruction_graph_generator][grid]") //
{
	SECTION("for non-overlapping boxes") {
		const std::vector input{
		    region{box<2>({0, 0}, {10, 10})},
		    region{box<2>({10, 0}, {20, 10})},
		    region{box<2>({0, 10}, {10, 20})},
		    region{box<2>({10, 10}, {20, 20})},
		};
		const auto expected = input;

		auto result = input;
		instruction_graph_generator_detail::symmetrically_split_overlapping_regions(result);
		CHECK(result == expected);

		test_utils::render_boxes(flatten_regions(input), "non-overlapping-input");
		test_utils::render_boxes(flatten_regions(expected), "non-overlapping-expected");
		test_utils::render_boxes(flatten_regions(result), "non-overlapping-result");
	}

	SECTION("for duplicate boxes") {
		const std::vector input{
		    region{box<2>({0, 0}, {10, 10})},
		    region{box<2>({0, 0}, {10, 10})},
		    region{box<2>({0, 0}, {10, 10})},
		};
		const std::vector expected{input[0]};

		auto result = input;
		instruction_graph_generator_detail::symmetrically_split_overlapping_regions(result);
		CHECK(result == expected);

		test_utils::render_boxes(flatten_regions(input), "duplicate-input");
		test_utils::render_boxes(flatten_regions(expected), "duplicate-expected");
		test_utils::render_boxes(flatten_regions(result), "duplicate-result");
	}

	SECTION("for fully-nested boxes") {
		const std::vector input{
		    region{box<2>({0, 0}, {20, 20})},
		    region{box<2>({0, 0}, {15, 15})},
		    region{box<2>({5, 5}, {15, 15})},
		    region{box<2>({8, 8}, {12, 12})},
		};

		// this test assumes that the implementation-defined order of regions stays intact - re-order this vector if necessary
		const std::vector expected{
		    region<2>({box<2>({0, 15}, {15, 20}), box<2>({15, 0}, {20, 20})}),
		    region<2>({box<2>({0, 0}, {5, 15}), box<2>({5, 0}, {15, 5})}),
		    region<2>({box<2>({5, 5}, {8, 15}), box<2>({8, 5}, {12, 8}), box<2>({8, 12}, {12, 15}), box<2>({12, 5}, {15, 15})}),
		    region<2>({box<2>({8, 8}, {12, 12})}),
		};

		auto result = input;
		instruction_graph_generator_detail::symmetrically_split_overlapping_regions(result);
		CHECK(result == expected);

		test_utils::render_boxes(flatten_regions(input), "nested-input");
		test_utils::render_boxes(flatten_regions(expected), "nested-expected");
		test_utils::render_boxes(flatten_regions(result), "nested-result");
	}

	SECTION("for partially-overlapping boxes") {
		const std::vector input{
		    region<2>({box<2>({0, 8}, {10, 18}), box<2>({8, 0}, {18, 10})}),
		    region<2>({box<2>({5, 13}, {15, 23}), box<2>({13, 5}, {23, 15})}),
		};

		// this test assumes that the implementation-defined order of regions stays intact - re-order this vector if necessary
		const std::vector expected{
		    region<2>({box<2>({0, 8}, {5, 18}), box<2>({5, 8}, {8, 13}), box<2>({8, 0}, {10, 13}), box<2>({10, 0}, {13, 10}), box<2>({13, 0}, {18, 5})}),
		    region<2>(
		        {box<2>({5, 18}, {10, 23}), box<2>({10, 13}, {13, 23}), box<2>({13, 10}, {15, 23}), box<2>({15, 10}, {18, 15}), box<2>({18, 5}, {23, 15})}),
		    region<2>({box<2>({5, 13}, {10, 18}), box<2>({13, 5}, {18, 10})}),
		};

		auto result = input;
		instruction_graph_generator_detail::symmetrically_split_overlapping_regions(result);
		CHECK(result == expected);

		test_utils::render_boxes(flatten_regions(input), "partially-overlapping-input");
		test_utils::render_boxes(flatten_regions(expected), "partially-overlapping-expected");
		test_utils::render_boxes(flatten_regions(result), "partially-overlapping-result");
	}

	// Test the "real" 3D case - we can't get renderings for these unfortunately
	SECTION("for a large set of randomized boxes") {
		std::vector<region<3>> input;
		for(size_t i = 0; i < 5; ++i) {
			input.push_back(region(test_utils::create_random_boxes<3>(20, 10, 5, 12345 + i)));
		}

		const auto input_union = region(flatten_regions(input));

		auto result = input;
		instruction_graph_generator_detail::symmetrically_split_overlapping_regions(result);
		CHECK(result.size() >= input.size());

		// we don't check individual coordinates, just that `result` is a decomposition of `input`

		region<3> result_union_acc;
		for(const auto& r : result) {
			CHECK(region_intersection(result_union_acc, r).empty());
			result_union_acc = region_union(result_union_acc, r);
		}
		CHECK(result_union_acc == input_union);
	}
}
