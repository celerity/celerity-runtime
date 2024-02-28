#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "grid_test_utils.h"


using namespace celerity;
using namespace celerity::detail;


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
