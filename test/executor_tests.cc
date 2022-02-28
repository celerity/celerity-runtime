#include "conflict_graph.h"

#include "test_utils.h"

namespace celerity::detail {

TEST_CASE("maximum independent set", "[conflict-graph]") {
	conflict_graph cg;
	for(command_id cid = 0; cid <= 11; ++cid) {
		cg.add_command(cid);
	}

	// fork
	cg.add_conflict(0, 1);
	cg.add_conflict(0, 2);
	cg.add_conflict(0, 3);

	// triangle
	cg.add_conflict(5, 6);
	cg.add_conflict(6, 7);
	cg.add_conflict(5, 7);

	// quadrilateral
	cg.add_conflict(8, 9);
	cg.add_conflict(8, 10);
	cg.add_conflict(9, 11);
	cg.add_conflict(10, 11);

	const auto cfs = cg.largest_conflict_free_subset(cg.get_commands());
	CHECK(cfs.size() == 7);

	CHECK(cfs.count(0) == 0);
	for(command_id cid = 1; cid <= 4; ++cid) {
		CHECK(cfs.count(cid) == 1);
	}

	CHECK(cfs.count(5) + cfs.count(6) + cfs.count(7) == 1);

	const auto quad_8_11 = cfs.count(8) + cfs.count(11) == 2;
	const auto quad_9_10 = cfs.count(9) + cfs.count(10) == 2;
	CHECK(quad_8_11 != quad_9_10);
}

TEST_CASE("maximum independent set 2", "[conflict-graph]") {
	conflict_graph cg;
	cg.add_command(2);
	cg.add_command(0);
	cg.add_conflict(0, 2);
	const auto cfs = cg.largest_conflict_free_subset({2});
	CHECK(cfs == conflict_graph::command_set{2});
}

TEST_CASE("conflict graph", "[conflict-graph]") {
	conflict_graph cg;
	for(command_id cid = 0; cid < 5; ++cid) {
		cg.add_command(cid);
	}

	CHECK(!cg.has_any_conflict(0));
	CHECK(!cg.has_any_conflict(1));
	cg.add_conflict(0, 1);
	CHECK(cg.has_conflict(0, 1));
	CHECK(cg.has_conflict(1, 0));
	CHECK(cg.has_any_conflict(0));
	CHECK(cg.has_any_conflict(1));

	cg.add_conflict(2, 3);
	CHECK(cg.has_conflict(2, 3));
	CHECK(cg.has_conflict(3, 2));
	CHECK(cg.has_any_conflict(2));
	CHECK(cg.has_any_conflict(3));
}

} // namespace celerity::detail