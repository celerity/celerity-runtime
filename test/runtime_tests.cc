#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <limits>

#include <SYCL/sycl.hpp>
#include <catch.hpp>

#include <celerity.h>

GridBox<3> make_grid_box(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
	const auto end = cl::sycl::range<3>(offset) + range;
	return GridBox<3>(celerity::detail::sycl_range_to_grid_point(cl::sycl::range<3>(offset)), celerity::detail::sycl_range_to_grid_point(end));
}

GridRegion<3> make_grid_region(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
	return GridRegion<3>(make_grid_box(range, offset));
}

namespace celerity {

TEST_CASE("Basic", "[buffer_state]") {
	detail::buffer_state bs(cl::sycl::range<3>(256, 1, 1), 2);

	auto sn = bs.get_source_nodes(make_grid_region({256, 1, 1}));
	REQUIRE(sn.size() == 1);
	REQUIRE(sn[0].first == make_grid_box({256, 1, 1}));
	REQUIRE(sn[0].second.size() == 2);
	REQUIRE(sn[0].second.count(0) == 1);
	REQUIRE(sn[0].second.count(1) == 1);
}

TEST_CASE("UpdateRegion", "[buffer_state]") {
	detail::buffer_state bs(cl::sycl::range<3>(256, 1, 1), 2);
	bs.update_region(make_grid_region({128, 1, 1}), {1});

	auto sn = bs.get_source_nodes(make_grid_region({32, 1, 1}, {32, 0, 0}));
	REQUIRE(sn.size() == 1);
	REQUIRE(sn[0].first == make_grid_box({32, 1, 1}, {32, 0, 0}));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(1) == 1);

	sn = bs.get_source_nodes(make_grid_region({256, 1, 1}));
	REQUIRE(sn.size() == 2);
	REQUIRE(sn[0].first == make_grid_box({128, 1, 1}, {128, 0, 0}));
	REQUIRE(sn[0].second.size() == 2);
	REQUIRE(sn[0].second.count(0) == 1);
	REQUIRE(sn[0].second.count(1) == 1);
	REQUIRE(sn[1].first == make_grid_box({128, 1, 1}));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);
}

TEST_CASE("CollapseRegions", "[buffer_state]") {
	// We test buffer_state<>::collapse_regions by observing the order of the
	// returned boxes. This somewhat relies on implementation details of
	// buffer_state<>::get_source_nodes.
	// TODO: We may want to test this directly instead
	detail::buffer_state bs(cl::sycl::range<3>(256, 1, 1), 2);
	bs.update_region(make_grid_region({64, 1, 1}, {64, 0, 0}), {1});
	bs.update_region(make_grid_region({64, 1, 1}, {192, 0, 0}), {1});

	auto sn = bs.get_source_nodes(make_grid_region({192, 1, 1}, {64, 0, 0}));
	REQUIRE(sn.size() == 3);
	REQUIRE(sn[0].first == make_grid_box({64, 1, 1}, {64, 0, 0}));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(1) == 1);

	// Since this one is returned before the [128,192) box,
	// the {[64,128), [192,256)} region must exist internally.
	// REQUIRE(sn[1].first == GridBox<1>(192, 256));
	REQUIRE(sn[1].first == make_grid_box({64, 1, 1}, {192, 0, 0}));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);

	REQUIRE(sn[2].first == make_grid_box({64, 1, 1}, {128, 0, 0}));
	REQUIRE(sn[2].second.size() == 2);
	REQUIRE(sn[2].second.count(0) == 1);
	REQUIRE(sn[2].second.count(1) == 1);
}

} // namespace celerity
