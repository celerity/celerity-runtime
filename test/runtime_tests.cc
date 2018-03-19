#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <catch.hpp>
#include <celerity.h>

namespace celerity {
TEST_CASE("Basic", "[buffer_state]") {
	detail::buffer_state<1> bs(cl::sycl::range<1>(256), 2);
	REQUIRE(bs.get_dimensions() == 1);

	auto sn = bs.get_source_nodes(GridRegion<1>(256));
	REQUIRE(sn.size() == 1);
	REQUIRE(sn[0].first == GridBox<1>(256));
	REQUIRE(sn[0].second.size() == 2);
	REQUIRE(sn[0].second.count(0) == 1);
	REQUIRE(sn[0].second.count(1) == 1);
}

TEST_CASE("UpdateRegion", "[buffer_state]") {
	detail::buffer_state<1> bs(cl::sycl::range<1>(256), 2);
	bs.update_region({0, 128}, {1});

	auto sn = bs.get_source_nodes(GridRegion<1>(32, 64));
	REQUIRE(sn.size() == 1);
	REQUIRE(sn[0].first == GridBox<1>(32, 64));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(1) == 1);

	sn = bs.get_source_nodes(GridRegion<1>(256));
	REQUIRE(sn.size() == 2);
	REQUIRE(sn[0].first == GridBox<1>(128, 256));
	REQUIRE(sn[0].second.size() == 2);
	REQUIRE(sn[0].second.count(0) == 1);
	REQUIRE(sn[0].second.count(1) == 1);
	REQUIRE(sn[1].first == GridBox<1>(0, 128));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);
}

TEST_CASE("CollapseRegions", "[buffer_state]") {
	// We test buffer_state<>::collapse_regions by observing the order of the
	// returned boxes. This somewhat relies on implementation details of
	// buffer_state<>::get_source_nodes.
	// TODO: We may want to test this directly instead
	detail::buffer_state<1> bs(cl::sycl::range<1>(256), 2);
	bs.update_region({64, 128}, {1});
	bs.update_region({192, 256}, {1});

	auto sn = bs.get_source_nodes(GridRegion<1>(64, 256));
	REQUIRE(sn.size() == 3);
	REQUIRE(sn[0].first == GridBox<1>(64, 128));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(1) == 1);

	// Since this one is returned before the [128,192) box,
	// the {[64,128), [192,256)} region must exist internally.
	REQUIRE(sn[1].first == GridBox<1>(192, 256));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);

	REQUIRE(sn[2].first == GridBox<1>(128, 192));
	REQUIRE(sn[2].second.size() == 2);
	REQUIRE(sn[2].second.count(0) == 1);
	REQUIRE(sn[2].second.count(1) == 1);
}
} // namespace celerity
