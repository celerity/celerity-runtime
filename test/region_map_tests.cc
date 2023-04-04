#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <celerity.h>

#include "ranges.h"
#include "region_map.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	GridBox<3> make_grid_box(range<3> range, id<3> offset = {}) { return {id_to_grid_point(offset), id_to_grid_point(offset + range)}; }

	GridRegion<3> make_grid_region(range<3> range, id<3> offset = {}) { return GridRegion<3>(make_grid_box(range, offset)); }

	TEST_CASE("region_map correctly handles region updates", "[region_map]") {
		region_map<std::string> rm(range<3>(256, 128, 1));

		rm.update_region(make_grid_region({256, 1, 1}), "foo");
		{
			const auto rvs = rm.get_region_values(make_grid_region({32, 1, 1}, {32, 0, 0}));
			REQUIRE(rvs.size() == 1);
			REQUIRE(rvs[0].first == make_grid_box({32, 1, 1}, {32, 0, 0}));
			REQUIRE(rvs[0].second == "foo");
		}

		rm.update_region(make_grid_region({64, 1, 1}), "baz");
		{
			const auto rvs = rm.get_region_values(make_grid_region({256, 1, 1}));
			REQUIRE(rvs.size() == 2);
			REQUIRE(rvs[1].first == make_grid_box({64, 1, 1}));
			REQUIRE(rvs[1].second == "baz");
			REQUIRE(rvs[0].first == make_grid_box({192, 1, 1}, {64, 0, 0}));
			REQUIRE(rvs[0].second == "foo");
		}
	}

	TEST_CASE("region_map collapses stored regions with the same values", "[region_map]") {
		// We test region_map<>::collapse_regions by observing the order of the
		// returned boxes. This somewhat relies on implementation details of
		// region_map<>::get_region_values.
		// TODO: We may want to test this directly instead
		region_map<std::unordered_set<size_t>> rm(range<3>(256, 1, 1));
		rm.update_region(make_grid_region({64, 1, 1}, {64, 0, 0}), {1});
		rm.update_region(make_grid_region({64, 1, 1}, {192, 0, 0}), {1});

		auto rvs = rm.get_region_values(make_grid_region({192, 1, 1}, {64, 0, 0}));
		REQUIRE(rvs.size() == 3);
		REQUIRE(rvs[0].first == make_grid_box({64, 1, 1}, {64, 0, 0}));
		REQUIRE(rvs[0].second.size() == 1);
		REQUIRE(rvs[0].second.count(1) == 1);

		// Since this one is returned before the [128,192) box,
		// the {[64,128), [192,256)} region must exist internally.
		REQUIRE(rvs[1].first == make_grid_box({64, 1, 1}, {192, 0, 0}));
		REQUIRE(rvs[1].second.size() == 1);
		REQUIRE(rvs[1].second.count(1) == 1);

		REQUIRE(rvs[2].first == make_grid_box({64, 1, 1}, {128, 0, 0}));
		// This is the default initialized region that was never updated
		REQUIRE(rvs[2].second.empty());
	}

	TEST_CASE("region_map correctly merges with other instance", "[region_map]") {
		region_map<size_t> rm1(range<3>(128, 64, 32));
		region_map<size_t> rm2(range<3>(128, 64, 32));
		rm1.update_region(make_grid_region({128, 64, 32}, {0, 0, 0}), 5);
		rm2.update_region(make_grid_region({128, 8, 1}, {0, 24, 0}), 1);
		rm2.update_region(make_grid_region({128, 24, 1}, {0, 0, 0}), 2);
		rm1.merge(rm2);

		const auto rvs = rm1.get_region_values(make_grid_region({128, 64, 32}));
		REQUIRE(rvs.size() == 4);
		REQUIRE(rvs[0].first == make_grid_box({128, 32, 31}, {0, 0, 1}));
		REQUIRE(rvs[0].second == 5);

		REQUIRE(rvs[1].first == make_grid_box({128, 32, 32}, {0, 32, 0}));
		REQUIRE(rvs[1].second == 5);

		REQUIRE(rvs[2].first == make_grid_box({128, 24, 1}, {0, 0, 0}));
		REQUIRE(rvs[2].second == 2);

		REQUIRE(rvs[3].first == make_grid_box({128, 8, 1}, {0, 24, 0}));
		REQUIRE(rvs[3].second == 1);

		// Attempting to merge region maps with incompatible extents should throw
		const region_map<size_t> rm_incompat(range<3>(128, 64, 30));
		REQUIRE_THROWS_WITH(rm1.merge(rm_incompat), Catch::Matchers::Equals("Incompatible region map"));
	}

} // namespace detail
} // namespace celerity
