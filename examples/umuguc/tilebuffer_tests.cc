#include "../test/test_utils.h"
#include "tilebuffer_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace celerity;
using namespace celerity::detail;

std::vector<uint32_t> compute_num_entries(const size_t num_ranks, const std::vector<uint32_t>& num_entries_by_rank) {
	std::vector<uint32_t> num_entries(num_entries_by_rank.size() / num_ranks);
	for(size_t j = 0; j < num_ranks; ++j) {
		for(size_t i = 0; i < num_entries.size(); ++i) {
			num_entries[i] += num_entries_by_rank[j * num_entries.size() + i];
		}
	}
	return num_entries;
}

std::vector<uint32_t> compute_num_entries_cumulative(const std::vector<uint32_t>& num_entries) {
	std::vector<uint32_t> num_entries_cumulative(num_entries.size());
	for(size_t i = 1; i < num_entries.size(); ++i) {
		num_entries_cumulative[i] = num_entries[i - 1] + num_entries_cumulative[i - 1];
	}
	return num_entries_cumulative;
}

std::vector<std::vector<subrange<1>>> compute_written_subranges_for_all_ranks(
    const std::vector<uint32_t>& num_entries, const std::vector<uint32_t>& num_entries_by_rank, const std::vector<uint32_t>& num_entries_cumulative) {
	std::vector<std::vector<subrange<1>>> written_subranges_for_all_ranks;
	for(size_t rank = 0; rank < num_entries_by_rank.size() / num_entries.size(); ++rank) {
		written_subranges_for_all_ranks.push_back(compute_written_subranges((int)rank, num_entries, num_entries_by_rank, num_entries_cumulative));
	}

	// Sanity check: No subrange must overlap with any other subrange
	std::vector<region<1>> regions_by_rank;
	for(size_t rank = 0; rank < num_entries_by_rank.size() / num_entries.size(); ++rank) {
		region<1> region;
		for(const auto& sr : written_subranges_for_all_ranks[rank]) {
			CHECK(region_intersection(region, box<1>{sr}).empty());
			region = region_union(region, box<1>{sr});
		}
		regions_by_rank.push_back(region);
	}
	for(size_t i = 0; i < regions_by_rank.size(); ++i) {
		for(size_t j = i + 1; j < regions_by_rank.size(); ++j) {
			CHECK(region_intersection(regions_by_rank[i], regions_by_rank[j]).empty());
		}
	}

	return written_subranges_for_all_ranks;
}

TEST_CASE("compute_written_subranges works as intended") {
	const size_t num_ranks = 2;

	const auto compute = [&](const std::vector<uint32_t>& num_entries_by_rank) {
		const auto num_entries = compute_num_entries(num_ranks, num_entries_by_rank);
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		return compute_written_subranges_for_all_ranks(num_entries, num_entries_by_rank, num_entries_cumulative);
	};

	// Only rank 0 writes
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 1, 1, 0, 0, 0, 0};
		const auto written_srs_by_rank = compute(num_entries_by_rank);
		REQUIRE(written_srs_by_rank[0].size() == 1);
		CHECK(written_srs_by_rank[0][0] == subrange<1>{0, 4});
		CHECK(written_srs_by_rank[1].empty());
	}

	// Only rank 1 writes
	{
		const std::vector<uint32_t> num_entries_by_rank = {0, 0, 0, 0, 1, 1, 1, 1};
		const auto written_srs_by_rank = compute(num_entries_by_rank);
		CHECK(written_srs_by_rank[0].empty());
		REQUIRE(written_srs_by_rank[1].size() == 1);
		CHECK(written_srs_by_rank[1][0] == subrange<1>{0, 4});
	}

	// Both write consecutive, disjoint groups of tiles
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 0, 0, 0, 0, 1, 1};
		const auto written_srs_by_rank = compute(num_entries_by_rank);
		REQUIRE(written_srs_by_rank[0].size() == 1);
		CHECK(written_srs_by_rank[0][0] == subrange<1>{0, 2});
		REQUIRE(written_srs_by_rank[1].size() == 1);
		CHECK(written_srs_by_rank[1][0] == subrange<1>{2, 2});
	}

	// Both write interleaved
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 1, 1, 0, 1, 0, 1};
		const auto written_srs_by_rank = compute(num_entries_by_rank);
		REQUIRE(written_srs_by_rank[0].size() == 2);
		CHECK(written_srs_by_rank[0][0] == subrange<1>{0, 2});
		CHECK(written_srs_by_rank[0][1] == subrange<1>{3, 2});
		REQUIRE(written_srs_by_rank[1].size() == 2);
		CHECK(written_srs_by_rank[1][0] == subrange<1>{2, 1});
		CHECK(written_srs_by_rank[1][1] == subrange<1>{5, 1});
	}

	// Both write interleaved, multiple values
	{
		const std::vector<uint32_t> num_entries_by_rank = {7, 0, 3, 8, 2, 4, 0, 5};
		const auto written_srs_by_rank = compute(num_entries_by_rank);
		REQUIRE(written_srs_by_rank[0].size() == 2);
		REQUIRE(written_srs_by_rank[1].size() == 2);
		CHECK(written_srs_by_rank[0][0] == subrange<1>{0, 7});
		CHECK(written_srs_by_rank[1][0] == subrange<1>{7, 6});
		CHECK(written_srs_by_rank[0][1] == subrange<1>{13, 11});
		CHECK(written_srs_by_rank[1][1] == subrange<1>{24, 5});
	}
}

TEST_CASE("compute_written_subranges_per_gpu works as intended") {
	const size_t num_ranks = 2;

	const auto compute = [&](const std::vector<uint32_t>& num_entries_by_rank, const int rank,
	                         const std::vector<std::vector<uint32_t>>& num_entries_per_device_on_rank) {
		const auto num_entries = compute_num_entries(num_ranks, num_entries_by_rank);
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		return compute_written_subranges_per_device(rank, num_entries, num_entries_by_rank, num_entries_cumulative, num_entries_per_device_on_rank);
	};

	// Only device 0 on rank 0 writes
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 1, 1, 0, 0, 0, 0};
		const std::vector<std::vector<uint32_t>> num_entries_by_device_on_0 = {{1, 1, 1, 1}, {0, 0, 0, 0}};
		const auto written_srs_by_device_on_0 = compute(num_entries_by_rank, 0, num_entries_by_device_on_0);
		CHECK(written_srs_by_device_on_0.at(0).at(0) == subrange<1>{0, 4});
		CHECK(written_srs_by_device_on_0.at(1).empty());
	}

	// Device 1 on rank 0 writes after device 0 on rank 0
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 1, 1, 0, 0, 0, 0};
		const std::vector<std::vector<uint32_t>> num_entries_by_device_on_0 = {{1, 1, 0, 0}, {0, 0, 1, 1}};
		const auto written_srs_by_device_on_0 = compute(num_entries_by_rank, 0, num_entries_by_device_on_0);
		CHECK(written_srs_by_device_on_0.at(0).at(0) == subrange<1>{0, 2});
		CHECK(written_srs_by_device_on_0.at(1).at(0) == subrange<1>{2, 2});
	}

	// Devices on rank 0 write alternating tiles
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 1, 1, 0, 0, 0, 0};
		const std::vector<std::vector<uint32_t>> num_entries_by_device_on_0 = {{1, 0, 1, 0}, {0, 1, 0, 1}};
		const auto written_srs_by_device_on_0 = compute(num_entries_by_rank, 0, num_entries_by_device_on_0);
		CHECK(written_srs_by_device_on_0.at(0).at(0) == subrange<1>{0, 1});
		CHECK(written_srs_by_device_on_0.at(0).at(1) == subrange<1>{2, 1});
		CHECK(written_srs_by_device_on_0.at(1).at(0) == subrange<1>{1, 1});
		CHECK(written_srs_by_device_on_0.at(1).at(1) == subrange<1>{3, 1});
	}

	// Both devices on rank 0 write all tiles
	{
		const std::vector<uint32_t> num_entries_by_rank = {2, 2, 2, 2, 0, 0, 0, 0};
		const std::vector<std::vector<uint32_t>> num_entries_by_device_on_0 = {{1, 1, 1, 1}, {1, 1, 1, 1}};
		const auto written_srs_by_device_on_0 = compute(num_entries_by_rank, 0, num_entries_by_device_on_0);
		CHECK(written_srs_by_device_on_0.at(0).at(0) == subrange<1>{0, 1});
		CHECK(written_srs_by_device_on_0.at(0).at(1) == subrange<1>{2, 1});
		CHECK(written_srs_by_device_on_0.at(0).at(2) == subrange<1>{4, 1});
		CHECK(written_srs_by_device_on_0.at(0).at(3) == subrange<1>{6, 1});

		CHECK(written_srs_by_device_on_0.at(1).at(0) == subrange<1>{1, 1});
		CHECK(written_srs_by_device_on_0.at(1).at(1) == subrange<1>{3, 1});
		CHECK(written_srs_by_device_on_0.at(1).at(2) == subrange<1>{5, 1});
		CHECK(written_srs_by_device_on_0.at(1).at(3) == subrange<1>{7, 1});
	}

	// Device 1 on rank 0 writes after device 0 on rank 0, rank 1 writes in between
	{
		const std::vector<uint32_t> num_entries_by_rank = {1, 1, 1, 1, 0, 1, 0, 0};
		const std::vector<std::vector<uint32_t>> num_entries_by_device_on_0 = {{1, 1, 0, 0}, {0, 0, 1, 1}};
		const auto written_srs_by_device_on_0 = compute(num_entries_by_rank, 0, num_entries_by_device_on_0);
		CHECK(written_srs_by_device_on_0.at(0).at(0) == subrange<1>{0, 2});
		CHECK(written_srs_by_device_on_0.at(1).at(0) == subrange<1>{3, 2});
	}

	// Both devices on rank 0 write interleaved, both ranks write interleaved, multiple values
	{
		const std::vector<uint32_t> num_entries_by_rank = {7, 0, 3, 8, 2, 4, 0, 5};
		const std::vector<std::vector<uint32_t>> num_entries_by_device_on_0 = {{0, 0, 2, 3}, {7, 0, 1, 5}};
		const auto written_srs_by_device_on_0 = compute(num_entries_by_rank, 0, num_entries_by_device_on_0);
		CHECK(written_srs_by_device_on_0.at(1).at(0) == subrange<1>{0, 7});
		CHECK(written_srs_by_device_on_0.at(0).at(0) == subrange<1>{13, 2});
		CHECK(written_srs_by_device_on_0.at(1).at(1) == subrange<1>{15, 1});
		CHECK(written_srs_by_device_on_0.at(0).at(1) == subrange<1>{16, 3});
		CHECK(written_srs_by_device_on_0.at(1).at(2) == subrange<1>{19, 5});
	}
}

TEST_CASE("compute_task_geometry attempts to evenly distribute tiles based on their number of entries") {
	const size_t num_ranks = 4;

	const auto compute = [&](const uint32_t num_entries, const std::vector<uint32_t>& num_entries_cumulative) {
		const auto geo = compute_task_geometry(num_ranks, num_entries, num_entries_cumulative);
		// Check that chunks do not overlap
		detail::region<1> region;
		for(const auto& [chunk, _, _2] : geo.assigned_chunks) {
			CHECK(region_intersection(region, box{subrange_cast<1>(chunk)}).empty());
			region = region_union(region, box{subrange_cast<1>(chunk)});
		}
		return geo.assigned_chunks;
	};

	// 4 tiles, all contain 10 entries
	{
		const uint32_t total_entries = 40;
		const auto num_entries_cumulative = compute_num_entries_cumulative({10, 10, 10, 10});
		const auto geo = compute(total_entries, num_entries_cumulative);
		REQUIRE(geo.size() == 4);
		CHECK(geo[0].sr == subrange_cast<3>(subrange<1>{0, 10}));
		CHECK(geo[1].sr == subrange_cast<3>(subrange<1>{10, 10}));
		CHECK(geo[2].sr == subrange_cast<3>(subrange<1>{20, 10}));
		CHECK(geo[3].sr == subrange_cast<3>(subrange<1>{30, 10}));
	}

	// More complex distribution, same result
	{
		const uint32_t total_entries = 40;
		const auto num_entries_cumulative = compute_num_entries_cumulative({5, 5, 3, 7, 3, 3, 3, 1, 2, 4, 4});
		const auto geo = compute(total_entries, num_entries_cumulative);
		REQUIRE(geo.size() == 4);
		CHECK(geo[0].sr == subrange_cast<3>(subrange<1>{0, 10}));
		CHECK(geo[1].sr == subrange_cast<3>(subrange<1>{10, 10}));
		CHECK(geo[2].sr == subrange_cast<3>(subrange<1>{20, 10}));
		CHECK(geo[3].sr == subrange_cast<3>(subrange<1>{30, 10}));
	}

	// Uneven split towards end
	{
		const uint32_t total_entries = 40;
		const auto num_entries_cumulative = compute_num_entries_cumulative({10, 10, 9, 4, 7});
		const auto geo = compute(total_entries, num_entries_cumulative);
		REQUIRE(geo.size() == 4);
		CHECK(geo[0].sr == subrange_cast<3>(subrange<1>{0, 10}));
		CHECK(geo[1].sr == subrange_cast<3>(subrange<1>{10, 10}));
		CHECK(geo[2].sr == subrange_cast<3>(subrange<1>{20, 13}));
		CHECK(geo[3].sr == subrange_cast<3>(subrange<1>{33, 7}));
	}

	// Suboptimal split
	// FIXME: This split is not optimal due to the greedy algorithm - we may want to revisit this
	//        (A better split would be 10, 9, 9, 12)
	//        => However, with enough tiles, this might not make a difference in practice
	{
		const uint32_t total_entries = 40;
		const auto num_entries_cumulative = compute_num_entries_cumulative({10, 9, 9, 11, 1});
		const auto geo = compute(total_entries, num_entries_cumulative);
		REQUIRE(geo.size() == 4);
		CHECK(geo[0].sr == subrange_cast<3>(subrange<1>{0, 10}));
		CHECK(geo[1].sr == subrange_cast<3>(subrange<1>{10, 18}));
		CHECK(geo[2].sr == subrange_cast<3>(subrange<1>{28, 11}));
		CHECK(geo[3].sr == subrange_cast<3>(subrange<1>{39, 1}));
	}

	// Fewer chunks than requested
	// FIXME: This is again due to the greedy algorithm, better split would be 10, 9, 9, 12
	{
		const uint32_t total_entries = 40;
		const auto num_entries_cumulative = compute_num_entries_cumulative({10, 9, 9, 1, 11});
		const auto geo = compute(total_entries, num_entries_cumulative);
		REQUIRE(geo.size() == 3);
		CHECK(geo[0].sr == subrange_cast<3>(subrange<1>{0, 10}));
		CHECK(geo[1].sr == subrange_cast<3>(subrange<1>{10, 18}));
		CHECK(geo[2].sr == subrange_cast<3>(subrange<1>{28, 12}));
	}
}

TEST_CASE("compute_neighborhood_reads_1d adds a 1-neighborhood to each chunk's reads along a single dimension") {
	const size_t num_ranks = 4;

	const auto compute = [&](const uint32_t total_entries, const std::vector<uint32_t>& num_entries, const std::vector<uint32_t>& num_entries_cumulative) {
		const auto geo = compute_task_geometry(num_ranks, total_entries, num_entries_cumulative);
		const auto reads = compute_neighborhood_reads_1d(geo, total_entries, num_entries, num_entries_cumulative);
		return reads;
	};

	// 4 tiles, all contain 10 entries
	{
		const uint32_t total_entries = 40;
		const std::vector<uint32_t> num_entries = {10, 10, 10, 10};
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		const auto reads = compute(total_entries, num_entries, num_entries_cumulative);
		REQUIRE(reads.size() == 4);
		CHECK(reads[0].at(0) == subrange_cast<3>(subrange<1>{0, 20}));
		CHECK(reads[1].at(0) == subrange_cast<3>(subrange<1>{0, 30}));
		CHECK(reads[2].at(0) == subrange_cast<3>(subrange<1>{10, 30}));
		CHECK(reads[3].at(0) == subrange_cast<3>(subrange<1>{20, 20}));
	}

	// More complex distribution
	{
		const uint32_t total_entries = 40;
		const std::vector<uint32_t> num_entries = {5, 5, 3, 7, 3, 3, 3, 1, 2, 4, 4};
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		const auto reads = compute(total_entries, num_entries, num_entries_cumulative);
		REQUIRE(reads.size() == 4);
		CHECK(reads[0].at(0) == subrange_cast<3>(subrange<1>{0, 13}));
		CHECK(reads[1].at(0) == subrange_cast<3>(subrange<1>{5, 18}));
		CHECK(reads[2].at(0) == subrange_cast<3>(subrange<1>{13, 19}));
		CHECK(reads[3].at(0) == subrange_cast<3>(subrange<1>{29, 11}));
	}

	// Same as above but with an empty tile between each chunk => no neighborhood
	{
		const uint32_t total_entries = 40;
		const std::vector<uint32_t> num_entries = {5, 5, 0, 3, 7, 0, 3, 3, 3, 1, 0, 2, 4, 4};
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		const auto reads = compute(total_entries, num_entries, num_entries_cumulative);
		REQUIRE(reads.size() == 4);
		CHECK(reads[0].at(0) == subrange_cast<3>(subrange<1>{0, 10}));
		CHECK(reads[1].at(0) == subrange_cast<3>(subrange<1>{10, 10}));
		CHECK(reads[2].at(0) == subrange_cast<3>(subrange<1>{20, 10}));
		CHECK(reads[3].at(0) == subrange_cast<3>(subrange<1>{30, 10}));
	}
}

TEST_CASE("compute_neighborhood_reads_2d adds a 1-neighborhood to each chunk's reads along two dimensions") {
	const size_t num_ranks = 4;

	const auto compute = [&](const range<2>& buffer_size, const uint32_t total_entries, const std::vector<uint32_t>& num_entries,
	                         const std::vector<uint32_t>& num_entries_cumulative) {
		const auto geo = compute_task_geometry(num_ranks, total_entries, num_entries_cumulative);
		const auto reads = compute_neighborhood_reads_2d(geo, buffer_size, total_entries, num_entries, num_entries_cumulative);
		return reads;
	};

	// Split neatly into rows, no left/right neighborhood
	{
		const uint32_t total_entries = 120;
		const range<2> buffer_size{4, 3};
		const std::vector<uint32_t> num_entries = {
		    10, 10, 10, //
		    10, 10, 10, //
		    10, 10, 10, //
		    10, 10, 10, //
		};
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		const auto reads = compute(buffer_size, total_entries, num_entries, num_entries_cumulative);
		REQUIRE(reads.size() == 4);

		CHECK(reads[0].at(0) == subrange_cast<3>(subrange<1>{0, 60}));
		CHECK(reads[1].at(0) == subrange_cast<3>(subrange<1>{0, 90}));
		CHECK(reads[2].at(0) == subrange_cast<3>(subrange<1>{30, 90}));
		CHECK(reads[3].at(0) == subrange_cast<3>(subrange<1>{60, 60}));
	}

	// Split spans multiple rows
	{
		const uint32_t total_entries = 120;
		const range<2> buffer_size{4, 3};
		const std::vector<uint32_t> num_entries = {
		    30, 10, 10, // chunk 0, 1
		    10, 30, 1,  // chunk 1, 2, 3
		    5, 4, 0,    // chunk 3
		    10, 0, 10,  // chunk 3
		};
		const auto num_entries_cumulative = compute_num_entries_cumulative(num_entries);
		const auto reads = compute(buffer_size, total_entries, num_entries, num_entries_cumulative);
		REQUIRE(reads.size() == 4);

		CHECK(reads[0].at(0) == subrange_cast<3>(subrange<1>{0, 30 + 10}));  // chunk + right neighbor
		CHECK(reads[0].at(1) == subrange_cast<3>(subrange<1>{50, 10 + 30})); // below chunk

		CHECK(reads[1].at(0) == subrange_cast<3>(subrange<1>{0, (30 + 10 + 10 + 10 + 30) + (1 + 5 + 4)})); // chunk + left and right neighbor, below chunk

		CHECK(reads[2].at(0)
		      == subrange_cast<3>(subrange<1>{0, (30 + 10 + 10) + (10 + 30 + 1) + (5 + 4 + 0)})); // above chunk, chunk + left and right neighbor, below chunk

		CHECK(reads[3].at(0) == subrange_cast<3>(subrange<1>{30, (10 + 10 + 10) + (30 + 1 + 5 + 4 + 0 + 10 + 0 + 10)})); // above chunk, chunk
	}

	// TODO: More disjoint subranges (needs larger grid, I think)
}
