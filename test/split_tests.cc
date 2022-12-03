#include <unordered_set>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "test_utils.h"

#include "distributed_graph_generator.h"

using namespace celerity;
using namespace celerity::detail;

namespace celerity::detail {
// FIXME: Reorganize split functions into dedicated TU
std::vector<chunk<3>> split_1d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);
std::vector<chunk<3>> split_2d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);
} // namespace celerity::detail

// TODO: Add test cases for 1D split

namespace {

template <int Dims>
chunk<3> make_full_chunk(range<Dims> range) {
	return {id<3>{}, range_cast<3>(range), range_cast<3>(range)};
}

/**
 * Checks whether a split conforms to a given set of expected ranges and offsets.
 *
 * Note: The following assumes the convention that dimension 0 specifies a "row" or "height",
 *       while dimension 1 specifies a "column" or "width".
 *
 * The expected ranges are specified in a "visual" representation of the form
 *
 * {
 *  { height_0, { width_0, width_1, ..., width_N }},
 *  ...
 *  { height_N, { width_0, width_1, ..., width_N }}
 * },
 *
 * where height_i refers to the height shared by all chunks within this row, and width_i refers
 * to the width of an individual chunk.
 */
void check_2d_split(
    const chunk<3>& full_chunk, const std::vector<chunk<3>>& split_chunks, const std::vector<std::pair<size_t, std::vector<size_t>>>& chunk_ranges) {
	REQUIRE(split_chunks.size() == std::accumulate(chunk_ranges.begin(), chunk_ranges.end(), size_t(0), [](size_t c, auto& p) { return c + p.second.size(); }));
	REQUIRE(std::all_of(chunk_ranges.begin(), chunk_ranges.end(), [&](auto& p) { return p.second.size() == chunk_ranges[0].second.size(); }));
	id<3> offset = full_chunk.offset;
	for(size_t j = 0; j < chunk_ranges.size(); ++j) {
		const auto& [height, widths] = chunk_ranges[j];
		for(size_t i = 0; i < widths.size(); ++i) {
			const auto& chnk = split_chunks[j * chunk_ranges[0].second.size() + i];
			REQUIRE_LOOP(chnk.offset == offset);
			REQUIRE_LOOP(chnk.range[0] == height);
			REQUIRE_LOOP(chnk.range[1] == widths[i]);
			REQUIRE_LOOP(chnk.global_size == full_chunk.global_size);
			offset[1] += widths[i];
		}
		offset[1] = full_chunk.offset[1];
		offset[0] += height;
	}
}

} // namespace

TEST_CASE("2D split produces perfect square chunks if possible", "[split]") {
	const auto full_chunk = make_full_chunk<2>({128, 128});
	const auto chunks = split_2d(full_chunk, unit_range, 4);
	check_2d_split(full_chunk, chunks,
	    {
	        {64, {64, 64}}, //
	        {64, {64, 64}}  //
	    });
}

// FIXME: These are simply transferred from jupyter notebook for now.

TEST_CASE("kitchen sink", "[split]") {
	// Simple constrained split
	{
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {8, 8, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {64, 64}}, //
		        {32, {64, 64}}, //
		        {32, {64, 64}}, //
		        {32, {64, 64}}  //
		    });
	}
	// Uneven, unconstrained split
	{
		const auto full_chunk = make_full_chunk<2>({100, 100});
		const auto chunks = split_2d(full_chunk, unit_range, 6);
		check_2d_split(full_chunk, chunks,
		    {
		        {34, {50, 50}}, //
		        {33, {50, 50}}, //
		        {33, {50, 50}}, //
		    });
	}
	// Uneven, constrained split
	{
		const auto full_chunk = make_full_chunk<2>({128, 120});
		const auto chunks = split_2d(full_chunk, {8, 8, 1}, 4);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {64, 56}}, //
		        {64, {64, 56}}, //
		    });
	}
	// Uneven, constrained split
	{
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {16, 8, 1}, 12);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {48, 40, 40}}, //
		        {32, {48, 40, 40}}, //
		        {32, {48, 40, 40}}, //
		        {32, {48, 40, 40}}, //
		    });
	}
	// Not sure what this one even demonstrates
	{
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {8, 32, 1}, 4);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {64, 64}}, //
		        {64, {64, 64}}, //
		    });
	}
	// Can't create requested number of chunks
	{
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {64, 64, 1}, 3);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {128}}, //
		        {64, {128}}, //
		    });
	}
	// Transpose to fit constraints (1/2)
	{
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {8, 64, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {64, 64}}, //
		        {32, {64, 64}}, //
		        {32, {64, 64}}, //
		        {32, {64, 64}}, //
		    });
	}
	// Transpose to fit constraints (2/2)
	{
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {64, 8, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {32, 32, 32, 32}}, //
		        {64, {32, 32, 32, 32}}, //
		    });
	}
	// Non-square full chunk, constrained to 1D split
	{
		const auto full_chunk = make_full_chunk<2>({256, 128});
		const auto chunks = split_2d(full_chunk, {8, 128, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {128}}, //
		        {32, {128}}, //
		        {32, {128}}, //
		        {32, {128}}, //
		        {32, {128}}, //
		        {32, {128}}, //
		        {32, {128}}, //
		        {32, {128}}, //
		    });
	}
	// Transpose split to minimize edge lengths (1/2)
	{
		const auto full_chunk = make_full_chunk<2>({256, 64});
		const auto chunks = split_2d(full_chunk, unit_range, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {32, 32}}, //
		        {64, {32, 32}}, //
		        {64, {32, 32}}, //
		        {64, {32, 32}}, //
		    });
	}
	// Transpose split to minimize edge lengths (2/2)
	{
		const auto full_chunk = make_full_chunk<2>({64, 256});
		const auto chunks = split_2d(full_chunk, unit_range, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {64, 64, 64, 64}}, //
		        {32, {64, 64, 64, 64}}, //
		    });
	}
}

TEST_CASE("2D split preserves offset of original chunk", "[split]") {
	const auto full_chunk = chunk<3>{{37, 42, 0}, {64, 64, 1}, {128, 128, 1}};
	const auto chunks = split_2d(full_chunk, unit_range, 4);

	CHECK(chunks[0].offset == id<3>{37, 42, 0});
	CHECK(chunks[1].offset == id<3>{37, 42 + 32, 0});
	CHECK(chunks[2].offset == id<3>{37 + 32, 42 + 0, 0});
	CHECK(chunks[3].offset == id<3>{37 + 32, 42 + 32, 0});

	check_2d_split(full_chunk, chunks,
	    {
	        {32, {32, 32}}, //
	        {32, {32, 32}}  //
	    });
}

// TODO:
// - 2D split preserves offset of original chunk
// - 2D split preserves third dimension
