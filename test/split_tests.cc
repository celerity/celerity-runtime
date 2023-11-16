#include <unordered_set>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "test_utils.h"

#include "split.h"

using namespace celerity;
using namespace celerity::detail;

namespace {

template <int Dims>
chunk<3> make_full_chunk(range<Dims> range) {
	return {id<3>{}, range_cast<3>(range), range_cast<3>(range)};
}

void check_1d_split(const chunk<3>& full_chunk, const std::vector<chunk<3>>& split_chunks, const std::vector<size_t>& chunk_ranges) {
	REQUIRE(split_chunks.size() == chunk_ranges.size());
	id<3> offset = full_chunk.offset;
	for(size_t i = 0; i < split_chunks.size(); ++i) {
		const auto& chnk = split_chunks[i];
		REQUIRE_LOOP(chnk.offset == offset);
		REQUIRE_LOOP(chnk.range[0] == chunk_ranges[i]);
		REQUIRE_LOOP(chnk.global_size == full_chunk.global_size);
		offset[0] += split_chunks[i].range[0];
	}
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

TEST_CASE("split_1d creates evenly sized chunks if possible", "[split]") {
	const auto full_chunk = make_full_chunk<1>({128});
	const auto chunks = split_1d(full_chunk, ones, 4);
	check_1d_split(full_chunk, chunks, {32, 32, 32, 32});
}

TEST_CASE("split_1d distributes remainder evenly", "[split]") {
	const auto full_chunk = make_full_chunk<1>({13});
	const auto chunks = split_1d(full_chunk, ones, 5);
	check_1d_split(full_chunk, chunks, {3, 3, 3, 2, 2});
}

TEST_CASE("split_1d respects granularity constraints", "[split]") {
	const auto full_chunk = make_full_chunk<1>({80});
	const auto chunks = split_1d(full_chunk, {16, 1, 1}, 3);
	check_1d_split(full_chunk, chunks, {32, 32, 16});
}

TEST_CASE("split_1d creates fewer chunks than requested if mandated by granularity", "[split]") {
	const auto full_chunk = make_full_chunk<1>({96});
	const auto chunks = split_1d(full_chunk, {48, 1, 1}, 3);
	check_1d_split(full_chunk, chunks, {48, 48});
}

TEST_CASE("split_1d preserves offset of original chunk", "[split]") {
	const auto full_chunk = chunk<3>{{37, 42, 7}, {128, 1, 1}, {128, 1, 1}};
	const auto chunks = split_1d(full_chunk, ones, 4);

	CHECK(chunks[0].offset == id<3>{37 + 0, 42, 7});
	CHECK(chunks[1].offset == id<3>{37 + 32, 42, 7});
	CHECK(chunks[2].offset == id<3>{37 + 64, 42, 7});
	CHECK(chunks[3].offset == id<3>{37 + 96, 42, 7});

	check_1d_split(full_chunk, chunks, {32, 32, 32, 32});
}

TEST_CASE("split_1d preserves ranges of original chunk in other dimensions", "[split]") {
	const auto full_chunk = make_full_chunk<3>({128, 42, 341});
	const auto chunks = split_1d(full_chunk, ones, 4);
	for(size_t i = 0; i < 4; ++i) {
		REQUIRE_LOOP(chunks[0].range == range<3>{32, 42, 341});
	}
}

TEST_CASE("split_2d produces perfectly square chunks if possible", "[split]") {
	const auto full_chunk = make_full_chunk<2>({128, 128});
	const auto chunks = split_2d(full_chunk, ones, 4);
	check_2d_split(full_chunk, chunks,
	    {
	        {64, {64, 64}},
	        {64, {64, 64}},
	    });
}

TEST_CASE("split_2d respects granularity constraints") {
	SECTION("simple constrained split") {
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {8, 8, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {64, 64}},
		        {32, {64, 64}},
		        {32, {64, 64}},
		        {32, {64, 64}},
		    });
	}
	SECTION("non-square full chunk, constrained to 1D split") {
		const auto full_chunk = make_full_chunk<2>({256, 128});
		const auto chunks = split_2d(full_chunk, {8, 128, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {128}},
		        {32, {128}},
		        {32, {128}},
		        {32, {128}},
		        {32, {128}},
		        {32, {128}},
		        {32, {128}},
		        {32, {128}},
		    });
	}
	SECTION("very imbalanced, constrained split") {
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {32, 8, 1}, 4);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {64, 64}},
		        {64, {64, 64}},
		    });
	}
}

TEST_CASE("split_2d distributes remainder evenly") {
	SECTION("unconstrained split") {
		const auto full_chunk = make_full_chunk<2>({100, 100});
		const auto chunks = split_2d(full_chunk, ones, 6);
		check_2d_split(full_chunk, chunks,
		    {
		        {34, {50, 50}},
		        {33, {50, 50}},
		        {33, {50, 50}},
		    });
	}
	SECTION("constrained split 1") {
		const auto full_chunk = make_full_chunk<2>({128, 120});
		const auto chunks = split_2d(full_chunk, {8, 8, 1}, 4);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {64, 56}},
		        {64, {64, 56}},
		    });
	}
	SECTION("constrained split 2") {
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {16, 8, 1}, 12);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {48, 40, 40}},
		        {32, {48, 40, 40}},
		        {32, {48, 40, 40}},
		        {32, {48, 40, 40}},
		    });
	}
}

TEST_CASE("split_2d creates fewer chunks than requested if mandated by granularity", "[split]") {
	const auto full_chunk = make_full_chunk<2>({128, 128});
	const auto chunks = split_2d(full_chunk, {64, 64, 1}, 3);
	check_2d_split(full_chunk, chunks,
	    {
	        {64, {128}},
	        {64, {128}},
	    });
}

TEST_CASE("split_2d transposes split to better fit granularity constraints", "[split]") {
	SECTION("case 1") {
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {8, 64, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {64, 64}},
		        {32, {64, 64}},
		        {32, {64, 64}},
		        {32, {64, 64}},
		    });
	}
	SECTION("case 2") {
		const auto full_chunk = make_full_chunk<2>({128, 128});
		const auto chunks = split_2d(full_chunk, {64, 8, 1}, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {32, 32, 32, 32}},
		        {64, {32, 32, 32, 32}},
		    });
	}
}

TEST_CASE("split_2d minimizes edge lengths for non-square domains") {
	SECTION("case 1") {
		const auto full_chunk = make_full_chunk<2>({256, 64});
		const auto chunks = split_2d(full_chunk, ones, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {64, {32, 32}},
		        {64, {32, 32}},
		        {64, {32, 32}},
		        {64, {32, 32}},
		    });
	}
	SECTION("case 2") {
		const auto full_chunk = make_full_chunk<2>({64, 256});
		const auto chunks = split_2d(full_chunk, ones, 8);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {64, 64, 64, 64}},
		        {32, {64, 64, 64, 64}},
		    });
	}
}

TEST_CASE("split_2d preserves offset of original chunk", "[split]") {
	const auto full_chunk = chunk<3>{{37, 42, 7}, {64, 64, 1}, {128, 128, 1}};
	const auto chunks = split_2d(full_chunk, ones, 4);
	CHECK(chunks[0].offset == id<3>{37, 42, 7});
	CHECK(chunks[1].offset == id<3>{37, 42 + 32, 7});
	CHECK(chunks[2].offset == id<3>{37 + 32, 42 + 0, 7});
	CHECK(chunks[3].offset == id<3>{37 + 32, 42 + 32, 7});
}

TEST_CASE("split_2d preserves ranges of original chunk in other dimensions", "[split]") {
	const auto full_chunk = make_full_chunk<3>({128, 128, 341});
	const auto chunks = split_2d(full_chunk, ones, 4);
	for(size_t i = 0; i < 4; ++i) {
		REQUIRE_LOOP(chunks[i].range == range<3>{64, 64, 341});
	}
}

TEST_CASE("the behavior of split_2d on 1-dimensional chunks is well-defined", "[split]") {
	SECTION("even split") {
		const auto full_chunk = make_full_chunk<1>({128});
		const auto chunks = split_2d(full_chunk, ones, 4);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {1}},
		        {32, {1}},
		        {32, {1}},
		        {32, {1}},
		    });
	}

	SECTION("uneven split") {
		const auto full_chunk = make_full_chunk<1>({13});
		const auto chunks = split_1d(full_chunk, ones, 5);
		check_2d_split(full_chunk, chunks,
		    {
		        {3, {1}},
		        {3, {1}},
		        {3, {1}},
		        {2, {1}},
		        {2, {1}},
		    });
	}

	SECTION("constrained split") {
		const auto full_chunk = make_full_chunk<1>({80});
		const auto chunks = split_1d(full_chunk, {16, 1, 1}, 3);
		check_2d_split(full_chunk, chunks,
		    {
		        {32, {1}},
		        {32, {1}},
		        {16, {1}},
		    });
	}
}
