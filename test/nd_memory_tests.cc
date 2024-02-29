#include "host_utils.h"
#include "nd_memory.h"
#include "test_utils.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;


TEMPLATE_TEST_CASE_SIG("nd_copy_host works correctly in all source- and destination layouts ", "[memory]", ((int Dims), Dims), 0, 1, 2, 3) {
	const auto copy_range = test_utils::truncate_range<Dims>({5, 6, 7});

	int source_shift[Dims];
	int dest_shift[Dims];
	if constexpr(Dims > 0) { source_shift[0] = GENERATE(values({-2, 0, 2})), dest_shift[0] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 1) { source_shift[1] = GENERATE(values({-2, 0, 2})), dest_shift[1] = GENERATE(values({-2, 0, 2})); }
	if constexpr(Dims > 2) { source_shift[2] = GENERATE(values({-2, 0, 2})), dest_shift[2] = GENERATE(values({-2, 0, 2})); }

	range<Dims> source_range = ones;
	range<Dims> dest_range = ones;
	id<Dims> offset_in_source = zeros;
	id<Dims> offset_in_dest = zeros;
	for(int i = 0; i < Dims; ++i) {
		source_range[i] = copy_range[i] + std::abs(source_shift[i]);
		offset_in_source[i] = std::max(0, source_shift[i]);
		dest_range[i] = copy_range[i] + std::abs(dest_shift[i]);
		offset_in_dest[i] = std::max(0, dest_shift[i]);
	}

	CAPTURE(source_range, dest_range, offset_in_source, offset_in_dest, copy_range);

	std::vector<int> source(source_range.size());
	std::iota(source.begin(), source.end(), 1);

	std::vector<int> expected_dest(dest_range.size());
	experimental::for_each_item(copy_range, [&](const item<Dims> it) {
		const auto linear_index_in_source = get_linear_index(source_range, offset_in_source + it.get_id());
		const auto linear_index_in_dest = get_linear_index(dest_range, offset_in_dest + it.get_id());
		expected_dest[linear_index_in_dest] = source[linear_index_in_source];
	});

	std::vector<int> dest(dest_range.size());
	nd_copy_host(source.data(), dest.data(), range_cast<3>(source_range), range_cast<3>(dest_range), id_cast<3>(offset_in_source), id_cast<3>(offset_in_dest),
	    range_cast<3>(copy_range), sizeof(int));

	CHECK(dest == expected_dest);
}
