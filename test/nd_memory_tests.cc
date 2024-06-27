#include "nd_memory.h"

#include "copy_test_utils.h"
#include "test_utils.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("layout_nd_copy selects the minimum number of strides", "[memory]") {
	// empty
	CHECK(layout_nd_copy({1, 1, 1}, {1, 2, 3}, {0, 0, 0}, {0, 0, 0}, {0, 1, 1}, 1) == nd_copy_layout{0, 0, 0, {}, 0});
	CHECK(layout_nd_copy({1, 1, 1}, {1, 2, 3}, {0, 0, 0}, {0, 0, 0}, {0, 1, 1}, 4) == nd_copy_layout{0, 0, 0, {}, 0});

	// all contiguous
	CHECK(layout_nd_copy({1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 1) == nd_copy_layout{0, 0, 0, {}, 1});
	CHECK(layout_nd_copy({1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, 4) == nd_copy_layout{0, 0, 0, {}, 4});
	CHECK(layout_nd_copy({1, 3, 1}, {1, 1, 1}, {0, 2, 0}, {0, 0, 0}, {1, 1, 1}, 1) == nd_copy_layout{2, 0, 0, {}, 1});
	CHECK(layout_nd_copy({1, 1, 1}, {1, 3, 1}, {0, 0, 0}, {0, 2, 0}, {1, 1, 1}, 1) == nd_copy_layout{0, 2, 0, {}, 1});
	CHECK(layout_nd_copy({5, 3, 2}, {1, 1, 2}, {0, 0, 0}, {0, 0, 0}, {1, 1, 2}, 1) == nd_copy_layout{0, 0, 0, {}, 2});
	CHECK(layout_nd_copy({5, 3, 2}, {1, 1, 2}, {2, 1, 0}, {0, 0, 0}, {1, 1, 2}, 1) == nd_copy_layout{14, 0, 0, {}, 2});
	CHECK(layout_nd_copy({1, 1, 2}, {5, 3, 2}, {0, 0, 0}, {2, 1, 0}, {1, 1, 2}, 1) == nd_copy_layout{0, 14, 0, {}, 2});
	CHECK(layout_nd_copy({5, 1, 3}, {2, 1, 3}, {0, 0, 0}, {0, 0, 0}, {2, 1, 3}, 1) == nd_copy_layout{0, 0, 0, {}, 6});
	CHECK(layout_nd_copy({5, 2, 3}, {7, 2, 3}, {2, 0, 0}, {1, 0, 0}, {2, 2, 3}, 1) == nd_copy_layout{12, 6, 0, {}, 12});
	CHECK(layout_nd_copy({5, 2, 3}, {7, 2, 3}, {2, 0, 0}, {1, 0, 0}, {2, 2, 3}, 2) == nd_copy_layout{24, 12, 0, {}, 24});
	CHECK(layout_nd_copy({5, 2, 3}, {4, 2, 3}, {0, 0, 0}, {0, 0, 0}, {2, 2, 3}, 1) == nd_copy_layout{0, 0, 0, {}, 12});
	CHECK(layout_nd_copy({5, 2, 3}, {4, 2, 3}, {0, 0, 0}, {0, 0, 0}, {2, 2, 3}, 4) == nd_copy_layout{0, 0, 0, {}, 48});

	// one stride
	CHECK(layout_nd_copy({1, 2, 3}, {1, 2, 1}, {0, 0, 0}, {0, 0, 0}, {1, 2, 1}, 1) == nd_copy_layout{0, 0, 1, {{3, 1, 2}}, 1});
	CHECK(layout_nd_copy({1, 2, 3}, {1, 2, 1}, {0, 0, 1}, {0, 0, 0}, {1, 2, 1}, 1) == nd_copy_layout{1, 0, 1, {{3, 1, 2}}, 1});
	CHECK(layout_nd_copy({1, 2, 3}, {1, 2, 1}, {0, 0, 1}, {0, 0, 0}, {1, 2, 1}, 4) == nd_copy_layout{4, 0, 1, {{12, 4, 2}}, 4});
	CHECK(layout_nd_copy({5, 2, 3}, {4, 2, 1}, {0, 0, 0}, {0, 0, 0}, {2, 1, 1}, 1) == nd_copy_layout{0, 0, 1, {{6, 2, 2}}, 1});
	CHECK(layout_nd_copy({4, 3, 2}, {4, 3, 2}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}, 1) == nd_copy_layout{0, 0, 1, {{6, 6, 2}}, 4});
	CHECK(layout_nd_copy({4, 3, 2}, {4, 3, 2}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}, 4) == nd_copy_layout{0, 0, 1, {{24, 24, 2}}, 16});
	CHECK(layout_nd_copy({4, 5, 2}, {4, 5, 2}, {0, 0, 0}, {0, 0, 0}, {2, 4, 2}, 1) == nd_copy_layout{0, 0, 1, {{10, 10, 2}}, 8});
	CHECK(layout_nd_copy({4, 5, 6}, {4, 5, 6}, {0, 0, 0}, {0, 0, 0}, {2, 4, 6}, 1) == nd_copy_layout{0, 0, 1, {{30, 30, 2}}, 24});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {1, 3, 1}, 1) == nd_copy_layout{0, 0, 1, {{3, 3, 3}}, 1});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {1, 2, 2}, 1) == nd_copy_layout{0, 0, 1, {{3, 3, 2}}, 2});
	CHECK(layout_nd_copy({4, 1, 4}, {4, 1, 4}, {0, 0, 0}, {0, 0, 0}, {2, 1, 2}, 1) == nd_copy_layout{0, 0, 1, {{4, 4, 2}}, 2});
	CHECK(layout_nd_copy({4, 1, 4}, {4, 1, 4}, {0, 0, 0}, {0, 0, 0}, {2, 1, 2}, 4) == nd_copy_layout{0, 0, 1, {{16, 16, 2}}, 8});

	// two strides
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}, 1) == nd_copy_layout{0, 0, 2, {{9, 9, 2}, {3, 3, 2}}, 2});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {2, 2, 2}, 4) == nd_copy_layout{0, 0, 2, {{36, 36, 2}, {12, 12, 2}}, 8});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {1, 0, 0}, {0, 0, 0}, {2, 2, 2}, 1) == nd_copy_layout{9, 0, 2, {{9, 9, 2}, {3, 3, 2}}, 2});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {1, 1, 0}, {0, 0, 0}, {2, 2, 2}, 1) == nd_copy_layout{12, 0, 2, {{9, 9, 2}, {3, 3, 2}}, 2});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {1, 0, 0}, {2, 2, 2}, 1) == nd_copy_layout{0, 9, 2, {{9, 9, 2}, {3, 3, 2}}, 2});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {0, 0, 0}, {1, 1, 0}, {2, 2, 2}, 1) == nd_copy_layout{0, 12, 2, {{9, 9, 2}, {3, 3, 2}}, 2});
	CHECK(layout_nd_copy({2, 3, 4}, {3, 6, 5}, {0, 0, 0}, {0, 0, 0}, {2, 3, 4}, 1) == nd_copy_layout{0, 0, 2, {{12, 30, 2}, {4, 5, 3}}, 4});
	CHECK(layout_nd_copy({3, 3, 3}, {3, 3, 3}, {1, 0, 0}, {0, 0, 0}, {2, 2, 2}, 2) == nd_copy_layout{18, 0, 2, {{18, 18, 2}, {6, 6, 2}}, 4});
}

void dumb_nd_copy_host(
    const void* const source_base, void* const dest_base, const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box, const size_t elem_size) //
{
	REQUIRE(source_box.covers(copy_box));
	REQUIRE(dest_box.covers(copy_box));

	id<3> i;
	for(i[0] = copy_box.get_min()[0]; i[0] < copy_box.get_max()[0]; ++i[0]) {
		for(i[1] = copy_box.get_min()[1]; i[1] < copy_box.get_max()[1]; ++i[1]) {
			for(i[2] = copy_box.get_min()[2]; i[2] < copy_box.get_max()[2]; ++i[2]) {
				const auto offset_in_source = get_linear_index(source_box.get_range(), i - source_box.get_offset()) * elem_size;
				const auto offset_in_dest = get_linear_index(dest_box.get_range(), i - dest_box.get_offset()) * elem_size;
				memcpy(static_cast<std::byte*>(dest_base) + offset_in_dest, static_cast<const std::byte*>(source_base) + offset_in_source, elem_size);
			}
		}
	}
}

TEST_CASE("nd_copy_host works correctly in all source- and destination layouts", "[memory]") {
	for(const auto& [source_box, dest_box, copy_box] : test_utils::generate_copy_test_layouts()) {
		CAPTURE(source_box, dest_box, copy_box);

		std::vector<int> source(source_box.get_area());
		std::iota(source.begin(), source.end(), 1);

		std::vector<int> expected_dest(dest_box.get_area());
		dumb_nd_copy_host(source.data(), expected_dest.data(), source_box, dest_box, copy_box, sizeof(int));

		std::vector<int> dest(dest_box.get_area());
		nd_copy_host(source.data(), dest.data(), source_box, dest_box, copy_box, sizeof(int));

		CHECK(dest == expected_dest);
	}
}
