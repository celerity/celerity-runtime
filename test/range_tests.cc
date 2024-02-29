#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <celerity.h>


namespace celerity::detail {

template <typename Coordinate>
inline constexpr Coordinate zero_coordinate;

template <int Dims>
inline constexpr id<Dims> zero_coordinate<id<Dims>> = id<Dims>();

template <int Dims>
inline constexpr range<Dims> zero_coordinate<range<Dims>> = zeros;

template <typename Coordinate>
inline constexpr Coordinate make_coordinate(const std::initializer_list<size_t>& init) {
	auto coord = zero_coordinate<Coordinate>;
	for(int i = 0; i < std::min(Coordinate::dimensions, static_cast<int>(init.size())); ++i) {
		coord[i] = init.begin()[i];
	}
	return coord;
}

TEMPLATE_TEST_CASE("coordinate operations behave as expected", "[range]", range<0>, range<1>, range<2>, range<3>, id<0>, id<1>, id<2>, id<3>) {
	constexpr auto coord = [](const auto... init) { return make_coordinate<TestType>({static_cast<size_t>(init)...}); };
	constexpr auto lvalue = [](TestType&& rvalue) -> TestType& { return rvalue; };

	CHECK(coord(1, 2, 3) == coord(1, 2, 3));
	CHECK_FALSE(coord(1, 2, 3) != coord(1, 2, 3));

	if(TestType::dimensions >= 1) {
		CHECK(coord(0, 2, 3) != coord(1, 2, 3));
		CHECK_FALSE(coord(0, 2, 3) == coord(1, 2, 3));
	}
	if(TestType::dimensions >= 2) {
		CHECK(coord(1, 0, 3) != coord(1, 2, 3));
		CHECK_FALSE(coord(1, 0, 3) == coord(1, 2, 3));
	}
	if(TestType::dimensions >= 3) {
		CHECK(coord(1, 2, 0) != coord(1, 2, 3));
		CHECK_FALSE(coord(1, 2, 0) == coord(1, 2, 3));
	}

	CHECK(coord(1, 2, 3) + coord(4, 5, 6) == coord(5, 7, 9));
	CHECK(coord(5, 7, 9) - coord(1, 2, 3) == coord(4, 5, 6));
	CHECK(coord(1, 2, 3) * coord(2, 4, 6) == coord(2, 8, 18));
	CHECK(coord(10, 16, 21) / coord(2, 4, 7) == coord(5, 4, 3));
	CHECK(coord(10, 17, 23) % coord(2, 4, 7) == coord(0, 1, 2));
	CHECK(coord(1, 2, 3) << coord(1, 2, 3) == coord(2, 8, 24));
	CHECK(coord(2, 8, 24) >> coord(1, 2, 3) == coord(1, 2, 3));
	CHECK((coord(1, 2, 4) & coord(3, 6, 12)) == coord(1, 2, 4));
	CHECK((coord(1, 2, 4) | coord(3, 6, 12)) == coord(3, 6, 12));
	CHECK((coord(1, 2, 4) ^ coord(3, 6, 12)) == coord(2, 4, 8));
	CHECK((coord(2, 0, 4) && coord(9, 0, 0)) == coord(1, 0, 0));
	CHECK((coord(2, 0, 4) || coord(9, 0, 0)) == coord(1, 0, 1));
	CHECK((coord(1, 3, 7) < coord(1, 6, 2)) == coord(0, 1, 0));
	CHECK((coord(1, 6, 2) > coord(1, 3, 7)) == coord(0, 1, 0));
	CHECK((coord(1, 3, 7) <= coord(1, 6, 2)) == coord(1, 1, 0));
	CHECK((coord(1, 6, 2) >= coord(1, 3, 7)) == coord(1, 1, 0));

	CHECK(coord(1, 2, 3) + 1 == coord(2, 3, 4));
	CHECK(coord(1, 2, 3) - 1 == coord(0, 1, 2));
	CHECK(coord(1, 2, 3) * 2 == coord(2, 4, 6));
	CHECK(coord(4, 5, 6) / 2 == coord(2, 2, 3));
	CHECK(coord(4, 5, 6) % 2 == coord(0, 1, 0));
	CHECK(coord(1, 2, 3) << 1 == coord(2, 4, 6));
	CHECK(coord(1, 2, 3) >> 1 == coord(0, 1, 1));
	CHECK((coord(1, 2, 3) & 1) == coord(1, 0, 1));
	CHECK((coord(1, 2, 3) | 1) == coord(1, 3, 3));
	CHECK((coord(1, 2, 3) > 2) == coord(0, 0, 1));
	CHECK((coord(1, 2, 3) < 2) == coord(1, 0, 0));
	CHECK((coord(1, 2, 3) >= 2) == coord(0, 1, 1));
	CHECK((coord(1, 2, 3) <= 2) == coord(1, 1, 0));
	CHECK((coord(1, 2, 0) && 1) == coord(1, 1, 0));
	CHECK((coord(1, 2, 0) || 0) == coord(1, 1, 0));

	CHECK((lvalue(coord(1, 2, 3)) += coord(4, 5, 6)) == coord(5, 7, 9));
	CHECK((lvalue(coord(5, 7, 9)) -= coord(1, 2, 3)) == coord(4, 5, 6));
	CHECK((lvalue(coord(1, 2, 3)) *= coord(2, 4, 6)) == coord(2, 8, 18));
	CHECK((lvalue(coord(10, 16, 21)) /= coord(2, 4, 7)) == coord(5, 4, 3));
	CHECK((lvalue(coord(10, 17, 23)) %= coord(2, 4, 7)) == coord(0, 1, 2));
	CHECK((lvalue(coord(1, 2, 3)) <<= coord(1, 2, 3)) == coord(2, 8, 24));
	CHECK((lvalue(coord(2, 8, 24)) >>= coord(1, 2, 3)) == coord(1, 2, 3));
	CHECK((lvalue(coord(1, 2, 4)) &= coord(3, 6, 12)) == coord(1, 2, 4));
	CHECK((lvalue(coord(1, 2, 4)) |= coord(3, 6, 12)) == coord(3, 6, 12));
	CHECK((lvalue(coord(1, 2, 4)) ^= coord(3, 6, 12)) == coord(2, 4, 8));

	CHECK((lvalue(coord(1, 2, 3)) + 1) == coord(2, 3, 4));
	CHECK((lvalue(coord(1, 2, 3)) - 1) == coord(0, 1, 2));
	CHECK((lvalue(coord(1, 2, 3)) * 2) == coord(2, 4, 6));
	CHECK((lvalue(coord(4, 5, 6)) / 2) == coord(2, 2, 3));
	CHECK((lvalue(coord(4, 5, 6)) % 2) == coord(0, 1, 0));
	CHECK((lvalue(coord(1, 2, 3)) << 1) == coord(2, 4, 6));
	CHECK((lvalue(coord(1, 2, 3)) >> 1) == coord(0, 1, 1));
	CHECK((lvalue(coord(1, 2, 3)) & 1) == coord(1, 0, 1));
	CHECK((lvalue(coord(1, 2, 3)) | 1) == coord(1, 3, 3));
	CHECK((lvalue(coord(1, 2, 3)) > 2) == coord(0, 0, 1));
	CHECK((lvalue(coord(1, 2, 3)) < 2) == coord(1, 0, 0));
	CHECK((lvalue(coord(1, 2, 3)) >= 2) == coord(0, 1, 1));
	CHECK((lvalue(coord(1, 2, 3)) <= 2) == coord(1, 1, 0));

	CHECK(1 + coord(1, 2, 3) == coord(2, 3, 4));
	CHECK(3 - coord(1, 2, 3) == coord(2, 1, 0));
	CHECK(2 * coord(1, 2, 3) == coord(2, 4, 6));
	CHECK(9 / coord(1, 2, 3) == coord(9, 4, 3));
	CHECK(9 % coord(4, 5, 6) == coord(1, 4, 3));
	CHECK(1 << coord(1, 2, 3) == coord(2, 4, 8));
	CHECK(2 >> coord(1, 2, 3) == coord(1, 0, 0));
	CHECK((1 & coord(1, 2, 3)) == coord(1, 0, 1));
	CHECK((1 | coord(1, 2, 3)) == coord(1, 3, 3));
	CHECK((2 > coord(1, 2, 3)) == coord(1, 0, 0));
	CHECK((2 < coord(1, 2, 3)) == coord(0, 0, 1));
	CHECK((2 >= coord(1, 2, 3)) == coord(1, 1, 0));
	CHECK((2 <= coord(1, 2, 3)) == coord(0, 1, 1));
	CHECK((1 && coord(1, 2, 0)) == coord(1, 1, 0));
	CHECK((0 || coord(1, 2, 0)) == coord(1, 1, 0));

	CHECK(+coord(1, 2, 3) == coord(1, 2, 3));
	CHECK(-coord(1, 2, 3) + coord(1, 2, 3) == coord(0, 0, 0));

	{
		auto pre = coord(1, 2, 3);
		CHECK(++pre == coord(2, 3, 4));
		CHECK(pre == coord(2, 3, 4));
	}

	{
		auto pre = coord(1, 2, 3);
		CHECK(--pre == coord(0, 1, 2));
		CHECK(pre == coord(0, 1, 2));
	}

	{
		auto post = coord(1, 2, 3);
		CHECK(post++ == coord(1, 2, 3));
		CHECK(post == coord(2, 3, 4));
	}

	{
		auto post = coord(1, 2, 3);
		CHECK(post-- == coord(1, 2, 3));
		CHECK(post == coord(0, 1, 2));
	}
}

TEST_CASE("0-dimensional ranges are empty types", "[range]") {
	if(!CELERITY_DETAIL_HAS_NO_UNIQUE_ADDRESS) SKIP("[[no_unique_address]] not available");

	// these checks are not static_asserts because they depend on an (optional) compiler layout optimization. Note that is_empty_v<T> does not imply
	// sizeof(T) == 1 (false at least for chunk<0>, which has two range<0> members that can not overlap due to strict-aliasing rules)
	CHECK(std::is_empty_v<range<0>>);
	CHECK(std::is_empty_v<id<0>>);
	CHECK(std::is_empty_v<nd_range<0>>);
	CHECK(std::is_empty_v<chunk<0>>);
	CHECK(std::is_empty_v<subrange<0>>);
}

TEST_CASE("for_each_item behaves as expected", "[host_utils]") {
	SECTION("for 0-dimensional partitions") {
		const auto test_partition_0d = detail::make_partition(range<0>(), subrange<0>());
		int call_count = 0;
		experimental::for_each_item(test_partition_0d, [&](const item<0> item) { call_count++; });
		CHECK(call_count == 1);
	}

	SECTION("for 1-dimensional partitions") {
		const celerity::range<1> global = {10};
		const celerity::subrange<1> range = {2, 5};
		const auto test_partition_1d = detail::make_partition(global, range);
		std::vector<int> call_counts(global.size(), 0);
		experimental::for_each_item(test_partition_1d, [&](const item<1> item) {
			CHECK(item.get_offset() == range.offset);
			CHECK(item.get_range() == global);
			call_counts[item.get_id(0)]++;
		});
		for(size_t i = 0; i < global[0]; ++i) {
			CHECK(call_counts[i] == ((i >= range.offset && i < range.offset + range.range).get(0) ? 1 : 0));
		}
	}

	SECTION("for 2-dimensional partitions") {
		const celerity::range<2> global = {10, 6};
		const celerity::subrange<2> range = {{4, 2}, {3, 1}};
		const auto test_partition_2d = detail::make_partition(global, range);
		std::vector<std::vector<int>> call_counts(global[0], std::vector<int>(global[1], 0));
		experimental::for_each_item(test_partition_2d, [&](const item<2> item) {
			CHECK(item.get_offset() == range.offset);
			CHECK(item.get_range() == global);
			call_counts[item[0]][item[1]]++;
		});
		for(size_t i = 0; i < global[0]; ++i) {
			for(size_t j = 0; j < global[1]; ++j) {
				const celerity::id<2> item = {i, j};
				CHECK(call_counts[i][j] == ((item >= range.offset && item < range.offset + range.range) == celerity::id{1, 1} ? 1 : 0));
			}
		}
	}

	SECTION("for 3-dimensional partitions") {
		const celerity::range<3> global = {8, 6, 3};
		const celerity::subrange<3> range = {{2, 0, 1}, {5, 1, 2}};
		const auto test_partition_3d = detail::make_partition(global, range);
		std::vector<std::vector<std::vector<int>>> call_counts(global[0], std::vector<std::vector<int>>(global[1], std::vector<int>(global[2], 0)));
		experimental::for_each_item(test_partition_3d, [&](const item<3> item) {
			CHECK(item.get_offset() == range.offset);
			CHECK(item.get_range() == global);
			call_counts[item[0]][item[1]][item[2]]++;
		});
		for(size_t i = 0; i < global[0]; ++i) {
			for(size_t j = 0; j < global[1]; ++j) {
				for(size_t k = 0; k < global[2]; ++k) {
					const celerity::id<3> item = {i, j, k};
					CHECK(call_counts[i][j][k] == ((item >= range.offset && item < range.offset + range.range) == celerity::id{1, 1, 1} ? 1 : 0));
				}
			}
		}
	}
}

} // namespace celerity::detail
