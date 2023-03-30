#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <celerity.h>


namespace celerity::detail {

template <typename Coordinate>
inline constexpr Coordinate zero_coordinate;

template <int Dims>
inline constexpr id<Dims> zero_coordinate<id<Dims>> = id<Dims>();

template <int Dims>
inline constexpr range<Dims> zero_coordinate<range<Dims>> = zero_range;

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

} // namespace celerity::detail