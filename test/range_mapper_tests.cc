#include <catch2/catch_test_macros.hpp>

#include "range_mapper.h"

using namespace celerity;
using namespace celerity::detail;


template <typename T>
extern const int range_dims;
template <int N>
constexpr inline int range_dims<range<N>> = N;

TEST_CASE("range mappers are only invocable with correctly-dimensioned chunks", "[range-mapper]") {
	auto rmfn1 = [](chunk<2> chnk) -> subrange<3> { return {}; };
	using rmfn1_t = decltype(rmfn1);
	static_assert(!is_range_mapper_invocable<rmfn1_t, 1>);
	static_assert(!is_range_mapper_invocable<rmfn1_t, 2>);
	static_assert(is_range_mapper_invocable<rmfn1_t, 3>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 1, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 2, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 3, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 1, 2>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 2, 2>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn1_t, 3, 2>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 1, 3>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 2, 3>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 3, 3>);

	auto rmfn2 = [](auto chnk, range<2>) -> subrange<2> { return {}; };
	using rmfn2_t = decltype(rmfn2);
	static_assert(!is_range_mapper_invocable<rmfn2_t, 1>);
	static_assert(is_range_mapper_invocable<rmfn2_t, 2>);
	static_assert(!is_range_mapper_invocable<rmfn2_t, 3>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 1, 1>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn2_t, 2, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 3, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 1, 2>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn2_t, 2, 2>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 3, 2>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 1, 3>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn2_t, 2, 3>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 3, 3>);

	auto rmfn3 = [](chunk<3> chnk, auto range) -> subrange<range_dims<decltype(range)>> { return {}; };
	using rmfn3_t = decltype(rmfn3);
	static_assert(is_range_mapper_invocable<rmfn3_t, 1>);
	static_assert(is_range_mapper_invocable<rmfn3_t, 2>);
	static_assert(is_range_mapper_invocable<rmfn3_t, 3>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 1, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 2, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 3, 1>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 1, 2>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 2, 2>);
	static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 3, 2>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn3_t, 1, 3>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn3_t, 2, 3>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn3_t, 3, 3>);

	auto rmfn4 = [](auto chnk, auto range) -> subrange<range_dims<decltype(range)>> { return {}; };
	using rmfn4_t = decltype(rmfn4);
	static_assert(is_range_mapper_invocable<rmfn4_t, 1>);
	static_assert(is_range_mapper_invocable<rmfn4_t, 2>);
	static_assert(is_range_mapper_invocable<rmfn4_t, 3>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 1, 1>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 2, 1>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 3, 1>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 1, 2>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 2, 2>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 3, 2>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 1, 3>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 2, 3>);
	static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 3, 3>);
}


template <int Dims>
subrange<Dims> rm_result_to_subrange(const region<Dims>& r) {
	REQUIRE(r.get_boxes().size() == 1);
	return r.get_boxes().front().get_subrange();
}


TEST_CASE("range mapper results are clamped to buffer range", "[range-mapper]") {
	const auto rmfn = [](chunk<3>) { return subrange<3>{{0, 100, 127}, {256, 64, 32}}; };
	range_mapper rm{rmfn, sycl::access::mode::read, range<3>{128, 128, 128}};
	auto sr = rm_result_to_subrange(rm.map_3(chunk<3>{}));
	REQUIRE(sr.offset == id<3>{0, 100, 127});
	REQUIRE(sr.range == range<3>{128, 28, 1});
}

TEST_CASE("one_to_one built-in range mapper behaves as expected", "[range-mapper]") {
	range_mapper rm{access::one_to_one{}, sycl::access::mode::read, range<2>{128, 128}};
	auto sr = rm_result_to_subrange(rm.map_2(chunk<2>{{64, 32}, {32, 4}, {128, 128}}));
	REQUIRE(sr.offset == id<2>{64, 32});
	REQUIRE(sr.range == range<2>{32, 4});
}

TEST_CASE("fixed built-in range mapper behaves as expected", "[range-mapper]") {
	range_mapper rm{access::fixed<1>({{3}, {97}}), sycl::access::mode::read, range<1>{128}};
	auto sr = rm_result_to_subrange(rm.map_1(chunk<2>{{64, 32}, {32, 4}, {128, 128}}));
	REQUIRE(sr.offset == id<1>{3});
	REQUIRE(sr.range == range<1>{97});
}

TEST_CASE("slice built-in range mapper behaves as expected", "[range-mapper]") {
	{
		range_mapper rm{access::slice<3>(0), sycl::access::mode::read, range<3>{128, 128, 128}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}}));
		REQUIRE(sr.offset == id<3>{0, 32, 32});
		REQUIRE(sr.range == range<3>{128, 32, 32});
	}
	{
		range_mapper rm{access::slice<3>(1), sycl::access::mode::read, range<3>{128, 128, 128}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}}));
		REQUIRE(sr.offset == id<3>{32, 0, 32});
		REQUIRE(sr.range == range<3>{32, 128, 32});
	}
	{
		range_mapper rm{access::slice<3>(2), sycl::access::mode::read, range<3>{128, 128, 128}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}}));
		REQUIRE(sr.offset == id<3>{32, 32, 0});
		REQUIRE(sr.range == range<3>{32, 32, 128});
	}
}

TEST_CASE("all built-in range mapper behaves as expected", "[range-mapper]") {
	{
		range_mapper rm{access::all{}, sycl::access::mode::read, range<1>{128}};
		auto sr = rm_result_to_subrange(rm.map_1(chunk<1>{}));
		REQUIRE(sr.offset == id<1>{0});
		REQUIRE(sr.range == range<1>{128});
	}
	{
		range_mapper rm{access::all{}, sycl::access::mode::read, range<2>{128, 64}};
		auto sr = rm_result_to_subrange(rm.map_2(chunk<1>{}));
		REQUIRE(sr.offset == id<2>{0, 0});
		REQUIRE(sr.range == range<2>{128, 64});
	}
	{
		range_mapper rm{access::all{}, sycl::access::mode::read, range<3>{128, 64, 32}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{}));
		REQUIRE(sr.offset == id<3>{0, 0, 0});
		REQUIRE(sr.range == range<3>{128, 64, 32});
	}
}

TEST_CASE("neighborhood built-in range mapper behaves as expected", "[range-mapper]") {
	{
		range_mapper rm{access::neighborhood<1>(10), sycl::access::mode::read, range<1>{128}};
		auto sr = rm_result_to_subrange(rm.map_1(chunk<1>{{15}, {10}, {128}}));
		REQUIRE(sr.offset == id<1>{5});
		REQUIRE(sr.range == range<1>{30});
	}
	{
		range_mapper rm{access::neighborhood<2>(10, 10), sycl::access::mode::read, range<2>{128, 128}};
		auto sr = rm_result_to_subrange(rm.map_2(chunk<2>{{5, 100}, {10, 20}, {128, 128}}));
		REQUIRE(sr.offset == id<2>{0, 90});
		REQUIRE(sr.range == range<2>{25, 38});
	}
	{
		range_mapper rm{access::neighborhood<3>(3, 4, 5), sycl::access::mode::read, range<3>{128, 128, 128}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<3>{{3, 4, 5}, {1, 1, 1}, {128, 128, 128}}));
		REQUIRE(sr.offset == id<3>{0, 0, 0});
		REQUIRE(sr.range == range<3>{7, 9, 11});
	}
}

TEST_CASE("direct_neighborhood built-in range mapper behaves as expected", "[range-mapper]") {
	{
		range_mapper rm{access::direct_neighborhood(10), sycl::access::mode::read, range<1>{128}};
		const auto r = rm.map_1(chunk<1>{{15}, {10}, {128}});
		CHECK(r == box<1>({5}, {35}));
	}
	{
		range_mapper rm{access::direct_neighborhood(10, 10), sycl::access::mode::read, range<2>{128, 128}};
		const auto r = rm.map_2(chunk<2>{{5, 100}, {10, 20}, {128, 128}});
		CHECK(r == region<2>({box<2>({0, 100}, {25, 120}), box<2>({5, 90}, {15, 128})}));
	}
	{
		range_mapper rm{access::direct_neighborhood(3, 4, 5), sycl::access::mode::read, range<3>{128, 128, 128}};
		const auto r = rm.map_3(chunk<3>{{3, 4, 5}, {1, 1, 1}, {128, 128, 128}});
		CHECK(r == region<3>({box<3>({0, 4, 5}, {7, 5, 6}), box<3>({3, 0, 5}, {4, 9, 6}), box<3>({3, 4, 0}, {4, 5, 11})}));
	}
}

TEST_CASE("even_split built-in range mapper behaves as expected", "[range-mapper]") {
	{
		range_mapper rm{experimental::access::even_split<3>(), sycl::access::mode::read, range<3>{128, 345, 678}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{0}, {1}, {8}}));
		REQUIRE(sr.offset == id<3>{0, 0, 0});
		REQUIRE(sr.range == range<3>{16, 345, 678});
	}
	{
		range_mapper rm{experimental::access::even_split<3>(), sycl::access::mode::read, range<3>{128, 345, 678}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{4}, {2}, {8}}));
		REQUIRE(sr.offset == id<3>{64, 0, 0});
		REQUIRE(sr.range == range<3>{32, 345, 678});
	}
	{
		range_mapper rm{experimental::access::even_split<3>(), sycl::access::mode::read, range<3>{131, 992, 613}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{5}, {2}, {7}}));
		REQUIRE(sr.offset == id<3>{95, 0, 0});
		REQUIRE(sr.range == range<3>{36, 992, 613});
	}
	{
		range_mapper rm{experimental::access::even_split<3>(range<3>(10, 1, 1)), sycl::access::mode::read, range<3>{128, 345, 678}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{0}, {1}, {8}}));
		REQUIRE(sr.offset == id<3>{0, 0, 0});
		REQUIRE(sr.range == range<3>{20, 345, 678});
	}
	{
		range_mapper rm{experimental::access::even_split<3>(range<3>(10, 1, 1)), sycl::access::mode::read, range<3>{131, 992, 613}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{0}, {1}, {7}}));
		REQUIRE(sr.offset == id<3>{0, 0, 0});
		REQUIRE(sr.range == range<3>{20, 992, 613});
	}
	{
		range_mapper rm{experimental::access::even_split<3>(range<3>(10, 1, 1)), sycl::access::mode::read, range<3>{131, 992, 613}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{5}, {2}, {7}}));
		REQUIRE(sr.offset == id<3>{100, 0, 0});
		REQUIRE(sr.range == range<3>{31, 992, 613});
	}
	{
		range_mapper rm{experimental::access::even_split<3>(range<3>(10, 1, 1)), sycl::access::mode::read, range<3>{236, 992, 613}};
		auto sr = rm_result_to_subrange(rm.map_3(chunk<1>{{6}, {1}, {7}}));
		REQUIRE(sr.offset == id<3>{200, 0, 0});
		REQUIRE(sr.range == range<3>{36, 992, 613});
	}
}
