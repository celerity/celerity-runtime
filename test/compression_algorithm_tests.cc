#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "compression.h"
#include "compression_algorithms.h"
#include "compression_impl.h"
#include "compression_wrapper.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

using Point = sycl::vec<float, 3>;

TEST_CASE("quantization round-trips scalar values", "[compression]") {
	const compressed<compression::quantization<float, uint8_t>, compression_category::element_wise> quant{0.0f, 1.0f};
	const auto max_error = 1.0f / 255.0f;

	for(const float value : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
		const auto compressed_value = quant.compress(value, id<1>{0});
		const auto decompressed_value = quant.decompress(compressed_value, id<1>{0});
		CHECK(decompressed_value == Catch::Approx(value).margin(max_error));
	}
}

TEST_CASE("quantization round-trips vec values component-wise", "[compression]") {
	using QuantVec = compressed<compression::quantization<sycl::vec<float, 3>, sycl::vec<uint8_t, 3>>, compression_category::local_memory>;
	const QuantVec quant{-1.0f, 1.0f};
	const auto max_error = 2.0f / 255.0f;

	const sycl::vec<float, 3> value{-0.5f, 0.0f, 0.75f};
	const auto compressed_value = quant.compress(value, id<2>{0, 0});
	const auto decompressed_value = quant.decompress(compressed_value, id<2>{0, 0});

	CHECK(decompressed_value.x() == Catch::Approx(value.x()).margin(max_error));
	CHECK(decompressed_value.y() == Catch::Approx(value.y()).margin(max_error));
	CHECK(decompressed_value.z() == Catch::Approx(value.z()).margin(max_error));
}

TEST_CASE("quantization defaults to [0, 1] bounds", "[compression]") {
	const compressed<compression::quantization<float, uint8_t>, compression_category::element_wise> quant{};
	CHECK(quant.get_lower_bound() == 0.0f);
	CHECK(quant.get_upper_bound() == 1.0f);

	const auto compressed_value = quant.compress(0.5f, id<1>{0});
	const auto decompressed_value = quant.decompress(compressed_value, id<1>{0});
	CHECK(decompressed_value == Catch::Approx(0.5f).margin(1.0f / 255.0f));
}

TEST_CASE("quantization bounds can be reconfigured", "[compression]") {
	compressed<compression::quantization<float, uint8_t>, compression_category::element_wise> quant{0.0f, 1.0f};
	CHECK(quant.get_lower_bound() == 0.0f);
	CHECK(quant.get_upper_bound() == 1.0f);

	quant.set_lower_bound(10.0f);
	quant.set_upper_bound(20.0f);
	CHECK(quant.get_lower_bound() == 10.0f);
	CHECK(quant.get_upper_bound() == 20.0f);

	const auto compressed_value = quant.compress(15.0f, id<1>{0});
	const auto decompressed_value = quant.decompress(compressed_value, id<1>{0});
	CHECK(decompressed_value == Catch::Approx(15.0f).margin((20.0f - 10.0f) / 255.0f));
}

TEST_CASE("conversion round-trips via lossless static_cast", "[compression]") {
	const compressed<compression::conversion<double, int32_t>, compression_category::element_wise> conv{};

	for(const double value : {-100.0, 0.0, 42.0, 1000.0}) {
		const auto compressed_value = conv.compress(value, id<1>{0});
		CHECK(compressed_value == static_cast<int32_t>(value));
		const auto decompressed_value = conv.decompress(compressed_value, id<1>{0});
		CHECK(decompressed_value == Catch::Approx(value));
	}
}

TEST_CASE("point_cloud stores points relative to tile center", "[compression]") {
	const Point min{0.0f, 0.0f, 0.0f};
	const int tile_size = 10;
	const compressed<compression::point_cloud<Point, Point>, compression_category::local_memory> point_cloud{min, tile_size};

	const Point value{25.0f, 35.0f, 5.0f};
	const id<2> tile_index{2, 3};

	const auto compressed_value = point_cloud.compress(value, tile_index);
	CHECK(compressed_value.x() == Catch::Approx(5.0f));
	CHECK(compressed_value.y() == Catch::Approx(5.0f));
	CHECK(compressed_value.z() == Catch::Approx(5.0f));

	const auto decompressed_value = point_cloud.decompress(compressed_value, tile_index);
	CHECK(decompressed_value.x() == Catch::Approx(value.x()));
	CHECK(decompressed_value.y() == Catch::Approx(value.y()));
	CHECK(decompressed_value.z() == Catch::Approx(value.z()));
}

TEST_CASE("point_cloud_with_dep stores points relative to external center", "[compression]") {
	const Point min{0.0f, 0.0f, 0.0f};
	const compressed<compression::point_cloud_with_dep<Point, Point>, compression_category::global_memory> point_cloud{min, 10};

	const Point center{20.0f, 30.0f, 0.0f};
	const Point value{25.0f, 35.0f, 5.0f};

	const auto compressed_value = point_cloud.compress(value, id<2>{2, 3}, center);
	CHECK(compressed_value.x() == Catch::Approx(5.0f));
	CHECK(compressed_value.y() == Catch::Approx(5.0f));
	CHECK(compressed_value.z() == Catch::Approx(5.0f));

	const auto decompressed_value = point_cloud.decompress(compressed_value, id<2>{2, 3}, center);
	CHECK(decompressed_value.x() == Catch::Approx(value.x()));
	CHECK(decompressed_value.y() == Catch::Approx(value.y()));
	CHECK(decompressed_value.z() == Catch::Approx(value.z()));
}

TEST_CASE("z_curve_hybrid morton-encodes x/y and carries z as a dependency", "[compression]") {
	const Point min{0.0f, 0.0f, 0.0f};
	const float xy_resolution = 0.01f;
	const compressed<compression::z_curve_hybrid<Point, uint32_t>, compression_category::global_memory> z_curve{min, xy_resolution};

	const Point value{1.23f, 4.56f, 7.89f};
	const auto compressed_value = z_curve.compress(value, id<2>{0, 0});

	const auto z_dep = static_cast<sycl::half>(value.z() - min.z());
	const auto decompressed_value = z_curve.decompress(compressed_value, id<2>{0, 0}, z_dep);

	CHECK(decompressed_value.x() == Catch::Approx(value.x()).margin(xy_resolution));
	CHECK(decompressed_value.y() == Catch::Approx(value.y()).margin(xy_resolution));
	CHECK(static_cast<float>(decompressed_value.z()) == Catch::Approx(value.z()).margin(0.01f));
}

TEST_CASE("z_curve_hybrid clamps values below minimum", "[compression]") {
	const Point min{10.0f, 10.0f, 0.0f};
	const compressed<compression::z_curve_hybrid<Point, uint32_t>, compression_category::global_memory> z_curve{min, 0.1f};

	const Point value{5.0f, 5.0f, 0.0f};
	const auto compressed_value = z_curve.compress(value, id<2>{0, 0});
	const auto decompressed_value = z_curve.decompress(compressed_value, id<2>{0, 0}, sycl::half{0});

	CHECK(decompressed_value.x() == Catch::Approx(min.x()));
	CHECK(decompressed_value.y() == Catch::Approx(min.y()));
}

TEST_CASE("z_curve_hybrid accepts single-element sycl::vec compressed type", "[compression]") {
	const Point min{0.0f, 0.0f, 0.0f};
	const compressed<compression::z_curve_hybrid<Point, sycl::vec<uint32_t, 1>>, compression_category::global_memory> z_curve{min, 0.01f};

	const Point value{1.23f, 4.56f, 7.89f};
	const auto compressed_value = z_curve.compress(value, id<2>{0, 0});

	const auto z_dep = static_cast<sycl::half>(value.z() - min.z());
	const auto decompressed_value = z_curve.decompress(compressed_value, id<2>{0, 0}, z_dep);

	CHECK(decompressed_value.x() == Catch::Approx(value.x()).margin(0.01f));
	CHECK(decompressed_value.y() == Catch::Approx(value.y()).margin(0.01f));
	CHECK(static_cast<float>(decompressed_value.z()) == Catch::Approx(value.z()).margin(0.01f));
}

namespace {
struct point_grid_3d {
	std::vector<Point> data;
	size_t dim1, dim2;
	const Point& operator[](const celerity::id<3>& idx) const { return data[idx[0] * dim1 * dim2 + idx[1] * dim2 + idx[2]]; }
};
struct point_grid_2d {
	std::vector<Point> data;
	size_t dim1;
	const Point& operator[](const celerity::id<2>& idx) const { return data[idx[0] * dim1 + idx[1]]; }
};
struct flat_grid_3d_u32 {
	std::vector<uint32_t> data;
	uint32_t operator[](const celerity::id<3>& idx) const { return data[idx[0]]; }
};
struct flat_grid_3d_half {
	std::vector<sycl::half> data;
	sycl::half operator[](const celerity::id<3>& idx) const { return data[idx[0]]; }
};

void check_point(const Point& p, float x, float y, float z, float margin = 0.0f) {
	CHECK(p.x() == Catch::Approx(x).margin(margin));
	CHECK(p.y() == Catch::Approx(y).margin(margin));
	CHECK(p.z() == Catch::Approx(z).margin(margin));
}
} // namespace

TEST_CASE("point_cloud_with_dep::decompress_data() bulk-converts with dependency grid", "[compression]") {
	const Point min{0.0f, 0.0f, 0.0f};
	const compressed<compression::point_cloud_with_dep<Point, Point>, compression_category::global_memory> point_cloud{min, 10};

	const point_grid_2d centers{{Point{0, 0, 0}, Point{10, 0, 0}, Point{0, 10, 0}, Point{10, 10, 0}}, 2};
	const point_grid_3d compressed_points{{Point{5, 5, 1}, Point{-5, 5, 2}, Point{5, -5, 3}, Point{-5, -5, 4}}, 2, 1};

	std::vector<Point> decompressed(4);
	point_cloud.decompress_data(compressed_points, decompressed, range<3>{2, 2, 1}, centers);

	check_point(decompressed[0], 5.0f, 5.0f, 1.0f);
	check_point(decompressed[1], 5.0f, 5.0f, 2.0f);
	check_point(decompressed[2], 5.0f, 5.0f, 3.0f);
	check_point(decompressed[3], 5.0f, 5.0f, 4.0f);
}

TEST_CASE("z_curve_hybrid::decompress_data() bulk-converts with z-dependency grid", "[compression]") {
	const Point min{0.0f, 0.0f, 0.0f};
	const compressed<compression::z_curve_hybrid<Point, uint32_t>, compression_category::global_memory> z_curve{min, 0.01f};

	const std::vector<Point> input{Point{1.0f, 2.0f, 3.0f}, Point{4.0f, 5.0f, 6.0f}};
	flat_grid_3d_u32 compressed_grid;
	flat_grid_3d_half z_dep_grid;
	for(size_t i = 0; i < input.size(); ++i) {
		compressed_grid.data.push_back(z_curve.compress(input[i], id<1>{i}));
		z_dep_grid.data.push_back(static_cast<sycl::half>(input[i].z() - min.z()));
	}

	std::vector<Point> output(input.size());
	z_curve.decompress_data(compressed_grid, output, range<3>{input.size(), 1, 1}, z_dep_grid);

	for(size_t i = 0; i < input.size(); ++i) {
		CAPTURE(i);
		check_point(output[i], input[i].x(), input[i].y(), input[i].z(), 0.05f); // xy_resolution + half precision on z
	}
}

TEST_CASE("downscale_point_cloud_buffer maps chunk to tile index", "[compression]") {
	const chunk<3> chnk{id<3>{0, 0, 0}, range<3>{4, 4, 10}, range<3>{8, 8, 10}, range<3>{4, 4, 10}};
	const range<2> tracking_buffer_size{1, 10};
	const detail::region<1> unused_sbr;
	const range<1> unused_main_size{1};

	const auto result = downscale_point_cloud_buffer{}(chnk, tracking_buffer_size, unused_sbr, unused_main_size);

	CHECK(result == detail::region<2>(detail::box<2>(id<2>{0, 0}, id<2>{1, 1})));
}

TEST_CASE("point_cloud_dep_buffer_mapper projects boxes onto XY footprint", "[compression]") {
	const chunk<1> unused_chunk{};
	const range<2> unused_buffer_size{10, 10};
	const detail::region<3> sbr{detail::box<3>(id<3>{1, 2, 5}, id<3>{4, 6, 9})};
	const range<3> unused_main_size{10, 10, 10};

	const auto result = point_cloud_dep_buffer_mapper{}(unused_chunk, unused_buffer_size, sbr, unused_main_size);

	CHECK(result == detail::region<2>(detail::box<2>(id<2>{1, 2}, id<2>{4, 6})));
}

TEST_CASE("full_subrange_passthrough_mapper passes region through unchanged", "[compression]") {
	const chunk<1> unused_chunk{};
	const range<3> unused_buffer_size{1, 1, 1};
	const detail::region<3> sbr{detail::box<3>(id<3>{1, 2, 3}, id<3>{4, 5, 6})};
	const range<3> unused_main_size{10, 10, 10};

	const auto result = full_subrange_passthrough_mapper{}(unused_chunk, unused_buffer_size, sbr, unused_main_size);

	CHECK(result == sbr);
}

TEST_CASE("algorithms declare correct dependency_bundle buffer count", "[compression]") {
	STATIC_REQUIRE(
	    std::tuple_size_v<typename detail::compression_dependency<quantization_compression<float, uint8_t, compression_category::local_memory>>::type> == 0);
	STATIC_REQUIRE(
	    std::tuple_size_v<typename detail::compression_dependency<point_cloud_compression<Point, Point, compression_category::local_memory>>::type> == 0);
	STATIC_REQUIRE(
	    std::tuple_size_v<typename detail::compression_dependency<conversion_compression<float, int32_t, compression_category::local_memory>>::type> == 0);

	STATIC_REQUIRE(
	    std::tuple_size_v<typename detail::compression_dependency<point_cloud_with_dep_compression<Point, Point, compression_category::local_memory>>::type>
	    == 1);

	STATIC_REQUIRE(
	    std::tuple_size_v<typename detail::compression_dependency<z_curve_hybrid_compression<Point, uint32_t, compression_category::local_memory>>::type> == 1);
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "buffers_from_descriptions() creates correctly-sized dependency buffers", "[compression]") {
	SECTION("point_cloud_with_dep: one 2D dependency buffer sized to the main buffer's XY footprint") {
		point_cloud_with_dep_compression<Point, Point, compression_category::local_memory> alg{Point{0, 0, 0}, 10};
		const auto buffers = alg.get_dependencies().buffers_from_descriptions(range<3>{4, 6, 100});
		STATIC_REQUIRE(std::tuple_size_v<std::decay_t<decltype(buffers)>> == 1);
		CHECK(std::get<0>(buffers).get_range() == range<2>{4, 6});
	}

	SECTION("z_curve_hybrid: one 3D dependency buffer matching the full main buffer range") {
		z_curve_hybrid_compression<Point, uint32_t, compression_category::local_memory> alg{Point{0, 0, 0}, 0.1f};
		const auto buffers = alg.get_dependencies().buffers_from_descriptions(range<3>{4, 6, 100});
		STATIC_REQUIRE(std::tuple_size_v<std::decay_t<decltype(buffers)>> == 1);
		CHECK(std::get<0>(buffers).get_range() == range<3>{4, 6, 100});
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "point_cloud_with_dep accessors round-trip via dependency buffer", "[compression][accessor]") {
	test_utils::allow_max_log_level(detail::log_level::warn);

	const Point min{0.0f, 0.0f, 0.0f};
	const int tile_size = 10;
	using CompressionAlgorithm = point_cloud_with_dep_compression<Point, Point, compression_category::global_memory>;
	CompressionAlgorithm compression(min, tile_size);

	constexpr size_t extent = 8, group = 4;
	buffer<Point, 2, CompressionAlgorithm> buf(range<2>{extent, extent}, compression);

	queue q;
	buf.init(q);

	buffer<Point, 2> decompressed_out(range<2>{extent, extent});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(nd_range<2>(buf.get_range(), range<2>{group, group}), [=](nd_item<2> item) {
			auto staged = acc.decompress_data(item);
			const auto local = item.get_local_id();
			const auto tile_offset = item.get_group().get_group_id() * item.get_local_range();
			const Point center{min.x() + tile_size * static_cast<float>(tile_offset[0]), min.y() + tile_size * static_cast<float>(tile_offset[1]), 0.0f};
			staged[item.get_global_id()] = center + Point{static_cast<float>(local[0]), static_cast<float>(local[1]), 0.0f};
		});
	});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::read_only};
		accessor out_acc{decompressed_out, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(nd_range<2>(buf.get_range(), range<2>{group, group}), [=](nd_item<2> item) {
			auto staged = acc.decompress_data(item);
			out_acc[item.get_global_id()] = staged[item.get_global_id()];
		});
	});

	std::vector<Point> result(extent * extent);
	q.submit([&](handler& cgh) {
		accessor out_acc{decompressed_out, cgh, access::all{}, celerity::read_only_host_task};
		cgh.host_task(on_master_node, [=, &result] {
			for(size_t x = 0; x < extent; ++x) {
				for(size_t y = 0; y < extent; ++y) {
					result[x * extent + y] = out_acc[id<2>{x, y}];
				}
			}
		});
	});
	q.wait();

	for(size_t x = 0; x < extent; ++x) {
		for(size_t y = 0; y < extent; ++y) {
			CAPTURE(x, y);
			const auto tile_x = (x / group) * group;
			const auto tile_y = (y / group) * group;
			const float expected_x = min.x() + tile_size * static_cast<float>(tile_x) + static_cast<float>(x % group);
			const float expected_y = min.y() + tile_size * static_cast<float>(tile_y) + static_cast<float>(y % group);
			check_point(result[x * extent + y], expected_x, expected_y, 0.0f);
		}
	}
}
