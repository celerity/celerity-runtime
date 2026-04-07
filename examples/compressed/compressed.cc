#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <celerity.h>

#include <compression.h>
#include <compression_impl.h>
#include <compression_wrapper.h>


#include "./binary_io.hpp"
#include "./eigen_decomposition.hpp"
#include "./floating_point_precision.hpp"
#include "./performance_counter.hpp"

// --- Data types ---
using Point = sycl::vec<DataTY, 3>;
using ShapeFactors = sycl::vec<DataTY, 3>;

// --- Compressed types ---
using CompressedPoint = sycl::vec<sycl::half, 3>;
using CompressedZCurve = uint32_t;

// -- Compression types --
// #define INTEGRATION_TEST

#define PERFORMANCE_TEST

#if defined(USE_UNCOMPRESSED)
#define USE_ELEMENT_COMPRESSION
#define USE_POINT_CLOUD_COMPRESSION
#endif

#if !defined(USE_ELEMENT_COMPRESSION) && !defined(USE_LOCAL_COMPRESSION) && !defined(USE_GLOBAL_COMPRESSION)
#define USE_LOCAL_COMPRESSION
#endif

#if !defined(USE_POINT_CLOUD_COMPRESSION) && !defined(USE_ZCURVE_HYBRID) && !defined(USE_POINT_CLOUD_WITH_DEP_COMPRESSION)
#define USE_POINT_CLOUD_COMPRESSION
#endif

#if defined(USE_UNCOMPRESSED)
constexpr auto compression_category = celerity::compression_category::element_wise;
#elif defined(USE_ELEMENT_COMPRESSION)
constexpr auto compression_category = celerity::compression_category::element_wise;
#elif defined(USE_LOCAL_COMPRESSION)
constexpr auto compression_category = celerity::compression_category::local_memory;
#elif defined(USE_GLOBAL_COMPRESSION)
constexpr auto compression_category = celerity::compression_category::global_memory;
#else
#error "Please define either USE_ELEMENT_COMPRESSION, USE_LOCAL_COMPRESSION or USE_GLOBAL_COMPRESSION"
#endif

// using compression_tile_type = celerity::compressed<celerity::compression::point_cloud<Point, CompressedPoint>, compression_category>;
#if defined(USE_UNCOMPRESSED)
using compression_tile_type = celerity::compression::uncompressed;
#elif defined(USE_POINT_CLOUD_COMPRESSION)
using compression_tile_type = celerity::point_cloud_compression<Point, CompressedPoint, compression_category>;
#elif defined(USE_ZCURVE_HYBRID)
using compression_tile_type = celerity::z_curve_hybrid_compression<Point, CompressedZCurve, compression_category>;
#elif defined(USE_POINT_CLOUD_WITH_DEP_COMPRESSION)
using compression_tile_type = celerity::point_cloud_with_dep_compression<Point, CompressedPoint, compression_category>;
#else
#error "Please define either USE_ELEMENT_COMPRESSION, USE_LOCAL_COMPRESSION, USE_GLOBAL_COMPRESSION, or USE_ZCURVE_HYBRID"
#endif

#if defined(USE_UNCOMPRESSED)
using compression_type = celerity::compression::uncompressed;
#else
using compression_type = celerity::quantization_compression<Point, sycl::vec<uint8_t, 3>, celerity::compression_category::element_wise>;
#endif
// using compression_type = celerity::compressed<celerity::compression::quantization<Point, sycl::vec<uint8_t, 3>>, compression_category>;

// --- Range mappers ---

template <int BufferDims>
struct full_third_dim {
	static_assert(BufferDims == 3, "BufferDims must be 3 for full_third_dim");

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<BufferDims> sbr;

		for(int i = 0; i < BufferDims; ++i) {
			if(i == 2) {
				sbr.offset[i] = 0;
				sbr.range[i] = buffer_size[i];
			} else {
				sbr.offset[i] = chnk.offset[i];
				sbr.range[i] = chnk.range[i];
			}
		}
		return sbr;
	}
};

template <int BufferDims>
struct full_third_dim_neighborhood {
	static_assert(BufferDims == 3, "BufferDims must be 3 for full_third_dim");

	full_third_dim_neighborhood<BufferDims>(size_t dim0, size_t dim1, size_t dim2) : m_dim0(dim0), m_dim1(dim1), m_dim2(dim2) {}

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<3> result = {celerity::detail::id_cast<3>(chnk.offset), celerity::detail::range_cast<3>(chnk.range)};
		const celerity::id<3> delta = {m_dim0 < result.offset[0] ? m_dim0 : result.offset[0], m_dim1 < result.offset[1] ? m_dim1 : result.offset[1],
		    m_dim2 < result.offset[2] ? m_dim2 : result.offset[2]};
		result.offset -= delta;
		result.range += celerity::range<3>{m_dim0 + delta[0], m_dim1 + delta[1], m_dim2 + delta[2]};
		result.offset[2] = 0;
		result.range[2] = buffer_size[2];

		return result;
	}

  private:
	size_t m_dim0 = 1;
	size_t m_dim1 = 1;
	size_t m_dim2 = 1;
};


template <int BufferDims>
struct three_d_to_two_d_neighborhood {
	static_assert(BufferDims == 2, "BufferDims must be 2 for full_third_dim");

	three_d_to_two_d_neighborhood<BufferDims>(size_t dim0, size_t dim1, size_t dim2) : m_dim0(dim0), m_dim1(dim1), m_dim2(dim2) {}

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<BufferDims> result = {{chnk.offset[0], chnk.offset[1]}, {chnk.range[0], chnk.range[1]}};
		const celerity::id<BufferDims> delta = {m_dim0 < result.offset[0] ? m_dim0 : result.offset[0], m_dim1 < result.offset[1] ? m_dim1 : result.offset[1]};
		result.offset -= delta;
		result.range += celerity::range<BufferDims>{m_dim0 + delta[0], m_dim1 + delta[1]};

		return result;
	}

  private:
	size_t m_dim0 = 1;
	size_t m_dim1 = 1;
	size_t m_dim2 = 1;
};

template <int BufferDims>
struct three_d_to_two_d {
	static_assert(BufferDims == 2, "BufferDims must be 2 for full_third_dim");

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<BufferDims> sbr;

		for(int i = 0; i < BufferDims; ++i) {
			sbr.offset[i] = chnk.offset[i];
			sbr.range[i] = chnk.range[i];
		}
		return sbr;
	}
};

// This function is used to split the work items in the parallel_for loop
template <int TileSize, int WorkItemsPerTile, int PointsPerTile, typename T, typename Q>
static void tiling_calculation(celerity::queue& queue, celerity::buffer<T, 1>& point_buffer, celerity::buffer<Q, 3, compression_tile_type>& tile_points,
    celerity::buffer<int, 2>& tile_point_count, const DataTY x_min, const DataTY y_min, const size_t points_size) {
	static_assert(PointsPerTile > 0, "POINTS_PER_TILE must be greater than 0");
	static_assert(WorkItemsPerTile > 0, "WORK_ITEMS_PER_TILE must be greater than 0");
	static_assert(TileSize > 0, "TILE_SIZE must be greater than 0");

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor point_acc{point_buffer, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor tile_point_acc{tile_points, cgh, full_third_dim<3>{}, celerity::write_only, celerity::no_init};

#if defined(USE_LOCAL_COMPRESSION)
		// Points per tile + some padding to avoid bank conflicts (TODO: THIS IS HARDCODED)
		celerity::local_accessor<Point> local_allocator({PointsPerTile + 10}, cgh);
#endif

		celerity::range<3> range = celerity::range<3>{tile_points.get_range().get(0), tile_points.get_range().get(1), WorkItemsPerTile};
		celerity::range<3> tile_point_range = tile_points.get_range();

		celerity::accessor tile_point_count_acc{tile_point_count, cgh, three_d_to_two_d<2>{}, celerity::write_only, celerity::no_init};

		celerity::experimental::constrain_split(cgh, celerity::range<3>(1, 1, WorkItemsPerTile));
		cgh.parallel_for(celerity::nd_range<3>(range, celerity::range<3>(1, 1, WorkItemsPerTile)), [=](celerity::nd_item<3> item) {
		// auto test = tile_point_acc.decompress_data(item, tile_point_count_acc, POINTS_PER_TILE, {x_min, y_min, 0}, TILE_SIZE);
#if defined(USE_ELEMENT_COMPRESSION) || defined(USE_UNCOMPRESSED)
			auto uncomp_tile_point_acc = tile_point_acc;

#elif defined(USE_LOCAL_COMPRESSION)
			auto alloc = make_local_allocator<Point>(WorkItemsPerTile, local_allocator); // TODO: THIS IS HARDCODED

			auto uncomp_tile_point_acc = tile_point_acc.decompress_data(item, alloc, {celerity::id<3>(item.get_group().get_group_id() * item.get_local_range()), {item.get_local_range(0), item.get_local_range(1), PointsPerTile}, tile_point_range});
#elif defined(USE_GLOBAL_COMPRESSION)
			auto uncomp_tile_point_acc = tile_point_acc.decompress_data(item, {celerity::id<3>(item.get_group().get_group_id() * item.get_local_range()), {item.get_local_range(0), item.get_local_range(1), PointsPerTile}, tile_point_range});
#endif

			size_t amount = points_size / WorkItemsPerTile;
			size_t num_points_per_runner = (amount + 1);

			for(size_t i = 0; i < num_points_per_runner; i++) {
				size_t idx = (i * WorkItemsPerTile) + item.get_local_id(2);

				if(idx >= points_size) { break; }

				T p = point_acc[idx];

				size_t pos_x = (p.x() - x_min) / TileSize;
				size_t pos_y = (p.y() - y_min) / TileSize;

				if(pos_x == item.get_global_id(0) && pos_y == item.get_global_id(1)) {
					sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
					    tile_point_count_acc[{item.get_global_id(0), item.get_global_id(1)}]};

					int x = atomic_ref_count.fetch_add(1);

					uncomp_tile_point_acc[{item.get_global_id(0), item.get_global_id(1), x}] = p;
				}
			}
		});
	});
}

static inline std::array<std::array<DataTY, 3>, 3> matmul(const Point p) {
	// create the matrix by calculating p * p^T
	std::array<std::array<DataTY, 3>, 3> matrix{};
	matrix[0][0] = p.x() * p.x();
	matrix[0][1] = p.x() * p.y();
	matrix[0][2] = p.x() * p.z();
	matrix[1][0] = p.y() * p.x();
	matrix[1][1] = p.y() * p.y();
	matrix[1][2] = p.y() * p.z();
	matrix[2][0] = p.z() * p.x();
	matrix[2][1] = p.z() * p.y();
	matrix[2][2] = p.z() * p.z();

	return matrix;
}

static inline bool is_different(const Point p1, const Point p2) { return p1.x() != p2.x() || p1.y() != p2.y() || p1.z() != p2.z(); }

template <int LocalSearchRadius, int WorkItemsPerTile, int PointsPerTile, typename T, typename U>
static void shape_factor_calculation(celerity::queue& queue, celerity::buffer<T, 3, compression_tile_type>& tile_points,
    celerity::buffer<int, 2>& tile_point_count, celerity::buffer<U, 3, compression_type>& shape_factors, const DataTY radius, const DataTY x_min,
    const DataTY y_min, const int tile_size) {
	size_t neighborhood_radius = static_cast<size_t>(LocalSearchRadius);
	CELERITY_DEBUG("Neighborhood radius: {}", neighborhood_radius);
	std::cout << "Neighborhood radius: " << neighborhood_radius << "\n";

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_point_acc{
		    tile_points, cgh, full_third_dim_neighborhood<3>{neighborhood_radius, neighborhood_radius, neighborhood_radius}, celerity::read_only};
		celerity::accessor tile_point_count_acc{
		    tile_point_count, cgh, three_d_to_two_d_neighborhood<2>{neighborhood_radius, neighborhood_radius, neighborhood_radius}, celerity::read_only};
		celerity::accessor shape_factor_acc{shape_factors, cgh, full_third_dim<3>{}, celerity::write_only, celerity::no_init};
		celerity::range<3> range = celerity::range<3>{tile_points.get_range().get(0), tile_points.get_range().get(1), WorkItemsPerTile};
		celerity::range<3> tile_point_range = tile_points.get_range();

#if defined(USE_LOCAL_COMPRESSION)
		// Points per tile + some padding to avoid bank conflicts (TODO: THIS IS HARDCODED)
		celerity::local_accessor<Point> local_allocator({static_cast<size_t>((PointsPerTile + 10) * 2)}, cgh);
#endif

		cgh.parallel_for<class shape_factors>(celerity::nd_range<3>{range, {1, 1, WorkItemsPerTile}}, [=](celerity::nd_item<3> item) {
			size_t amount = PointsPerTile / WorkItemsPerTile;
			size_t num_points_per_runner = (amount + 1);

			// auto current_tile = tile_point_acc.decompress_data(item, tile_point_count_acc, {x_min, y_min, 0}, tile_size);
			// auto current_tile = [&]() -> auto {
			// 	if constexpr(compression_method == celerity::compression_category::element_wise) {
			// 		return tile_point_acc;
			// 	} else {
			// 		return std::move(tile_point_acc.decompress_data(item, tile_point_count_acc, {x_min, y_min, 0}, tile_size));
			// 	}
			// }();
#if defined(USE_ELEMENT_COMPRESSION) || defined(USE_UNCOMPRESSED)
			auto current_tile = tile_point_acc;
#elif defined(USE_LOCAL_COMPRESSION)
			auto alloc = make_local_allocator<Point>((PointsPerTile + 10) * 2, local_allocator); // TODO: THIS IS HARDCODED
			auto test = tile_point_count_acc[{item.get_global_id(0), item.get_global_id(1)}];
			auto current_tile = tile_point_acc.decompress_data(item, alloc, {celerity::id<3>(item.get_global_id(0), item.get_global_id(1), 0), {1, 1, test}, tile_point_range});
#elif defined(USE_GLOBAL_COMPRESSION)
			auto test = tile_point_count_acc[{item.get_global_id(0), item.get_global_id(1)}];
			auto current_tile = tile_point_acc.decompress_data(item, {celerity::id<3>(item.get_global_id(0), item.get_global_id(1), 0), {1, 1, test}, tile_point_range});
#endif

			for(size_t i = 0; i < num_points_per_runner; i++) {
				size_t idx = i + (item.get_global_id(2) * num_points_per_runner);

				Point p = {0, 0, 0};
				if(idx < PointsPerTile) { p = current_tile[{item.get_global_id(0), item.get_global_id(1), idx}]; }

				std::array<std::array<DataTY, 3>, 3> matrix{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};

				DataTY sum_fermi = 0.0;
				for(int j = -LocalSearchRadius; j <= LocalSearchRadius; j++) {
					for(int k = -LocalSearchRadius; k <= LocalSearchRadius; k++) {
						// auto neighborhood = [&]() -> auto {
						// 	if constexpr(compression_method == celerity::compression_category::element_wise) {
						// 		return tile_point_acc;
						// 	} else {
						// 		return std::move(tile_point_acc.decompress_data(item, j, k, tile_point_count_acc, {x_min, y_min, 0}, tile_size));
						// 	}
						// }();
						// auto neighborhood = tile_point_acc.decompress_data(item, j, k, tile_point_count_acc, {x_min, y_min, 0}, tile_size);
#if defined(USE_ELEMENT_COMPRESSION) || defined(USE_UNCOMPRESSED)
						auto neighborhood = tile_point_acc;
#elif defined(USE_LOCAL_COMPRESSION)
						auto test = tile_point_count_acc[{item.get_global_id(0) + j, item.get_global_id(1) + k}];
						auto neighborhood = tile_point_acc.decompress_data(item, alloc, {celerity::id<3>(item.get_global_id(0) + j, item.get_global_id(1) + k, 0), {1, 1, test}, tile_point_range});
#elif defined(USE_GLOBAL_COMPRESSION)
						auto test = tile_point_count_acc[{item.get_global_id(0) + j, item.get_global_id(1) + k}];
						auto neighborhood = tile_point_acc.decompress_data(item, {celerity::id<3>(item.get_global_id(0) + j, item.get_global_id(1) + k, 0), {1, 1, test}, tile_point_range});
#endif

						if(idx >= PointsPerTile) { continue; }

						if(item.get_global_id(0) + j >= 0 && item.get_global_id(0) + j < item.get_global_range(0) && item.get_global_id(1) + k >= 0
						    && item.get_global_id(1) + k < item.get_global_range(1)) {
							for(int l = 0; l < tile_point_count_acc[{item.get_global_id(0) + j, item.get_global_id(1) + k}]; l++) {
								const Point p2 = neighborhood[{item.get_global_id(0) + j, item.get_global_id(1) + k, l}];
								const Point p3 = p - p2;

								auto distance_p_p2 = sycl::length(p3);

								if(is_different(p, p2) && distance_p_p2 <= radius + 0.01) {
									const DataTY fermi = 1 / (std::exp((distance_p_p2 / radius) - 0.6) / 0.1 + 1);
									sum_fermi += fermi;
									std::array<std::array<DataTY, 3>, 3> matrix2 = matmul(p3);

									for(int m = 0; m < 3; m++) {
										for(int n = 0; n < 3; n++) {
											matrix[m][n] += fermi * matrix2[m][n];
										}
									}
								}
							}
						}
					}
				}

				if(idx >= PointsPerTile) { continue; }

				for(int j = 0; j < 3; j++) {
					for(int k = 0; k < 3; k++) {
						matrix[j][k] = (1 / sum_fermi) * matrix[j][k];
					}
				}

				std::array<std::array<DataTY, 3>, 3> V{};
				std::array<DataTY, 3> d{0, 0, 0};
				eigen_decomposition<3>(matrix, V, d);

				U sf;

				auto sum = d[0] + d[1] + d[2];
				sf.z() = (3 * d[0]) / sum;
				sf.y() = (2 * (d[1] - d[0])) / sum;
				sf.x() = (d[2] - d[1]) / sum;

				shape_factor_acc[{item.get_global_id(0), item.get_global_id(1), idx}] = sf;
			}
		});
	});
}

static constexpr int get_radius(DataTY radius, int tile_size) {
	int tmp_radius = static_cast<int>(radius) / tile_size;
	return tmp_radius * tile_size == radius ? tmp_radius : tmp_radius + 1;
}

int main(int argc, char* argv[]) {
	constexpr int tile_size = 10;
	constexpr DataTY radius = 10;
	constexpr int work_items_per_tile = 128;
	constexpr int points_per_tile = 105;

#ifndef INTEGRATION_TEST
	if(argc != 4) {
		std::cerr << "Usage: " << argv[0] << " <input point file> <output tile file> <output shape factor file>" << std::endl;
		return 1;
	}
#else
	if(argc != 5) {
		std::cerr << "Usage: " << argv[0] << " <input point file> <input tile file> <output tile file> <output shape factor file>" << std::endl;
		return 1;
	}
#endif

#ifndef INTEGRATION_TEST
	std::string input_point_file = argv[1];
	std::string output_tile_file_name = argv[2];
	std::string output_shape_factors_file_name = argv[3];
#else
	std::string input_point_file = argv[1];
	std::string input_tile_file_name = argv[2];
	std::string output_tile_file_name = argv[3];
	std::string output_shape_factors_file_name = argv[4];
#endif

	std::filesystem::path p = std::filesystem::current_path();
	std::cout << "Current path is " << p << std::endl;

	std::cout << "Reading from file " << input_point_file << std::endl;

	std::vector<Point> points = binary_io::read_point_file<Point>(input_point_file);

	DataTY x_min = std::numeric_limits<DataTY>::max();
	DataTY y_min = std::numeric_limits<DataTY>::max();
	DataTY z_min = std::numeric_limits<DataTY>::max();

	DataTY x_max = std::numeric_limits<DataTY>::min();
	DataTY y_max = std::numeric_limits<DataTY>::min();
	DataTY z_max = std::numeric_limits<DataTY>::min();

	for(const Point& p : points) {
		x_max = std::max(x_max, p.x());
		x_min = std::min(x_min, p.x());
		y_max = std::max(y_max, p.y());
		y_min = std::min(y_min, p.y());
		z_max = std::max(z_max, p.z());
		z_min = std::min(z_min, p.z());
	}

	size_t width = static_cast<size_t>(x_max - x_min) + 1;
	size_t height = static_cast<size_t>(y_max - y_min) + 1;

	std::cout << width << " " << height << std::endl;

	width = width / tile_size + 1;
	height = height / tile_size + 1;

	std::cout << width << " " << height << std::endl;

	std::cout << points.size() << " points" << std::endl;

	celerity::queue queue;
	celerity::buffer<Point, 1> point_buffer(points.data(), points.size());

#if defined(USE_UNCOMPRESSED)
	celerity::buffer<Point, 3, compression_tile_type> tile_points({width, height, points_per_tile});
#elif defined(USE_ZCURVE_HYBRID)
	compression_tile_type compression_tile{Point{x_min, y_min, 0}, static_cast<DataTY>(1)};
	celerity::buffer<Point, 3, compression_tile_type> tile_points({width, height, points_per_tile}, compression_tile);
#else
	compression_tile_type compression_tile{Point{x_min, y_min, 0}, tile_size};
	celerity::buffer<Point, 3, compression_tile_type> tile_points({width, height, points_per_tile}, compression_tile);
#endif
	// compression_tile.get_dependencies().tracking().calculate_size({width, height, points_per_tile});

	celerity::buffer<int, 2> tile_point_count({width, height});
	celerity::debug::set_buffer_name(tile_point_count, "tile_point_count");

#if defined(USE_UNCOMPRESSED)
	celerity::buffer<ShapeFactors, 3, compression_type> shape_factors({width, height, points_per_tile});
#else
	compression_type compression(0.0_FT, 1.0_FT);
	celerity::buffer<ShapeFactors, 3, compression_type> shape_factors({width, height, points_per_tile}, compression);
#endif

#if defined(USE_GLOBAL_COMPRESSION)
	tile_points.init(queue);
#endif

	queue.submit([&](celerity::handler& cgh) {
		// set the tile_point_count to 0
		celerity::accessor tile_point_count_acc{tile_point_count, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class set_tile_point_count>(tile_point_count.get_range(), [=](celerity::item<2> item) { tile_point_count_acc[item] = 0; });
	});

	queue.wait(celerity::experimental::barrier);

#ifdef PERFORMANCE_TEST
	performance_counter<3> pc;
#else
	counter_stub<3> pc;
#endif

	pc.record<0>();

	tiling_calculation<tile_size, work_items_per_tile, points_per_tile>(queue, point_buffer, tile_points, tile_point_count, x_min, y_min, points.size());

#ifdef PERFORMANCE_TEST
	queue.wait(celerity::experimental::barrier);
#endif
	pc.record<1>();


#ifdef INTEGRATION_TEST
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_points_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor tile_points_acc{tile_points, cgh, celerity::access::all{}, celerity::read_only_host_task};
		std::pair<size_t, size_t> xy = {tile_points.get_range()[0], tile_points.get_range()[1]};
#if defined(USE_UNCOMPRESSED)
		auto tile_capacity = tile_points.get_range()[2];
#endif

		// celerity::range<3> tile_points_range = tile_points.get_range();
		cgh.host_task(celerity::on_master_node, [=]() {
#if defined(USE_UNCOMPRESSED)
			std::vector<Point> vec;
			vec.reserve(static_cast<size_t>(xy.first) * static_cast<size_t>(xy.second) * tile_capacity);
			for(size_t i = 0; i < static_cast<size_t>(xy.first); ++i) {
				for(size_t j = 0; j < static_cast<size_t>(xy.second); ++j) {
					for(size_t k = 0; k < tile_capacity; ++k) {
						vec.push_back(tile_points_acc[celerity::id<3>(i, j, k)]);
					}
				}
			}
#else
			auto vec = tile_points_acc.decompress_data();
#endif

			binary_io::write_grid_file<Point>(output_tile_file_name, vec, xy, tile_points_count_acc);
		});
	});

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_points_acc{tile_points, cgh, celerity::access::all{}, celerity::write_only_host_task};
		celerity::accessor tile_points_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::write_only_host_task};

		cgh.host_task(celerity::on_master_node, [=]() { binary_io::read_grid_file<Point>(input_tile_file_name, tile_points_acc, tile_points_count_acc); });
	});
#endif

	constexpr int local_search_radius = get_radius(radius, tile_size);

	shape_factor_calculation<local_search_radius, work_items_per_tile, points_per_tile>(
	    queue, tile_points, tile_point_count, shape_factors, radius, x_min, y_min, tile_size);

#ifdef PERFORMANCE_TEST
	queue.wait(celerity::experimental::barrier);
#endif

	pc.record<2>();

	int rank = 0;

	if(rank == 0) {
		pc.print();

		std::cout << "Writing to file " << output_tile_file_name << std::endl;
		std::cout << "Writing to file " << output_shape_factors_file_name << std::endl;
	}

#ifndef INTEGRATION_TEST
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_points_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor tile_points_acc{tile_points, cgh, celerity::access::all{}, celerity::read_only_host_task};
		std::pair<size_t, size_t> xy = {tile_points.get_range()[0], tile_points.get_range()[1]};
#if defined(USE_UNCOMPRESSED)
		auto tile_capacity = tile_points.get_range()[2];
#endif

		// celerity::range<3> tile_points_range = tile_points.get_range();
		cgh.host_task(celerity::on_master_node, [=]() {
#if defined(USE_UNCOMPRESSED)
			std::vector<Point> vec;
			vec.reserve(static_cast<size_t>(xy.first) * static_cast<size_t>(xy.second) * tile_capacity);
			for(size_t i = 0; i < static_cast<size_t>(xy.first); ++i) {
				for(size_t j = 0; j < static_cast<size_t>(xy.second); ++j) {
					for(size_t k = 0; k < tile_capacity; ++k) {
						vec.push_back(tile_points_acc[celerity::id<3>(i, j, k)]);
					}
				}
			}
#else
			auto vec = tile_points_acc.decompress_data();
#endif

			binary_io::write_grid_file<Point>(output_tile_file_name, vec, xy, tile_points_count_acc);
		});
	});
#endif

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor shape_factor_acc{shape_factors, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor tile_point_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::read_only_host_task};
		std::pair<int, int> xy = {shape_factors.get_range()[0], shape_factors.get_range()[1]};
#if defined(USE_UNCOMPRESSED)
		auto shape_capacity = shape_factors.get_range()[2];
#endif
		cgh.host_task(celerity::on_master_node, [=]() {
#if defined(USE_UNCOMPRESSED)
			std::vector<ShapeFactors> vec;
			vec.reserve(static_cast<size_t>(xy.first) * static_cast<size_t>(xy.second) * shape_capacity);
			for(size_t i = 0; i < static_cast<size_t>(xy.first); ++i) {
				for(size_t j = 0; j < static_cast<size_t>(xy.second); ++j) {
					for(size_t k = 0; k < shape_capacity; ++k) {
						vec.push_back(shape_factor_acc[celerity::id<3>(i, j, k)]);
					}
				}
			}
#else
			auto vec = shape_factor_acc.decompress_data();
#endif

			binary_io::write_grid_file<ShapeFactors>(output_shape_factors_file_name, vec, xy, tile_point_count_acc);
		});
	});

	queue.wait(celerity::experimental::barrier);
	return 0;
}
