#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <celerity.h>

#include "../compression_local_wave_sim/compression.h"

#include "./binary_io.hpp"
#include "./floating_point_precision.hpp"
#include "./performance_counter.hpp"

// --- Data types ---
using Point = sycl::vec<DataTY, 3>;
using ShapeFactors = sycl::vec<DataTY, 3>;

// --- Compressed types ---
using CompressedPoint = sycl::vec<sycl::half, 3>;

// -- Compression types --

#define USE_GLOBAL_COMPRESSION

#if defined(USE_ELEMENT_COMPRESSION)
constexpr auto compression_category = celerity::compression_category::element_wise;
#elif defined(USE_LOCAL_COMPRESSION)
constexpr auto compression_category = celerity::compression_category::local_memory;
#elif defined(USE_GLOBAL_COMPRESSION)
constexpr auto compression_category = celerity::compression_category::global_memory;
#else
#error "Please define either USE_ELEMENT_COMPRESSION, USE_LOCAL_COMPRESSION or USE_GLOBAL_COMPRESSION"
#endif

// using compression_tile_type = celerity::compressed<celerity::compression::point_cloud<Point, CompressedPoint>, compression_category>;
using compression_tile_type = celerity::point_cloud_compression<Point, CompressedPoint, compression_category>;
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
void tiling_calculation(celerity::queue& queue, celerity::buffer<T, 1>& point_buffer, celerity::buffer<Q, 3, compression_tile_type>& tile_points,
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
#if defined(USE_ELEMENT_COMPRESSION)
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
				size_t idx = i * WorkItemsPerTile + item.get_local_id(2);

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

constexpr int get_radius(DataTY radius, int tile_size) {
	int tmp_radius = static_cast<int>(radius) / tile_size;
	return tmp_radius * tile_size == radius ? tmp_radius : tmp_radius + 1;
}

int main(int argc, char* argv[]) {
	constexpr int tile_size = 10;
	// constexpr DataTY radius = 10;
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

	compression_tile_type compression_tile{Point{x_min, y_min, 0}, tile_size};
	std::cout << "test" << std::endl;
	compression_tile.get_dependencies().tracking().calculate_size({width, height, points_per_tile});
	std::cout << "test2" << std::endl;

	celerity::buffer<Point, 3, compression_tile_type> tile_points({width, height, points_per_tile}, compression_tile);

	celerity::buffer<int, 2> tile_point_count({width, height});
	celerity::debug::set_buffer_name(tile_point_count, "tile_point_count");

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

		// celerity::range<3> tile_points_range = tile_points.get_range();
		cgh.host_task(celerity::on_master_node, [=]() {
			auto vec = tile_points_acc.decompress_data();

			binary_io::write_grid_file<Point>(output_tile_file_name, vec, xy, tile_points_count_acc);
		});
	});
#endif

	queue.wait(celerity::experimental::barrier);
	return 0;
}
