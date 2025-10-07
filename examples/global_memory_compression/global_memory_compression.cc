#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

#include "./binary_io.hpp"
#include "./celerity_shape_factors.hpp"
#include "./celerity_tiling.hpp"
#include "./compression.h"
#include "./umuguc_types.hpp"

#include "./direct_compression.hpp"
#include "./quantization.hpp"
#include "./umuguc_types.hpp"

#include <celerity.h>

#include "./performance_counter.hpp"


constexpr int get_radius(DataTY radius, int tile_size) {
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

	compression_tile_type compression_tile;

	celerity::buffer<Point, 3, compression_tile_type> tile_points({width, height, points_per_tile}, compression_tile);

	tile_points.init(queue);

	celerity::buffer<int, 2> tile_point_count({width, height});
	celerity::debug::set_buffer_name(tile_point_count, "tile_point_count");

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
	std::cout << "Tiling make" << std::endl;

	tiling_calculation<tile_size, work_items_per_tile, points_per_tile>(queue, point_buffer, tile_points, tile_point_count, x_min, y_min, points.size());
	queue.wait(celerity::experimental::barrier);
	std::cout << "Tiling done" << std::endl;

	queue.wait(celerity::experimental::barrier);

#ifdef INTEGRATION_TEST
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_points_acc{tile_points, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor tile_points_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::read_only_host_task};
		std::pair<int, int> xy = {tile_points.get_range()[0], tile_points.get_range()[1]};
		cgh.host_task(
		    celerity::on_master_node, [=]() { binary_io::write_grid_file<Point>(output_tile_file_name, tile_points_acc, xy, tile_points_count_acc); });
	});


	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_points_acc{tile_points, cgh, celerity::access::all{}, celerity::write_only_host_task};
		celerity::accessor tile_points_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::write_only_host_task};

		cgh.host_task(celerity::on_master_node, [=]() { binary_io::read_grid_file<Point>(input_tile_file_name, tile_points_acc, tile_points_count_acc); });
	});
#endif


#ifdef PERFORMANCE_TEST
	queue.wait(celerity::experimental::barrier);
#endif
	pc.record<1>();

	compression_type compression(0.0_FT, 1.0_FT);

	celerity::buffer<ShapeFactors, 3, compression_type> shape_factors({width, height, points_per_tile}, compression);

	constexpr int local_search_radius = get_radius(radius, tile_size);

	std::cout << "Shape start" << std::endl;
	shape_factor_calculation<local_search_radius, work_items_per_tile, points_per_tile>(
	    queue, tile_points, tile_point_count, shape_factors, radius, x_min, y_min, tile_size);
	queue.wait(celerity::experimental::barrier);
	std::cout << "Shape done" << std::endl;

	queue.wait(celerity::experimental::barrier);

	pc.record<2>();

	int rank = 0;
	// MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0) {
		pc.print();

		std::cout << "Writing to file " << output_tile_file_name << std::endl;
		std::cout << "Writing to file " << output_shape_factors_file_name << std::endl;
	}

#ifndef INTEGRATION_TEST
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_points_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor tile_points_acc{tile_points, cgh, celerity::access::all{}, celerity::read_only_host_task};
		std::pair<int, int> xy = {tile_points.get_range()[0], tile_points.get_range()[1]};
		cgh.host_task(celerity::on_master_node, [=]() {
			auto vec = tile_points_acc.decompress_data(width, height, tile_points_count_acc);

			binary_io::write_grid_file<Point>(output_tile_file_name, vec, xy, tile_points_count_acc);
		});
	});
#endif

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor shape_factor_acc{shape_factors, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor tile_point_count_acc{tile_point_count, cgh, celerity::access::all{}, celerity::read_only_host_task};
		std::pair<int, int> xy = {shape_factors.get_range()[0], shape_factors.get_range()[1]};
		cgh.host_task(celerity::on_master_node,
		    [=]() { binary_io::write_grid_file<ShapeFactors>(output_shape_factors_file_name, shape_factor_acc, xy, tile_point_count_acc); });
	});

	queue.wait(celerity::experimental::barrier);

	return 0;
}