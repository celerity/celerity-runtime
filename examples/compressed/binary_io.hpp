#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "./floating_point_precision.hpp"
#include <celerity.h>

namespace binary_io {

namespace detail {
	struct grid_index {
		int x, y;
	};
} // namespace detail

template <typename Point, typename Grid, typename SizeRetriever>
void write_grid_file_internal(const std::string& filename, const Grid& grid, const std::pair<int, int>& xy, const SizeRetriever& retrieve_size) {
	std::ofstream binary_file(filename, std::ios::binary);
	if(!binary_file) {
		std::cerr << "Cannot open the binary file for writing." << std::endl;
		return;
	}

	const size_t width = static_cast<size_t>(xy.first);
	const size_t height = static_cast<size_t>(xy.second);
	const size_t tile_capacity = (width == 0 || height == 0) ? 0 : grid.size() / (width * height);

	for(int i = 0; i < xy.first; i++) {
		for(int j = 0; j < xy.second; j++) {
			detail::grid_index grid_index{};
			grid_index.x = i;
			grid_index.y = j;

			binary_file.write(reinterpret_cast<char*>(&grid_index), sizeof(detail::grid_index));
			int size = retrieve_size(i, j);
			binary_file.write(reinterpret_cast<char*>(&size), sizeof(int));
			for(int k = 0; k < size; k++) {
				// Flattened layout is [x][y][z] with a fixed z-capacity per tile.
				size_t linear_idx = (((static_cast<size_t>(i) * height) + static_cast<size_t>(j)) * tile_capacity) + static_cast<size_t>(k);
				const Point point = grid[linear_idx];
				binary_file.write(reinterpret_cast<const char*>(&point), sizeof(DataTY) * 3);
			}
		}
	}
}

template <typename Point, typename Grid, typename Sizes>
void write_grid_file(const std::string& filename, const Grid& grid, const std::pair<int, int>& xy, Sizes& size) {
	write_grid_file_internal<Point>(filename, grid, xy, [&size](size_t i, size_t j) { return size[celerity::id<2>(i, j)]; });
}

template <typename Point>
void write_grid_file(const std::string& filename, const std::vector<std::vector<std::vector<Point>>>& grid) {
	write_grid_file_internal<Point>(
	    filename, grid, {grid.size(), grid.empty() ? 0 : grid[0].size()}, [&grid](size_t i, size_t j) { return grid[i][j].size(); });
}

template <typename Point>
std::vector<std::vector<std::vector<Point>>> read_grid_file(const std::string& filename) {
	std::vector<std::vector<std::vector<Point>>> grid{};

	std::ifstream binary_file(filename, std::ios::binary);
	if(!binary_file) {
		std::cerr << "Cannot open the binary file for reading." << std::endl;
		return grid;
	}

	while(true) {
		int amount = 0;
		detail::grid_index grid_index{};
		if(!binary_file.read(reinterpret_cast<char*>(&grid_index), sizeof(detail::grid_index))) { break; }
		if(!binary_file.read(reinterpret_cast<char*>(&amount), sizeof(int))) { break; }

		if(static_cast<size_t>(grid_index.x) >= grid.size()) { grid.push_back(std::vector<std::vector<Point>>()); }

		if(static_cast<size_t>(grid_index.y) >= grid[grid_index.x].size()) { grid[grid_index.x].push_back(std::vector<Point>()); }

		for(int i = 0; i < amount; i++) {
			Point point;
			if(!binary_file.read(reinterpret_cast<char*>(&point), sizeof(DataTY) * 3)) { return grid; }

			grid[grid_index.x][grid_index.y].push_back(point);
		}
	}

	return grid;
}

template <typename Point, typename PointGrid, typename AmountGrid>
void read_grid_file(const std::string& filename, PointGrid& grid, AmountGrid& amount_grid) {
	std::ifstream binary_file(filename, std::ios::binary);
	if(!binary_file) {
		std::cerr << "Cannot open the binary file for reading." << std::endl;
		return;
	}

	while(true) {
		int amount = 0;
		detail::grid_index grid_index{};
		if(!binary_file.read(reinterpret_cast<char*>(&grid_index), sizeof(detail::grid_index))) { break; }
		if(!binary_file.read(reinterpret_cast<char*>(&amount), sizeof(int))) { break; }

		amount_grid[celerity::id<2>(grid_index.x, grid_index.y)] = amount;

		for(int i = 0; i < amount; i++) {
			Point point;
			if(!binary_file.read(reinterpret_cast<char*>(&point), sizeof(DataTY) * 3)) { return; }

			// Use linear indexing for 1D decompressed data
			// size_t linear_idx = (static_cast<size_t>(grid_index.x) * 105000) + (static_cast<size_t>(grid_index.y) * 105) + (static_cast<size_t>(i));
			grid[celerity::id<3>(grid_index.x, grid_index.y, i)] = point;
		}
	}
}

template <typename Point>
void write_point_file(const std::string& filename, const std::vector<Point>& points) {
	std::ofstream binary_file(filename, std::ios::binary);
	if(!binary_file) {
		std::cerr << "Cannot open the binary file for writing." << std::endl;
		return;
	}

	for(const Point& point : points) {
		binary_file.write(reinterpret_cast<const char*>(&point), sizeof(DataTY) * 3);
	}
}

template <typename Point>
std::vector<Point> read_point_file(const std::string& filename) {
	std::vector<Point> points;

	std::ifstream binary_file(filename, std::ios::binary);
	if(!binary_file) {
		std::cerr << "Cannot open the binary file for reading." << std::endl;
		return points;
	}

	Point point;
	while(binary_file.read(reinterpret_cast<char*>(&point), sizeof(DataTY) * 3)) {
		points.push_back(point);
	}

	return points;
}

} // namespace binary_io