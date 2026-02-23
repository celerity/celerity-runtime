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

	for(int i = 0; i < xy.first; i++) {
		for(int j = 0; j < xy.second; j++) {
			detail::grid_index grid_index{};
			grid_index.x = i;
			grid_index.y = j;

			binary_file.write(reinterpret_cast<char*>(&grid_index), sizeof(detail::grid_index));
			int size = retrieve_size(i, j);
			binary_file.write(reinterpret_cast<char*>(&size), sizeof(int));
			for(int k = 0; k < retrieve_size(i, j); k++) {
				const Point point = grid[celerity::detail::get_linear_index({static_cast<size_t>(xy.first), static_cast<size_t>(xy.second), 105},
				    {static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k)})]; // Assuming the grid is stored in a linearized manner
				binary_file.write(reinterpret_cast<const char*>(&point), sizeof(DataTY) * 3);
			}
		}
	}
}

template <typename Point, typename Grid, typename Sizes>
void write_grid_file(const std::string& filename, const Grid& grid, const std::pair<int, int>& xy, Sizes& size) {
	write_grid_file_internal<Point>(filename, grid, xy, [&size](size_t i, size_t j) { return size[{i, j}]; });
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

	while(!binary_file.eof()) {
		int amount = 0;
		detail::grid_index grid_index{};
		binary_file.read(reinterpret_cast<char*>(&grid_index), sizeof(detail::grid_index));
		binary_file.read(reinterpret_cast<char*>(&amount), sizeof(int)); // Read the closing parenthesis

		if(static_cast<size_t>(grid_index.x) >= grid.size()) { grid.push_back(std::vector<std::vector<Point>>()); }

		if(static_cast<size_t>(grid_index.y) >= grid[grid_index.x].size()) { grid[grid_index.x].push_back(std::vector<Point>()); }

		for(int i = 0; i < amount; i++) {
			Point point;
			binary_file.read(reinterpret_cast<char*>(&point), sizeof(DataTY) * 3);

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

	while(!binary_file.eof()) {
		int amount = 0;
		detail::grid_index grid_index{};
		binary_file.read(reinterpret_cast<char*>(&grid_index), sizeof(detail::grid_index));
		binary_file.read(reinterpret_cast<char*>(&amount), sizeof(int)); // Read the closing parenthesis

		amount_grid[grid_index.x][grid_index.y] = amount;

		for(int i = 0; i < amount; i++) {
			Point point;
			binary_file.read(reinterpret_cast<char*>(&point), sizeof(DataTY) * 3);

			grid[grid_index.x][grid_index.y][i] = point;
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