#pragma once

#include <celerity.h>

#include <array>

#include "./eigen_decomposition.hpp"
#include "./umuguc_types.hpp"

inline std::array<std::array<DataTY, 3>, 3> matmul(const Point p) {
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

inline bool is_different(const Point p1, const Point p2) { return p1.x() != p2.x() || p1.y() != p2.y() || p1.z() != p2.z(); }

inline void print_matrix(const std::array<std::array<DataTY, 3>, 3>& matrix) {
	printf("Matrix: \n");
	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			printf("%lf ", matrix[i][j]);
		}
		printf("\n");
	}
}

template <int LocalSearchRadius, int WorkItemsPerTile, int PointsPerTile, typename T, typename U>
void shape_factor_calculation(celerity::queue& queue, celerity::buffer<T, 3, compression_tile_type>& tile_points, celerity::buffer<int, 2>& tile_point_count,
    celerity::buffer<U, 3, compression_type>& shape_factors, const DataTY radius, const DataTY x_min, const DataTY y_min, const int tile_size) {
	size_t neighborhood_radius = static_cast<size_t>(LocalSearchRadius);
	CELERITY_DEBUG("Neighborhood radius: {}", neighborhood_radius);

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor tile_point_acc{
		    tile_points, cgh, full_third_dim_neighborhood<3>{neighborhood_radius, neighborhood_radius, neighborhood_radius}, celerity::read_only};
		celerity::accessor tile_point_count_acc{
		    tile_point_count, cgh, three_d_to_two_d_neighborhood<2>{neighborhood_radius, neighborhood_radius, neighborhood_radius}, celerity::read_only};
		celerity::accessor shape_factor_acc{shape_factors, cgh, full_third_dim<3>{}, celerity::write_only, celerity::no_init};
		celerity::range<3> range = celerity::range<3>{tile_points.get_range().get(0), tile_points.get_range().get(1), WorkItemsPerTile};

		cgh.parallel_for<class shape_factors>(celerity::nd_range<3>{range, {1, 1, WorkItemsPerTile}}, [=](celerity::nd_item<3> item) {
			size_t amount = PointsPerTile / WorkItemsPerTile;
			size_t num_points_per_runner = (amount + 1);

			auto current_tile = tile_point_acc.decompress_data(item, tile_point_count_acc, {x_min, y_min, 0}, tile_size);

			for(size_t i = 0; i < num_points_per_runner; i++) {
				size_t idx = i + (item.get_global_id(2) * num_points_per_runner);

				Point p = {0, 0, 0};
				if(idx < PointsPerTile) { p = current_tile[{item.get_global_id(0), item.get_global_id(1), idx}]; }

				std::array<std::array<DataTY, 3>, 3> matrix{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};

				DataTY sum_fermi = 0.0;
				for(int j = -LocalSearchRadius; j <= LocalSearchRadius; j++) {
					for(int k = -LocalSearchRadius; k <= LocalSearchRadius; k++) {
						auto neighborhood = tile_point_acc.decompress_data(item, j, k, tile_point_count_acc, {x_min, y_min, 0}, tile_size);

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
