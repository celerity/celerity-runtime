#pragma once

#include <celerity.h>

#include "./umuguc_types.hpp"

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

		celerity::range<3> range = celerity::range<3>{tile_points.get_range().get(0), tile_points.get_range().get(1), WorkItemsPerTile};

		celerity::accessor tile_point_count_acc{tile_point_count, cgh, three_d_to_two_d<2>{}, celerity::write_only, celerity::no_init};

		celerity::experimental::constrain_split(cgh, celerity::range<3>(1, 1, WorkItemsPerTile));
		cgh.parallel_for(celerity::nd_range<3>(range, celerity::range<3>(1, 1, WorkItemsPerTile)), [=](celerity::nd_item<3> item) {
			// make celerity Debug messages
			// CELERITY_DEBUG("Tiling item {} {} {}", item.get_global_id(0), item.get_global_id(1), item.get_global_id(2));
			auto test = tile_point_acc.decompress_data(item, tile_point_count_acc, {x_min, y_min, 0}, TileSize);
			// CELERITY_DEBUG("Decompressed item {} {} {}", item.get_global_id(0), item.get_global_id(1), item.get_global_id(2));
			// printf("Decompressed item %ld %ld %ld\n", item.get_global_id(0), item.get_global_id(1), item.get_global_id(2));

			size_t amount = points_size / WorkItemsPerTile;
			size_t num_points_per_runner = (amount + 1);

			for(size_t i = 0; i < num_points_per_runner; i++) {
				size_t idx = i * WorkItemsPerTile + item.get_local_id(2);

				if(idx >= points_size) { break; }

				T p = point_acc[idx];

				int pos_x = (p.x() - x_min) / TileSize;
				int pos_y = (p.y() - y_min) / TileSize;

				if(static_cast<size_t>(pos_x) == item.get_global_id(0) && static_cast<size_t>(pos_y) == item.get_global_id(1)) {
					sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
					    tile_point_count_acc[{item.get_global_id(0), item.get_global_id(1)}]};

					int x = atomic_ref_count.fetch_add(1);
					// printf("ADDING POINT %ld %ld %ld at %d\n", p.x(), p.y(), p.z(), x);

					// printf("linear index %ld %ld %ld %ld\n", item.get_global_id(0), item.get_global_id(1), x,
					//     celerity::detail::get_linear_index({tile_points.get_range().get(0), tile_points.get_range().get(1), PointsPerTile},
					//         {item.get_global_id(0), item.get_global_id(1), static_cast<size_t>(x)}));
					test[{item.get_global_id(0), item.get_global_id(1), x}] = point_acc[idx];
				}
			}
		});
	});
}
