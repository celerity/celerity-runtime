#pragma once

#include "grid.h"

#include <vector>

namespace celerity::test_utils {

struct copy_test_layout {
	detail::box<3> source_box;
	detail::box<3> dest_box;
	detail::box<3> copy_box;
};

inline constexpr range<3> copy_test_max_range{8, 8, 8}; // default range 4 in each dimension, plus potentially 2x2 padding

inline std::vector<copy_test_layout> generate_copy_test_layouts() {
	enum padding { none = 0b00, left = 0b01, right = 0b10, both = 0b11 };
	constexpr size_t padding_width = 2;
	const std::vector<padding> no_padding = {none};
	const std::vector<padding> all_paddings = {none, left, right, both};

	std::vector<copy_test_layout> layouts;
	for(int dims = 0; dims < 3; ++dims) {
		id<3> copy_min{3, 4, 5};
		id<3> copy_max{7, 8, 9};
		for(int d = dims; d < 3; ++d) {
			copy_min[d] = 0;
			copy_max[d] = 1;
		}

		for(const auto source_padding_x : dims > 0 ? all_paddings : no_padding) {
			for(const auto dest_padding_x : dims > 0 ? all_paddings : no_padding) {
				for(const auto source_padding_y : dims > 1 ? all_paddings : no_padding) {
					for(const auto dest_padding_y : dims > 1 ? all_paddings : no_padding) {
						for(const auto source_padding_z : dims > 2 ? all_paddings : no_padding) {
							for(const auto dest_padding_z : dims > 2 ? all_paddings : no_padding) {
								id<3> source_min = copy_min;
								id<3> source_max = copy_max;
								id<3> dest_min = copy_min;
								id<3> dest_max = copy_max;
								const padding source_padding[] = {source_padding_x, source_padding_y, source_padding_z};
								const padding dest_padding[] = {dest_padding_x, dest_padding_y, dest_padding_z};
								for(int d = 0; d < dims; ++d) {
									if((source_padding[d] & left) != 0) { source_min[d] -= padding_width; }
									if((source_padding[d] & right) != 0) { source_max[d] += padding_width; }
									if((dest_padding[d] & left) != 0) { dest_min[d] -= padding_width; }
									if((dest_padding[d] & right) != 0) { dest_max[d] += padding_width; }
								}
								layouts.push_back({
								    detail::box<3>{source_min, source_max},
								    detail::box<3>{dest_min, dest_max},
								    detail::box<3>{copy_min, copy_max},
								});
							}
						}
					}
				}
			}
		}
	}

	return layouts;
}

} // namespace celerity::test_utils
