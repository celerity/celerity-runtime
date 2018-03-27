#pragma once

#include <SYCL/sycl.hpp>
#include <allscale/api/user/data/grid.h>

#include "subrange.h"

using namespace allscale::api::user::data;

namespace celerity {

namespace detail {

	inline GridPoint<1> sycl_range_to_grid_point(cl::sycl::range<1> range) { return GridPoint<1>(range[0]); }

	inline GridPoint<2> sycl_range_to_grid_point(cl::sycl::range<2> range) { return GridPoint<2>(range[0], range[1]); }

	inline GridPoint<3> sycl_range_to_grid_point(cl::sycl::range<3> range) { return GridPoint<3>(range[0], range[1], range[2]); }

	template <int Dims>
	void clamp_range(cl::sycl::range<Dims>& range, const cl::sycl::range<Dims>& max) {
		if(range[0] > max[0]) { range[0] = max[0]; }
		if(range[1] > max[1]) { range[1] = max[1]; }
		if(range[2] > max[2]) { range[2] = max[2]; }
	}

	inline GridRegion<1> subrange_to_grid_region(const subrange<1>& sr) {
		auto end = sr.start + sr.range;
		clamp_range(end, sr.global_size);
		return GridRegion<1>(sycl_range_to_grid_point(sr.start), sycl_range_to_grid_point(end));
	}

	inline GridRegion<2> subrange_to_grid_region(const subrange<2>& sr) {
		auto end = sr.start + sr.range;
		clamp_range(end, sr.global_size);
		return GridRegion<2>(sycl_range_to_grid_point(sr.start), sycl_range_to_grid_point(end));
	}

	inline GridRegion<3> subrange_to_grid_region(const subrange<3>& sr) {
		auto end = sr.start + sr.range;
		clamp_range(end, sr.global_size);
		return GridRegion<3>(sycl_range_to_grid_point(sr.start), sycl_range_to_grid_point(end));
	}

	inline subrange<1> grid_box_to_subrange(const GridBox<1>& box) {
		const auto& min = box.get_min();
		const auto& max = box.get_max();
		const cl::sycl::range<1> size(max[0] - min[0]);
		return subrange<1>{cl::sycl::range<1>(min[0]), size, size};
	}

	inline subrange<2> grid_box_to_subrange(const GridBox<2>& box) {
		const auto& min = box.get_min();
		const auto& max = box.get_max();
		const cl::sycl::range<2> size(max[0] - min[0], max[1] - min[1]);
		return subrange<2>{cl::sycl::range<2>(min[0], min[1]), size, size};
	}

	inline subrange<3> grid_box_to_subrange(const GridBox<3>& box) {
		const auto& min = box.get_min();
		const auto& max = box.get_max();
		const cl::sycl::range<3> size(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
		return subrange<3>{cl::sycl::range<3>(min[0], min[1], min[2]), size, size};
	}

} // namespace detail

} // namespace celerity
