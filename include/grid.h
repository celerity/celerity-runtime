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

} // namespace detail

} // namespace celerity
