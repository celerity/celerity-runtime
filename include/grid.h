#pragma once

#include <CL/sycl.hpp>
#include <allscale/api/user/data/grid.h>
#undef assert_fail // Incompatible with fmt

#include "ranges.h"

namespace celerity {
namespace detail {

	using namespace allscale::api::user::data;

	inline GridPoint<1> sycl_id_to_grid_point(cl::sycl::range<1> range) { return GridPoint<1>(range[0]); }

	inline GridPoint<2> sycl_id_to_grid_point(cl::sycl::range<2> range) { return GridPoint<2>(range[0], range[1]); }

	inline GridPoint<3> sycl_id_to_grid_point(cl::sycl::range<3> range) { return GridPoint<3>(range[0], range[1], range[2]); }

	// The AllScale classes use a different template type for dimensions (size_t), which can lead to some type inference issues.
	// We thus have to provide all instantiations explicitly as overloads below.
	namespace impl {

		template <int Dims>
		GridBox<Dims> subrange_to_grid_box(const subrange<Dims>& sr) {
			const auto end = detail::range_cast<Dims>(sr.offset) + sr.range;
			return GridBox<Dims>(sycl_id_to_grid_point(detail::range_cast<Dims>(sr.offset)), sycl_id_to_grid_point(end));
		}

		template <int Dims>
		subrange<Dims> grid_box_to_subrange(const GridBox<Dims>& box) {
			const auto& box_min = box.get_min();
			const auto& box_max = box.get_max();
			cl::sycl::id<Dims> min;
			cl::sycl::id<Dims> max;
			for(int i = 0; i < Dims; ++i) {
				min[i] = box_min[i];
				max[i] = box_max[i];
			}
			return subrange<Dims>{min, range_cast<Dims>(max - min)};
		}

	} // namespace impl


	inline GridBox<1> subrange_to_grid_box(const subrange<1>& sr) { return impl::subrange_to_grid_box<1>(sr); }
	inline GridBox<2> subrange_to_grid_box(const subrange<2>& sr) { return impl::subrange_to_grid_box<2>(sr); }
	inline GridBox<3> subrange_to_grid_box(const subrange<3>& sr) { return impl::subrange_to_grid_box<3>(sr); }

	inline subrange<1> grid_box_to_subrange(const GridBox<1>& box) { return impl::grid_box_to_subrange<1>(box); }
	inline subrange<2> grid_box_to_subrange(const GridBox<2>& box) { return impl::grid_box_to_subrange<2>(box); }
	inline subrange<3> grid_box_to_subrange(const GridBox<3>& box) { return impl::grid_box_to_subrange<3>(box); }

} // namespace detail
} // namespace celerity
