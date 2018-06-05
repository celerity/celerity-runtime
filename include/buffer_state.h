#pragma once

#include <unordered_set>
#include <vector>

#include <CL/sycl.hpp>

#include "grid.h"
#include "types.h"

namespace celerity {
namespace detail {

	class buffer_state {
	  public:
		buffer_state(cl::sycl::range<3> size, size_t num_nodes);

		std::vector<std::pair<GridBox<3>, std::unordered_set<node_id>>> get_source_nodes(GridRegion<3> request) const;
		void update_region(const GridRegion<3>& region, const std::unordered_set<node_id>& nodes);

	  private:
		// TODO: Look into using a different data structure for this.
		// Maybe order descending by area?
		std::vector<std::pair<GridRegion<3>, std::unordered_set<node_id>>> region_nodes;

		void collapse_regions();
	};

} // namespace detail
} // namespace celerity
