#pragma once

#include <cassert>
#include <set>
#include <unordered_set>
#include <vector>

#include <CL/sycl.hpp>

#include "grid.h"
#include "types.h"

namespace celerity {
namespace detail {

	class buffer_state_base {
	  public:
		virtual size_t get_dimensions() const = 0;
		virtual ~buffer_state_base(){};
	};

	template <int Dims>
	class buffer_state : public buffer_state_base {
		static_assert(Dims >= 1 && Dims <= 3, "Unsupported dimensionality");

	  public:
		buffer_state(cl::sycl::range<Dims> size, size_t num_nodes) {
			std::unordered_set<node_id> all_nodes(num_nodes);
			for(auto i = 0u; i < num_nodes; ++i)
				all_nodes.insert(i);
			region_nodes.push_back(std::make_pair(GridRegion<Dims>(sycl_range_to_grid_point(size)), all_nodes));
		}

		size_t get_dimensions() const override { return Dims; }

		std::vector<std::pair<GridBox<Dims>, std::unordered_set<node_id>>> get_source_nodes(GridRegion<Dims> request) const {
			std::vector<std::pair<GridBox<Dims>, std::unordered_set<node_id>>> result;

			// Locate entire region by iteratively removing the largest overlaps
			GridRegion<Dims> remaining = request;
			while(remaining.area() > 0) {
				size_t largest_overlap = 0;
				size_t largest_overlap_i = -1;
				for(auto i = 0u; i < region_nodes.size(); ++i) {
					auto r = GridRegion<Dims>::intersect(region_nodes[i].first, remaining);
					auto area = r.area();
					if(area > largest_overlap) {
						largest_overlap = area;
						largest_overlap_i = i;
					}
				}

				assert(largest_overlap > 0);
				auto r = GridRegion<Dims>::intersect(region_nodes[largest_overlap_i].first, remaining);
				remaining = GridRegion<Dims>::difference(remaining, r);
				r.scanByBoxes([this, &result, largest_overlap_i](
				                  const GridBox<Dims>& b) { result.push_back(std::make_pair(b, region_nodes[largest_overlap_i].second)); });
			}

			return result;
		}

		void update_region(const GridRegion<Dims>& region, const std::unordered_set<node_id>& nodes) {
			auto num_regions = region_nodes.size();
			for(auto i = 0u; i < num_regions; ++i) {
				const size_t overlap = GridRegion<Dims>::intersect(region_nodes[i].first, region).area();
				if(overlap == 0) continue;
				const auto diff = GridRegion<Dims>::difference(region_nodes[i].first, region);
				if(diff.area() == 0) {
					// New region is larger / equal to stored region - update it
					region_nodes[i].first = region;
					region_nodes[i].second = nodes;
				} else {
					// Stored region needs to be updated as well
					region_nodes[i].first = diff;
					region_nodes.push_back(std::make_pair(region, nodes));
				}
			}

			collapse_regions();
		}

	  private:
		// TODO: Look into using a different data structure for this.
		// Maybe order descending by area?
		std::vector<std::pair<GridRegion<Dims>, std::unordered_set<node_id>>> region_nodes;

		void collapse_regions() {
			std::set<size_t> erase_indices;
			for(auto i = 0u; i < region_nodes.size(); ++i) {
				const auto& nodes_i = region_nodes[i].second;
				for(auto j = i + 1; j < region_nodes.size(); ++j) {
					const auto& nodes_j = region_nodes[j].second;
					std::vector<node_id> intersection;
					std::set_intersection(nodes_i.cbegin(), nodes_i.cend(), nodes_j.cbegin(), nodes_j.cend(), std::back_inserter(intersection));
					if(intersection.size() == nodes_i.size()) {
						region_nodes[i].first = GridRegion<Dims>::merge(region_nodes[i].first, region_nodes[j].first);
						erase_indices.insert(j);
					}
				}
			}

			for(auto it = erase_indices.rbegin(); it != erase_indices.rend(); ++it) {
				region_nodes.erase(region_nodes.begin() + *it);
			}
		}
	};

} // namespace detail
} // namespace celerity
