#pragma once

#include <set>
#include <vector>

#include <CL/sycl.hpp>

#include "grid.h"

namespace celerity {
namespace detail {

	/**
	 * The region_map class maintains a mapping of regions to arbitrary values.
	 *
	 * This can for example be used to store the command_id that last wrote to a particular buffer subrange.
	 *
	 * @tparam ValueType The value type stored within the data structure. Needs to be EqualityComparable.
	 *
	 * TODO: The semantics of this class are a bit unclear, especially in regards to merging. Try to find a nicer solution.
	 * For instance right now, two region_maps can be initialized using different default values. If the second then gets merged
	 * into the first, none of the values of the second one will be transferred, as they are considered default values.
	 */
	template <typename ValueType>
	class region_map {
		friend struct region_map_testspy;

		using region_values_t = std::vector<std::pair<GridRegion<3>, ValueType>>;

	  public:
		/**
		 * @param extent The maximum extent of a region that can be stored within the map (i.e. all regions are subsets of this).
		 * @param default_value The default value is used to initialize the entire extent
		 */
		region_map(cl::sycl::range<3> extent, ValueType default_value = ValueType{}) : extent(extent) {
			default_initialized = GridRegion<3>(sycl_id_to_grid_point(extent));
			region_values.push_back(std::make_pair(default_initialized, default_value));
		}

		region_map(const region_map<ValueType>& other) = default;

		/**
		 * @brief Given a region request, returns all values that belong to regions intersecting with the request.
		 *
		 * @return A collection of boxes and their associated values. This may contain default initialized regions.
		 */
		std::vector<std::pair<GridBox<3>, ValueType>> get_region_values(GridRegion<3> request) const {
			std::vector<std::pair<GridBox<3>, ValueType>> result;

			// Locate entire region by iteratively removing the largest overlaps
			GridRegion<3> remaining = request;
			while(remaining.area() > 0) {
				size_t largest_overlap = 0;
				size_t largest_overlap_i = -1;
				for(auto i = 0u; i < region_values.size(); ++i) {
					auto r = GridRegion<3>::intersect(region_values[i].first, remaining);
					const auto area = r.area();
					if(area > largest_overlap) {
						largest_overlap = area;
						largest_overlap_i = i;
					}
				}

				assert(largest_overlap > 0);
				auto r = GridRegion<3>::intersect(region_values[largest_overlap_i].first, remaining);
				remaining = GridRegion<3>::difference(remaining, r);
				r.scanByBoxes(
				    [this, &result, largest_overlap_i](const GridBox<3>& b) { result.push_back(std::make_pair(b, region_values[largest_overlap_i].second)); });
			}

			return result;
		}

		void update_region(const GridRegion<3>& region, const ValueType& value) {
			if(!default_initialized.empty()) { default_initialized = GridRegion<3>::difference(default_initialized, region); }

			region_values_t new_region_values;
			// Reserve enough elements in case we need to add a region.
			new_region_values.reserve(region_values.size() + 1);
			for(const auto& region_value : region_values) {
				auto rest = GridRegion<3>::difference(region_value.first, region);
				if(rest.empty()) continue;
				new_region_values.push_back({rest, region_value.second});
			}
			new_region_values.push_back({region, value});
			region_values = std::move(new_region_values);

			// Since we only add regions in this function it's important to collapse afterwards.
			collapse_regions();
		}

		template <typename Functor>
		void apply_to_values(Functor f) {
			for(auto& pair : region_values) {
				pair.second = f(pair.second);
			}
			collapse_regions();
		}

		/**
		 * @brief Merges with a given region_map \p other
		 *
		 * Updated (i.e. non-default-initialized) regions within \p other take precedence over regions in the current region_map.
		 */
		void merge(const region_map<ValueType>& other) {
			if(extent != other.extent) { throw std::runtime_error("Incompatible region map"); }
			for(auto& p : other.region_values) {
				if(GridRegion<3>::intersect(other.default_initialized, p.first).empty()) { update_region(p.first, p.second); }
			}
		}

	  private:
		const cl::sycl::range<3> extent;
		// We keep track which parts are default initialized for merging
		GridRegion<3> default_initialized;
		// TODO: Look into using a different data structure for this.
		// Maybe order descending by area?
		region_values_t region_values;


		/**
		 * Merge regions with the same values.
		 */
		void collapse_regions() {
			std::set<size_t> erase_indices;
			for(auto i = 0u; i < region_values.size(); ++i) {
				const auto& values_i = region_values[i].second;
				for(auto j = i + 1; j < region_values.size(); ++j) {
					const auto& values_j = region_values[j].second;
					if(values_i == values_j) {
						region_values[i].first = GridRegion<3>::merge(region_values[i].first, region_values[j].first);
						erase_indices.insert(j);
					}
				}
			}

			for(auto it = erase_indices.rbegin(); it != erase_indices.rend(); ++it) {
				region_values.erase(region_values.begin() + *it);
			}
		}
	};

} // namespace detail
} // namespace celerity
