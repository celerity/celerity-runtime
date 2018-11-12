#include "task.h"

#include <algorithm>

namespace celerity {
namespace detail {

	std::vector<buffer_id> compute_task::get_accessed_buffers() const {
		std::vector<buffer_id> result;
		std::transform(range_mappers.cbegin(), range_mappers.cend(), std::back_inserter(result), [](auto& p) { return p.first; });
		return result;
	}

	std::unordered_set<cl::sycl::access::mode> compute_task::get_access_modes(buffer_id bid) const {
		std::unordered_set<cl::sycl::access::mode> result;
		for(auto& rm : range_mappers.at(bid)) {
			result.insert(rm->get_access_mode());
		}
		return result;
	}

	template <int KernelDims>
	subrange<3> apply_range_mapper(range_mapper_base const* rm, chunk<KernelDims> chnk) {
		switch(rm->get_buffer_dimensions()) {
		case 1: return rm->map_1(chnk);
		case 2: return rm->map_2(chnk);
		case 3: return rm->map_3(chnk);
		default: assert(false);
		}
		return subrange<3>{};
	}

	GridRegion<3> compute_task::get_requirements(buffer_id bid, cl::sycl::access::mode mode, const subrange<3>& sr) const {
		GridRegion<3> result;
		for(auto& rm : range_mappers.at(bid)) {
			if(rm->get_access_mode() != mode) continue;
			subrange<3> req;
			switch(dimensions) {
			case 1:
				req = apply_range_mapper<1>(rm.get(), chunk<1>(cl::sycl::id<1>(sr.offset), cl::sycl::range<1>(sr.range), cl::sycl::range<1>(global_size)));
				break;
			case 2:
				req = apply_range_mapper<2>(rm.get(), chunk<2>(cl::sycl::id<2>(sr.offset), cl::sycl::range<2>(sr.range), cl::sycl::range<2>(global_size)));
				break;
			case 3:
				req = apply_range_mapper<3>(rm.get(), chunk<3>(cl::sycl::id<3>(sr.offset), cl::sycl::range<3>(sr.range), cl::sycl::range<3>(global_size)));
				break;
			default: assert(false);
			}
			result = GridRegion<3>::merge(result, subrange_to_grid_region(req));
		}
		return result;
	}

	std::vector<buffer_id> master_access_task::get_accessed_buffers() const {
		std::vector<buffer_id> result;
		std::transform(buffer_accesses.cbegin(), buffer_accesses.cend(), std::back_inserter(result), [](auto& p) { return p.first; });
		return result;
	}

	std::unordered_set<cl::sycl::access::mode> master_access_task::get_access_modes(buffer_id bid) const {
		std::unordered_set<cl::sycl::access::mode> result;
		for(auto& bai : buffer_accesses.at(bid)) {
			result.insert(bai.mode);
		}
		return result;
	}

	GridRegion<3> master_access_task::get_requirements(buffer_id bid, cl::sycl::access::mode mode) const {
		GridRegion<3> result;
		for(auto& bai : buffer_accesses.at(bid)) {
			if(bai.mode != mode) continue;
			result = GridRegion<3>::merge(result, subrange_to_grid_region(bai.sr));
		}
		return result;
	}

} // namespace detail
} // namespace celerity
