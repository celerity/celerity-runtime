#include "task.h"

#include "access_modes.h"
#include "grid.h"
#include "range_mapper.h"
#include "ranges.h"
#include "types.h"
#include "utils.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


namespace celerity::detail {

template <int KernelDims>
region<3> apply_range_mapper(const range_mapper_base* rm, const chunk<KernelDims>& chnk) {
	switch(rm->get_buffer_dimensions()) {
	case 0: return region_cast<3>(region(box<0>()));
	case 1: return region_cast<3>(rm->map_1(chnk));
	case 2: return region_cast<3>(rm->map_2(chnk));
	case 3: return rm->map_3(chnk);
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

region<3> apply_range_mapper(const range_mapper_base* rm, const chunk<3>& chnk, int kernel_dims) {
	switch(kernel_dims) {
	case 0: return apply_range_mapper<0>(rm, chunk_cast<0>(chnk));
	case 1: return apply_range_mapper<1>(rm, chunk_cast<1>(chnk));
	case 2: return apply_range_mapper<2>(rm, chunk_cast<2>(chnk));
	case 3: return apply_range_mapper<3>(rm, chunk_cast<3>(chnk));
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

buffer_access_map::buffer_access_map(std::vector<buffer_access>&& accesses, const task_geometry& geometry)
    : m_accesses(std::move(accesses)), m_task_geometry(geometry) {
	std::unordered_map<buffer_id, region_builder<3>> consumed_regions;
	std::unordered_map<buffer_id, region_builder<3>> produced_regions;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [bid, mode, rm] = m_accesses[i];
		m_accessed_buffers.insert(bid);
		const auto req = apply_range_mapper(rm.get(), chunk<3>{geometry.global_offset, geometry.global_size, geometry.global_size}, geometry.dimensions);
		auto& cons = consumed_regions[bid]; // allow default-insert
		auto& prod = produced_regions[bid]; // allow default-insert
		if(access::mode_traits::is_consumer(mode)) { cons.add(req); }
		if(access::mode_traits::is_producer(mode)) { prod.add(req); }
	}
	for(auto& [bid, builder] : consumed_regions) {
		m_task_consumed_regions.emplace(bid, std::move(builder).into_region());
	}
	for(auto& [bid, builder] : produced_regions) {
		m_task_produced_regions.emplace(bid, std::move(builder).into_region());
	}
}

region<3> buffer_access_map::get_requirements_for_nth_access(const size_t n, const box<3>& execution_range) const {
	const auto sr = execution_range.get_subrange();
	return apply_range_mapper(m_accesses[n].range_mapper.get(), chunk<3>{sr.offset, sr.range, m_task_geometry.global_size}, m_task_geometry.dimensions);
}

region<3> buffer_access_map::compute_consumed_region(const buffer_id bid, const box<3>& execution_range) const {
	region_builder<3> builder;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [b, m, _] = m_accesses[i];
		if(b != bid || !access::mode_traits::is_consumer(m)) continue;
		builder.add(get_requirements_for_nth_access(i, execution_range));
	}
	return std::move(builder).into_region();
}

region<3> buffer_access_map::compute_produced_region(const buffer_id bid, const box<3>& execution_range) const {
	region_builder<3> builder;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [b, m, _] = m_accesses[i];
		if(b != bid || !access::mode_traits::is_producer(m)) continue;
		builder.add(get_requirements_for_nth_access(i, execution_range));
	}
	return std::move(builder).into_region();
}

box_vector<3> buffer_access_map::compute_required_contiguous_boxes(const buffer_id bid, const box<3>& execution_range) const {
	box_vector<3> boxes;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [b, a_mode, _] = m_accesses[i];
		if(b == bid) {
			const auto accessed_region = get_requirements_for_nth_access(i, execution_range);
			if(!accessed_region.empty()) { boxes.push_back(bounding_box(accessed_region)); }
		}
	}
	return boxes;
}

} // namespace celerity::detail

namespace celerity {
namespace detail {

	void side_effect_map::add_side_effect(const host_object_id hoid, const experimental::side_effect_order order) {
		// TODO for multiple side effects on the same hoid, find the weakest order satisfying all of them
		emplace(hoid, order);
	}

	std::string print_task_debug_label(const task& tsk, bool title_case) {
		return utils::make_task_debug_label(tsk.get_type(), tsk.get_id(), tsk.get_debug_name(), title_case);
	}

	std::unordered_map<buffer_id, region<3>> detect_overlapping_writes(const task& tsk, const box_vector<3>& chunks) {
		const box<3> scalar_reduction_box({0, 0, 0}, {1, 1, 1});

		auto& bam = tsk.get_buffer_access_map();

		// track the union of writes we have checked so far in order to detect an overlap between that union and the next write
		std::unordered_map<buffer_id, region<3>> buffer_write_accumulators;
		// collect overlapping writes in order to report all of them before throwing
		std::unordered_map<buffer_id, region<3>> overlapping_writes;

		for(const auto bid : bam.get_accessed_buffers()) {
			for(const auto& ck : chunks) {
				const auto writes = bam.compute_produced_region(bid, ck.get_subrange());
				if(!writes.empty()) {
					auto& write_accumulator = buffer_write_accumulators[bid]; // allow default-insert
					if(const auto overlap = region_intersection(write_accumulator, writes); !overlap.empty()) {
						auto& full_overlap = overlapping_writes[bid]; // allow default-insert
						full_overlap = region_union(full_overlap, overlap);
					}
					write_accumulator = region_union(write_accumulator, writes);
				}
			}
		}

		// we already check for accessor-reduction overlaps on task generation, but we still repeat the sanity-check here
		for(const auto& rinfo : tsk.get_reductions()) {
			auto& write_accumulator = buffer_write_accumulators[rinfo.bid]; // allow default-insert
			if(const auto overlap = region_intersection(write_accumulator, scalar_reduction_box); !overlap.empty()) {
				auto& full_overlap = overlapping_writes[rinfo.bid]; // allow default-insert
				full_overlap = region_union(full_overlap, overlap);
			}
			write_accumulator = region_union(write_accumulator, scalar_reduction_box);
		}

		return overlapping_writes;
	}

} // namespace detail
} // namespace celerity
