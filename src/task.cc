#include "task.h"

#include "cgf.h"
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
    : m_accesses(std::move(accesses)), m_task_global_size(get_global_size(geometry)), m_task_dimensions(get_dimensions(geometry)) {
	std::unordered_map<buffer_id, region_builder<3>> consumed_regions;
	std::unordered_map<buffer_id, region_builder<3>> produced_regions;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [bid, mode, rm, _] = m_accesses[i];
		m_accessed_buffers.insert(bid);
		const auto req = matchbox::match(
		    rm,
		    [&](const std::unique_ptr<range_mapper_base>& rm) {
			    return matchbox::match(
			        geometry,
			        [&](const basic_task_geometry& geo) {
				        return apply_range_mapper(rm.get(), chunk<3>{geo.global_offset, geo.global_size, geo.global_size}, geo.dimensions);
			        },
			        [&](const custom_task_geometry_desc& geo) {
				        // TODO: Or get "union chunk" from geometry.. directly?
				        region<3> result;
				        for(const auto& [chnk, _, _2] : geo.assigned_chunks) {
					        result = region_union(result, apply_range_mapper(rm.get(), chunk<3>{chnk.offset, chnk.range, geo.global_size}, geo.dimensions));
				        }
				        return result;
			        });
		    },
		    [&](const expert_mapper& em) { return em.get_task_requirements(); });
		auto& cons = consumed_regions[bid]; // allow default-insert
		auto& prod = produced_regions[bid]; // allow default-insert
		if(is_consumer_mode(mode)) { cons.add(req); }
		if(is_producer_mode(mode)) { prod.add(req); }
	}
	for(auto& [bid, builder] : consumed_regions) {
		m_task_consumed_regions.emplace(bid, std::move(builder).into_region());
	}
	for(auto& [bid, builder] : produced_regions) {
		m_task_produced_regions.emplace(bid, std::move(builder).into_region());
	}
}

region<3> buffer_access_map::get_requirements_for_nth_access(const size_t n, const std::optional<box<3>>& execution_range) const {
	return matchbox::match(
	    m_accesses[n].range_mapper,
	    [&](const std::unique_ptr<range_mapper_base>& rm) {
		    if(execution_range.has_value()) {
			    const auto sr = execution_range->get_subrange();
			    return apply_range_mapper(rm.get(), chunk<3>{sr.offset, sr.range, m_task_global_size}, m_task_dimensions);
		    } else {
			    return apply_range_mapper(rm.get(), chunk<3>{zeros, m_task_global_size, m_task_global_size}, m_task_dimensions);
		    }
	    },
	    [&](const expert_mapper& em) {
		    if(execution_range.has_value()) {
			    const auto sr = execution_range->get_subrange();
			    return em.get_chunk_requirements(chunk<3>{sr.offset, sr.range, m_task_global_size});
		    } else {
			    return em.get_task_requirements();
		    }
	    });
}

region<3> buffer_access_map::compute_consumed_region(const buffer_id bid, const box<3>& execution_range) const {
	region_builder<3> builder;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [b, m, _, _2] = m_accesses[i];
		if(b != bid || !is_consumer_mode(m)) continue;
		builder.add(get_requirements_for_nth_access(i, execution_range));
	}
	return std::move(builder).into_region();
}

region<3> buffer_access_map::compute_produced_region(const buffer_id bid, const box<3>& execution_range) const {
	region_builder<3> builder;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [b, m, _, _2] = m_accesses[i];
		if(b != bid || !is_producer_mode(m)) continue;
		builder.add(get_requirements_for_nth_access(i, execution_range));
	}
	return std::move(builder).into_region();
}

box_vector<3> buffer_access_map::compute_required_contiguous_boxes(const buffer_id bid, const box<3>& execution_range) const {
	box_vector<3> boxes;
	for(size_t i = 0; i < m_accesses.size(); ++i) {
		const auto& [b, a_mode, _, _2] = m_accesses[i];
		if(b == bid) {
			const auto accessed_region = get_requirements_for_nth_access(i, execution_range);
			if(!accessed_region.empty()) { boxes.push_back(bounding_box(accessed_region)); }
		}
	}
	return boxes;
}

std::unique_ptr<detail::task> make_command_group_task(const detail::task_id tid, const size_t num_collective_nodes, raw_command_group&& cg) {
	std::unique_ptr<detail::task> task;
	switch(cg.task_type.value()) {
	case detail::task_type::host_compute: {
		assert(!cg.collective_group_id.has_value());
		auto& geometry = cg.geometry.value();
		if(get_global_size(geometry).size() == 0) {
			// TODO this can be easily supported by not creating a task in case the execution range is empty
			throw std::runtime_error{"The execution range of distributed host tasks must have at least one item"};
		}
		auto& launcher = std::get<detail::host_task_launcher>(cg.launcher.value());
		buffer_access_map bam(std::move(cg.buffer_accesses), geometry);
		side_effect_map sem(cg.side_effects);
		task = detail::task::make_host_compute(tid, std::move(geometry), std::move(launcher), std::move(bam), std::move(sem), std::move(cg.reductions));
		break;
	}
	case detail::task_type::device_compute: {
		assert(!cg.collective_group_id.has_value());
		auto& geometry = cg.geometry.value();
		if(get_global_size(geometry).size() == 0) {
			// TODO unless reductions are involved, this can be easily supported by not creating a task in case the execution range is empty.
			// Edge case: If the task includes reductions that specify property::reduction::initialize_to_identity, we need to create a task that sets
			// the buffer state to an empty pending_reduction_state in the graph_generator. This will cause a trivial reduction_command to be generated on
			// each node that reads from the reduction output buffer, initializing it to the identity value locally.
			throw std::runtime_error{"The execution range of device tasks must have at least one item"};
		}
		auto& launcher = std::get<detail::device_kernel_launcher>(cg.launcher.value());
		buffer_access_map bam(std::move(cg.buffer_accesses), geometry);
		// Note that cgf_diagnostics has a similar check, but we don't catch void side effects there.
		if(!cg.side_effects.empty()) { throw std::runtime_error{"Side effects cannot be used in device kernels"}; }
		task = detail::task::make_device_compute(tid, std::move(geometry), std::move(launcher), std::move(bam), std::move(cg.reductions));
		break;
	}
	case detail::task_type::collective: {
		assert(!cg.geometry.has_value());
		const basic_task_geometry geometry{// geometry is dependent on num_collective_nodes, so it is not set in raw_command_group
		    .dimensions = 1,
		    .global_size = detail::range_cast<3>(range(num_collective_nodes)),
		    .global_offset = zeros,
		    .granularity = ones};
		const auto cgid = cg.collective_group_id.value();
		auto& launcher = std::get<detail::host_task_launcher>(cg.launcher.value());
		buffer_access_map bam(std::move(cg.buffer_accesses), geometry);
		side_effect_map sem(cg.side_effects);
		assert(cg.reductions.empty());
		task = detail::task::make_collective(tid, geometry, cgid, num_collective_nodes, std::move(launcher), std::move(bam), std::move(sem));
		break;
	}
	case detail::task_type::master_node: {
		assert(!cg.collective_group_id.has_value());
		assert(!cg.geometry.has_value());
		auto& launcher = std::get<detail::host_task_launcher>(cg.launcher.value());
		buffer_access_map bam(std::move(cg.buffer_accesses), task_geometry{});
		side_effect_map sem(cg.side_effects);
		assert(cg.reductions.empty());
		task = detail::task::make_master_node(tid, std::move(launcher), std::move(bam), std::move(sem));
		break;
	}
	case detail::task_type::horizon:
	case detail::task_type::fence:
	case detail::task_type::epoch: //
		detail::utils::unreachable();
	}
	for(auto& h : cg.hints) {
		task->add_hint(std::move(h));
	}
	if(cg.task_name.has_value()) { task->set_debug_name(*cg.task_name); }
	task->perf_assertions = cg.perf_assertions;
	task->perf_assertions.tid = tid;
	task->perf_assertions.task_debug_name = task->get_debug_name();
	return task;
}

} // namespace celerity::detail

namespace celerity {
namespace detail {

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
			// NOCOMMIT HACK - Need to restructure the entire thing to work in terms of nth access
			// => We currently skip the check for the entire buffer if one of the accesses is replicated
			// TODO: Add tests as well
			bool is_replicated = false;
			for(size_t i = 0; i < bam.get_num_accesses(); ++i) {
				if(bam.get_nth_access(i).first == bid && bam.is_replicated(i)) {
					is_replicated = true;
					break;
				}
			}
			if(is_replicated) continue;

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
