#include "task.h"

#include "access_modes.h"
#include "utils.h"


namespace celerity {
namespace detail {

	std::unordered_set<buffer_id> buffer_access_map::get_accessed_buffers() const {
		std::unordered_set<buffer_id> result;
		for(const auto& [bid, _] : m_accesses) {
			result.emplace(bid);
		}
		return result;
	}

	std::unordered_set<cl::sycl::access::mode> buffer_access_map::get_access_modes(buffer_id bid) const {
		std::unordered_set<cl::sycl::access::mode> result;
		for(const auto& [b, rm] : m_accesses) {
			if(b == bid) { result.insert(rm->get_access_mode()); }
		}
		return result;
	}

	template <int KernelDims>
	subrange<3> apply_range_mapper(const range_mapper_base* rm, const chunk<KernelDims>& chnk) {
		switch(rm->get_buffer_dimensions()) {
		case 0: return subrange_cast<3>(subrange<0>());
		case 1: return subrange_cast<3>(rm->map_1(chnk));
		case 2: return subrange_cast<3>(rm->map_2(chnk));
		case 3: return rm->map_3(chnk);
		default: assert(false); return subrange<3>{};
		}
	}

	subrange<3> apply_range_mapper(const range_mapper_base* rm, const chunk<3>& chnk, int kernel_dims) {
		switch(kernel_dims) {
		case 0: return apply_range_mapper<0>(rm, chunk_cast<0>(chnk));
		case 1: return apply_range_mapper<1>(rm, chunk_cast<1>(chnk));
		case 2: return apply_range_mapper<2>(rm, chunk_cast<2>(chnk));
		case 3: return apply_range_mapper<3>(rm, chunk_cast<3>(chnk));
		default: assert(!"Unreachable"); return subrange<3>{};
		}
	}

	region<3> buffer_access_map::get_mode_requirements(
	    const buffer_id bid, const access_mode mode, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		box_vector<3> boxes;
		for(size_t i = 0; i < m_accesses.size(); ++i) {
			if(m_accesses[i].first != bid || m_accesses[i].second->get_access_mode() != mode) continue;
			boxes.push_back(get_requirements_for_nth_access(i, kernel_dims, sr, global_size));
		}
		return region(std::move(boxes));
	}

	box<3> buffer_access_map::get_requirements_for_nth_access(const size_t n, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		return apply_range_mapper(m_accesses[n].second.get(), chunk<3>{sr.offset, sr.range, global_size}, kernel_dims);
	}

	box_vector<3> buffer_access_map::get_required_contiguous_boxes(
	    const buffer_id bid, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const {
		box_vector<3> boxes;
		for(const auto& [a_bid, a_rm] : m_accesses) {
			if(a_bid == bid) {
				const auto accessed_box = box(apply_range_mapper(a_rm.get(), chunk<3>{sr.offset, sr.range, global_size}, kernel_dims));
				if(!accessed_box.empty()) { boxes.push_back(accessed_box); }
			}
		}
		return boxes;
	}

	void side_effect_map::add_side_effect(const host_object_id hoid, const experimental::side_effect_order order) {
		// TODO for multiple side effects on the same hoid, find the weakest order satisfying all of them
		emplace(hoid, order);
	}

	std::string print_task_debug_label(const task& tsk, bool title_case) {
		const auto type_string = [&] {
			switch(tsk.get_type()) {
			case task_type::epoch: return "epoch";
			case task_type::host_compute: return "host-compute task";
			case task_type::device_compute: return "device kernel";
			case task_type::collective: return "collective host task";
			case task_type::master_node: return "master-node host task";
			case task_type::horizon: return "horizon";
			case task_type::fence: return "fence";
			default: return "unknown task";
			}
		}();

		auto label = fmt::format("{} T{}", type_string, tsk.get_id());
		if(title_case) { label[0] = static_cast<char>(std::toupper(label[0])); }
		if(!tsk.get_debug_name().empty()) { fmt::format_to(std::back_inserter(label), " \"{}\"", tsk.get_debug_name()); }
		return label;
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
				region<3> writes;
				for(const auto mode : bam.get_access_modes(bid)) {
					if(access::mode_traits::is_producer(mode)) {
						const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), ck.get_subrange(), tsk.get_global_size());
						writes = region_union(writes, req);
					}
				}
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
