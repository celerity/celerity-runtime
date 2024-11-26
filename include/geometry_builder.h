#pragma once

#include "log.h"
#include "ranges.h"
#include "runtime.h"
#include "split.h"

#include <vector>


namespace celerity {

template <int Dims>
class geometry_builder {
  public:
	geometry_builder(const celerity::range<2>& global_size) : m_global_size(global_size) {}

	// TODO API: Should these things be mutators, or pure functions that return a new geometry..?
	// TODO API: Should optionally receive number of chunks
	// TODO API: Should assignment be a separate step?
	void split_2d() {
		auto& rt = celerity::detail::runtime::get_instance();
		// NOCOMMIT: We assume a uniform number of devices per node here
		//           => Ideally we should simply not create per-device chunks for remote nodes
		// NOCOMMIT: This is not equivalent to the recursive split we do in IGGEN (=> could support both approaches)
		const size_t num_devices = rt.NOCOMMIT_get_num_local_devices();
		const size_t num_chunks = rt.NOCOMMIT_get_num_nodes() * num_devices;
		if(rt.NOCOMMIT_get_num_nodes() > 1 && num_devices > 1) {
			static bool warning_printed = false;
			if(!warning_printed) {
				warning_printed = true;
				CELERITY_CRITICAL("2D split assignment is still stupid, especially interaction w/ setup - beware");
			}
		}
		const auto chunks = celerity::detail::split_2d(celerity::detail::box<3>::full_range(range_cast<3>(m_global_size)), celerity::detail::ones, num_chunks);
		assert(chunks.size() <= num_chunks);
		for(size_t i = 0; i < chunks.size(); ++i) {
			m_chunks.emplace_back(chunks[i], celerity::detail::node_id(i / num_devices), celerity::detail::device_id(i % num_devices));
		}
	}

	void split_2d_but_recursive_and_only_for_local_chunks() {
		auto& rt = celerity::detail::runtime::get_instance();
		const size_t num_chunks = rt.NOCOMMIT_get_num_nodes();
		const auto chunks = celerity::detail::split_2d(celerity::detail::box<3>::full_range(range_cast<3>(m_global_size)), celerity::detail::ones, num_chunks);
		assert(chunks.size() <= num_chunks);
		for(size_t i = 0; i < chunks.size(); ++i) {
			if(i == rt.NOCOMMIT_get_local_nid()) {
				const auto local_chunks = celerity::detail::split_2d(chunks[i], celerity::detail::ones, rt.NOCOMMIT_get_num_local_devices());
				for(size_t j = 0; j < local_chunks.size(); ++j) {
					m_chunks.emplace_back(local_chunks[j], celerity::detail::node_id(i), celerity::detail::device_id(j));
				}
			} else {
				m_chunks.emplace_back(chunks[i], celerity::detail::node_id(i), std::nullopt);
			}
		}
	}

	void split_2d_but_recursive_and_only_for_local_chunks_v2_electric_boogaloo(size_t num_nodes, size_t local_devices, detail::node_id local_nid) {
		const size_t num_chunks = num_nodes;
		const auto chunks = celerity::detail::split_2d(celerity::detail::box<3>::full_range(range_cast<3>(m_global_size)), celerity::detail::ones, num_chunks);
		assert(chunks.size() <= num_chunks);
		for(size_t i = 0; i < chunks.size(); ++i) {
			if(i == local_nid) {
				const auto local_chunks = celerity::detail::split_2d(chunks[i], celerity::detail::ones, local_devices);
				for(size_t j = 0; j < local_chunks.size(); ++j) {
					m_chunks.emplace_back(local_chunks[j], celerity::detail::node_id(i), celerity::detail::device_id(j));
				}
			} else {
				m_chunks.emplace_back(chunks[i], celerity::detail::node_id(i), std::nullopt);
			}
		}
	}

	void split_1d() {
		auto& rt = celerity::detail::runtime::get_instance();
		// NOCOMMIT: We assume a uniform number of devices per node here
		//           => Ideally we should simply not create per-device chunks for remote nodes
		// NOCOMMIT: This is not equivalent to the recursive split we do in IGGEN (=> could support both approaches)
		const size_t num_devices = rt.NOCOMMIT_get_num_local_devices();
		const size_t num_chunks = rt.NOCOMMIT_get_num_nodes() * num_devices;
		const auto chunks = celerity::detail::split_1d(celerity::detail::box<3>::full_range(range_cast<3>(m_global_size)), celerity::detail::ones, num_chunks);
		assert(chunks.size() <= num_chunks);
		for(size_t i = 0; i < chunks.size(); ++i) {
			m_chunks.emplace_back(chunks[i], celerity::detail::node_id(i / num_devices), celerity::detail::device_id(i % num_devices));
		}
	}

	// TODO API: Naming????
	void splice(const geometry_builder& another_one) {
		if(another_one.is_overlapping()) { throw std::runtime_error("Splicing with overlapping geometry - not sure what to do here??"); }

		// TODO: Can we maybe do all of the operations in a geometry lazily? Only once we actually require them we manifest them? So we don't need to compute
		// the splice for all remote chunks as well?
		// => Maybe we should distinguish between local and remote chunks in storage?
		std::vector<celerity::detail::region<3>> subtracted_from(m_chunks.size());
		std::vector<celerity::detail::region<3>> subtracted_with(m_chunks.size());
		for(size_t i = 0; i < m_chunks.size(); ++i) {
			const auto& chunk = m_chunks[i];
			subtracted_from[i] = celerity::detail::box<3>(m_chunks[i].sr);
			for(const auto& other_chunk : another_one.m_chunks) {
				// if(chunk.sr == other_chunk.sr) {
				// 	if(chunk.nid != other_chunk.nid || chunk.did != other_chunk.did) {
				// 		throw std::runtime_error("Same chunk but different node and/or device assignment - not sure what to do here??");
				// 	}
				// 	// Nothing to do, skip
				// 	continue;
				// }
				if(chunk.nid != other_chunk.nid || chunk.did != other_chunk.did) {
					// TODO: Does that make sense?
					continue;
				}

				const auto intersection = celerity::detail::region_intersection(subtracted_from[i], celerity::detail::box<3>(other_chunk.sr));
				if(!intersection.empty()) {
					subtracted_from[i] = celerity::detail::region_difference(subtracted_from[i], celerity::detail::box<3>(other_chunk.sr));
					subtracted_with[i] = celerity::detail::region_union(subtracted_with[i], intersection);
				}
			}
		}

		std::vector<celerity::geometry_chunk> new_chunks;
		for(size_t i = 0; i < m_chunks.size(); ++i) {
			for(const auto& box : subtracted_from[i].get_boxes()) {
				new_chunks.emplace_back(box, m_chunks[i].nid, m_chunks[i].did);
			}
			for(const auto& box : subtracted_with[i].get_boxes()) {
				new_chunks.emplace_back(box, m_chunks[i].nid, m_chunks[i].did);
			}
		}
		m_chunks = std::move(new_chunks);
	}

	// TODO API: Not sure if this is something we actually want. Also naming.
	void replicate() {
		auto& rt = celerity::detail::runtime::get_instance();
		const size_t num_devices = rt.NOCOMMIT_get_num_local_devices();
		const size_t num_chunks = rt.NOCOMMIT_get_num_nodes() * num_devices;
		for(size_t i = 0; i < num_chunks; ++i) {
			m_chunks.emplace_back(celerity::subrange<3>{{}, range_cast<3>(m_global_size)}, celerity::detail::node_id(i / num_devices),
			    celerity::detail::device_id(i % num_devices));
		}
	}

	// TODO API: Several options - clamped to global size, free, non-overlapping (only grow towards outside), ...
	// We currently do clamped, overlapping
	void outset(const size_t amount) {
		if(m_chunks.empty()) throw std::runtime_error("no chunks"); // TODO: Should we start with a single big chunk?
		for(auto& chunk : m_chunks) {
			for(int d = 0; d < Dims; ++d) {
				size_t min = chunk.sr.offset[d];
				size_t max = chunk.sr.offset[d] + chunk.sr.range[d];
				if(const auto delta = chunk.sr.offset[d]; delta > 0) { //
					min -= std::min(delta, amount);
				}
				if(const auto delta = m_global_size[d] - chunk.sr.offset[d] - chunk.sr.range[d]; delta > 0) { //
					max += std::min(delta, amount);
				}
				chunk.sr.offset[d] = min;
				chunk.sr.range[d] = max - min;
			}
		}
	}

	celerity::custom_task_geometry<Dims> make() const {
		// TODO API: Add support for global offset
		// TODO API: Add support for nd-range kernels / local size
		return celerity::custom_task_geometry<Dims>{range_cast<3>(m_global_size), celerity::detail::zeros, celerity::detail::ones, m_chunks};
	}

  private:
	celerity::range<2> m_global_size;
	std::vector<celerity::geometry_chunk> m_chunks;

	bool is_overlapping() const {
		celerity::detail::region<3> current_region;
		for(const auto& chunk : m_chunks) {
			if(!celerity::detail::region_intersection(current_region, celerity::detail::box<3>(chunk.sr)).empty()) return true;
			current_region = celerity::detail::region_union(current_region, celerity::detail::box<3>(chunk.sr));
		}
		return false;
	}
};

} // namespace celerity
