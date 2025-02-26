#pragma once

#include <algorithm>
#include <vector>

#include <fmt/format.h>

#include "grid.h"
#include "print_utils.h"

namespace celerity {

// NOTES ON EXPERT MAPPER:
// - For writes we don't need to know exactly what everybody else is doing, only the union of all writes
// - For reads we actually also only need to know which peers require our data. Thee options:
//		- Always provide all chunks + requirements on all nodes
//			- Potentially in a "hierarchical" fashion, where we don't need to specify individual GPU-chunks (basically what we do now with CDAG/IDAG)
//		- Only provide local requirements, then automatically exchange with peers
//		- Provide local requirements, as well as list of all peers that require local data (and which parts)
// - We *could* provide the option of automatically exchanging access information with peers for when it is needed (e.g. for replicated writes),
//   maybe by having the user explicitly dispatch an async exchange as early as possible (CGF then waits for completion).
//   => Actually we could also do this for remote chunks.
// Q: How do we compute task dependencies going forward? In order to compute them correctly we pretty much have to do a full CDAG-style analysis...
//    (Not that it matters for the correctness of our execution model)


// TODO API: Naming - not a range mapper (also change file name!)
// TODO API: Should we make these have dimensionality as well?
// TODO API: Should it be possible to use these in conjunction with a non-custom geometry task? Probably not. But where to diagnose this?
class expert_mapper {
  public:
	expert_mapper(detail::region<3> union_access, std::vector<std::pair<detail::box<3>, detail::region<3>>> per_chunk_accesses)
	    : m_union_region(std::move(union_access)) {
		m_per_chunk_accesses.reserve(per_chunk_accesses.size());
		for(auto& [chunk, region] : per_chunk_accesses) {
			m_per_chunk_accesses.push_back({chunk, std::move(region)});
			// NOTE: We could also compute the union region here, but that means that every rank has to known all other rank's exact accesses.
			//       In the current variant it only needs to know the union.
			// TODO API: Have two overloads of the ctor, one that receives the union, and one that computes it?
			// m_union_region = detail::region_union(m_union_region, detail::region{std::move(boxes)});
		}
		// TODO API: If we are running single node, we should check whether union_region == union(boxes) (not sure where though; CDAG generator?)
	}

	detail::region<3> get_chunk_requirements(const chunk<3>& chnk) const { return get_region_for_chunk(chnk); }

	detail::region<3> get_task_requirements() const { return m_union_region; }

  private:
	std::vector<std::pair<detail::box<3>, detail::region<3>>> m_per_chunk_accesses;
	detail::region<3> m_union_region;

	const detail::region<3>& get_region_for_chunk(const chunk<3>& chnk) const {
		// TODO: Can we avoid the linear search here? Would need to introduce notion of "chunk index" or similar
		// => Should we store a map instead? Make chunk hashable?
		// NOCOMMIT TODO: It is very not ideal that box<> has a ctor that takes min/max id, and range is implicitly convertible to id!!
		const auto box = detail::box<3>{subrange<3>{chnk.offset, chnk.range}};
		const auto it = std::find_if(m_per_chunk_accesses.begin(), m_per_chunk_accesses.end(), [&](const auto& p) { return p.first == box; });
		if(it == m_per_chunk_accesses.end()) { throw std::runtime_error(fmt::format("No region found for chunk: {}", box)); }
		return it->second;
	}
};
} // namespace celerity
