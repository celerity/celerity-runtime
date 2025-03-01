#pragma once

#include <algorithm>
#include <vector>

#include <fmt/format.h>

#include "grid.h"
#include "print_utils.h"
#include "range_mapper.h"
#include "task_geometry.h"
#include "utils.h"

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

struct data_requirement_options {
	// TODO: This currently doesn't change allocation behavior, it only checks that the allocation is exact
	bool allocate_exactly = false;
	bool use_local_indexing = false;
};

// TODO: Passing buffer range is kind of awkward
template <int KernelDims, int BufferDims, typename RangeMapperFn>
std::vector<std::pair<detail::box<3>, detail::region<3>>> from_range_mapper(
    const custom_task_geometry<KernelDims>& geo, const range<BufferDims>& buffer_range, RangeMapperFn&& fn) {
	std::vector<std::pair<detail::box<3>, detail::region<3>>> result;
	detail::range_mapper rm{fn, buffer_range};
	for(const auto& achunk : geo.assigned_chunks) {
		const auto sr = detail::subrange_cast<KernelDims>(achunk.box.get_subrange());
		const chunk<KernelDims> chnk = {sr.offset, sr.range, range_cast<KernelDims>(geo.global_size)};
		detail::region<3> r;
		switch(BufferDims) {
		case 0: r = region_cast<3>(detail::region(detail::box<0>())); break;
		case 1: r = region_cast<3>(rm.map_1(chnk)); break;
		case 2: r = region_cast<3>(rm.map_2(chnk)); break;
		case 3: r = rm.map_3(chnk); break;
		default: detail::utils::unreachable(); // LCOV_EXCL_LINE
		}
		result.push_back({achunk.box, r});
	}
	return result;
}

// TODO API: Naming - not a range mapper (also change file name!)
// TODO API: Should we make these have dimensionality as well?
// TODO API: Should it be possible to use these in conjunction with a non-custom geometry task? Probably not. But where to diagnose this?
class expert_mapper {
  public:
	// This overload receives the union access explicitly from the user.
	// Useful if the user does not want to declare all remote chunks and their dependencies explicitly.
	expert_mapper(detail::region<3> union_access, std::vector<std::pair<detail::box<3>, detail::region<3>>> per_chunk_accesses)
	    : m_union_region(std::move(union_access)) {
		m_per_chunk_accesses.reserve(per_chunk_accesses.size());
		for(auto& [chunk, region] : per_chunk_accesses) {
			m_per_chunk_accesses.push_back({chunk, std::move(region)});
		}
		// TODO API: If we are running single node, we should check whether union_region == union(boxes) (not sure where though; CDAG generator?)
	}

	// Same as above, but union access is specified as a range. Useful if an entire buffer is being accessed.
	expert_mapper(range<3> union_access, std::vector<std::pair<detail::box<3>, detail::region<3>>> per_chunk_accesses)
	    : expert_mapper(detail::box<3>::full_range(union_access), std::move(per_chunk_accesses)) {}

	expert_mapper(std::vector<std::pair<detail::box<3>, detail::region<3>>> per_chunk_accesses)
	    : expert_mapper(compute_union_region(per_chunk_accesses), std::move(per_chunk_accesses)) {}

	detail::region<3> get_chunk_requirements(const chunk<3>& chnk) const { return get_region_for_chunk(chnk); }

	detail::region<3> get_task_requirements() const { return m_union_region; }

	// FIXME HACK Prototyping - figure out proper encapsulation
	data_requirement_options options;

  private:
	std::vector<std::pair<detail::box<3>, detail::region<3>>> m_per_chunk_accesses;
	detail::region<3> m_union_region;

	static detail::region<3> compute_union_region(const std::vector<std::pair<detail::box<3>, detail::region<3>>>& per_chunk_accesses) {
		detail::region_builder<3> builder;
		for(const auto& [_, region] : per_chunk_accesses) {
			builder.add(region);
		}
		return std::move(builder).into_region();
	}

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
