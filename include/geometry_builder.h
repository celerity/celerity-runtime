#pragma once

#include "log.h"
#include "ranges.h"
#include "runtime.h"
#include "split.h"

#include <vector>


namespace celerity {

// TODO: Naming - partitioning, domain_partition, domain, cartesian_partition ...?
//		=> Partition is already used for host tasks...
// 		=> Is this even CARTESIAN, if the cells can be of different size (unequal split?). Or is it RECTILINEAR?
// SHOULD BOTH THIS AND GEOMETRY_BUILDER EXIST? I think so. A partition is a subset: Fully covers the domain, no overlap.
// => Also partitions (I think?) don't have any node/device assignments
//		=> When assigning we could then choose to only assign to a subset of nodes
// NOCOMMIT MOVE
// TODO: Should it also be possible to just create a partition with a given grid size, and unit-sized cells?
// TODO: It should be possible to go from a 2D partition to a 1D partition (by laying it out row-wise). Vice versa?
// TODO: It should be possible to scale a partition down or up (e.g. for thread coarsening)
// => OR: Should this be done in geometry builder? If we do it here we could also apply it in reverse though (for data requirements)
template <int Dims>
class cartesian_grid {
	using box = celerity::detail::box<Dims>;

  public:
	struct cell {
		id<Dims> pos;
		box box;
	};

	// TODO: Have static factory methods instead? Or a builder?
	cartesian_grid(box extent) : m_extent(std::move(extent)) {
		if(m_extent.get_area() == 0) { throw std::runtime_error("Extent must have non-zero area"); }
	}

	// TODO: Naming - a partition should already be split. Also it should only be possible to call this once.
	// TODO: Terminology - are these "chunks", "blocks", "tiles", "cells"..?
	// TODO: Constraint policy - exact (fail if not possible) or pad (create larger/smaller chunks at end)
	// TODO: Option to specify number of chunks in each dimension (or number of cuts?)
	void split(const size_t num_cells, const celerity::range<2>& constraints = {1, 1}) {
		const auto split_factors = celerity::detail::find_best_split_factors_2d(box_cast<3>(m_extent), range_cast<3>(constraints), num_cells);
		if(split_factors[0] * split_factors[1] != num_cells) { throw std::runtime_error("Failed to create requested number of cells - what now?"); }
		split(split_factors, constraints);
	}

	// TODO: Get rid of array, too annoying
	void split(const celerity::range<2>& num_cells, const celerity::range<2>& constraints = {1, 1}) {
		split(std::array<size_t, 2>{num_cells[0], num_cells[1]}, constraints);
	}

	void split(const std::array<size_t, 2>& num_cells, const celerity::range<2>& constraints = {1, 1}) {
		if(num_cells[0] * num_cells[1] == 0) { throw std::runtime_error("Number of cells must be greater than zero"); }
		if(m_extent.get_range() % celerity::range<2>(num_cells[0], num_cells[1]) != celerity::detail::zeros) {
			throw std::runtime_error("Cell size does not divide extent");
		}
		// TODO: Also check that constraints are possible
		m_grid_size = num_cells;
		const auto cells = celerity::detail::split_2d(box_cast<3>(m_extent), range_cast<3>(constraints), m_grid_size);
		assert(m_grid_size[0] * m_grid_size[1] != m_cells.size()); // FIXME 1D/3D

		// TODO: We don't actually need to compute the boxes themselves. It would suffice to get their number and shape in each dimension.
		m_cells.reserve(cells.size());
		static_assert(Dims == 2); // 1D/3D positioning NYI
		id<2> pos = {};
		for(const auto& cell : cells) {
			if(cell.get_min()[1] == m_extent.get_min()[1]) {
				if(!m_cells.empty()) {
					pos[0]++;
					pos[1] = 0;
				}
			}
			m_cells.push_back({pos, box_cast<2>(cell)});
			pos[1]++;
		}
		CELERITY_CRITICAL("Created a grid of size {}x{}", m_grid_size[0], m_grid_size[1]);
	}

	// TODO: Naming - domain_extent, domain_size..?
	const box& get_extent() const { return m_extent; }

	// TODO: Naming - grid? get_size?
	std::array<size_t, 2> get_grid_size() const { return m_grid_size; }

	const std::vector<cell>& get_cells() const { return m_cells; }

	const box& get_cell(const id<Dims>& pos) const {
		static_assert(Dims == 2); // 1D/3D Linearization NYI
		const auto idx = pos[0] * m_grid_size[1] + pos[1];
		if(idx >= m_cells.size()) { throw std::runtime_error("Cell index out of bounds"); }
		return m_cells[idx].box;
	}

  private:
	box m_extent;
	std::array<size_t, 2> m_grid_size;
	std::vector<cell> m_cells;
};

// TODO: This should be the output of the builder (to configure assignment, partial materialization, ...)
//		=> Probably also to do device splitting/assignments
// TODO: Should this just have a "view as 1D" functionality? (For optimized GEMM kernel)
//		=> THE PROBLEM with the current approach is that the chunks won't match what is stored in expert mapper. So we'd have to do this BEFORE creating the
//         mapper. This could be remedied by switching to chunk indices instead.
template <int Dims>
class grid_geometry {
  public:
	grid_geometry(cartesian_grid<Dims> grid, const range<Dims>& local_size) : m_grid(std::move(grid)), m_local_size(local_size) {
		if(m_grid.get_cells().empty()) { throw std::runtime_error("Grid has not been split yet"); }
		for(auto& cell : m_grid.get_cells()) {
			if(cell.box.get_range() % local_size != detail::zeros) { throw std::runtime_error("Local size does not divide cell size"); }
		}
	}

	// TODO: Can we support a assignment where neighboring blocks are assigned to the same node? Or just use hierarchical for that?
	void assign(const size_t num_nodes, const size_t devices_per_node) {
		// Block-cyclic distribution
		for(size_t i = 0; i < m_grid.get_cells().size(); ++i) {
			const size_t row = i / m_grid.get_grid_size()[1];
			const size_t col = i % m_grid.get_grid_size()[1];
			const size_t node_id = (row * m_grid.get_grid_size()[1] + col) % num_nodes;
			const size_t device_id = ((row * m_grid.get_grid_size()[1] + col) / num_nodes) % devices_per_node;
			m_assignment.push_back({detail::node_id(node_id), detail::device_id(device_id)});
		}
	}

	// TODO: Or inherit from custom_task_geometry?
	operator custom_task_geometry<Dims>() const {
		if(m_assignment.empty()) { throw std::runtime_error("No assignment has been made"); }
		std::vector<geometry_chunk> chunks;
		for(size_t i = 0; i < m_grid.get_cells().size(); ++i) {
			const auto& cell = m_grid.get_cells()[i];
			const auto [nid, did] = m_assignment[i];
			chunks.push_back({box_cast<3>(cell.box), nid, did});
		}
		custom_task_geometry<Dims> geo{.global_size = range_cast<3>(m_grid.get_extent().get_range()),
		    .global_offset = id_cast<3>(m_grid.get_extent().get_offset()),
		    .local_size = range_cast<3>(m_local_size),
		    .assigned_chunks = std::move(chunks)};
		return geo;
	}

	// NOCOMMIT This is beyond stupid
	operator nd_custom_task_geometry<Dims>() const {
		custom_task_geometry geo = this->operator custom_task_geometry<Dims>();
		return nd_custom_task_geometry<Dims>{.global_size = geo.global_size,
		    .global_offset = geo.global_offset,
		    .local_size = geo.local_size,
		    .assigned_chunks = std::move(geo.assigned_chunks)};
	}

	const cartesian_grid<Dims>& get_grid() const { return m_grid; }

  private:
	cartesian_grid<Dims> m_grid;
	range<Dims> m_local_size;
	std::vector<std::pair<detail::node_id, detail::device_id>> m_assignment;
};

// TODO: The builder should probably use a chaining pattern so we can control what operations can be done in what order
// => One potential way of implementing this would be to have a intermediate type with template parameters that indicate what operations have already been done
//		=> Member functions could then be enabled/disabled based on that
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

	// TODO API: Naming???? Divide? That's what affinity designer (and illustrator IIRC) calls it
	void splice(const geometry_builder& another_one) {
		if(another_one.is_overlapping()) { throw std::runtime_error("Splicing with overlapping geometry - not sure what to do here??"); }

		// TODO: Can we maybe do all of the operations in a geometry lazily? Only once we actually require them we manifest them? So we don't need to compute
		// the splice for all remote chunks as well?
		// => Maybe we should distinguish between local and remote chunks in storage?
		std::vector<celerity::detail::region<3>> subtracted_from(m_chunks.size());
		std::vector<celerity::detail::region<3>> subtracted_with(m_chunks.size());
		for(size_t i = 0; i < m_chunks.size(); ++i) {
			const auto& chunk = m_chunks[i];
			subtracted_from[i] = m_chunks[i].box;
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
			auto sr = chunk.box.get_subrange();
			for(int d = 0; d < Dims; ++d) {
				size_t min = sr.offset[d];
				size_t max = sr.offset[d] + sr.range[d];
				if(const auto delta = sr.offset[d]; delta > 0) { //
					min -= std::min(delta, amount);
				}
				if(const auto delta = m_global_size[d] - sr.offset[d] - sr.range[d]; delta > 0) { //
					max += std::min(delta, amount);
				}
				sr.offset[d] = min;
				sr.range[d] = max - min;
			}
			chunk.box = sr;
		}
	}

	celerity::custom_task_geometry<Dims> make() const {
		// TODO API: Add support for global offset
		// TODO API: Add support for nd-range kernels / local size
		return celerity::custom_task_geometry<Dims>{range_cast<3>(m_global_size), celerity::detail::zeros, celerity::detail::ones, m_chunks};
	}

	celerity::nd_custom_task_geometry<Dims> make_nd(const range<3> local_size) const {
		if(std::any_of(
		       m_chunks.begin(), m_chunks.end(), [=](const geometry_chunk& chunk) { return chunk.box.get_range() % local_size != celerity::detail::zeros; })) {
			throw std::runtime_error("Local size does not divide chunk size");
		}
		return celerity::nd_custom_task_geometry<Dims>{range_cast<3>(m_global_size), celerity::detail::zeros, local_size, m_chunks};
	}

  private:
	// TODO: Should this be a box, like in partition? We could then pass the offset as global offset into task_geometry.
	// => Altough it's not clear we even need global_offset (and size, for that matter) in custom_task_geometry at all - see comments there
	celerity::range<2> m_global_size;
	std::vector<celerity::geometry_chunk> m_chunks;

	bool is_overlapping() const {
		celerity::detail::region<3> current_region;
		for(const auto& chunk : m_chunks) {
			if(!celerity::detail::region_intersection(current_region, chunk.box).empty()) return true;
			current_region = celerity::detail::region_union(current_region, chunk.box);
		}
		return false;
	}
};

} // namespace celerity
