#pragma once

#include <celerity.h>

#include <algorithm>
#include <utility>
#include <vector>

// TODO API: Are there any primitives in here that we could provide as generic utilities for custom data structures in the Celerity "expert interface"?

/// Computes the set of contiguous subranges in the 1D tile buffer written by a particular rank
/// Lower ranks get ordered before higher ranks in case multiple ranks write to the same tile
// TODO: Inputs are somewhat redundant; if we don't need them elsewhere we could also compute them here
inline std::vector<celerity::subrange<1>> compute_written_subranges(
    int rank, const std::vector<uint32_t>& num_entries, const std::vector<uint32_t>& num_entries_by_rank, const std::vector<uint32_t>& num_entries_cumulative) {
	const uint32_t* const my_counts = num_entries_by_rank.data() + rank * num_entries.size();
	std::vector<celerity::subrange<1>> written_subranges;
	std::optional<celerity::subrange<1>> current_sr;
	for(size_t i = 0; i < num_entries.size(); ++i) {
		if(my_counts[i] == 0) {
			if(num_entries[i] != 0 && current_sr.has_value()) {
				// We're not writing to this tile, but somebody else is - finalize current subrange
				written_subranges.push_back(current_sr.value());
				current_sr.reset();
			}
			continue;
		}

		if(num_entries[i] == my_counts[i]) {
			// Simple case: We're the only ones writing to this tile
			if(current_sr.has_value()) {
				current_sr->range[0] += my_counts[i];
			} else {
				current_sr = celerity::subrange<1>{num_entries_cumulative[i], my_counts[i]};
			}
		} else {
			// Begin by figuring out how many other ranks write to this tile before me
			uint32_t other_counts = 0;
			for(size_t j = 0; j < size_t(rank); ++j) {
				other_counts += num_entries_by_rank[j * num_entries.size() + i];
			}
			if(other_counts == 0) {
				// We're the first rank to write to this tile
				if(current_sr.has_value()) {
					current_sr->range[0] += my_counts[i];
				} else {
					current_sr = celerity::subrange<1>{num_entries_cumulative[i], my_counts[i]};
				}
			} else {
				// We're not the first rank to write to this tile
				if(current_sr.has_value()) {
					// We need to finalize the current subrange
					written_subranges.push_back(current_sr.value());
				}
				current_sr = celerity::subrange<1>{num_entries_cumulative[i] + other_counts, my_counts[i]};
			}
			if(other_counts + my_counts[i] != num_entries[i]) {
				// We're not the last rank to write to this tile
				written_subranges.push_back(current_sr.value());
				current_sr.reset();
			}
		}
	}
	if(current_sr.has_value()) { written_subranges.push_back(current_sr.value()); }
	return written_subranges;
}

// Same as above, but works for multiple devices per rank
// TODO: Return regions instead? (also above)
inline std::vector<std::vector<celerity::subrange<1>>> compute_written_subranges_per_device(int rank, const std::vector<uint32_t>& num_entries,
    const std::vector<uint32_t>& num_entries_by_rank, const std::vector<uint32_t>& num_entries_cumulative,
    const std::vector<std::vector<uint32_t>>& num_entries_per_device) {
	using celerity::subrange;
	using celerity::detail::device_id;

#ifndef NDEBUG
	// Sanity check: The sum of entries across devices must be equal to the number of entries for this rank in each tile
	for(size_t i = 0; i < num_entries.size(); ++i) {
		size_t sum_devices = 0;
		for(const auto& per_device : num_entries_per_device) {
			sum_devices += per_device[i];
		}
		assert(sum_devices == num_entries_by_rank[rank * num_entries.size() + i]);
	}
#endif

	// TODO: Not sure if its better to compute the whole thing from scratch or start with per-rank ranges. For now we'll do the latter.
	const auto rank_srs = compute_written_subranges(rank, num_entries, num_entries_by_rank, num_entries_cumulative);

	const size_t num_devices = num_entries_per_device.size();
	std::vector<std::vector<celerity::subrange<1>>> result(num_devices);

	size_t current_idx = 0;
	for(const auto& rsr : rank_srs) {
		std::vector<size_t> current_linear_tile_index(num_devices);
		device_id current_did = ([&] {
			// Find the first tile for which at least one device has entries
			for(; current_idx < num_entries.size(); ++current_idx) {
				for(device_id did = 0; did < num_devices; ++did) {
					if(num_entries_per_device[did][current_idx] != 0) { return did; }
				}
			}
			// This should never happen
			abort();
		})(); // IIFE

		subrange<1> current_sr = {rsr.offset[0], 0};
		size_t consumed = 0;
		while(consumed < rsr.range[0]) {
			for(device_id did = 0; did < num_devices; ++did) {
				const auto entries = num_entries_per_device[did][current_idx];
				if(entries == 0) continue;
				if(did == current_did) {
					current_sr.range[0] += entries;
					consumed += entries;
					continue;
				}
				result[current_did].push_back(current_sr);
				current_sr = {rsr.offset[0] + consumed, entries};
				consumed += entries;
				current_did = did;
			}
			current_idx++;
		}
		result[current_did].push_back(current_sr);
	}

	return result;
}

// DEPRECATED - I think we have to separate the two things after all (I forgot that we need a 2D neighborhood...)
#if 0
// TODO: Store one additional element in num_entries_cumulative, which would be num_entries?
inline std::pair<celerity::TASK_GEOMETRY, std::vector<std::vector<celerity::subrange<3>>>> compute_task_geometry_and_neighborhood_reads(
    const int num_ranks, const uint32_t total_entries, const std::vector<uint32_t>& num_entries_cumulative) {
	using celerity::subrange;
	using celerity::detail::subrange_cast;

	celerity::TASK_GEOMETRY geometry;
	const auto ideal_points_per_rank = total_entries / num_ranks;
	uint32_t current_offset = 0;
	uint32_t prev_end_tile = 0;
	std::vector<std::vector<subrange<3>>> per_chunk_neighborhood_reads;
	for(int i = 0; i < num_ranks; ++i) {
		const auto end_it =
		    std::lower_bound(num_entries_cumulative.begin() + prev_end_tile, num_entries_cumulative.end(), current_offset + ideal_points_per_rank);
		const auto end_tile = end_it != num_entries_cumulative.end() ? end_it - num_entries_cumulative.begin() : -1;
		const uint32_t end_offset = end_it != num_entries_cumulative.end() ? num_entries_cumulative[end_tile] : total_entries;

		assert(end_offset != current_offset);
		const subrange<1> chunk{{current_offset}, {end_offset - current_offset}};
		geometry.push_back({subrange_cast<3>(chunk), i});

		subrange<1> read_range = {current_offset, end_offset - current_offset};
		if(i > 0) {
			read_range.offset[0] = num_entries_cumulative[prev_end_tile - 1];
			read_range.range[0] = end_offset - num_entries_cumulative[prev_end_tile - 1];
		}
		if(end_it != num_entries_cumulative.end()) { //
			read_range.range[0] = (end_tile < num_entries_cumulative.size() - 1 ? num_entries_cumulative[end_tile + 1] : total_entries) - read_range.offset[0];
		}
		per_chunk_neighborhood_reads.push_back({subrange_cast<3>(read_range)});

		prev_end_tile = end_tile;
		current_offset = end_offset;

		// If we run into this it means we're unable to create enough chunks with ideal_points_per_rank
		// TODO: We might be able to create more chunks with slightly less than ideal_points_per_rank
		if(end_offset == total_entries) { break; }
	}

	return {geometry, per_chunk_neighborhood_reads};
}
#endif

// TODO: This should ideally try to assign chunks in a way that minimizes required data movement
//          => We don't have to assign chunks to nodes here, could do that in a separate step
// TODO: Store one additional element in num_entries_cumulative, which would be num_entries?
// TODO: Rename to indicate its attempting to do an even split on number of elements?
inline celerity::custom_task_geometry<1> compute_task_geometry(
    const int num_ranks, const uint32_t total_entries, const std::vector<uint32_t>& num_entries_cumulative) {
	using celerity::subrange;
	using celerity::detail::subrange_cast;

	celerity::custom_task_geometry geometry;
	geometry.global_size = range_cast<3>(celerity::range<1>(total_entries));
	const auto ideal_points_per_rank = total_entries / num_ranks;
	uint32_t current_offset = 0;
	uint32_t prev_end_tile = 0;
	for(int i = 0; i < num_ranks; ++i) {
		const auto end_it =
		    std::lower_bound(num_entries_cumulative.begin() + prev_end_tile, num_entries_cumulative.end(), current_offset + ideal_points_per_rank);
		const auto end_tile = end_it != num_entries_cumulative.end() ? end_it - num_entries_cumulative.begin() : -1;
		const uint32_t end_offset = end_it != num_entries_cumulative.end() ? num_entries_cumulative[end_tile] : total_entries;

		assert(end_offset != current_offset);
		const subrange<1> chunk{{current_offset}, {end_offset - current_offset}};
		geometry.assigned_chunks.push_back({subrange_cast<3>(chunk), i, {}});

		prev_end_tile = end_tile;
		current_offset = end_offset;

		// If we run into this it means we're unable to create enough chunks with ideal_points_per_rank
		// TODO: We might be able to create more chunks with slightly less than ideal_points_per_rank
		if(end_offset == total_entries) { break; }
	}

	return geometry;
}

// TODO: We don't actually need the geometry (as in, with node assignments), just the ranges
// TODO: This could also be simplified by including one additional entry in num_entries_cumulative
inline std::vector<std::vector<celerity::subrange<3>>> compute_neighborhood_reads_1d(const celerity::custom_task_geometry<1>& geometry,
    const uint32_t total_entries, const std::vector<uint32_t>& num_entries, const std::vector<uint32_t>& num_entries_cumulative) {
	using celerity::subrange;
	using celerity::detail::subrange_cast;
	assert(num_entries_cumulative.size() == num_entries.size());

	std::vector<std::vector<subrange<3>>> per_chunk_neighborhood_reads;
	for(auto [box, _, _2] : geometry.assigned_chunks) {
		assert(box.get_effective_dims() == 1);
		const auto sr = box.get_subrange();

		// Find start and end tile for this subrange
		// upper_bound - 1: We want to find the first tile that contains any elements, because only on those will we
		// do any computations (and thus require a neighborhood read)
		// Using lower_bound instead returns the first empty tile after the previous chunk, thus overestimating the required reads
		const auto start_it = std::upper_bound(num_entries_cumulative.begin(), num_entries_cumulative.end(), sr.offset[0]) - 1;
		assert(start_it != num_entries_cumulative.end());
		const auto end_it = std::lower_bound(start_it, num_entries_cumulative.end(), sr.offset[0] + sr.range[0]);
		assert(end_it != num_entries_cumulative.end() || sr.offset[0] + sr.range[0] == total_entries);

		const uint32_t start_tile = start_it - num_entries_cumulative.begin();
		const uint32_t end_tile = end_it != num_entries_cumulative.end() ? end_it - num_entries_cumulative.begin() : -1;

		subrange<1> read_range = subrange_cast<1>(sr);

		if(start_tile > 0) {
			read_range.offset[0] -= num_entries[start_tile - 1];
			read_range.range[0] += num_entries[start_tile - 1];
		}
		if(end_it != num_entries_cumulative.end()) { //
			read_range.range[0] += num_entries[end_tile];
		}

		per_chunk_neighborhood_reads.push_back({subrange_cast<3>(read_range)});
	}

	return per_chunk_neighborhood_reads;
}

// TODO: We don't actually need num entries here - could do the same above (by computing coordinates first, then converting to offsets)
inline std::vector<std::vector<celerity::subrange<3>>> compute_neighborhood_reads_2d(const celerity::custom_task_geometry<1>& geometry,
    const celerity::range<2>& buffer_size, const uint32_t total_entries, const std::vector<uint32_t>& num_entries,
    const std::vector<uint32_t>& num_entries_cumulative) {
	using celerity::subrange;
	using celerity::detail::region;
	using celerity::detail::subrange_cast;
	assert(num_entries.size() == buffer_size.size());
	assert(num_entries_cumulative.size() == buffer_size.size());

	std::vector<std::vector<subrange<3>>> per_chunk_neighborhood_reads;
	for(auto [box, _, _2] : geometry.assigned_chunks) {
		// The split is still in 1D
		assert(box.get_effective_dims() == 1);

		const auto sr = box.get_subrange();

		// Find start and end tile for this subrange
		// upper_bound - 1: We want to find the first tile that contains any elements, because only on those will we
		// do any computations (and thus require a neighborhood read)
		// Using lower_bound instead returns the first empty tile after the previous chunk, thus overestimating the required reads
		const auto start_it = std::upper_bound(num_entries_cumulative.begin(), num_entries_cumulative.end(), sr.offset[0]) - 1;
		assert(start_it != num_entries_cumulative.end());
		const auto end_it = std::lower_bound(start_it, num_entries_cumulative.end(), sr.offset[0] + sr.range[0]);
		assert(end_it != num_entries_cumulative.end() || sr.offset[0] + sr.range[0] == total_entries);

		const uint32_t start_tile = start_it - num_entries_cumulative.begin();
		const uint32_t end_tile = end_it != num_entries_cumulative.end() ? end_it - num_entries_cumulative.begin() : num_entries.size();

		// Compute inclusive start and end coordinates in 2D
		const celerity::id<2> first = {start_tile / buffer_size[1], start_tile % buffer_size[1]};
		const celerity::id<2> last = celerity::id<2>{(end_tile - 1) / buffer_size[1], (end_tile - 1) % buffer_size[1]};

		celerity::detail::box_vector<1> read_boxes;
		const auto add_box = [&](const celerity::id<2>& min, const celerity::id<2>& max) {
			const uint32_t start_1d = min[0] * buffer_size[1] + min[1];
			const uint32_t end_1d = max[0] * buffer_size[1] + max[1] + 1; // +1: Convert back to exclusive
			read_boxes.push_back({num_entries_cumulative[start_1d], end_1d < num_entries.size() ? num_entries_cumulative[end_1d] : total_entries});
		};

		// Add main chunk range + left and right neighbors
		celerity::id<2> main_neighborhood_start = first;
		celerity::id<2> main_neighborhood_end = last;
		if(first[1] > 0) { main_neighborhood_start[1]--; }
		if(last[1] < buffer_size[1] - 1) { main_neighborhood_end[1]++; }
		add_box(main_neighborhood_start, main_neighborhood_end);

		// Add top neighbors
		if(last[0] > 0) {
			auto adjusted_start = main_neighborhood_start;
			if(first[0] == 0) { adjusted_start = {1, 0}; }
			const celerity::id<2> top_neighborhood_start = {adjusted_start[0] - 1, adjusted_start[1]};
			const celerity::id<2> top_neighborhood_end = {main_neighborhood_end[0] - 1, main_neighborhood_end[1]};
			add_box(top_neighborhood_start, top_neighborhood_end);
		}

		// Add bottom neighbors
		if(first[0] < buffer_size[0] - 1) {
			auto adjusted_end = main_neighborhood_end;
			if(last[0] == buffer_size[0] - 1) { adjusted_end = {buffer_size[0] - 2, buffer_size[1] - 1}; }
			const celerity::id<2> bottom_neighborhood_start = {main_neighborhood_start[0] + 1, main_neighborhood_start[1]};
			const celerity::id<2> bottom_neighborhood_end = {adjusted_end[0] + 1, adjusted_end[1]};
			add_box(bottom_neighborhood_start, bottom_neighborhood_end);
		}

#if 0 // Naive implementation: Iterate over all tiles in range and add their neighborhood
		{
			celerity::detail::box_vector<1> read_boxes_naive;
			const auto add_single_tile = [&](const celerity::id<2>& coords) {
				const uint32_t start_1d = coords[0] * buffer_size[1] + coords[1];
				const uint32_t end_1d = start_1d + 1;
				read_boxes_naive.push_back({num_entries_cumulative[start_1d], end_1d < num_entries.size() ? num_entries_cumulative[end_1d] : total_entries});
			};

			for(uint32_t i = start_tile; i < end_tile; ++i) {
				const celerity::id<2> tile_coords = {i / buffer_size[1], i % buffer_size[1]};
				// iterate over all neighboring cells (this includes the center cell as well)
				for(int dx = -1; dx <= 1; ++dx) {
					for(int dy = -1; dy <= 1; ++dy) {
						const celerity::id<2> neighbor_coords = {tile_coords[0] + dy, tile_coords[1] + dx};
						if(neighbor_coords[0] >= buffer_size[0] || neighbor_coords[1] >= buffer_size[1]) { // only check larger b/c of unsigned underflow
							continue;
						}
						add_single_tile(neighbor_coords);
					}
				}
			}

			// Check that naive variant produced same set of boxes
			{
				auto read_boxes_copy = read_boxes;
				auto read_boxes_naive_copy = read_boxes_naive;
				auto read_boxes_region = celerity::detail::region<1>{std::move(read_boxes_copy)};
				auto read_boxes_naive_region = celerity::detail::region<1>{std::move(read_boxes_naive_copy)};
				if(read_boxes_region != read_boxes_naive_region) { //
					throw std::runtime_error(fmt::format("Naive and optimized neighborhood read computation produced different results. NAIVE: {}, NEW: {}",
					    read_boxes_naive_region, read_boxes_region));
				}
			}
		}
#endif

		// We collect all reads in a region so overlapping/connected ranges are automatically merged
		// TODO: We should probably just return a region from this function
		std::vector<subrange<3>> srs;
		for(auto& box : celerity::detail::region<1>{std::move(read_boxes)}.into_boxes()) {
			srs.push_back({subrange_cast<3>(box.get_subrange())});
		}
		per_chunk_neighborhood_reads.push_back(srs);
	}

	return per_chunk_neighborhood_reads;
}

// TODO: Make this a Celerity built-in function?
template <int Dims>
std::vector<celerity::detail::region<Dims>> allgather_regions(celerity::detail::region<Dims> local_region, const size_t num_ranks, const size_t rank) {
	const auto& local_box_vector = local_region.get_boxes();
	std::vector<int> num_boxes_per_rank(num_ranks);
	num_boxes_per_rank[rank] = local_box_vector.size();
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, num_boxes_per_rank.data(), 1, MPI_INT, MPI_COMM_WORLD);
	std::accumulate(num_boxes_per_rank.begin(), num_boxes_per_rank.end(), 0);
	size_t total_num_boxes = 0;
	std::vector<int> displs(num_ranks);
	std::vector<int> recv_counts(num_ranks);
	for(size_t i = 0; i < num_ranks; ++i) {
		displs[i] = total_num_boxes * sizeof(celerity::detail::box<Dims>); // displacement is in elements (which is bytes)
		total_num_boxes += num_boxes_per_rank[i];
		recv_counts[i] = num_boxes_per_rank[i] * sizeof(celerity::detail::box<Dims>);
	}
	celerity::detail::box_vector<Dims> all_boxes(total_num_boxes);
	MPI_Allgatherv(local_box_vector.data(), local_box_vector.size() * sizeof(celerity::detail::box<Dims>), MPI_BYTE, all_boxes.data(), recv_counts.data(),
	    displs.data(), MPI_BYTE, MPI_COMM_WORLD);
	std::vector<celerity::detail::region<Dims>> result(num_ranks);
	size_t next_rank_start = 0;
	for(size_t i = 0; i < num_ranks; ++i) {
		if(i == rank) {
			result[i] = std::move(local_region);
			next_rank_start += num_boxes_per_rank[i];
			continue;
		}
		celerity::detail::region_builder<Dims> builder;
		for(size_t j = next_rank_start; j < next_rank_start + num_boxes_per_rank[i]; ++j) {
			builder.add(all_boxes[j]);
		}
		result[i] = std::move(builder).into_region();
		next_rank_start += num_boxes_per_rank[i];
	}
	return result;
}

struct tilebuffer_item {
	celerity::id<2> slot;
	uint32_t index = 0;
	bool within_bounds = false;
};

/// Returns the index of the current "item" (point in a tile) being processed by the given thread
/// TODO: Only works for 2D buffers
inline tilebuffer_item get_current_item(
    const celerity::id<1>& thread_idx, const uint32_t total_count, const celerity::range<2>& buffer_extent, const uint32_t* num_entries_cumulative) {
	if(thread_idx[0] >= total_count) { return {{}, 0, false}; }

	const auto buffer_size = buffer_extent.size();
	uint32_t min = 0;
	uint32_t max = buffer_size;
	auto midpoint = (min + max) / 2;

	// do binary search until we find an index i such that num_entries_cumulative[i] <= thread_idx and
	// num_entries_cumulative[i+1] > thread_idx
	while(midpoint > 0 && midpoint < buffer_size) {
		if(num_entries_cumulative[midpoint] <= thread_idx[0]) {
			if(midpoint == buffer_size - 1 || num_entries_cumulative[midpoint + 1] > thread_idx[0]) { break; }
			min = midpoint;
		} else {
			max = midpoint;
		}
		midpoint = (min + max) / 2;
	}

	// Now compute actual index
	const uint32_t index = thread_idx[0] - num_entries_cumulative[midpoint];
	celerity::id<2> slot;
	slot[0] = midpoint / buffer_extent[1];
	slot[1] = midpoint % buffer_extent[1];
	return {slot, index, true};
}
