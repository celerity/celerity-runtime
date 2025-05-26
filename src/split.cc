#include "split.h"

#include "grid.h"
#include "ranges.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>


namespace {

using namespace celerity;
using namespace celerity::detail;

[[maybe_unused]] void sanity_check_split(const box<3>& full_chunk, const std::vector<box<3>>& split) {
	region<3> reconstructed_chunk;
	for(auto& chnk : split) {
		assert(region_intersection(reconstructed_chunk, chnk).empty());
		reconstructed_chunk = region_union(chnk, reconstructed_chunk);
	}
	assert(region_difference(reconstructed_chunk, full_chunk).empty());
}

template <int Dims>
std::tuple<range<Dims>, range<Dims>, range<Dims>> compute_small_and_large_chunks(
    const box<3>& full_chunk, const range<3>& granularity, const std::array<size_t, Dims>& actual_num_chunks) {
	range<Dims> small_chunk_size{zeros};
	range<Dims> large_chunk_size{zeros};
	range<Dims> num_large_chunks{zeros};
	for(int d = 0; d < Dims; ++d) {
		const size_t ideal_chunk_size = full_chunk.get_range()[d] / actual_num_chunks[d];
		small_chunk_size[d] = (ideal_chunk_size / granularity[d]) * granularity[d];
		large_chunk_size[d] = small_chunk_size[d] + granularity[d];
		num_large_chunks[d] = (full_chunk.get_range()[d] - small_chunk_size[d] * actual_num_chunks[d]) / granularity[d];
	}
	return {small_chunk_size, large_chunk_size, num_large_chunks};
}

/**
 * Given a factorization of `num_chunks` (i.e., `f0 * f1 = num_chunks`), try to find the assignment of factors to
 * dimensions that produces more chunks under the given constraints. If they are tied, try to find the assignment
 * that results in a "nicer" split according to some heuristics (see below).
 *
 * The single argument `factor` specifies both factors, as `f0 = factor` and `f1 = num_chunks / factor`.
 *
 * @returns The number of chunks that can be created in dimension 0 and dimension 1, respectively. These are at most
 *          (f0, f1) or (f1, f0), however may be less if constrained by the split granularity.
 */
std::array<size_t, 2> assign_split_factors_2d(const box<3>& full_chunk, const range<3>& granularity, const size_t factor, const size_t num_chunks) {
	assert(num_chunks % factor == 0);
	const size_t max_chunks[2] = {full_chunk.get_range()[0] / granularity[0], full_chunk.get_range()[1] / granularity[1]};
	const size_t f0 = factor;
	const size_t f1 = num_chunks / factor;

	// Decide in which direction to split by first checking which
	// factor assignment produces more chunks under the given constraints.
	const std::array<size_t, 2> split_0_1 = {std::min(f0, max_chunks[0]), std::min(f1, max_chunks[1])};
	const std::array<size_t, 2> split_1_0 = {std::min(f1, max_chunks[0]), std::min(f0, max_chunks[1])};
	const auto count0 = split_0_1[0] * split_0_1[1];
	const auto count1 = split_1_0[0] * split_1_0[1];

	if(count0 > count1) { return split_0_1; }
	if(count0 < count1) { return split_1_0; }

	// If we're tied for the number of chunks we can create, try some heuristics to decide.

	// If domain is square(-ish), prefer splitting along slower dimension.
	// (These bounds have been chosen arbitrarily!)
	const double squareishness = std::sqrt(full_chunk.get_area()) / static_cast<double>(full_chunk.get_range()[0]);
	if(squareishness > 0.95 && squareishness < 1.05) { return (f0 >= f1) ? split_0_1 : split_1_0; }

	// For non-square domains, prefer split that produces shorter edges (compare sum of circumferences)
	const auto circ0 = full_chunk.get_range()[0] / split_0_1[0] + full_chunk.get_range()[1] / split_0_1[1];
	const auto circ1 = full_chunk.get_range()[0] / split_1_0[0] + full_chunk.get_range()[1] / split_1_0[1];
	return circ0 < circ1 ? split_0_1 : split_1_0;

	// TODO: Yet another heuristic we may want to consider is how even chunk sizes are,
	// i.e., how balanced the workload is.
}

} // namespace

namespace celerity::detail {

std::vector<box<3>> split_1d(const box<3>& full_box, const range<3>& granularity, const size_t num_boxs) {
#ifndef NDEBUG
	assert(num_boxs > 0);
	for(int d = 0; d < 3; ++d) {
		assert(granularity[d] > 0);
		assert(full_box.get_range()[d] % granularity[d] == 0);
	}
#endif

	// Due to split granularity requirements or if num_workers > global_size[0],
	// we may not be able to create the requested number of boxs.
	const std::array<size_t, 1> actual_num_boxs = {std::min(num_boxs, full_box.get_range()[0] / granularity[0])};
	const auto [small_chunk_size, large_chunk_size, num_large_chunks] = compute_small_and_large_chunks<1>(full_box, granularity, actual_num_boxs);

	std::vector<box<3>> result;
	result.reserve(actual_num_boxs[0]);
	for(auto i = 0u; i < num_large_chunks[0]; ++i) {
		id<3> min = full_box.get_min();
		id<3> max = full_box.get_max();
		min[0] += i * large_chunk_size[0];
		max[0] = min[0] + large_chunk_size[0];
		result.emplace_back(min, max);
	}
	for(auto i = num_large_chunks[0]; i < actual_num_boxs[0]; ++i) {
		id<3> min = full_box.get_min();
		id<3> max = full_box.get_max();
		min[0] += num_large_chunks[0] * large_chunk_size[0] + (i - num_large_chunks[0]) * small_chunk_size[0];
		max[0] = min[0] + small_chunk_size[0];
		result.emplace_back(min, max);
	}

#ifndef NDEBUG
	sanity_check_split(full_box, result);
#endif

	return result;
}

// TODO: Make the split dimensions configurable for 3D chunks?
std::vector<box<3>> split_2d(const box<3>& full_box, const range<3>& granularity, const size_t num_boxs) {
#ifndef NDEBUG
	assert(num_boxs > 0);
	for(int d = 0; d < 3; ++d) {
		assert(granularity[d] > 0);
		assert(full_box.get_range()[d] % granularity[d] == 0);
	}
#endif

	// Factorize num_boxs
	// We start out with an initial guess of `factor = floor(sqrt(num_boxs))` (the other one is implicitly given by `num_boxs / factor`),
	// and work our way down, keeping track of the best factorization we've found so far, until we find a factorization that produces
	// the requested number of chunks, or until we reach (1, num_boxs), i.e., a 1D split.
	size_t factor = std::floor(std::sqrt(num_boxs));
	std::array<size_t, 2> best_chunk_counts = {0, 0};
	while(factor >= 1) {
		while(factor > 1 && num_boxs % factor != 0) {
			factor--;
		}
		// The returned counts are at most (factor, num_boxs / factor), however may be less if constrained by the split granularity.
		const auto chunk_counts = assign_split_factors_2d(full_box, granularity, factor, num_boxs);
		if(chunk_counts[0] * chunk_counts[1] > best_chunk_counts[0] * best_chunk_counts[1]) { best_chunk_counts = chunk_counts; }
		if(chunk_counts[0] * chunk_counts[1] == num_boxs) { break; }
		factor--;
	}
	const auto actual_num_chunks = best_chunk_counts;
	const auto [small_chunk_size, large_chunk_size, num_large_chunks] = compute_small_and_large_chunks<2>(full_box, granularity, actual_num_chunks);

	std::vector<box<3>> result;
	result.reserve(actual_num_chunks[0] * actual_num_chunks[1]);
	id<3> offset = full_box.get_min();

	for(size_t j = 0; j < actual_num_chunks[0]; ++j) {
		range<2> chunk_size = {(j < num_large_chunks[0]) ? large_chunk_size[0] : small_chunk_size[0], 0};
		for(size_t i = 0; i < actual_num_chunks[1]; ++i) {
			chunk_size[1] = (i < num_large_chunks[1]) ? large_chunk_size[1] : small_chunk_size[1];
			const id<3> min = offset;
			id<3> max = full_box.get_max();
			max[0] = min[0] + chunk_size[0];
			max[1] = min[1] + chunk_size[1];
			result.emplace_back(min, max);
			offset[1] += chunk_size[1];
		}
		offset[0] += chunk_size[0];
		offset[1] = full_box.get_min()[1];
	}

#ifndef NDEBUG
	sanity_check_split(full_box, result);
#endif

	return result;
}
} // namespace celerity::detail
