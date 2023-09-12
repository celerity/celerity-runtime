#include "grid.h"

namespace celerity::detail::grid_detail {

// Regions have a storage dimensionality (the `Dims` template parameter of `class region`) and an effective dimensionality that is smaller iff all contained
// boxes are effectively the result of casting e.g. box<2> to box<3>, or the described region "accidentally" is a lower-dimensional slice of the full space.
// This property is detected at runtime through {box,region}::get_effective_dims(), and all region-algorithm implementations are generic over both StorageDims
// and EffectiveDims to optimize for the embedding of arbitrary-dimensional regions into region<3> as it commonly happens in the runtime.

// Like detail::box_intersection, but aware of effective dimensionality
template <int EffectiveDims, int StorageDims>
box<StorageDims> box_intersection(const box<StorageDims>& box1, const box<StorageDims>& box2) {
	static_assert(EffectiveDims <= StorageDims);

	id<StorageDims> min;
	id<StorageDims> max;
	for(int d = 0; d < EffectiveDims; ++d) {
		min[d] = std::max(box1.get_min()[d], box2.get_min()[d]);
		max[d] = std::min(box1.get_max()[d], box2.get_max()[d]);
		if(min[d] >= max[d]) return {};
	}
	for(int d = EffectiveDims; d < StorageDims; ++d) {
		min[d] = 0;
		max[d] = 1;
	}
	return make_box<StorageDims>(non_empty, min, max);
}

// Like box::covers, but aware of effective dimensionality
template <int EffectiveDims, int StorageDims>
bool box_covers(const box<StorageDims>& top, const box<StorageDims>& bottom) {
	static_assert(EffectiveDims <= StorageDims);

	// empty boxes are normalized and thus may not intersect in coordinates
	if(bottom.empty()) return true;

	for(int d = 0; d < EffectiveDims; ++d) {
		if(bottom.get_min()[d] < top.get_min()[d]) return false;
		if(bottom.get_max()[d] > top.get_max()[d]) return false;
	}
	return true;
}

// In a range of boxes that are identical in all dimensions except MergeDim, merge all connected boxes ("unconditional directional merge")
template <int MergeDim, typename BidirectionalIterator>
BidirectionalIterator merge_connected_intervals(BidirectionalIterator first, BidirectionalIterator last) {
	using box_type = typename std::iterator_traits<BidirectionalIterator>::value_type;

	if(first == last || std::next(first) == last) return last; // common-case shortcut: no merge is possible

	// Sort by interval starting point
	std::sort(first, last, [](const box_type& lhs, const box_type& rhs) { return lhs.get_min()[MergeDim] < rhs.get_min()[MergeDim]; });

	// The range is both read and written from left-to-right, avoiding repeated left-shifts for compaction
	auto last_out = first;

	// Merge all connected boxes along MergeDim in O(N) by replacing each connected sequence with its bounding box
	while(first != last) {
		const auto merged_min = first->get_min();
		auto merged_max = first->get_max();
		for(++first; first != last && first->get_min()[MergeDim] <= merged_max[MergeDim]; ++first) {
			merged_max[MergeDim] = std::max(merged_max[MergeDim], first->get_max()[MergeDim]);
		}
		*last_out++ = make_box<box_type::dimensions>(grid_detail::non_empty, merged_min, merged_max);
	}

	return last_out;
}

// In an arbitrary range of boxes, merge all boxes that are identical in all dimensions except MergeDim ("conditional directional merge").
template <int MergeDim, int EffectiveDims, typename BidirectionalIterator>
BidirectionalIterator merge_connected_boxes_along_dim(const BidirectionalIterator first, const BidirectionalIterator last) {
	using box_type = typename std::iterator_traits<BidirectionalIterator>::value_type;
	static_assert(EffectiveDims <= box_type::dimensions);
	static_assert(MergeDim < EffectiveDims);

	constexpr auto orthogonal_to_merge_dim = [](const box_type& lhs, const box_type& rhs) {
		for(int d = 0; d < EffectiveDims; ++d) {
			if(d == MergeDim) continue;
			// arbitrary but consistent ordering along all orthogonal dimensions
			if(lhs.get_min()[d] < rhs.get_min()[d]) return true;
			if(lhs.get_min()[d] > rhs.get_min()[d]) return false;
			if(lhs.get_max()[d] < rhs.get_max()[d]) return true;
			if(lhs.get_max()[d] > rhs.get_max()[d]) return false;
		}
		return false;
	};

	if constexpr(EffectiveDims == 1) {
		return merge_connected_intervals<MergeDim>(first, last);
	} else {
		// partition [first, last) into sequences of boxes that are potentially mergeable wrt/ the dimensions orthogonal to MergeDim.
		// This reduces complexity from O(n^3) to O(n log n) + O(m^3), where m is the longest mergeable sequence in that regard.
		std::sort(first, last, orthogonal_to_merge_dim);

		// we want the result to be contiguous in [first, last_out), so in each iteration, we merge all boxes of a MergeDim-equal partition at their original
		// position in the iterator range; and then shift the merged range back to fill any gap left by merge of a previous partition.
		auto last_out = first;

		for(auto first_equal = first; first_equal != last;) {
			// O(n) std::find_if could be replaced by O(log n) std::partition_point, but we expect the number of "equal" elements to be small
			const auto last_equal = std::find_if(std::next(first_equal), last, [&](const box_type& box) {
				return orthogonal_to_merge_dim(*first_equal, box); // true if box is in a partition _after_ *first_equal
			});
			const auto last_merged = merge_connected_intervals<MergeDim>(first_equal, last_equal);
			// shift the newly merged boxes to the left to close any gap opened by the merge of a previous partition
			last_out = std::move(first_equal, last_merged, last_out);
			first_equal = last_equal;
		}

		return last_out;
	}
}

// explicit instantiations for tests (might otherwise be inlined)
template box_vector<1>::iterator merge_connected_boxes_along_dim<0, 1>(box_vector<1>::iterator first, box_vector<1>::iterator last);
template box_vector<2>::iterator merge_connected_boxes_along_dim<0, 2>(box_vector<2>::iterator first, box_vector<2>::iterator last);
template box_vector<2>::iterator merge_connected_boxes_along_dim<1, 2>(box_vector<2>::iterator first, box_vector<2>::iterator last);
template box_vector<3>::iterator merge_connected_boxes_along_dim<0, 3>(box_vector<3>::iterator first, box_vector<3>::iterator last);
template box_vector<3>::iterator merge_connected_boxes_along_dim<1, 3>(box_vector<3>::iterator first, box_vector<3>::iterator last);
template box_vector<3>::iterator merge_connected_boxes_along_dim<2, 3>(box_vector<3>::iterator first, box_vector<3>::iterator last);

// For higher-dimensional regions, the order in which dimensions are merged is relevant for the shape of the resulting box set. We merge along the last
// ("fastest") dimension first to make sure the resulting boxes cover the largest possible extent of contiguous memory when are applied to buffers.
template <int MergeDim, int EffectiveDims, typename BidirectionalIterator>
BidirectionalIterator merge_connected_boxes_recurse(const BidirectionalIterator first, BidirectionalIterator last) {
	static_assert(MergeDim >= 0 && MergeDim < EffectiveDims);
	last = merge_connected_boxes_along_dim<MergeDim, EffectiveDims>(first, last);
	if constexpr(MergeDim > 0) { last = merge_connected_boxes_recurse<MergeDim - 1, EffectiveDims>(first, last); }
	return last;
}

// Merge all adjacent boxes that are connected and identical in all except a single dimension.
template <int EffectiveDims, typename BidirectionalIterator>
BidirectionalIterator merge_connected_boxes(const BidirectionalIterator first, BidirectionalIterator last) {
	using box_type = typename std::iterator_traits<BidirectionalIterator>::value_type;
	static_assert(EffectiveDims <= box_type::dimensions);
	if constexpr(EffectiveDims > 0) { last = merge_connected_boxes_recurse<EffectiveDims - 1, EffectiveDims>(first, last); }
	return last;
}

// Split a box into parts according to dissection lines in `cuts`, where `cuts` is indexed by component dimension. This function is not generic
// over EffectiveDims, rather, `cuts` will have 1 <= n <= StorageDims entries to indicate along how many dimensions the box should be dissected.
template <int StorageDims>
void dissect_box(const box<StorageDims>& in_box, const std::vector<std::vector<size_t>>& cuts, box_vector<StorageDims>& out_dissected, int dim) {
	assert(dim < static_cast<int>(cuts.size()));

	const auto& dim_cuts = cuts[static_cast<size_t>(dim)];
	assert(std::is_sorted(dim_cuts.begin(), dim_cuts.end()));

	// start of the first (current) dissected box
	size_t start = in_box.get_min()[dim];
	// find the first cut that lies inside the box (dim_cuts is sorted)
	auto cut_it = std::lower_bound(dim_cuts.begin(), dim_cuts.end(), /* not less or equal */ start + 1);

	for(;;) {
		// the end of the current box is either the last cut that lies inside the box, or the end of in_box
		size_t end;
		if(cut_it != dim_cuts.end() && *cut_it < in_box.get_max()[dim]) {
			end = *cut_it++;
		} else {
			end = in_box.get_max()[dim];
		}
		if(end == start) break;

		// compute coordinates for the dissected box along `dim`, and recursively dissect it further along `dim + 1`
		auto min = in_box.get_min();
		auto max = in_box.get_max();
		min[dim] = start;
		max[dim] = end;
		const auto small_box = make_box<StorageDims>(grid_detail::non_empty, min, max);
		if(dim + 1 < static_cast<int>(cuts.size())) {
			dissect_box(small_box, cuts, out_dissected, dim + 1);
		} else {
			out_dissected.push_back(small_box);
		}

		start = end;
	}
}

// explicit instantiations for tests (might otherwise be inlined)
template void dissect_box(const box<2>& in_box, const std::vector<std::vector<size_t>>& cuts, box_vector<2>& out_dissected, int dim);
template void dissect_box(const box<3>& in_box, const std::vector<std::vector<size_t>>& cuts, box_vector<3>& out_dissected, int dim);

// Apply dissect_box to all boxes in a range, with a shortcut if no cuts are to be done.
template <typename InputIterator>
void dissect_boxes(const InputIterator first, const InputIterator last, const std::vector<std::vector<size_t>>& cuts,
    box_vector<std::iterator_traits<InputIterator>::value_type::dimensions>& out_dissected) {
	if(!cuts.empty()) {
		for(auto it = first; it != last; ++it) {
			dissect_box(*it, cuts, out_dissected, 0);
		}
	} else {
		out_dissected.insert(out_dissected.end(), first, last);
	}
}

// Collect the sorted, unique list of box start- and end points along a single dimension. These can then be used in dissect_boxes.
template <typename InputIterator>
std::vector<size_t> collect_dissection_lines(const InputIterator first, const InputIterator last, int dim) {
	std::vector<size_t> cuts;
	// allocating 2*N integers might seem wasteful, but this has negligible runtime in the profiler and is already algorithmically optimal at O(N log N)
	cuts.reserve(std::distance(first, last) * 2);
	for(auto it = first; it != last; ++it) {
		cuts.push_back(it->get_min()[dim]);
		cuts.push_back(it->get_max()[dim]);
	}
	std::sort(cuts.begin(), cuts.end());
	cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
	assert(first == last || cuts.size() >= 2);
	return cuts;
}

template <int EffectiveDims, int StorageDims>
void normalize_impl(box_vector<StorageDims>& boxes) {
	static_assert(EffectiveDims <= StorageDims);
	assert(!boxes.empty());

	if constexpr(EffectiveDims == 0) {
		// all 0d boxes are identical
		boxes.resize(1, box<StorageDims>());
	} else if constexpr(EffectiveDims == 1) {
		// merge_connected_boxes will sort and merge - this is already the complete 1d normalization
		boxes.erase(merge_connected_boxes<EffectiveDims>(boxes.begin(), boxes.end()), boxes.end());
		assert(!boxes.empty());
		assert(std::is_sorted(boxes.begin(), boxes.end(), box_coordinate_order()));
	} else {
		// 0. (hopefully) fast path: attempt to merge without dissecting first
		boxes.erase(merge_connected_boxes<EffectiveDims>(boxes.begin(), boxes.end()), boxes.end());
		assert(!boxes.empty());
		if(boxes.size() == 1) return;

		// 1. dissect boxes along the edges of all other boxes (except the last, "fastest" dim) to create the "maximally mergeable set" of boxes for step 2
		std::vector<std::vector<size_t>> cuts(EffectiveDims - 1);
		for(int d = 0; d < EffectiveDims - 1; ++d) {
			cuts[static_cast<size_t>(d)] = collect_dissection_lines(boxes.begin(), boxes.end(), d);
		}

		box_vector<StorageDims> dissected_boxes;
		dissect_boxes(boxes.begin(), boxes.end(), cuts, dissected_boxes);
		boxes = std::move(dissected_boxes);

		// 2. the dissected tiling of boxes only potentially overlaps in the fastest dimension - merge where possible
		boxes.erase(merge_connected_boxes<EffectiveDims>(boxes.begin(), boxes.end()), boxes.end());

		// 3. normalize box order
		std::sort(boxes.begin(), boxes.end(), box_coordinate_order());
	}
}

// Use together with a generic functor to dispatch the EffectiveDims template parameter at runtime
template <int StorageDims, typename F>
decltype(auto) dispatch_effective_dims(int effective_dims, F&& f) {
	assert(effective_dims <= StorageDims);

	// clang-format off
	switch(effective_dims) {
	case 0: if constexpr(StorageDims >= 0) { return f(std::integral_constant<int, 0>()); } [[fallthrough]];
	case 1: if constexpr(StorageDims >= 1) { return f(std::integral_constant<int, 1>()); } [[fallthrough]];
	case 2: if constexpr(StorageDims >= 2) { return f(std::integral_constant<int, 2>()); } [[fallthrough]];
	case 3: if constexpr(StorageDims >= 3) { return f(std::integral_constant<int, 3>()); } [[fallthrough]];
	default: abort(); // unreachable with the explicit instantiations in this file
	}
	// clang-format on
}

// For any set of boxes, find the unique box tiling that covers the same points and is subject to the following constraints:
//   1. the extent of every box is maximized along the last dimension, then along the second-to-last dimension, and so forth.
//   2. no two boxes within the tiling intersect (i.e. cover a common point).
//   3. the tiling contains no empty boxes.
//   4. the normalized sequence is sorted according to box_coordinate_order.
// There is exactly one sequence of boxes for any set of points that fulfills 1-4, meaning that an "==" comparison of normalized tilings would be equivalent
// to an equality comparision of the covered point sets.
template <int Dims>
void normalize(box_vector<Dims>& boxes) {
	boxes.erase(std::remove_if(boxes.begin(), boxes.end(), std::mem_fn(&box<Dims>::empty)), boxes.end());
	if(boxes.size() <= 1) return;

	const auto effective_dims = get_effective_dims(boxes.begin(), boxes.end());
	assert(effective_dims <= Dims);

	dispatch_effective_dims<Dims>(effective_dims, [&](const auto effective_dims) { //
		normalize_impl<effective_dims.value>(boxes);
	});
}

// explicit instantiations for tests (might otherwise be inlined into region::region)
template void normalize(box_vector<0>& boxes);
template void normalize(box_vector<1>& boxes);
template void normalize(box_vector<2>& boxes);
template void normalize(box_vector<3>& boxes);

template <int EffectiveDims, int StorageDims>
region<StorageDims> region_intersection_impl(const region<StorageDims>& lhs, const region<StorageDims>& rhs) {
	static_assert(EffectiveDims <= StorageDims);

	// O(N * M). This can probably be improved for large inputs by dissecting either lhs or rhs by the lines of the other and then performing an interval
	// search similar to how remove_pairwise_covered operates.
	box_vector<StorageDims> intersection;
	for(const auto& left : lhs.get_boxes()) {
		for(const auto& right : rhs.get_boxes()) {
			if(const auto box = grid_detail::box_intersection<EffectiveDims>(left, right); !box.empty()) { intersection.push_back(box); }
		}
	}

	// No dissection step is necessary as the intersection of two normalized tilings is already "maximally mergeable".
	const auto first = intersection.begin();
	auto last = intersection.end();
	last = grid_detail::merge_connected_boxes<EffectiveDims>(first, last);

	// intersected_boxes retains the sorting from lhs, but for Dims > 1, the intersection can shift min-points such that the box_coordinate_order reverses.
	if constexpr(EffectiveDims > 1) {
		std::sort(first, last, box_coordinate_order());
	} else {
		assert(std::is_sorted(first, last, box_coordinate_order()));
	}

	intersection.erase(last, intersection.end());
	return grid_detail::make_region<StorageDims>(grid_detail::normalized, std::move(intersection));
}

// Complete the region_difference operation with an already dissected left-hand side and knowledge of effective dimensionality.
template <int EffectiveDims, int StorageDims>
void apply_region_difference(box_vector<StorageDims>& dissected_left, const region<StorageDims>& rhs) {
	static_assert(EffectiveDims <= StorageDims);

	// O(N * M) remove all dissected boxes from lhs that are fully covered by any box in rhs
	const auto first_left = dissected_left.begin();
	auto last_left = dissected_left.end();
	for(const auto& right : rhs.get_boxes()) {
		for(auto left_it = first_left; left_it != last_left;) {
			if(grid_detail::box_covers<EffectiveDims>(right, *left_it)) {
				*left_it = *--last_left;
			} else {
				++left_it;
			}
		}
	}

	// merge the now non-overlapping boxes
	last_left = grid_detail::merge_connected_boxes<EffectiveDims>(first_left, last_left);
	dissected_left.erase(last_left, dissected_left.end());
}

} // namespace celerity::detail::grid_detail

namespace celerity::detail {

template <int Dims>
region<Dims>::region(const box& single_box) : region(box_vector{single_box}) {} // still need to normalize in case single_box is empty

template <int Dims>
region<Dims>::region(const subrange<Dims>& single_sr) : region(box(single_sr)) {}

template <int Dims>
region<Dims>::region(box_vector&& boxes) : region(grid_detail::normalized, (/* in-place */ grid_detail::normalize(boxes), /* then */ std::move(boxes))) {}

template <int Dims>
region<Dims>::region(grid_detail::normalized_t /* tag */, box_vector&& boxes) : m_boxes(std::move(boxes)) {}

template class region<0>;
template class region<1>;
template class region<2>;
template class region<3>;

template <int Dims>
region<Dims> region_union(const region<Dims>& lhs, const region<Dims>& rhs) {
	// shortcut-evaluate trivial cases
	if(lhs.empty()) return rhs;
	if(rhs.empty()) return lhs;

	box_vector<Dims> box_union;
	box_union.reserve(lhs.get_boxes().size() + rhs.get_boxes().size());
	box_union.insert(box_union.end(), lhs.get_boxes().begin(), lhs.get_boxes().end());
	box_union.insert(box_union.end(), rhs.get_boxes().begin(), rhs.get_boxes().end());
	return region<Dims>(std::move(box_union));
}

template region<0> region_union(const region<0>& lhs, const region<0>& rhs);
template region<1> region_union(const region<1>& lhs, const region<1>& rhs);
template region<2> region_union(const region<2>& lhs, const region<2>& rhs);
template region<3> region_union(const region<3>& lhs, const region<3>& rhs);

template <int Dims>
region<Dims> region_intersection(const region<Dims>& lhs, const region<Dims>& rhs) {
	// shortcut-evaluate trivial cases
	if(lhs.empty() || rhs.empty()) return {};

	const auto effective_dims = std::max(lhs.get_effective_dims(), rhs.get_effective_dims());
	return grid_detail::dispatch_effective_dims<Dims>(effective_dims, [&](const auto effective_dims) { //
		return grid_detail::region_intersection_impl<effective_dims.value>(lhs, rhs);
	});
}

template region<0> region_intersection(const region<0>& lhs, const region<0>& rhs);
template region<1> region_intersection(const region<1>& lhs, const region<1>& rhs);
template region<2> region_intersection(const region<2>& lhs, const region<2>& rhs);
template region<3> region_intersection(const region<3>& lhs, const region<3>& rhs);

template <int Dims>
region<Dims> region_difference(const region<Dims>& lhs, const region<Dims>& rhs) {
	// shortcut-evaluate trivial cases
	if(lhs.empty()) return {};
	if(rhs.empty()) return lhs;

	// the resulting effective_dims can never be greater than the lhs dimension, but the difference operator must still operate on all available dimensions
	// to correctly identify overlapping boxes
	const auto effective_dims = std::max(lhs.get_effective_dims(), rhs.get_effective_dims());
	assert(effective_dims <= Dims);

	// 1. collect dissection lines (in *all* dimensions) from rhs
	std::vector<std::vector<size_t>> cuts(effective_dims);
	for(int d = 0; d < effective_dims; ++d) {
		cuts[static_cast<size_t>(d)] = grid_detail::collect_dissection_lines(rhs.get_boxes().begin(), rhs.get_boxes().end(), d);
	}

	// 2. dissect lhs according to the lines of rhs, so that any overlap between lhs and rhs is turned into an lhs box fully covered by an rhs box
	box_vector<Dims> dissected_left;
	grid_detail::dissect_boxes(lhs.get_boxes().begin(), lhs.get_boxes().end(), cuts, dissected_left);

	grid_detail::dispatch_effective_dims<Dims>(effective_dims, [&](const auto effective_dims) { //
		grid_detail::apply_region_difference<effective_dims.value>(dissected_left, rhs);
	});
	std::sort(dissected_left.begin(), dissected_left.end(), box_coordinate_order());

	return grid_detail::make_region<Dims>(grid_detail::normalized, std::move(dissected_left));
}

template region<0> region_difference(const region<0>& lhs, const region<0>& rhs);
template region<1> region_difference(const region<1>& lhs, const region<1>& rhs);
template region<2> region_difference(const region<2>& lhs, const region<2>& rhs);
template region<3> region_difference(const region<3>& lhs, const region<3>& rhs);

} // namespace celerity::detail