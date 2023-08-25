#pragma once

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>

#include <gch/small_vector.hpp>

#include "ranges.h"
#include "workaround.h"

namespace celerity::detail {

template <int Dims>
class box;

template <int Dims>
class region;

} // namespace celerity::detail

namespace celerity::detail::grid_detail {

struct normalized_t {
} inline constexpr normalized;

struct non_empty_t {
} inline constexpr non_empty;

template <int Dims, typename... Params>
box<Dims> make_box(Params&&... args) {
	return box<Dims>(std::forward<Params>(args)...);
}

template <int Dims, typename... Params>
region<Dims> make_region(Params&&... args) {
	return region<Dims>(std::forward<Params>(args)...);
}

template <typename InputIterator>
int get_min_dimensions(const InputIterator first, const InputIterator last) {
	return std::accumulate(first, last, 0, [](const int min_dims, const auto& box) { return std::max(min_dims, box.get_min_dimensions()); });
}

} // namespace celerity::detail::grid_detail

namespace celerity::detail {

/// An arbitrary-dimensional box described by its minimum and maximum points.
template <int Dims>
class box /* class instead of struct: enforces min <= max invariant */ {
  public:
	static_assert(Dims >= 0);
	static constexpr int dimensions = Dims;

	/// Construct an empty box for Dims > 0, and a unit-sized box for Dims == 0
	box() = default;

	/// Construct a box from two points where `min` must be less or equal to `max` in every dimension.
	/// Empty boxes are normalized to [0,0,0] - [0,0,0], meaning that every box-shaped set of points has a unique representation.
	box(const id<Dims>& min, const id<Dims>& max) {
		bool non_empty = true;
		for(int d = 0; d < Dims; ++d) {
			// Ideally all coordinates would be signed types, but since id and range must be unsigned to conform with SYCL, we trap size_t overflows and
			// incorrect casts from negative integers in user code in this assertion.
			CELERITY_DETAIL_ASSERT_ON_HOST(std::max(min[d], max[d]) < std::numeric_limits<size_t>::max() / 2 && "potential integer overflow detected");
			// Precondition:
			CELERITY_DETAIL_ASSERT_ON_HOST(min[d] <= max[d]);
			non_empty &= min[d] < max[d];
		}
		m_min = non_empty ? min : id<Dims>{};
		m_max = non_empty ? max : id<Dims>{};
	}

	box(const subrange<Dims>& other) : box(other.offset, other.offset + other.range) {
#if CELERITY_DETAIL_ENABLE_DEBUG
		for(int d = 0; d < Dims; ++d) {
			CELERITY_DETAIL_ASSERT_ON_HOST(other.range[d] < std::numeric_limits<size_t>::max() - other.offset[d]);
		}
#endif
	}

	bool empty() const {
		if constexpr(Dims > 0) {
			return m_max[0] == 0; // empty boxes are normalized to [0,0,0] - [0,0,0]
		} else {
			return false; // edge case: min == max, but 0-dimensional boxes are always size 1
		}
	}

	const id<Dims>& get_min() const { return m_min; }
	const id<Dims>& get_max() const { return m_max; }

	const id<Dims>& get_offset() const { return m_min; }
	range<Dims> get_range() const { return range_cast<Dims>(m_max - m_min); }
	subrange<Dims> get_subrange() const { return {get_offset(), get_range()}; }
	operator subrange<Dims>() const { return get_subrange(); }

	/// Counts the number of points covered by the region.
	size_t get_area() const { return get_range().size(); }

	/// Returns the smallest dimensionality that `*this` can be `box_cast` to.
	int get_min_dimensions() const {
		if(empty()) return 1; // edge case: a 0-dimensional box is always non-empty
		for(int dims = Dims; dims > 0; --dims) {
			if(m_max[dims - 1] > 1) { return dims; }
		}
		return 0;
	}

	bool covers(const box& other) const {
		// empty boxes are normalized and thus may not intersect in coordinates
		if(other.empty()) return true;

		for(int d = 0; d < Dims; ++d) {
			if(other.m_min[d] < m_min[d]) return false;
			if(other.m_max[d] > m_max[d]) return false;
		}
		return true;
	}

	friend bool operator==(const box& lhs, const box& rhs) { return lhs.m_min == rhs.m_min && lhs.m_max == rhs.m_max; }
	friend bool operator!=(const box& lhs, const box& rhs) { return !operator==(lhs, rhs); }

  private:
	template <int D, typename... P>
	friend box<D> grid_detail::make_box(P&&... args);

	id<Dims> m_min;
	id<Dims> m_max;

	// fast code path for grid algorithms that does not attempt to normalize empty boxes
	box(grid_detail::non_empty_t /* tag */, const id<Dims>& min, const id<Dims>& max) : m_min(min), m_max(max) {
#if CELERITY_DETAIL_ENABLE_DEBUG
		for(int d = 0; d < Dims; ++d) {
			CELERITY_DETAIL_ASSERT_ON_HOST(min[d] < max[d]);
		}
#endif
	}
};

/// Boxes can be cast between dimensionalities as long as no information is lost (i.e. a cast to a higher dimensionality is always round-trip safe).
template <int DimsOut, int DimsIn>
box<DimsOut> box_cast(const box<DimsIn>& in) {
	CELERITY_DETAIL_ASSERT_ON_HOST(in.get_min_dimensions() <= DimsOut);
	return box<DimsOut>(subrange_cast<DimsOut>(in.get_subrange())); // cast through subrange to fill missing range dimensions with 1s
}

template <int Dims>
box<Dims> bounding_box(const box<Dims>& box1, const box<Dims>& box2) {
	// empty boxes are covered by any other box, but their normalized coordinates should not contribute to the bounding box
	if(box1.empty()) return box2;
	if(box2.empty()) return box1;

	const auto min = id_min(box1.get_min(), box2.get_min());
	const auto max = id_max(box1.get_max(), box2.get_max());
	return box(min, max);
}

template <typename InputIterator>
auto bounding_box(InputIterator first, const InputIterator last) {
	using box_type = typename std::iterator_traits<InputIterator>::value_type;
	if(first == last) {
		assert(box_type::dimensions > 0); // box<0> can never be empty
		return box_type();
	}

	const auto init = *first;
	return std::accumulate(++first, last, init, bounding_box<box_type::dimensions>);
}

template <typename Range>
auto bounding_box(const Range& range) {
	using std::begin, std::end;
	return bounding_box(begin(range), end(range));
}

template <int Dims>
box<Dims> box_intersection(const box<Dims>& box1, const box<Dims>& box2) {
	const auto min = id_max(box1.get_min(), box2.get_min());
	const auto max = id_min(box1.get_max(), box2.get_max());
	for(int d = 0; d < Dims; ++d) {
		if(min[d] >= max[d]) return {};
	}
	return {min, max};
}

/// Comparison operator (similar to std::less) that orders boxes by their minimum, then their maximum, both starting with the first ("slowest") dimension.
/// This ordering is somewhat arbitrary but allows equality comparisons between ordered sequences of boxes (i.e., regions)
struct box_coordinate_order {
	template <int Dims>
	bool operator()(const box<Dims>& lhs, const box<Dims>& rhs) const {
		for(int d = 0; d < Dims; ++d) {
			if(lhs.get_min()[d] < rhs.get_min()[d]) return true;
			if(lhs.get_min()[d] > rhs.get_min()[d]) return false;
		}
		for(int d = 0; d < Dims; ++d) {
			if(lhs.get_max()[d] < rhs.get_max()[d]) return true;
			if(lhs.get_max()[d] > rhs.get_max()[d]) return false;
		}
		return false;
	}
};

template <int Dims>
using box_vector = gch::small_vector<box<Dims>>;

template <int DimsOut, int DimsIn>
box_vector<DimsOut> boxes_cast(const box_vector<DimsIn>& in) {
	assert(grid_detail::get_min_dimensions(in.begin(), in.end()) <= DimsOut);
	box_vector<DimsOut> out(in.size(), box<DimsOut>());
	std::transform(in.begin(), in.end(), out.begin(), box_cast<DimsOut, DimsIn>);
	return out;
}

/// An arbitrary-dimensional set of points described by a normalized tiling of boxes.
template <int Dims>
class region {
  public:
	constexpr static int dimensions = Dims;
	using box = detail::box<Dims>;
	using box_vector = detail::box_vector<Dims>;

	region() = default;
	region(const box& single_box);
	region(const subrange<Dims>& single_sr);

	/// Constructs a region by normalizing an arbitrary, potentially-overlapping tiling of boxes.
	explicit region(box_vector&& boxes);

	const box_vector& get_boxes() const& { return m_boxes; }

	box_vector into_boxes() && { return std::move(m_boxes); }

	bool empty() const { return m_boxes.empty(); }

	/// Counts the number of points covered by the region.
	size_t get_area() const {
		return std::accumulate(m_boxes.begin(), m_boxes.end(), size_t{0}, [](const size_t area, const box& box) { return area + box.get_area(); });
	}

	/// Returns the smallest dimensionality that `*this` can be `region_cast` to.
	int get_min_dimensions() const { return grid_detail::get_min_dimensions(m_boxes.begin(), m_boxes.end()); }

	friend bool operator==(const region& lhs, const region& rhs) { return lhs.m_boxes == rhs.m_boxes; }
	friend bool operator!=(const region& lhs, const region& rhs) { return !(lhs == rhs); }

  private:
	template <int D, typename... P>
	friend region<D> grid_detail::make_region(P&&... args);

	box_vector m_boxes;

	region(grid_detail::normalized_t, box_vector&& boxes);
};

} // namespace celerity::detail

namespace celerity::detail::grid_detail {

// forward-declaration for tests (explicitly instantiated)
template <int StorageDims>
void dissect_box(const box<StorageDims>& in_box, const std::vector<std::vector<size_t>>& cuts, box_vector<StorageDims>& out_dissected, int dim);

// forward-declaration for tests (explicitly instantiated)
template <int MergeDim, int EffectiveDims, typename BidirectionalIterator>
BidirectionalIterator merge_connected_boxes_along_dim(const BidirectionalIterator first, const BidirectionalIterator last);

// forward-declaration for tests (explicitly instantiated)
template <int Dims>
void normalize(box_vector<Dims>& boxes);

// rvalue shortcut for normalize(lvalue)
template <int Dims>
box_vector<Dims>&& normalize(box_vector<Dims>&& boxes) {
	normalize(boxes);
	return std::move(boxes);
}

} // namespace celerity::detail::grid_detail

namespace celerity::detail {

template <int DimsOut, int DimsIn>
region<DimsOut> region_cast(const region<DimsIn>& in) {
	assert(in.get_min_dimensions() <= DimsOut);
	// a normalized region will remain normalized after the cast
	return grid_detail::make_region<DimsOut>(grid_detail::normalized, boxes_cast<DimsOut>(in.get_boxes()));
}

template <int Dims>
box<Dims> bounding_box(const region<Dims>& region) {
	return bounding_box(region.get_boxes().begin(), region.get_boxes().end());
}

template <int Dims>
region<Dims> region_union(const region<Dims>& lhs, const region<Dims>& rhs);

template <int Dims>
region<Dims> region_union(const region<Dims>& lhs, const box<Dims>& rhs) {
	return region_union(lhs, region(rhs));
}

template <int Dims>
region<Dims> region_union(const box<Dims>& lhs, const region<Dims>& rhs) {
	return region_union(region(lhs), rhs);
}

template <int Dims>
region<Dims> region_union(const box<Dims>& lhs, const box<Dims>& rhs) {
	return region(box_vector<Dims>{lhs, rhs});
}

template <int Dims>
region<Dims> region_intersection(const region<Dims>& lhs, const region<Dims>& rhs);

template <int Dims>
region<Dims> region_intersection(const region<Dims>& lhs, const box<Dims>& rhs) {
	return region_intersection(lhs, region(rhs));
}

template <int Dims>
region<Dims> region_intersection(const box<Dims>& lhs, const region<Dims>& rhs) {
	return region_intersection(region(lhs), rhs);
}

template <int Dims>
region<Dims> region_difference(const region<Dims>& lhs, const region<Dims>& rhs);

template <int Dims>
region<Dims> region_difference(const region<Dims>& lhs, const box<Dims>& rhs) {
	return region_difference(lhs, region(rhs));
}

template <int Dims>
region<Dims> region_difference(const box<Dims>& lhs, const region<Dims>& rhs) {
	return region_difference(region(lhs), rhs);
}

template <int Dims>
region<Dims> region_difference(const box<Dims>& lhs, const box<Dims>& rhs) {
	return region_difference(region(lhs), region(rhs));
}

} // namespace celerity::detail
