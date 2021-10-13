#pragma once

#include "ranges.h"

namespace celerity {

template <int Dims>
class item;

namespace detail {

	template <int Dims>
	inline item<Dims> make_item(cl::sycl::id<Dims> absolute_global_id, cl::sycl::id<Dims> global_offset, cl::sycl::range<Dims> global_range) {
		return item<Dims>{absolute_global_id, global_offset, global_range};
	}

} // namespace detail

// We replace sycl::item with celerity::item to correctly expose the cluster global size instead of the chunk size to the user.
template <int Dims>
class item {
  public:
	item() = delete;

	friend bool operator==(const item& lhs, const item& rhs) {
		return lhs.absolute_global_id == rhs.absolute_global_id && lhs.global_offset == rhs.global_offset && lhs.global_range == rhs.global_range;
	}

	friend bool operator!=(const item& lhs, const item& rhs) { return !(lhs == rhs); }

	cl::sycl::id<Dims> get_id() const { return absolute_global_id; }

	size_t get_id(int dimension) const { return absolute_global_id[dimension]; }

	operator cl::sycl::id<Dims>() const { return absolute_global_id; } // NOLINT(google-explicit-constructor)

	size_t operator[](int dimension) const { return absolute_global_id[dimension]; }

	cl::sycl::range<Dims> get_range() const { return global_range; }

	size_t get_range(int dimension) const { return global_range[dimension]; }

	size_t get_linear_id() const { return detail::get_linear_index(global_range, absolute_global_id - global_offset); }

	cl::sycl::id<Dims> get_offset() const { return global_offset; }

  private:
	template <int D>
	friend item<D> celerity::detail::make_item(cl::sycl::id<D>, cl::sycl::id<D>, cl::sycl::range<D>);

	cl::sycl::id<Dims> absolute_global_id;
	cl::sycl::id<Dims> global_offset;
	cl::sycl::range<Dims> global_range;

	explicit item(cl::sycl::id<Dims> absolute_global_id, cl::sycl::id<Dims> global_offset, cl::sycl::range<Dims> global_range)
	    : absolute_global_id(absolute_global_id), global_offset(global_offset), global_range(global_range) {}
};

} // namespace celerity