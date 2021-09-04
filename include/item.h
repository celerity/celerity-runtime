#pragma once

#include "ranges.h"

namespace celerity {

template <int Dims>
class item;

namespace detail {

	template <int Dims>
	item<Dims> make_item(cl::sycl::id<Dims> global_id, cl::sycl::range<Dims> global_range) {
		return item<Dims>{global_id, global_range};
	}

} // namespace detail

// We replace sycl::item with celerity::item to correctly expose the cluster global size instead of the chunk size to the user.
template <int Dims>
class item {
  public:
	item() = delete;

	cl::sycl::id<Dims> get_id() const { return id; }

	size_t get_id(int dimension) const { return id[dimension]; }

	operator cl::sycl::id<Dims>() const { return id; } // NOLINT(google-explicit-constructor)

	size_t operator[](int dimension) const { return id[dimension]; }

	cl::sycl::range<Dims> get_range() const { return range; }

	size_t get_range(int dimension) const { return range[dimension]; }

	size_t get_linear_id() const { return detail::get_linear_index(range, id); }

  private:
	template <int D>
	friend item<D> celerity::detail::make_item(cl::sycl::id<D>, cl::sycl::range<D>);

	cl::sycl::id<Dims> id;
	cl::sycl::range<Dims> range;

	explicit item(cl::sycl::id<Dims> id, cl::sycl::range<Dims> range) : id(id), range(range) {}
};

} // namespace celerity