#pragma once

#include "grid.h"
#include "ranges.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <fmt/format.h>


namespace celerity {
namespace detail {

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_chunk_only = std::is_invocable_r_v<region<BufferDims>, const Functor&, const celerity::chunk<KernelDims>&>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_chunk_and_global_size =
	    std::is_invocable_r_v<region<BufferDims>, const Functor&, const celerity::chunk<KernelDims>&, const range<BufferDims>&>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_kernel = is_range_mapper_invocable_for_chunk_only<Functor, BufferDims, KernelDims> //
	                                                      || is_range_mapper_invocable_for_chunk_and_global_size<Functor, BufferDims, KernelDims>;

	template <typename Functor, int BufferDims>
	constexpr bool is_range_mapper_invocable = is_range_mapper_invocable_for_kernel<Functor, BufferDims, 0>    //
	                                           || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 1> //
	                                           || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 2> //
	                                           || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 3>;

	[[noreturn]] inline void throw_invalid_range_mapper_args(int expect_kernel_dims, int expect_buffer_dims) {
		throw std::runtime_error(fmt::format("Invalid range mapper dimensionality: {0}-dimensional kernel submitted with a requirement whose range mapper "
		                                     "is neither invocable for chunk<{0}> nor (chunk<{0}>, range<{1}>) to produce subrange<{1}>",
		    expect_kernel_dims, expect_buffer_dims));
	}

	[[noreturn]] inline void throw_invalid_range_mapper_result(int expect_sr_dims, int actual_sr_dims, int kernel_dims) {
		throw std::runtime_error(fmt::format("Range mapper produces subrange of wrong dimensionality: Expecting subrange<{}>, got subrange<{}> for chunk<{}>",
		    expect_sr_dims, actual_sr_dims, kernel_dims));
	}

	template <typename Functor, int KernelDims, int BufferDims>
	region<BufferDims> invoke_range_mapper(Functor&& fn, const celerity::chunk<KernelDims>& chunk, const range<BufferDims>& buffer_size) {
		static_assert(KernelDims >= 0 && KernelDims <= 3 && BufferDims >= 0 && BufferDims <= 3);
		if constexpr(is_range_mapper_invocable_for_chunk_and_global_size<Functor, BufferDims, KernelDims>) {
			return std::forward<Functor>(fn)(chunk, buffer_size);
		} else if constexpr(is_range_mapper_invocable_for_chunk_only<Functor, BufferDims, KernelDims>) {
			return std::forward<Functor>(fn)(chunk);
		} else {
			throw_invalid_range_mapper_args(KernelDims, BufferDims);
		}
	}

	template <int BufferDims>
	region<BufferDims> clamp_region_to_buffer_size(const region<BufferDims>& r, const range<BufferDims>& buffer_size) {
		return region_intersection(r, box<BufferDims>::full_range(buffer_size));
	}

	class range_mapper_base {
	  public:
		range_mapper_base() = default;

		range_mapper_base(const range_mapper_base& other) = delete;
		range_mapper_base(range_mapper_base&& other) = delete;
		range_mapper_base& operator=(const range_mapper_base& other) = delete;
		range_mapper_base& operator=(range_mapper_base&& other) = delete;

		virtual int get_buffer_dimensions() const = 0;

		virtual region<1> map_1(const chunk<0>& chnk) const = 0;
		virtual region<1> map_1(const chunk<1>& chnk) const = 0;
		virtual region<1> map_1(const chunk<2>& chnk) const = 0;
		virtual region<1> map_1(const chunk<3>& chnk) const = 0;
		virtual region<2> map_2(const chunk<0>& chnk) const = 0;
		virtual region<2> map_2(const chunk<1>& chnk) const = 0;
		virtual region<2> map_2(const chunk<2>& chnk) const = 0;
		virtual region<2> map_2(const chunk<3>& chnk) const = 0;
		virtual region<3> map_3(const chunk<0>& chnk) const = 0;
		virtual region<3> map_3(const chunk<1>& chnk) const = 0;
		virtual region<3> map_3(const chunk<2>& chnk) const = 0;
		virtual region<3> map_3(const chunk<3>& chnk) const = 0;

		virtual ~range_mapper_base() = default;
	};

	template <int BufferDims, typename Functor>
	class range_mapper final : public range_mapper_base {
	  public:
		range_mapper(Functor rmfn, range<BufferDims> buffer_size) : m_rmfn(rmfn), m_buffer_size(buffer_size) {}

		int get_buffer_dimensions() const override { return BufferDims; }

		region<1> map_1(const chunk<0>& chnk) const override { return map<1>(chnk); }
		region<1> map_1(const chunk<1>& chnk) const override { return map<1>(chnk); }
		region<1> map_1(const chunk<2>& chnk) const override { return map<1>(chnk); }
		region<1> map_1(const chunk<3>& chnk) const override { return map<1>(chnk); }
		region<2> map_2(const chunk<0>& chnk) const override { return map<2>(chnk); }
		region<2> map_2(const chunk<1>& chnk) const override { return map<2>(chnk); }
		region<2> map_2(const chunk<2>& chnk) const override { return map<2>(chnk); }
		region<2> map_2(const chunk<3>& chnk) const override { return map<2>(chnk); }
		region<3> map_3(const chunk<0>& chnk) const override { return map<3>(chnk); }
		region<3> map_3(const chunk<1>& chnk) const override { return map<3>(chnk); }
		region<3> map_3(const chunk<2>& chnk) const override { return map<3>(chnk); }
		region<3> map_3(const chunk<3>& chnk) const override { return map<3>(chnk); }

	  private:
		Functor m_rmfn;
		range<BufferDims> m_buffer_size;

		template <int OtherBufferDims, int KernelDims>
		region<OtherBufferDims> map(const chunk<KernelDims>& chnk) const {
			if constexpr(OtherBufferDims == BufferDims) {
				const auto r = invoke_range_mapper(m_rmfn, chnk, m_buffer_size);
				return clamp_region_to_buffer_size(r, m_buffer_size);
			} else {
				throw_invalid_range_mapper_result(OtherBufferDims, BufferDims, KernelDims);
			}
		}
	};

	struct range_mapper_testspy;

} // namespace detail


// --------------------------- Convenience range mappers ---------------------------

/// Optional parameter to the constructor of `access::neighborhood` to specify in what shape the accessed region should extend from the work item.
enum class neighborhood_shape {
	along_axes,   ///< The neighborhood extends along each axis separately, but not diagonally (in 2D, into a "+" shape).
	bounding_box, ///< The neighborhood extends along in all dimensions simultaneously to produce a single bounding box.
};

namespace access {

	struct one_to_one {
		template <int Dims>
		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			return chnk;
		}
	};

	template <int BufferDims>
	struct fixed {
		fixed(const subrange<BufferDims>& sr) : m_sr(sr) {}

		template <int KernelDims>
		subrange<BufferDims> operator()(const chunk<KernelDims>& /* chnk */) const {
			return m_sr;
		}

	  private:
		subrange<BufferDims> m_sr;
	};

	template <int Dims>
	struct slice {
		explicit slice(const size_t dim_idx) : m_dim_idx(dim_idx) { assert(dim_idx < Dims && "Invalid slice dimension index (starts at 0)"); }

		subrange<Dims> operator()(const chunk<Dims>& chnk, const range<Dims>& buffer_size) const {
			subrange<Dims> result = chnk;
			result.offset[m_dim_idx] = 0;
			result.range[m_dim_idx] = buffer_size[m_dim_idx];
			return result;
		}

	  private:
		size_t m_dim_idx;
	};

	struct all {
		template <int KernelDims, int BufferDims>
		subrange<BufferDims> operator()(const chunk<KernelDims>& /* chnk */, const range<BufferDims>& buffer_size) const {
			return {{}, buffer_size};
		}
	};

	/// Declares a buffer access that extends from the current work item by a symmetric boundary offset, either in all directions in the shape of a bounding
	/// box (default), or along each axis separately (without "diagonal" boundary elements). Buffer and kernel dimensions must both match the `Dims` parameter.
	///
	/// This is typically used in stencil applications. For bounding-box neighborhoods,
	/// - `neighborhood({1})` declares the read for a 1D 3-point stencil,
	/// - `neighborhood({1, 1})` for a 2D 9-point stencil, and
	/// - `neighborhood({1, 1, 1})` for a 3D 27-point stencil.
	///
	/// For neighborhoods defined along axes only,
	/// - `neighborhood({1}, neighborhood_shape::along_axes)` declares the read for a 1D 3-point stencil,
	/// - `neighborhood({1, 1}, neighborhood_shape::along_axes)` for a 2D 5-point stencil, and
	/// - `neighborhood({1, 1, 1}, neighborhood_shape::along_axes)` for a 3D 7-point stencil.
	///
	/// For reads, `neighborhood_shape::bounding_box` is functionally correct whenever `neighborhood_shape::along_axes` is, but will lead to unnecessary copies
	/// and transfers between diagonal neighbors in 2D-split work assignments when the application does not actually read from those buffer elements.
	template <int Dims>
	struct neighborhood {
		explicit neighborhood(const range<Dims>& extent, const neighborhood_shape shape = neighborhood_shape::bounding_box)
		    : m_extent(extent), m_shape(shape) {}

		[[deprecated("Use the neighborhood({a, b} [, shape]) instead of neighborhood(a, b)")]]
		explicit neighborhood(const size_t dim0, const size_t dim1)
		    requires(Dims == 2)
		    : neighborhood({dim0, dim1}) {}

		[[deprecated("Use the neighborhood({a, b, c} [, shape]) instead of neighborhood(a, b, c)")]]
		explicit neighborhood(const size_t dim0, const size_t dim1, const size_t dim2)
		    requires(Dims == 3)
		    : neighborhood({dim0, dim1, dim2}) {}

		detail::region<Dims> operator()(const chunk<Dims>& chnk) const {
			const detail::box interior(subrange(chnk.offset, chnk.range));
			if(m_shape == neighborhood_shape::along_axes) {
				detail::region_builder<Dims> boxes;
				boxes.add(interior);
				for(int d = 0; d < Dims; ++d) {
					boxes.add(extend_axis(interior, d));
				}
				return std::move(boxes).into_region();
			} else {
				auto bounding_box = interior;
				for(int d = 0; d < Dims; ++d) {
					bounding_box = extend_axis(bounding_box, d);
				}
				return bounding_box;
			}
		}

	  private:
		friend struct celerity::detail::range_mapper_testspy;

		range<Dims> m_extent;
		neighborhood_shape m_shape;

		inline detail::box<Dims> extend_axis(const detail::box<Dims>& box, const int d) const {
			auto min = box.get_min();
			auto max = box.get_max();
			min[d] -= std::min(m_extent[d], min[d]);
			max[d] += std::min(m_extent[d], std::numeric_limits<size_t>::max() - max[d]);
			return detail::box(min, max);
		}
	};

	neighborhood(size_t) -> neighborhood<1>;
	neighborhood(size_t, size_t) -> neighborhood<2>;
	neighborhood(size_t, size_t, size_t) -> neighborhood<3>;

	// Explicit CTAD guides allow deducing Dims from `neighborhood{{1, 1}}`.
	neighborhood(range<1>) -> neighborhood<1>;
	neighborhood(range<2>) -> neighborhood<2>;
	neighborhood(range<3>) -> neighborhood<3>;
	neighborhood(range<1>, neighborhood_shape) -> neighborhood<1>;
	neighborhood(range<2>, neighborhood_shape) -> neighborhood<2>;
	neighborhood(range<3>, neighborhood_shape) -> neighborhood<3>;

} // namespace access

namespace experimental::access {

	/**
	 * For a 1D kernel, splits an nD-buffer evenly along its slowest dimension.
	 *
	 * This range mapper is unique in the sense that the chunk parameter (i.e. the iteration space) is unrelated to the buffer indices it maps to.
	 * It is designed to distribute a buffer in contiguous portions between nodes for collective host tasks, allowing each node to output its portion in
	 * I/O operations. See `accessor::get_allocation_window` on how to access the resulting host memory.
	 */
	template <int BufferDims>
	class even_split {
		static_assert(BufferDims > 0);

	  public:
		even_split() = default;
		explicit even_split(const range<BufferDims>& granularity) : m_granularity(granularity) {}

		subrange<BufferDims> operator()(const chunk<1>& chunk, const range<BufferDims>& buffer_size) const {
			if(chunk.global_size[0] == 0) { return {}; }

			// Equal splitting has edge cases when buffer_size is not a multiple of global_size * granularity. Splitting is performed in a manner that
			// distributes the remainder as equally as possible while adhering to granularity. In case buffer_size is not even a multiple of granularity,
			// only last chunk should be oddly sized so that only one node needs to deal with misaligned buffers.

			// 1. Each slice has at least buffer_size / global_size items, rounded down to the nearest multiple of the granularity.
			// 2. The first chunks in the range receive one additional granularity-sized block each to distribute most of the remainder
			// 3. The last chunk additionally receives the not-granularity-sized part of the remainder, if any.

			auto dim0_step = buffer_size[0] / (chunk.global_size[0] * m_granularity[0]) * m_granularity[0];
			auto dim0_remainder = buffer_size[0] - chunk.global_size[0] * dim0_step;
			auto dim0_range_in_this_chunk = chunk.range[0] * dim0_step;
			auto sum_dim0_remainder_in_prev_chunks = std::min(dim0_remainder / m_granularity[0] * m_granularity[0], chunk.offset[0] * m_granularity[0]);
			if(dim0_remainder > sum_dim0_remainder_in_prev_chunks) {
				dim0_range_in_this_chunk +=
				    std::min(chunk.range[0], (dim0_remainder - sum_dim0_remainder_in_prev_chunks) / m_granularity[0]) * m_granularity[0];
				if(chunk.offset[0] + chunk.range[0] == chunk.global_size[0]) { dim0_range_in_this_chunk += dim0_remainder % m_granularity[0]; }
			}
			auto dim0_offset_in_this_chunk = chunk.offset[0] * dim0_step + sum_dim0_remainder_in_prev_chunks;

			subrange<BufferDims> sr;
			sr.offset[0] = dim0_offset_in_this_chunk;
			sr.range = buffer_size;
			sr.range[0] = dim0_range_in_this_chunk;

			return sr;
		}

	  private:
		range<BufferDims> m_granularity = detail::range_cast<BufferDims>(range<3>(1, 1, 1));
	};

} // namespace experimental::access

} // namespace celerity
