#pragma once

#include <stdexcept>
#include <type_traits>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "ranges.h"

namespace celerity {

namespace detail {


	template <int KernelDims, int BufferDims>
	using range_mapper_fn = std::function<subrange<BufferDims>(chunk<KernelDims> chnk, cl::sycl::range<BufferDims> buffer_size)>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_kernel =
	    std::is_invocable_v<Functor&&, const celerity::chunk<KernelDims>&,
	        const cl::sycl::range<BufferDims>&> || std::is_invocable_v<Functor&&, const celerity::chunk<KernelDims>&>;

	template <typename Functor, int BufferDims>
	constexpr bool is_range_mapper_invocable =
	    is_range_mapper_invocable_for_kernel<Functor, BufferDims,
	        1> || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 2> || is_range_mapper_invocable_for_kernel<Functor, BufferDims, 3>;

	template <int KernelDims, int BufferDims, typename Functor>
	subrange<BufferDims> invoke_range_mapper_for_kernel_dims(
	    Functor&& fn, const celerity::chunk<KernelDims>& chunk, const cl::sycl::range<BufferDims>& buffer_size) {
		static_assert(KernelDims >= 1 && KernelDims <= 3 && BufferDims >= 1 && BufferDims <= 3);
		if constexpr(std::is_invocable_v<Functor&&, const celerity::chunk<KernelDims>&, const cl::sycl::range<BufferDims>&>) {
			return std::forward<Functor>(fn)(chunk, buffer_size);
		} else if constexpr(std::is_invocable_v<Functor&&, const celerity::chunk<KernelDims>&>) {
			return std::forward<Functor>(fn)(chunk);
		} else {
			throw std::runtime_error(fmt::format("Invalid range mapper dimensionality: {0}-dimensional kernel submitted with a requirement whose range mapper "
			                                     "is neither invocable for chunk<{0}> nor (chunk<{0}>, range<{1}>)",
			    KernelDims, BufferDims));
		}
	}

	template <int BufferDims, typename Functor>
	subrange<BufferDims> invoke_range_mapper(int kernel_dims, Functor&& fn, const celerity::chunk<3>& chunk, const cl::sycl::range<BufferDims>& buffer_size) {
		static_assert(is_range_mapper_invocable<Functor, BufferDims>);
		switch(kernel_dims) {
		case 0:
			[[fallthrough]]; // cl::sycl::range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
			                 // anyway, we require range mappers to take at least 1d subranges
		case 1: return invoke_range_mapper_for_kernel_dims<1>(std::forward<Functor>(fn), chunk_cast<1>(chunk), buffer_size);
		case 2: return invoke_range_mapper_for_kernel_dims<2>(std::forward<Functor>(fn), chunk_cast<2>(chunk), buffer_size);
		case 3: return invoke_range_mapper_for_kernel_dims<3>(std::forward<Functor>(fn), chunk_cast<3>(chunk), buffer_size);
		default: assert(!"Unreachable"); return {};
		}
	}

	template <int KernelDims, int BufferDims, typename Functor>
	auto bind_range_mapper(Functor fn) {
		return [fn](const chunk<KernelDims>& chunk, const cl::sycl::range<BufferDims>& buffer_size) {
			return invoke_range_mapper_for_kernel_dims<KernelDims>(fn, chunk, buffer_size);
		};
	}

	class range_mapper_base {
	  public:
		range_mapper_base(cl::sycl::access::mode am) : access_mode(am) {}
		range_mapper_base(const range_mapper_base& other) = delete;
		range_mapper_base(range_mapper_base&& other) = delete;

		cl::sycl::access::mode get_access_mode() const { return access_mode; }

		virtual int get_kernel_dimensions() const = 0;
		virtual int get_buffer_dimensions() const = 0;

		virtual subrange<1> map_1(chunk<1> chnk) const { throw create_dimension_mismatch_error(1, 1); }
		virtual subrange<1> map_1(chunk<2> chnk) const { throw create_dimension_mismatch_error(2, 1); }
		virtual subrange<1> map_1(chunk<3> chnk) const { throw create_dimension_mismatch_error(3, 1); }
		virtual subrange<2> map_2(chunk<1> chnk) const { throw create_dimension_mismatch_error(1, 2); }
		virtual subrange<2> map_2(chunk<2> chnk) const { throw create_dimension_mismatch_error(2, 2); }
		virtual subrange<2> map_2(chunk<3> chnk) const { throw create_dimension_mismatch_error(3, 2); }
		virtual subrange<3> map_3(chunk<1> chnk) const { throw create_dimension_mismatch_error(1, 3); }
		virtual subrange<3> map_3(chunk<2> chnk) const { throw create_dimension_mismatch_error(2, 3); }
		virtual subrange<3> map_3(chunk<3> chnk) const { throw create_dimension_mismatch_error(3, 3); }

		virtual ~range_mapper_base() = default;

	  private:
		cl::sycl::access::mode access_mode;

		std::runtime_error create_dimension_mismatch_error(int wrong_kernel_dims, int buffer_dims) const {
			const int kernel_dims = get_kernel_dimensions();
			return std::runtime_error(fmt::format("Range mapper maps chunk<{}> -> subrange<{}>, but should map chunk<{}> -> subrange<{}>.", wrong_kernel_dims,
			    buffer_dims, kernel_dims, buffer_dims));
		}
	};

	template <int BufferDims>
	subrange<BufferDims> clamp_subrange_to_buffer_size(subrange<BufferDims> sr, cl::sycl::range<BufferDims> buffer_size) {
		auto end = sr.offset + sr.range;
		if(BufferDims > 0 && end[0] > buffer_size[0]) { sr.range[0] = sr.offset[0] <= buffer_size[0] ? buffer_size[0] - sr.offset[0] : 0; }
		if(BufferDims > 1 && end[1] > buffer_size[1]) { sr.range[1] = sr.offset[1] <= buffer_size[1] ? buffer_size[1] - sr.offset[1] : 0; }
		if(BufferDims > 2 && end[2] > buffer_size[2]) { sr.range[2] = sr.offset[2] <= buffer_size[2] ? buffer_size[2] - sr.offset[2] : 0; }
		return sr;
	}

	template <int KernelDims, int BufferDims>
	class range_mapper : public range_mapper_base {
	  public:
		template <typename RangeMapperFn>
		range_mapper(RangeMapperFn fn, cl::sycl::access::mode am, cl::sycl::range<BufferDims> buffer_size)
		    : range_mapper_base(am), rmfn(bind_range_mapper<KernelDims, BufferDims>(fn)), buffer_size(buffer_size) {}

		int get_kernel_dimensions() const override { return KernelDims; }
		int get_buffer_dimensions() const override { return BufferDims; }

		subrange<1> map_1(chunk<KernelDims> chnk) const override { return map_1_impl(chnk); }
		subrange<2> map_2(chunk<KernelDims> chnk) const override { return map_2_impl(chnk); }
		subrange<3> map_3(chunk<KernelDims> chnk) const override { return map_3_impl(chnk); }

	  private:
		range_mapper_fn<KernelDims, BufferDims> rmfn;
		cl::sycl::range<BufferDims> buffer_size;

		template <int D = BufferDims>
		typename std::enable_if<D == 1, subrange<1>>::type map_1_impl(chunk<KernelDims> chnk) const {
			return clamp_subrange_to_buffer_size(rmfn(chnk, buffer_size), buffer_size);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 1, subrange<1>>::type map_1_impl(chunk<KernelDims> chnk) const {
			return range_mapper_base::map_1(chnk);
		}

		template <int D = BufferDims>
		typename std::enable_if<D == 2, subrange<2>>::type map_2_impl(chunk<KernelDims> chnk) const {
			return clamp_subrange_to_buffer_size(rmfn(chnk, buffer_size), buffer_size);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 2, subrange<2>>::type map_2_impl(chunk<KernelDims> chnk) const {
			return range_mapper_base::map_2(chnk);
		}

		template <int D = BufferDims>
		typename std::enable_if<D == 3, subrange<3>>::type map_3_impl(chunk<KernelDims> chnk) const {
			return clamp_subrange_to_buffer_size(rmfn(chnk, buffer_size), buffer_size);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 3, subrange<3>>::type map_3_impl(chunk<KernelDims> chnk) const {
			return range_mapper_base::map_3(chnk);
		}
	};

} // namespace detail


// --------------------------- Convenience range mappers ---------------------------

namespace access {

	template <int Dims>
	struct one_to_one {
		subrange<Dims> operator()(chunk<Dims> chnk) const { return chnk; }
	};

	template <int KernelDims, int BufferDims = KernelDims>
	struct fixed {
		fixed(subrange<BufferDims> sr) : sr(sr) {}

		subrange<BufferDims> operator()(chunk<KernelDims>) const { return sr; }

	  private:
		subrange<BufferDims> sr;
	};

	template <int Dims>
	struct slice {
		slice(size_t dim_idx) : dim_idx(dim_idx) { assert(dim_idx < Dims && "Invalid slice dimension index (starts at 0)"); }

		subrange<Dims> operator()(chunk<Dims> chnk) const {
			subrange<Dims> result = chnk;
			result.offset[dim_idx] = 0;
			// Since we don't know the range of the buffer, we just set it way too high and let it be clamped to the correct range
			result.range[dim_idx] = std::numeric_limits<size_t>::max();
			return result;
		}

	  private:
		size_t dim_idx;
	};

	template <int KernelDims, int BufferDims = KernelDims>
	struct all {
		subrange<BufferDims> operator()(chunk<KernelDims>, cl::sycl::range<BufferDims> buffer_size) const { return {{}, buffer_size}; }
	};

	template <int Dims>
	struct neighborhood {
		neighborhood(size_t dim0) : dim0(dim0), dim1(0), dim2(0) {}

		template <int D = Dims, std::enable_if_t<D >= 2, void*>...>
		neighborhood(size_t dim0, size_t dim1) : dim0(dim0), dim1(dim1), dim2(0) {}

		template <int D = Dims, std::enable_if_t<D == 3, void*>...>
		neighborhood(size_t dim0, size_t dim1, size_t dim2) : dim0(dim0), dim1(dim1), dim2(dim2) {}

		subrange<Dims> operator()(chunk<Dims> chnk) const {
			subrange<3> result = {celerity::detail::id_cast<3>(chnk.offset), celerity::detail::range_cast<3>(chnk.range)};
			const cl::sycl::id<3> delta = {dim0 < result.offset[0] ? dim0 : result.offset[0], dim1 < result.offset[1] ? dim1 : result.offset[1],
			    dim2 < result.offset[2] ? dim2 : result.offset[2]};
			result.offset -= delta;
			result.range += cl::sycl::range<3>{dim0 + delta[0], dim1 + delta[1], dim2 + delta[2]};
			return detail::subrange_cast<Dims>(result);
		}

	  private:
		size_t dim0, dim1, dim2;
	};

} // namespace access

namespace experimental::access {

	/**
	 * For a 1D kernel, splits an nD-buffer evenly along its slowest dimension.
	 *
	 * This range mapper is unique in the sense that the chunk parameter (i.e. the iteration space) is unrelated to the buffer indices it maps to.
	 * It is designed to distribute a buffer in contiguous portions between nodes for collective host tasks, allowing each node to output its portion in
	 * I/O operations. See `accessor::get_host_memory` on how to access the resulting host memory.
	 */
	template <int BufferDims>
	class even_split {
		static_assert(BufferDims > 0);

	  public:
		even_split() = default;
		explicit even_split(cl::sycl::range<BufferDims> granularity) : granularity(granularity) {}

		subrange<BufferDims> operator()(chunk<1> chunk, cl::sycl::range<BufferDims> buffer_size) const {
			if(chunk.global_size[0] == 0) { return {}; }

			// Equal splitting has edge cases when buffer_size is not a multiple of global_size * granularity. Splitting is performed in a manner that
			// distributes the remainder as equally as possible while adhering to granularity. In case buffer_size is not even a multiple of granularity,
			// only last chunk should be oddly sized so that only one node needs to deal with misaligned buffers.

			// 1. Each slice has at least buffer_size / global_size items, rounded down to the nearest multiple of the granularity.
			// 2. The first chunks in the range receive one additional granularity-sized block each to distribute most of the remainder
			// 3. The last chunk additionally receives the not-granularity-sized part of the remainder, if any.

			auto dim0_step = buffer_size[0] / (chunk.global_size[0] * granularity[0]) * granularity[0];
			auto dim0_remainder = buffer_size[0] - chunk.global_size[0] * dim0_step;
			auto dim0_range_in_this_chunk = chunk.range[0] * dim0_step;
			auto sum_dim0_remainder_in_prev_chunks = std::min(dim0_remainder / granularity[0] * granularity[0], chunk.offset[0] * granularity[0]);
			if(dim0_remainder > sum_dim0_remainder_in_prev_chunks) {
				dim0_range_in_this_chunk += std::min(chunk.range[0], (dim0_remainder - sum_dim0_remainder_in_prev_chunks) / granularity[0]) * granularity[0];
				if(chunk.offset[0] + chunk.range[0] == chunk.global_size[0]) { dim0_range_in_this_chunk += dim0_remainder % granularity[0]; }
			}
			auto dim0_offset_in_this_chunk = chunk.offset[0] * dim0_step + sum_dim0_remainder_in_prev_chunks;

			subrange<BufferDims> sr;
			sr.offset[0] = dim0_offset_in_this_chunk;
			sr.range = buffer_size;
			sr.range[0] = dim0_range_in_this_chunk;

			return sr;
		}

	  private:
		cl::sycl::range<BufferDims> granularity = detail::range_cast<BufferDims>(cl::sycl::range<3>(1, 1, 1));
	};

} // namespace experimental::access

} // namespace celerity
