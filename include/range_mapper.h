#pragma once

#include <stdexcept>
#include <type_traits>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "ranges.h"

namespace celerity {

namespace detail {

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_chunk_only = std::is_invocable_r_v<subrange<BufferDims>, const Functor&, const celerity::chunk<KernelDims>&>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_chunk_and_global_size =
	    std::is_invocable_r_v<subrange<BufferDims>, const Functor&, const celerity::chunk<KernelDims>&, const range<BufferDims>&>;

	template <typename Functor, int BufferDims, int KernelDims>
	constexpr bool is_range_mapper_invocable_for_kernel = is_range_mapper_invocable_for_chunk_only<Functor, BufferDims, KernelDims> //
	                                                      || is_range_mapper_invocable_for_chunk_and_global_size<Functor, BufferDims, KernelDims>;

	template <typename Functor, int BufferDims>
	constexpr bool is_range_mapper_invocable = is_range_mapper_invocable_for_kernel<Functor, BufferDims, 1>    //
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

	template <int KernelDims, int BufferDims, typename Functor>
	subrange<BufferDims> invoke_range_mapper_for_kernel(Functor&& fn, const celerity::chunk<KernelDims>& chunk, const range<BufferDims>& buffer_size) {
		static_assert(KernelDims >= 1 && KernelDims <= 3 && BufferDims >= 1 && BufferDims <= 3);
		if constexpr(is_range_mapper_invocable_for_chunk_and_global_size<Functor, BufferDims, KernelDims>) {
			return std::forward<Functor>(fn)(chunk, buffer_size);
		} else if constexpr(is_range_mapper_invocable_for_chunk_only<Functor, BufferDims, KernelDims>) {
			return std::forward<Functor>(fn)(chunk);
		} else {
			throw_invalid_range_mapper_args(KernelDims, BufferDims);
		}
	}

	template <int BufferDims>
	subrange<BufferDims> clamp_subrange_to_buffer_size(subrange<BufferDims> sr, range<BufferDims> buffer_size) {
		auto end = sr.offset + sr.range;
		if(BufferDims > 0 && end[0] > buffer_size[0]) { sr.range[0] = sr.offset[0] <= buffer_size[0] ? buffer_size[0] - sr.offset[0] : 0; }
		if(BufferDims > 1 && end[1] > buffer_size[1]) { sr.range[1] = sr.offset[1] <= buffer_size[1] ? buffer_size[1] - sr.offset[1] : 0; }
		if(BufferDims > 2 && end[2] > buffer_size[2]) { sr.range[2] = sr.offset[2] <= buffer_size[2] ? buffer_size[2] - sr.offset[2] : 0; }
		return sr;
	}

	template <int BufferDims, typename Functor>
	subrange<BufferDims> invoke_range_mapper(int kernel_dims, Functor fn, const celerity::chunk<3>& chunk, const range<BufferDims>& buffer_size) {
		static_assert(is_range_mapper_invocable<Functor, BufferDims>);
		subrange<BufferDims> sr;
		switch(kernel_dims) {
		case 0:
			[[fallthrough]]; // range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
			                 // anyway, we require range mappers to take at least 1d subranges
		case 1: sr = invoke_range_mapper_for_kernel(fn, chunk_cast<1>(chunk), buffer_size); break;
		case 2: sr = invoke_range_mapper_for_kernel(fn, chunk_cast<2>(chunk), buffer_size); break;
		case 3: sr = invoke_range_mapper_for_kernel(fn, chunk_cast<3>(chunk), buffer_size); break;
		default: assert(!"Unreachable"); return {};
		}
		return clamp_subrange_to_buffer_size(sr, buffer_size);
	}

	class range_mapper_base {
	  public:
		explicit range_mapper_base(cl::sycl::access::mode am) : access_mode(am) {}
		range_mapper_base(const range_mapper_base& other) = delete;
		range_mapper_base(range_mapper_base&& other) = delete;

		cl::sycl::access::mode get_access_mode() const { return access_mode; }

		virtual int get_buffer_dimensions() const = 0;

		virtual subrange<1> map_1(const chunk<1>& chnk) const = 0;
		virtual subrange<1> map_1(const chunk<2>& chnk) const = 0;
		virtual subrange<1> map_1(const chunk<3>& chnk) const = 0;
		virtual subrange<2> map_2(const chunk<1>& chnk) const = 0;
		virtual subrange<2> map_2(const chunk<2>& chnk) const = 0;
		virtual subrange<2> map_2(const chunk<3>& chnk) const = 0;
		virtual subrange<3> map_3(const chunk<1>& chnk) const = 0;
		virtual subrange<3> map_3(const chunk<2>& chnk) const = 0;
		virtual subrange<3> map_3(const chunk<3>& chnk) const = 0;

		virtual ~range_mapper_base() = default;

	  private:
		cl::sycl::access::mode access_mode;
	};

	template <int BufferDims, typename Functor>
	class range_mapper : public range_mapper_base {
	  public:
		range_mapper(Functor rmfn, cl::sycl::access::mode am, range<BufferDims> buffer_size) : range_mapper_base(am), rmfn(rmfn), buffer_size(buffer_size) {}

		int get_buffer_dimensions() const override { return BufferDims; }

		subrange<1> map_1(const chunk<1>& chnk) const override { return map<1>(chnk); }
		subrange<1> map_1(const chunk<2>& chnk) const override { return map<1>(chnk); }
		subrange<1> map_1(const chunk<3>& chnk) const override { return map<1>(chnk); }
		subrange<2> map_2(const chunk<1>& chnk) const override { return map<2>(chnk); }
		subrange<2> map_2(const chunk<2>& chnk) const override { return map<2>(chnk); }
		subrange<2> map_2(const chunk<3>& chnk) const override { return map<2>(chnk); }
		subrange<3> map_3(const chunk<1>& chnk) const override { return map<3>(chnk); }
		subrange<3> map_3(const chunk<2>& chnk) const override { return map<3>(chnk); }
		subrange<3> map_3(const chunk<3>& chnk) const override { return map<3>(chnk); }

	  private:
		Functor rmfn;
		range<BufferDims> buffer_size;

		template <int OtherBufferDims, int KernelDims>
		subrange<OtherBufferDims> map(const chunk<KernelDims>& chnk) const {
			if constexpr(OtherBufferDims == BufferDims) {
				auto sr = invoke_range_mapper_for_kernel(rmfn, chnk, buffer_size);
				return clamp_subrange_to_buffer_size(sr, buffer_size);
			} else {
				throw_invalid_range_mapper_result(OtherBufferDims, BufferDims, KernelDims);
			}
		}
	};

} // namespace detail


// --------------------------- Convenience range mappers ---------------------------

namespace access {

	template <int Dims = 0>
	struct one_to_one;

	template <>
	struct one_to_one<0> {
		template <int Dims>
		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			return chnk;
		}
	};

	template <int Dims>
	struct [[deprecated("Explicitly-dimensioned range mappers are deprecated, remove template arguments from celerity::one_to_one")]] one_to_one
	    : one_to_one<0>{};

	one_to_one()->one_to_one<>;

	template <int KernelDims, int BufferDims = KernelDims>
	struct fixed;

	template <int BufferDims>
	struct fixed<BufferDims, BufferDims> {
		fixed(const subrange<BufferDims>& sr) : sr(sr) {}

		template <int KernelDims>
		subrange<BufferDims> operator()(const chunk<KernelDims>&) const {
			return sr;
		}

	  private:
		subrange<BufferDims> sr;
	};

	template <int KernelDims, int BufferDims>
	struct fixed : fixed<BufferDims, BufferDims> {
		[[deprecated("Explicitly-dimensioned range mappers are deprecated, remove first template argument from celerity::fixed")]] //
		fixed(const subrange<BufferDims>& sr)
		    : fixed<BufferDims, BufferDims>(sr) {}
	};

	template <int BufferDims>
	fixed(subrange<BufferDims>) -> fixed<BufferDims>;

	template <int Dims>
	struct slice {
		slice(size_t dim_idx) : dim_idx(dim_idx) { assert(dim_idx < Dims && "Invalid slice dimension index (starts at 0)"); }

		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			subrange<Dims> result = chnk;
			result.offset[dim_idx] = 0;
			// Since we don't know the range of the buffer, we just set it way too high and let it be clamped to the correct range
			result.range[dim_idx] = std::numeric_limits<size_t>::max();
			return result;
		}

	  private:
		size_t dim_idx;
	};

	template <int KernelDims = 0, int BufferDims = KernelDims>
	struct all;

	template <>
	struct all<0, 0> {
		template <int KernelDims, int BufferDims>
		subrange<BufferDims> operator()(const chunk<KernelDims>&, const range<BufferDims>& buffer_size) const {
			return {{}, buffer_size};
		}
	};

	template <int KernelDims, int BufferDims>
	struct [[deprecated("Explicitly-dimensioned range mappers are deprecated, remove template arguments from celerity::all")]] all : all<0, 0>{};

	all()->all<>;

	template <int Dims>
	struct neighborhood {
		neighborhood(size_t dim0) : dim0(dim0), dim1(0), dim2(0) {}

		template <int D = Dims, std::enable_if_t<D >= 2, void*>...>
		neighborhood(size_t dim0, size_t dim1) : dim0(dim0), dim1(dim1), dim2(0) {}

		template <int D = Dims, std::enable_if_t<D == 3, void*>...>
		neighborhood(size_t dim0, size_t dim1, size_t dim2) : dim0(dim0), dim1(dim1), dim2(dim2) {}

		subrange<Dims> operator()(const chunk<Dims>& chnk) const {
			subrange<3> result = {celerity::detail::id_cast<3>(chnk.offset), celerity::detail::range_cast<3>(chnk.range)};
			const id<3> delta = {dim0 < result.offset[0] ? dim0 : result.offset[0], dim1 < result.offset[1] ? dim1 : result.offset[1],
			    dim2 < result.offset[2] ? dim2 : result.offset[2]};
			result.offset -= delta;
			result.range += range<3>{dim0 + delta[0], dim1 + delta[1], dim2 + delta[2]};
			return detail::subrange_cast<Dims>(result);
		}

	  private:
		size_t dim0, dim1, dim2;
	};

	neighborhood(size_t)->neighborhood<1>;
	neighborhood(size_t, size_t)->neighborhood<2>;
	neighborhood(size_t, size_t, size_t)->neighborhood<3>;

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
		explicit even_split(const range<BufferDims>& granularity) : granularity(granularity) {}

		subrange<BufferDims> operator()(const chunk<1>& chunk, const range<BufferDims>& buffer_size) const {
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
		range<BufferDims> granularity = detail::range_cast<BufferDims>(range<3>(1, 1, 1));
	};

} // namespace experimental::access

} // namespace celerity
