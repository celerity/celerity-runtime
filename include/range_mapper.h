#pragma once

#include <stdexcept>
#include <type_traits>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "ranges.h"

namespace celerity {

namespace detail {


	template <int KernelDims, int BufferDims>
	using range_mapper_fn = std::function<subrange<BufferDims>(chunk<KernelDims> chnk)>;

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
		range_mapper(range_mapper_fn<KernelDims, BufferDims> fn, cl::sycl::access::mode am, cl::sycl::range<BufferDims> buffer_size)
		    : range_mapper_base(am), rmfn(fn), buffer_size(buffer_size) {}

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
			return clamp_subrange_to_buffer_size(rmfn(chnk), buffer_size);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 1, subrange<1>>::type map_1_impl(chunk<KernelDims> chnk) const {
			return range_mapper_base::map_1(chnk);
		}

		template <int D = BufferDims>
		typename std::enable_if<D == 2, subrange<2>>::type map_2_impl(chunk<KernelDims> chnk) const {
			return clamp_subrange_to_buffer_size(rmfn(chnk), buffer_size);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 2, subrange<2>>::type map_2_impl(chunk<KernelDims> chnk) const {
			return range_mapper_base::map_2(chnk);
		}

		template <int D = BufferDims>
		typename std::enable_if<D == 3, subrange<3>>::type map_3_impl(chunk<KernelDims> chnk) const {
			return clamp_subrange_to_buffer_size(rmfn(chnk), buffer_size);
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
		subrange<BufferDims> operator()(chunk<KernelDims>) const {
			subrange<BufferDims> result;
			result.offset = detail::id_cast<BufferDims>(cl::sycl::id<3>{0, 0, 0});
			const auto max_num = std::numeric_limits<size_t>::max();
			// Since we don't know the range of the buffer, we just set it way too high and let it be clamped to the correct range
			result.range = detail::range_cast<BufferDims>(cl::sycl::range<3>{max_num, max_num, max_num});
			return result;
		}
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
			return subrange<Dims>(result);
		}

	  private:
		size_t dim0, dim1, dim2;
	};

} // namespace access

} // namespace celerity
