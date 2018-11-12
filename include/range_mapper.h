#pragma once

#include <stdexcept>
#include <type_traits>

#include <SYCL/sycl.hpp>
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
		if(end[0] > buffer_size[0]) { sr.range[0] = sr.offset[0] <= buffer_size[0] ? buffer_size[0] - sr.offset[0] : 0; }
		if(end[1] > buffer_size[1]) { sr.range[1] = sr.offset[1] <= buffer_size[1] ? buffer_size[1] - sr.offset[1] : 0; }
		if(end[2] > buffer_size[2]) { sr.range[2] = sr.offset[2] <= buffer_size[2] ? buffer_size[2] - sr.offset[2] : 0; }
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

	template <int KernelDims, int BufferDims>
	struct fixed {
		fixed(subrange<BufferDims> sr) : sr(sr) {}

		subrange<BufferDims> operator()(chunk<KernelDims>) const { return sr; }

	  private:
		subrange<BufferDims> sr;
	};

} // namespace access

} // namespace celerity
