#pragma once

#include <stdexcept>
#include <type_traits>

#include <SYCL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "subrange.h"

namespace celerity {

namespace detail {

	template <int KernelDims, int BufferDims>
	using range_mapper_fn = std::function<subrange<BufferDims>(subrange<KernelDims> range)>;

	class range_mapper_base {
	  public:
		range_mapper_base(cl::sycl::access::mode am) : access_mode(am) {}
		range_mapper_base(const range_mapper_base& other) = delete;
		range_mapper_base(range_mapper_base&& other) = delete;

		cl::sycl::access::mode get_access_mode() const { return access_mode; }

		virtual size_t get_kernel_dimensions() const = 0;

		virtual subrange<1> map_1(subrange<1> range) const { throw create_dimension_mismatch_error(1, 1); }
		virtual subrange<1> map_1(subrange<2> range) const { throw create_dimension_mismatch_error(2, 1); }
		virtual subrange<1> map_1(subrange<3> range) const { throw create_dimension_mismatch_error(3, 1); }
		virtual subrange<2> map_2(subrange<1> range) const { throw create_dimension_mismatch_error(1, 2); }
		virtual subrange<2> map_2(subrange<2> range) const { throw create_dimension_mismatch_error(2, 2); }
		virtual subrange<2> map_2(subrange<3> range) const { throw create_dimension_mismatch_error(3, 2); }
		virtual subrange<3> map_3(subrange<1> range) const { throw create_dimension_mismatch_error(1, 3); }
		virtual subrange<3> map_3(subrange<2> range) const { throw create_dimension_mismatch_error(2, 3); }
		virtual subrange<3> map_3(subrange<3> range) const { throw create_dimension_mismatch_error(3, 3); }

		virtual ~range_mapper_base() = default;

	  private:
		cl::sycl::access::mode access_mode;

		std::runtime_error create_dimension_mismatch_error(int wrong_kernel_dims, int buffer_dims) const {
			const int kernel_dims = get_kernel_dimensions();
			return std::runtime_error(fmt::format("Range mapper maps subrange<{}> -> subrange<{}>, but should map subrange<{}> -> subrange<{}>.",
			    wrong_kernel_dims, buffer_dims, kernel_dims, buffer_dims));
		}
	};

	template <int KernelDims, int BufferDims>
	class range_mapper : public range_mapper_base {
	  public:
		range_mapper(range_mapper_fn<KernelDims, BufferDims> fn, cl::sycl::access::mode am) : range_mapper_base(am), rmfn(fn) {}

		size_t get_kernel_dimensions() const override { return KernelDims; }

		subrange<1> map_1(subrange<KernelDims> range) const override { return map_1_impl(range); }
		subrange<2> map_2(subrange<KernelDims> range) const override { return map_2_impl(range); }
		subrange<3> map_3(subrange<KernelDims> range) const override { return map_3_impl(range); }

	  private:
		range_mapper_fn<KernelDims, BufferDims> rmfn;

		template <int D = BufferDims>
		typename std::enable_if<D == 1, subrange<1>>::type map_1_impl(subrange<KernelDims> range) const {
			return rmfn(range);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 1, subrange<1>>::type map_1_impl(subrange<KernelDims> range) const {
			return range_mapper_base::map_1(range);
		}

		template <int D = BufferDims>
		typename std::enable_if<D == 2, subrange<2>>::type map_2_impl(subrange<KernelDims> range) const {
			return rmfn(range);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 2, subrange<2>>::type map_2_impl(subrange<KernelDims> range) const {
			return range_mapper_base::map_2(range);
		}

		template <int D = BufferDims>
		typename std::enable_if<D == 3, subrange<3>>::type map_3_impl(subrange<KernelDims> range) const {
			return rmfn(range);
		}

		template <int D = BufferDims>
		typename std::enable_if<D != 3, subrange<3>>::type map_3_impl(subrange<KernelDims> range) const {
			return range_mapper_base::map_3(range);
		}
	};

} // namespace detail


// --------------------------- Convenience range mappers ---------------------------


namespace access {
	template <int Dims>
	struct one_to_one {
		subrange<Dims> operator()(subrange<Dims> range) const { return range; }
	};
} // namespace access

} // namespace celerity
