#pragma once

#include <celerity.h>

#include "./floating_point_precision.hpp"

// --- Data types ---
using Point = sycl::vec<DataTY, 3>;
using ShapeFactors = sycl::vec<DataTY, 3>;

// --- Compressed types ---
using CompressedPoint = sycl::vec<sycl::half, 3>;

namespace celerity::compression {
template <typename T, typename Q>
struct quantization {
	using value_type = T;
	using quant_type = Q;
};

template <typename T, typename Q>
struct point_cloud {
	using value_type = T;
	using compression_type = Q;
};
} // namespace celerity::compression

// template <typename T>
// struct compression_category_extractor {
// 	static constexpr celerity::compression_category value = celerity::compression_category::none; // default
// };

// template <typename Algorithm, celerity::compression_category Category>
// struct compression_category_extractor<celerity::compressed<Algorithm, Category>> {
// 	static constexpr celerity::compression_category value = Category;
// };

// // If given a celerity::accessor, delegate to the accessor's Compression parameter
// template <typename DataT, int Dims, celerity::access_mode Mode, celerity::target Target, typename Comp>
// struct compression_category_extractor<celerity::accessor<DataT, Dims, Mode, Target, Comp>> {
// 	static constexpr celerity::compression_category value = compression_category_extractor<Comp>::value;
// };

// // If given a celerity::buffer<T, N, Compression> also delegate (optional)
// template <typename DataT, int Dims, typename Comp>
// struct compression_category_extractor<celerity::buffer<DataT, Dims, Comp>> {
// 	static constexpr celerity::compression_category value = compression_category_extractor<Comp>::value;
// };

// -- Compression types --
constexpr celerity::compression_category compression_method = celerity::compression_category::global_memory;

using compression_type = celerity::compressed<celerity::compression::quantization<Point, sycl::vec<uint8_t, 3>>, celerity::compression_category::element_wise>;
using compression_tile_type = celerity::compressed<celerity::compression::point_cloud<Point, CompressedPoint>, compression_method>;

// --- Range mappers ---

template <int BufferDims>
struct full_third_dim {
	static_assert(BufferDims == 3, "BufferDims must be 3 for full_third_dim");

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<BufferDims> sbr;

		for(int i = 0; i < BufferDims; ++i) {
			if(i == 2) {
				sbr.offset[i] = 0;
				sbr.range[i] = buffer_size[i];
			} else {
				sbr.offset[i] = chnk.offset[i];
				sbr.range[i] = chnk.range[i];
			}
		}
		return sbr;
	}
};

template <int BufferDims>
struct full_third_dim_neighborhood {
	static_assert(BufferDims == 3, "BufferDims must be 3 for full_third_dim");

	full_third_dim_neighborhood<BufferDims>(size_t dim0, size_t dim1, size_t dim2) : m_dim0(dim0), m_dim1(dim1), m_dim2(dim2) {}

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<3> result = {celerity::detail::id_cast<3>(chnk.offset), celerity::detail::range_cast<3>(chnk.range)};
		const celerity::id<3> delta = {m_dim0 < result.offset[0] ? m_dim0 : result.offset[0], m_dim1 < result.offset[1] ? m_dim1 : result.offset[1],
		    m_dim2 < result.offset[2] ? m_dim2 : result.offset[2]};

		CELERITY_DEBUG("S Delta: {} {} {}, m_dim: {} {} {}, offset: {} {} {}, range: {} {} {}", delta[0], delta[1], delta[2], m_dim0, m_dim1, m_dim2,
		    result.offset[0], result.offset[1], result.offset[2], result.range[0], result.range[1], result.range[2]);
		result.offset -= delta;
		result.range += celerity::range<3>{m_dim0 + delta[0], m_dim1 + delta[1], m_dim2 + delta[2]};
		result.offset[2] = 0;
		result.range[2] = buffer_size[2];

		CELERITY_DEBUG("E Delta: {} {} {}, m_dim: {} {} {}, offset: {} {} {}, range: {} {} {}", delta[0], delta[1], delta[2], m_dim0, m_dim1, m_dim2,
		    result.offset[0], result.offset[1], result.offset[2], result.range[0], result.range[1], result.range[2]);

		return result;
	}

  private:
	size_t m_dim0 = 1;
	size_t m_dim1 = 1;
	size_t m_dim2 = 1;
};


template <int BufferDims>
struct three_d_to_two_d_neighborhood {
	static_assert(BufferDims == 2, "BufferDims must be 2 for full_third_dim");

	three_d_to_two_d_neighborhood<BufferDims>(size_t dim0, size_t dim1, size_t dim2) : m_dim0(dim0), m_dim1(dim1), m_dim2(dim2) {}

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<BufferDims> result = {{chnk.offset[0], chnk.offset[1]}, {chnk.range[0], chnk.range[1]}};
		const celerity::id<BufferDims> delta = {m_dim0 < result.offset[0] ? m_dim0 : result.offset[0], m_dim1 < result.offset[1] ? m_dim1 : result.offset[1]};
		result.offset -= delta;
		result.range += celerity::range<BufferDims>{m_dim0 + delta[0], m_dim1 + delta[1]};

		return result;
	}

  private:
	size_t m_dim0 = 1;
	size_t m_dim1 = 1;
	size_t m_dim2 = 1;
};

template <int BufferDims>
struct three_d_to_two_d {
	static_assert(BufferDims == 2, "BufferDims must be 2 for full_third_dim");

	template <int KernelDims>
	celerity::subrange<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		celerity::subrange<BufferDims> sbr;

		for(int i = 0; i < BufferDims; ++i) {
			sbr.offset[i] = chnk.offset[i];
			sbr.range[i] = chnk.range[i];
		}
		return sbr;
	}
};