#pragma once

#include "compression_impl.h"

// The concrete compression algorithms built on top of the generic machinery in compression_impl.h: quantization,
// conversion, point_cloud, point_cloud_with_dep, and z_curve_hybrid.

namespace celerity {
namespace compression {
	template <typename T, typename C>
	struct conversion {
		using value_type = T;
		using compressed_type = C;
	};

	template <typename T, typename C>
	struct quantization {
		using value_type = T;
		using compressed_type = C;
	};

	template <typename T, typename C>
	struct point_cloud {
		using value_type = T;
		using compressed_type = C;
	};

	template <typename T, typename C>
	struct point_cloud_with_dep {
		using value_type = T;
		using compressed_type = C;
	};

	template <typename T, typename C>
	struct z_curve_hybrid {
		using value_type = T;
		using compressed_type = C;
	};
} // namespace compression

struct downscale_point_cloud_buffer {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);
		auto linearized_range = chnk.range.size() / chnk.local_range.size();
		auto linearized_offset =
		    celerity::detail::get_linear_index(celerity::range<2>{chnk.global_size[0], chnk.global_size[1]}, celerity::id<2>{chnk.offset[0], chnk.offset[1]});

		auto my_region = celerity::detail::region_builder<BufferDims>();
		my_region.add({{linearized_id, linearized_offset}, {linearized_id + 1, linearized_offset + linearized_range}});

		return std::move(my_region).into_region();
	}
};

struct point_cloud_dep_buffer_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		auto boxes = sbr.get_boxes();
		auto my_region = celerity::detail::region_builder<BufferDims>();

		for(const auto& box : boxes) {
			auto offset = box.get_offset();
			auto range = box.get_range();
			my_region.add({{offset[0], offset[1]}, {offset[0] + range[0], offset[1] + range[1]}});
		}

		return std::move(my_region).into_region();
	}
};

struct full_subrange_passthrough_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		(void)chnk;
		(void)buffer_size;
		(void)main_buffer_size;
		static_assert(BufferDims == MainBufferDims, "full_subrange_passthrough_mapper requires matching dimensions");

		auto boxes = sbr.get_boxes();
		auto my_region = celerity::detail::region_builder<BufferDims>();

		for(const auto& box : boxes) {
			auto offset = box.get_offset();
			auto range = box.get_range();
			my_region.add({offset, offset + range});
		}

		return std::move(my_region).into_region();
	}
};

template <typename T, typename C, compression_category Category>
class compressed<compression::point_cloud<T, C>, Category>
    : public compressed_default<compressed<compression::point_cloud<T, C>, Category>,
          compression_category::element_wise | compression_category::local_memory | compression_category::global_memory> {
  public:
	using value_type = typename celerity::compression::point_cloud<T, C>::value_type;
	using compressed_type = typename celerity::compression::point_cloud<T, C>::compressed_type;

	compressed(const value_type& min, int tile_size) : m_min(min), m_tile_size(tile_size) {}

	template <int Dims>
	compressed_type compress(const value_type number, [[maybe_unused]] const id<Dims>& item) const {
		value_type center = {m_min.x() + (m_tile_size * item[0]), m_min.y() + (m_tile_size * item[1]), 0};
		auto tmp = number - center;
		compressed_type compressed_point = {tmp.x(), tmp.y(), tmp.z()};
		return compressed_point;
	}

	template <int Dims>
	value_type decompress(const compressed_type number, [[maybe_unused]] const id<Dims>& item) const {
		value_type center = {m_min.x() + (m_tile_size * item[0]), m_min.y() + (m_tile_size * item[1]), 0};
		value_type decompressed_p;

		decompressed_p.x() = number.x();
		decompressed_p.y() = number.y();
		decompressed_p.z() = number.z();

		decompressed_p += center;
		return decompressed_p;
	}

	template <specialization_of_item Item, int Dim>
	inline celerity::id<1> calculate_tile_tracking_idx(const Item& item, const id<Dim>& offset) const {
		return celerity::detail::get_linear_index(
		    celerity::detail::range_cast<2>(item.get_global_range()) / celerity::detail::range_cast<2>(item.get_local_range()),
		    celerity::detail::id_cast<2>(offset) / celerity::detail::range_cast<2>(item.get_local_range()));
	}

  private:
	const value_type m_min;
	const int m_tile_size;
};

template <typename T, typename C, compression_category Category>
class compressed<compression::point_cloud_with_dep<T, C>, Category> : public compressed_default<compressed<compression::point_cloud_with_dep<T, C>, Category>,
                                                                          compression_category::local_memory | compression_category::global_memory> {
  public:
	using value_type = typename celerity::compression::point_cloud_with_dep<T, C>::value_type;
	using compressed_type = typename celerity::compression::point_cloud_with_dep<T, C>::compressed_type;

	compressed(const value_type& min, int tile_size) : m_min(min), m_tile_size(tile_size) {}

	template <int Dims>
	inline compressed_type compress(const value_type number, [[maybe_unused]] const id<Dims>& item, const value_type& dep) const {
		auto tmp = number - dep;
		compressed_type compressed_point = {tmp.x(), tmp.y(), tmp.z()};
		return compressed_point;
	}

	template <int Dims>
	inline value_type decompress(const compressed_type number, [[maybe_unused]] const id<Dims>& item, const value_type& dep) const {
		value_type decompressed_p;

		decompressed_p.x() = number.x();
		decompressed_p.y() = number.y();
		decompressed_p.z() = number.z();

		decompressed_p += dep;
		return decompressed_p;
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void compress_memory_chunk(
	    celerity::nd_item<Dims>& item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, const Dependency& dep) const {
		auto chunk_offset = uncompressed_data_acc.get_offset();
		auto tile_x = chunk_offset[0];
		auto tile_y = chunk_offset[1];

		if(tile_x >= item.get_global_range(0) || tile_y >= item.get_global_range(1)) { return; }

		auto center = value_type{m_min.x() + (m_tile_size * tile_x), m_min.y() + (m_tile_size * tile_y), 0};
		dep[{tile_x, tile_y}] = center;

		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;

		for(size_t i = 0; i < calculations_per_item; ++i) {
			auto idx = (i * amount) + item.get_local_linear_id();
			if(idx >= chunk_range.size()) { break; }

			celerity::id<Dims> local_index = celerity::detail::get_nd_index(idx, chunk_range);
			celerity::id<Dims> global_index = uncompressed_data_acc.get_offset() + local_index;

			if(!detail::all_true(global_index < item.get_global_range())) { continue; }

			compressed_data_acc[global_index] = compress(uncompressed_data_acc[global_index], global_index, center);
		}
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void decompress_memory_chunk(
	    celerity::nd_item<Dims>& item, const CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, const Dependency& dep) const {
		auto chunk_offset = uncompressed_data_acc.get_offset();
		auto tile_x = chunk_offset[0];
		auto tile_y = chunk_offset[1];

		if(tile_x >= item.get_global_range(0) || tile_y >= item.get_global_range(1)) { return; }

		auto& center = dep[{tile_x, tile_y}];
		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;

		for(size_t i = 0; i < calculations_per_item; ++i) {
			auto idx = (i * amount) + item.get_local_linear_id();
			if(idx >= chunk_range.size()) { break; }

			celerity::id<Dims> local_index = celerity::detail::get_nd_index<Dims>(idx, chunk_range);
			celerity::id<Dims> global_index = uncompressed_data_acc.get_offset() + local_index;

			// since size_t -1 = uint64_max we do not need to check for underflow
			if(!detail::all_true(global_index < item.get_global_range())) { continue; }

			uncompressed_data_acc[global_index] = decompress(compressed_data_acc[global_index], global_index, center);
		}
	}

	template <typename CompressedData, typename UncompressedData, typename Dependency>
	void compress_data(const UncompressedData* data, CompressedData& compressed_data, size_t size, const Dependency& dep) const {
		(void)data;
		(void)compressed_data;
		(void)size;
		(void)dep;
		assert(true);
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void decompress_data(
	    const CompressedData& compressed_data, UncompressedData& uncompressed_data, const celerity::range<Dims> range, const Dependency& dep) const {
		for(size_t i = 0; i < range[0]; i++) {
			for(size_t j = 0; j < range[1]; j++) {
				auto& center = dep[{i, j}];
				for(size_t k = 0; k < range[2]; k++) {
					uncompressed_data[celerity::detail::get_linear_index(range, {i, j, k})] =
					    decompress(compressed_data[{i, j, k}], celerity::id<3>{i, j, k}, center);
				}
			}
		}
	}


	template <specialization_of_item Item, int Dim>
	inline celerity::id<1> calculate_tile_tracking_idx(const Item& item, const id<Dim>& offset) const {
		return celerity::detail::get_linear_index(
		    celerity::detail::range_cast<2>(item.get_global_range()) / celerity::detail::range_cast<2>(item.get_local_range()),
		    celerity::detail::id_cast<2>(offset) / celerity::detail::range_cast<2>(item.get_local_range()));
	}

  private:
	const value_type m_min;
	const int m_tile_size;
};

template <typename T, typename C, compression_category Category>
class compressed<compression::z_curve_hybrid<T, C>, Category> : public compressed_default<compressed<compression::z_curve_hybrid<T, C>, Category>,
                                                                    compression_category::local_memory | compression_category::global_memory> {
  public:
	using value_type = typename celerity::compression::z_curve_hybrid<T, C>::value_type;
	using compressed_type = typename celerity::compression::z_curve_hybrid<T, C>::compressed_type;
	using scalar_type = typename vec_element_type<value_type>::type;
	using compressed_scalar_type = typename vec_element_type<compressed_type>::type;

	compressed(const value_type& min, scalar_type xy_resolution) : m_min(min), m_xy_resolution(xy_resolution) {
		static_assert(std::is_integral_v<compressed_scalar_type>, "z_curve_hybrid compressed type must be integral");
		static_assert(vec_size<compressed_type>::value == 1, "z_curve_hybrid compressed type must be scalar or sycl::vec<integral, 1>");
	}

	template <int Dims>
	inline compressed_type compress(const value_type number, [[maybe_unused]] const id<Dims>& item) const {
		const auto quantized_x = quantize(number.x(), m_min.x());
		const auto quantized_y = quantize(number.y(), m_min.y());
		const uint32_t morton = morton_encode_2d(quantized_x, quantized_y);
		return pack_morton(morton);
	}

	template <int Dims>
	inline value_type decompress(const compressed_type number, [[maybe_unused]] const id<Dims>& item, const sycl::half z_dep) const {
		const uint32_t morton = unpack_morton(number);
		const uint32_t quantized_x = morton_decode_x(morton);
		const uint32_t quantized_y = morton_decode_y(morton);

		value_type out{};
		out.x() = m_min.x() + static_cast<scalar_type>(quantized_x) * m_xy_resolution;
		out.y() = m_min.y() + static_cast<scalar_type>(quantized_y) * m_xy_resolution;
		out.z() = m_min.z() + static_cast<scalar_type>(z_dep);
		return out;
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void compress_memory_chunk(
	    celerity::nd_item<Dims>& item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, const Dependency& z_dep) const {
		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;

		for(size_t i = 0; i < calculations_per_item; ++i) {
			auto idx = (i * amount) + item.get_local_linear_id();
			if(idx >= chunk_range.size()) { break; }

			celerity::id<Dims> local_index = celerity::detail::get_nd_index(idx, chunk_range);
			celerity::id<Dims> global_index = uncompressed_data_acc.get_offset() + local_index;

			if(!detail::all_true(global_index < item.get_global_range())) { continue; }

			const auto point = uncompressed_data_acc[global_index];
			compressed_data_acc[global_index] = compress(point, global_index);
			z_dep[global_index] = static_cast<sycl::half>(point.z() - m_min.z());
		}
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void decompress_memory_chunk(celerity::nd_item<Dims>& item, const CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc,
	    const Dependency& z_dep) const {
		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;

		for(size_t i = 0; i < calculations_per_item; ++i) {
			auto idx = (i * amount) + item.get_local_linear_id();
			if(idx >= chunk_range.size()) { break; }

			celerity::id<Dims> local_index = celerity::detail::get_nd_index(idx, chunk_range);
			celerity::id<Dims> global_index = uncompressed_data_acc.get_offset() + local_index;

			if(!detail::all_true(global_index < item.get_global_range())) { continue; }

			uncompressed_data_acc[global_index] = decompress(compressed_data_acc[global_index], global_index, z_dep[global_index]);
		}
	}

	template <typename CompressedData, typename UncompressedData, typename Dependency>
	void compress_data(const UncompressedData* data, CompressedData& compressed_data, size_t size, const Dependency& z_dep) const {
		(void)data;
		(void)compressed_data;
		(void)size;
		(void)z_dep;
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void decompress_data(
	    const CompressedData& compressed_data, UncompressedData& uncompressed_data, const celerity::range<Dims> range, const Dependency& z_dep) const {
		for(size_t i = 0; i < range[0]; i++) {
			for(size_t j = 0; j < range[1]; j++) {
				for(size_t k = 0; k < range[2]; k++) {
					const celerity::id<3> idx{i, j, k};
					uncompressed_data[celerity::detail::get_linear_index(range, idx)] = decompress(compressed_data[idx], idx, z_dep[idx]);
				}
			}
		}
	}

	template <specialization_of_item Item, int Dim>
	inline celerity::id<1> calculate_tile_tracking_idx(const Item& item, const id<Dim>& offset) const {
		return celerity::detail::get_linear_index(
		    celerity::detail::range_cast<2>(item.get_global_range()) / celerity::detail::range_cast<2>(item.get_local_range()),
		    celerity::detail::id_cast<2>(offset) / celerity::detail::range_cast<2>(item.get_local_range()));
	}

  private:
	static constexpr uint32_t max_quantized_value = (1u << 16) - 1;

	inline uint32_t quantize(const scalar_type value, const scalar_type minimum) const {
		if(m_xy_resolution <= static_cast<scalar_type>(0)) { return 0; }
		const auto normalized = (value - minimum) / m_xy_resolution;
		if(normalized <= static_cast<scalar_type>(0)) { return 0; }
		const auto rounded = static_cast<uint32_t>(normalized + static_cast<scalar_type>(0.5));
		return std::min(rounded, max_quantized_value);
	}

	static inline uint32_t spread_bits_16(uint32_t v) {
		v &= 0x0000ffffu;
		v = (v | (v << 8)) & 0x00FF00FFu;
		v = (v | (v << 4)) & 0x0F0F0F0Fu;
		v = (v | (v << 2)) & 0x33333333u;
		v = (v | (v << 1)) & 0x55555555u;
		return v;
	}

	static inline uint32_t compact_bits_16(uint32_t v) {
		v &= 0x55555555u;
		v = (v | (v >> 1)) & 0x33333333u;
		v = (v | (v >> 2)) & 0x0F0F0F0Fu;
		v = (v | (v >> 4)) & 0x00FF00FFu;
		v = (v | (v >> 8)) & 0x0000FFFFu;
		return v;
	}

	static inline uint32_t morton_encode_2d(uint32_t x, uint32_t y) { return spread_bits_16(x) | (spread_bits_16(y) << 1); }
	static inline uint32_t morton_decode_x(uint32_t morton) { return compact_bits_16(morton); }
	static inline uint32_t morton_decode_y(uint32_t morton) { return compact_bits_16(morton >> 1); }

	inline compressed_type pack_morton(const uint32_t morton) const {
		if constexpr(is_vec<compressed_type>::value) {
			compressed_type packed{};
			packed.x() = static_cast<compressed_scalar_type>(morton);
			return packed;
		} else {
			return static_cast<compressed_type>(morton);
		}
	}

	inline uint32_t unpack_morton(const compressed_type value) const {
		if constexpr(is_vec<compressed_type>::value) {
			return static_cast<uint32_t>(value.x());
		} else {
			return static_cast<uint32_t>(value);
		}
	}

	const value_type m_min;
	const scalar_type m_xy_resolution;
};

template <typename T, typename C, compression_category Category>
class compressed<celerity::compression::quantization<T, C>, Category>
    : public compressed_default<compressed<celerity::compression::quantization<T, C>, Category>,
          compression_category::element_wise | compression_category::local_memory | compression_category::global_memory> {
	using compressed_type = typename celerity::compression::quantization<T, C>::compressed_type;
	using value_type = typename celerity::compression::quantization<T, C>::value_type;

	using vec_value_type = typename vec_element_type<value_type>::type;
	using vec_compressed_type = typename vec_element_type<compressed_type>::type;

  public:
	compressed()
	    : m_lower_bound(0), m_upper_bound(1), m_decompression_factor(calculate_decompression_factor(m_lower_bound, m_upper_bound)),
	      m_compression_factor(calculate_compression_factor(m_lower_bound, m_upper_bound)) {}

	template <typename ValueT>
	compressed(ValueT lower_bound, ValueT upper_bound)
	    : m_lower_bound(lower_bound), m_upper_bound(upper_bound), m_decompression_factor(calculate_decompression_factor(m_lower_bound, m_upper_bound)),
	      m_compression_factor(calculate_compression_factor(m_lower_bound, m_upper_bound)) {
		static_assert(std::is_same<ValueT, vec_value_type>(), "Value type isn't the same");
		static_assert(is_vec<value_type>::value == is_vec<compressed_type>::value, "Value and Quant type must be both either sycl::vec or fundamental");
	}

	vec_value_type get_upper_bound() const { return m_upper_bound; }
	vec_value_type get_lower_bound() const { return m_lower_bound; }

	void set_upper_bound(vec_value_type upper_bound) {
		m_upper_bound = upper_bound;
		m_decompression_factor = calculate_decompression_factor(m_lower_bound, m_upper_bound);
		m_compression_factor = calculate_compression_factor(m_lower_bound, m_upper_bound);
	}
	void set_lower_bound(vec_value_type lower_bound) {
		m_lower_bound = lower_bound;
		m_decompression_factor = calculate_decompression_factor(m_lower_bound, m_upper_bound);
		m_compression_factor = calculate_compression_factor(m_lower_bound, m_upper_bound);
	}

	template <int Dims>
	compressed_type compress(const value_type number, [[maybe_unused]] const id<Dims>& item) const {
		if constexpr(is_vec<value_type>::value) {
			compressed_type result;

			for(int i = 0; i < vec_size<compressed_type>::value; ++i) {
				result[i] = static_cast<vec_compressed_type>(std::round((number[i] - m_lower_bound) * m_compression_factor));
			}

			return result;
		} else {
			return static_cast<compressed_type>(std::round((number - m_lower_bound) * m_compression_factor));
		}
	}

	template <int Dims>
	value_type decompress(const compressed_type number, [[maybe_unused]] const id<Dims>& item) const {
		if constexpr(is_vec<value_type>::value) {
			value_type result;

			for(int i = 0; i < vec_size<value_type>::value; ++i) {
				result[i] = static_cast<vec_value_type>(number[i]) * m_decompression_factor + m_lower_bound;
			}

			return result;
		} else {
			return (static_cast<value_type>(number) * m_decompression_factor) + m_lower_bound;
		}
	}

  private:
	static vec_value_type calculate_decompression_factor(vec_value_type lower_bound, vec_value_type upper_bound) {
		return 1 / static_cast<vec_value_type>(std::numeric_limits<vec_compressed_type>::max()) * (upper_bound - lower_bound);
	}
	static vec_value_type calculate_compression_factor(vec_value_type lower_bound, vec_value_type upper_bound) {
		return (1 / (upper_bound - lower_bound)) * std::numeric_limits<vec_compressed_type>::max();
	}

	vec_value_type m_lower_bound;
	vec_value_type m_upper_bound;
	vec_value_type m_decompression_factor;
	vec_value_type m_compression_factor;
};

template <typename T, typename C, compression_category Category>
class point_cloud_compression : public compression_object_skeleton<point_cloud_compression<T, C, Category>> {
  public:
	template <typename... Args>
	    requires std::constructible_from<compressed<compression::point_cloud<T, C>, Category>, Args...>
	point_cloud_compression(Args&&... args)
	    : m_compression_object(std::forward<Args>(args)...),
	      m_dependencies({{[](const celerity::range<3>& r) { return celerity::range<2>{r[0], r[1]}.size(); }, downscale_point_cloud_buffer{}}}) {}

	auto get_compression_object() const { return m_compression_object; }

	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<compression::point_cloud<T, C>, Category> m_compression_object;

	dependency_bundle<buffer_access_description<T, downscale_point_cloud_buffer>> m_dependencies;
};

template <typename T, typename C, compression_category Category>
class point_cloud_with_dep_compression : public compression_object_skeleton<point_cloud_with_dep_compression<T, C, Category>> {
  public:
	template <typename... Args>
	    requires std::constructible_from<compressed<compression::point_cloud_with_dep<T, C>, Category>, Args...>
	point_cloud_with_dep_compression(Args&&... args)
	    : m_compression_object(std::forward<Args>(args)...),
	      m_dependencies({{[](const celerity::range<3>& r) { return celerity::range<2>{r[0], r[1]}.size(); }, downscale_point_cloud_buffer{}},
	          {[](const celerity::range<3>& r) { return celerity::range<2>{r[0], r[1]}; }, point_cloud_dep_buffer_mapper{}}}) {}

	auto get_compression_object() const { return m_compression_object; }
	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<compression::point_cloud_with_dep<T, C>, Category> m_compression_object;

	dependency_bundle<buffer_access_description<T, downscale_point_cloud_buffer>, buffer_access_description<T, point_cloud_dep_buffer_mapper, 2>>
	    m_dependencies;
};

template <typename T, typename C, compression_category Category>
class z_curve_hybrid_compression : public compression_object_skeleton<z_curve_hybrid_compression<T, C, Category>> {
  public:
	template <typename... Args>
	    requires std::constructible_from<compressed<compression::z_curve_hybrid<T, C>, Category>, Args...>
	z_curve_hybrid_compression(Args&&... args)
	    : m_compression_object(std::forward<Args>(args)...),
	      m_dependencies({{[](const celerity::range<3>& r) { return celerity::range<2>{r[0], r[1]}.size(); }, downscale_point_cloud_buffer{}},
	          {[](const celerity::range<3>& r) { return celerity::range<3>{r[0], r[1], r[2]}; }, full_subrange_passthrough_mapper{}}}) {}

	auto get_compression_object() const { return m_compression_object; }
	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<compression::z_curve_hybrid<T, C>, Category> m_compression_object;

	dependency_bundle<buffer_access_description<T, downscale_point_cloud_buffer>, buffer_access_description<sycl::half, full_subrange_passthrough_mapper, 3>>
	    m_dependencies;
};

template <typename T, typename C, compression_category Category>
class quantization_compression : public compression_object_skeleton<quantization_compression<T, C, Category>> {
  public:
	template <typename... Args>
	    requires std::constructible_from<compressed<compression::quantization<T, C>, Category>, Args...>
	quantization_compression(Args&&... args)
	    : m_compression_object(std::forward<Args>(args)...),
	      m_dependencies({[](const celerity::range<3>& r) { return celerity::range<2>{r[0], r[1]}.size(); }, downscale_device_specific_mapper{}}) {
	} 

	auto get_compression_object() const { return m_compression_object; }

	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<compression::quantization<T, C>, Category> m_compression_object;

	dependency_bundle<buffer_access_description<T, downscale_device_specific_mapper>> m_dependencies;
};


template <typename T, typename C, compression_category Category>
class compressed<compression::conversion<T, C>, Category>
    : public compressed_default<compressed<compression::conversion<T, C>, Category>,
          compression_category::element_wise | compression_category::local_memory | compression_category::global_memory> {
  public:
	using value_type = typename celerity::compression::conversion<T, C>::value_type;
	using compressed_type = typename celerity::compression::conversion<T, C>::compressed_type;

	template <int Dims>
	compressed_type compress(const value_type number, [[maybe_unused]] const id<Dims>& item) const {
		return static_cast<compressed_type>(number);
	}

	template <int Dims>
	value_type decompress(const compressed_type number, [[maybe_unused]] const id<Dims>& item) const {
		return static_cast<value_type>(number);
	}
};


template <typename T, typename C, compression_category Category>
class conversion_compression : public compression_object_skeleton<conversion_compression<T, C, Category>> {
  public:
	template <typename... Args>
	    requires std::constructible_from<compressed<compression::conversion<T, C>, Category>, Args...>
	conversion_compression(Args&&... args) : m_compression_object(std::forward<Args>(args)...) {}

	auto get_compression_object() const { return m_compression_object; }

	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<compression::conversion<T, C>, Category> m_compression_object;
	dependency_bundle<buffer_access_description<T, downscale_device_specific_mapper>> m_dependencies;
};
} // namespace celerity
