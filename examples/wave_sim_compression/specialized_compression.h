#pragma once

#include <celerity.h>
#include <compression.h>
#include <compression_impl.h>
#include <compression_wrapper.h>

namespace celerity::compression {
template <typename T, typename C>
struct specialized_quantization {
	using value_type = T;
	using compressed_type = C;
};
} // namespace celerity::compression


namespace celerity {
template <typename T, typename C, compression_category Category>
class compressed<celerity::compression::specialized_quantization<T, C>, Category>
    : public compressed_default<compressed<celerity::compression::specialized_quantization<T, C>, Category>,
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

	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void compress_memory_chunk(celerity::nd_item<Dims> item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc) const {
		auto global_id = item.get_global_id();

		compressed_data_acc[global_id] = compress(uncompressed_data_acc[global_id], global_id);
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void decompress_memory_chunk(celerity::nd_item<Dims> item, const CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc) const {
		auto global_id = item.get_global_id();
		auto chunk_size = uncompressed_data_acc.get_chunk_size();

		if(item.get_local_range(0) < chunk_size[0] || item.get_local_range(1) < chunk_size[1]) {
			auto global_range = item.get_global_range();

			if(item.get_local_id(0) == item.get_local_range(0) - 1) {
				size_t py = global_id[0] < global_range[0] - 1 ? global_id[0] + 1 : global_id[0];
				uncompressed_data_acc[{py, global_id[1]}] = decompress(compressed_data_acc[{py, global_id[1]}], global_id);
			}

			if(item.get_local_id(0) == 0) {
				size_t my = global_id[0] > 0 ? global_id[0] - 1 : global_id[0];
				uncompressed_data_acc[{my, global_id[1]}] = decompress(compressed_data_acc[{my, global_id[1]}], global_id);
			}

			if(item.get_local_id(1) == item.get_local_range(1) - 1) {
				size_t px = global_id[1] < global_range[1] - 1 ? global_id[1] + 1 : global_id[1];
				uncompressed_data_acc[{global_id[0], px}] = decompress(compressed_data_acc[{global_id[0], px}], global_id);
			}

			if(item.get_local_id(1) == 0) {
				size_t mx = global_id[1] > 0 ? global_id[1] - 1 : global_id[1];
				uncompressed_data_acc[{global_id[0], mx}] = decompress(compressed_data_acc[{global_id[0], mx}], global_id);
			}
		}

		uncompressed_data_acc[global_id] = decompress(compressed_data_acc[global_id], global_id);
	}

	// offset = maybe unused
	template <specialization_of_item Item, int Dim>
	inline celerity::id<1> calculate_tile_tracking_idx(const Item& item, [[maybe_unused]] const id<Dim>& offset) const {
		return celerity::detail::get_linear_index(
		    celerity::detail::range_cast<2>(item.get_global_range()) / celerity::detail::range_cast<2>(item.get_local_range()),
		    celerity::detail::id_cast<2>(item.get_global_id()) / celerity::detail::range_cast<2>(item.get_local_range()));
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
class specialized_quantization_compression : public compression_object_skeleton<specialized_quantization_compression<T, C, Category>> {
  public:
	template <typename... Args>
	    requires std::constructible_from<compressed<compression::specialized_quantization<T, C>, Category>, Args...>
	specialized_quantization_compression(Args&&... args)
	    : m_compression_object(std::forward<Args>(args)...),
	      m_dependencies({[](const celerity::range<3>& r) { return celerity::range<2>{r[0], r[1]}.size(); }, downscale_device_specific_mapper{}}) {}

	auto get_compression_object() const { return m_compression_object; }

	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<compression::specialized_quantization<T, C>, Category> m_compression_object;

	dependency_bundle<buffer_access_description<T, downscale_device_specific_mapper>> m_dependencies;
};

} // namespace celerity