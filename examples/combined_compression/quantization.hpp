#pragma once

#include <celerity.h>

#include "./compression_utils.hpp"
#include "umuguc_types.hpp"
// #include "./direct_compression.hpp"

namespace celerity {
template <typename T, typename Q, compression_category Category>
class compressed<celerity::compression::quantization<T, Q>, Category> {
  private:
	using quant_type = typename celerity::compression::quantization<T, Q>::quant_type;
	using value_type = typename celerity::compression::quantization<T, Q>::value_type;

	using vec_value_type = typename vec_element_type<value_type>::type;
	using vec_quant_type = typename vec_element_type<quant_type>::type;

  public:
	compressed() : m_lower_bound(0), m_upper_bound(1) {}

	template <typename ValueT>
	compressed(ValueT lower_bound, ValueT upper_bound) : m_lower_bound(lower_bound), m_upper_bound(upper_bound) {
		static_assert(std::is_same<ValueT, vec_value_type>(), "Value type isn't the same");
		static_assert(is_vec<value_type>::value == is_vec<quant_type>::value, "Value and Quant type must be both either sycl::vec or fundamental");
	}

	vec_value_type get_upper_bound() const { return m_upper_bound; }
	vec_value_type get_lower_bound() const { return m_lower_bound; }

	void set_upper_bound(vec_value_type upper_bound) { m_upper_bound = upper_bound; }
	void set_lower_bound(vec_value_type lower_bound) { m_lower_bound = lower_bound; }

	quant_type compress(const value_type number) const {
		if constexpr(is_vec<value_type>::value) {
			quant_type result;
			for(int i = 0; i < vec_size<quant_type>::value; ++i) {
				result[i] = static_cast<vec_quant_type>(
				    std::round((number[i] - m_lower_bound) / (m_upper_bound - m_lower_bound) * std::numeric_limits<vec_quant_type>::max()));
			}

			return result;
		} else {
			return static_cast<quant_type>(std::round((number - m_lower_bound) / (m_upper_bound - m_lower_bound) * std::numeric_limits<quant_type>::max()));
		}
	}

	value_type decompress(const quant_type number) const {
		if constexpr(is_vec<value_type>::value) {
			value_type result;

			for(int i = 0; i < vec_size<value_type>::value; ++i) {
				result[i] = static_cast<vec_value_type>(number[i]) / static_cast<vec_value_type>(std::numeric_limits<vec_quant_type>::max())
				                * (m_upper_bound - m_lower_bound)
				            + m_lower_bound;
			}

			return result;
		} else {
			return static_cast<value_type>(number) / static_cast<value_type>(std::numeric_limits<quant_type>::max()) * (m_upper_bound - m_lower_bound)
			       + m_lower_bound;
		}
	}

	std::vector<quant_type> compress_data(const value_type* data, const size_t size) {
		std::vector<quant_type> keep_alive(size);

		if constexpr(is_vec<value_type>::value) {
			if(m_upper_bound == m_lower_bound) {
				vec_value_type max = m_upper_bound;
				vec_value_type min = m_lower_bound;

				for(size_t i = 0; i < size; ++i) {
					for(int j = 0; j < vec_size<value_type>::value; ++j) {
						max = std::max(max, data[i][j]);
						min = std::min(min, data[i][j]);
					}
				}

				m_upper_bound = max;
				m_lower_bound = min;
			}
		} else {
			if(m_upper_bound == m_lower_bound) {
				m_upper_bound = *std::max_element(data, data + size);
				m_lower_bound = *std::min_element(data, data + size);
			}
		}

		std::transform(data, data + size, keep_alive.begin(), [&](const value_type& number) { return compress(number); });
		return std::move(keep_alive);
	}

  private:
	vec_value_type m_lower_bound;
	vec_value_type m_upper_bound;
};
} // namespace celerity
