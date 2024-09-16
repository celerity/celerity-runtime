#pragma once

#include <celerity.h>

#include <algorithm>
#include <numeric>

namespace celerity {
// Primary template for is_vec
template <typename T>
struct is_vec : std::false_type {};

// Specialization for sycl::vec
template <typename T, int N>
struct is_vec<sycl::vec<T, N>> : std::true_type {};

template <typename T>
struct vec_element_type {
	using type = T;
};

template <typename T, int N>
struct vec_element_type<sycl::vec<T, N>> {
	using type = T;
};

template <typename T>
struct vec_size {
	static constexpr int value = 1;
};

template <typename T, int N>
struct vec_size<sycl::vec<T, N>> {
	static constexpr int value = N;
};

template <typename T, typename Q>
class compressed<celerity::compression::quantization<T, Q>> {
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

template <typename T, typename Q>
struct uncompressed_wrapper_const {
  public:
	uncompressed_wrapper_const(const Q& compressed_ref, const compressed<compression::quantization<T, Q>>& compression)
	    : m_compressed_ref(compressed_ref), m_compression(compression) {}

	operator T() const { return m_compression.decompress(m_compressed_ref); }

  private:
	const Q& m_compressed_ref;
	const compressed<compression::quantization<T, Q>>& m_compression;
};

template <typename T, typename Q>
struct uncompressed_wrapper {
  public:
	uncompressed_wrapper(Q& compressed_ref, const compressed<compression::quantization<T, Q>>& compression)
	    : m_compressed_ref(compressed_ref), m_compression(compression) {}

	uncompressed_wrapper& operator=(T value) {
		m_compressed_ref = m_compression.compress(value);
		return *this;
	}

	operator T() const { return m_compression.decompress(m_compressed_ref); }
	explicit operator Q() const { return m_compressed_ref; }

  private:
	Q& m_compressed_ref;
	const compressed<compression::quantization<T, Q>>& m_compression;
};


// buffer specialization compressed buffer initialization
template <typename DataT, int Dims, typename Intype>
class buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;

	buffer(const Intype* data, range<Dims> range, compressed<compression>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression) {}

	buffer(range<Dims> range, compressed<compression>& compression) : base(range), m_compression(compression) {
		assert(m_compression.get_lower_bound() != m_compression.get_upper_bound() && "Lower bound is equal to upper bound");
	}

	// move constructor
	buffer(buffer&& other) noexcept : base(std::move(other)), m_compression(other.m_compression) {}

	// copy constructor
	buffer(const buffer& other) : base(other), m_compression(other.m_compression) {}

	// move assignment
	buffer& operator=(buffer&& other) noexcept {
		base::operator=(std::move(other));
		m_compression = other.m_compression;
		return *this;
	}

	// copy assignment
	buffer& operator=(const buffer& other) {
		base::operator=(other);
		m_compression = other.m_compression;
		return *this;
	}

	const compressed<compression>& get_compression() const { return m_compression; }

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression) {}

	std::vector<DataT> m_data;

	compressed<celerity::compression::quantization<Intype, DataT>> m_compression;
};


template <typename DataT, int Dims, typename Intype, access_mode Mode, target Target>
class accessor<DataT, Dims, Mode, Target, compressed<celerity::compression::quantization<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, Target, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, Target, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;
	using quant_type = typename compression::quant_type;
	using value_type = typename compression::value_type;
	using retval = std::conditional_t<detail::is_producer_mode(Mode), uncompressed_wrapper<Intype, DataT>, const uncompressed_wrapper_const<Intype, DataT>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, Target> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, Target> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}


	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, Target> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	template <access_mode M = Mode>
	inline retval operator[](const id<Dims>& index) const {
		return {base::operator[](index), m_compression};
	}

	template <int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t dim0) const {
		return detail::subscript<Dims>(*this, dim0);
	}

	template <target T = Target, std::enable_if_t<T == target::host_task, int> = 0>
	inline const std::vector<value_type> get_pointer(range<Dims> new_range) const {
		auto* new_buff = base::get_pointer();

		std::vector<value_type> uncompressed_data(new_range.size());
		std::transform(
		    new_buff, new_buff + new_range.size(), uncompressed_data.begin(), [&](const quant_type& number) { return m_compression.decompress(number); });

		return std::move(uncompressed_data);
	}

  private:
	compressed<compression> m_compression;
};

template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target,
    template <typename, typename> typename SelectedCompression>
accessor(const buffer<Intype, Dims, compressed<SelectedCompression<Intype, DataT>>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<Mode, ModeNoInit, Target> tag) -> accessor<DataT, Dims, Mode, Target, compressed<SelectedCompression<Intype, DataT>>>;

template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode TagMode, target Target,
    template <typename, typename> typename SelectedCompression>
accessor(const buffer<Intype, Dims, compressed<SelectedCompression<Intype, DataT>>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<TagMode, Mode, Target> tag,
    const property::no_init& prop) -> accessor<DataT, Dims, Mode, Target, compressed<SelectedCompression<Intype, DataT>>>;

template <typename DataT, int Dims, typename Intype, access_mode TagMode, access_mode TagModeNoInit, target Target,
    template <typename, typename> typename SelectedCompression>
accessor(buffer<Intype, Dims, compressed<SelectedCompression<Intype, DataT>>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, Target> tag,
    const property_list& prop_list) -> accessor<DataT, Dims, TagModeNoInit, Target, compressed<SelectedCompression<Intype, DataT>>>;
} // namespace celerity
