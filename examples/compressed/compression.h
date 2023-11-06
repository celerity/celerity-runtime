#pragma once

#include <celerity.h>

#include <algorithm>
#include <numeric>

namespace celerity {
template <typename T>
class compressed {};

template <typename T, typename Q>
class compressed<celerity::compression::quantization<T, Q>> {
	using quant_type = typename celerity::compression::quantization<T, Q>::quant_type;
	using value_type = typename celerity::compression::quantization<T, Q>::value_type;

  public:
	compressed() : m_upper_bound(0), m_lower_bound(0) {}

	template <typename ValueT>
	compressed(ValueT upper_bound, ValueT lower_bound) : m_upper_bound(upper_bound), m_lower_bound(lower_bound) {
		static_assert(std::is_same<ValueT, value_type>(), "Value type isn't the same");
	}

	value_type get_upper_bound() const { return m_upper_bound; }
	value_type get_lower_bound() const { return m_lower_bound; }

	void set_upper_bound(value_type upper_bound) { m_upper_bound = upper_bound; }
	void set_lower_bound(value_type lower_bound) { m_lower_bound = lower_bound; }

	quant_type compress(const value_type number) const {
		return static_cast<quant_type>(std::round((number - m_lower_bound) / (m_upper_bound - m_lower_bound) * std::numeric_limits<quant_type>::max()));
	}

	value_type decompress(const quant_type number) const {
		return static_cast<value_type>(number) / static_cast<value_type>(std::numeric_limits<quant_type>::max()) * (m_upper_bound - m_lower_bound)
		       + m_lower_bound;
	}

	// compress value_type* to keep_alive
	const std::vector<quant_type> compress_data(const value_type* data, const size_t size) {
		std::vector<quant_type> keep_alive(size);

		// find upper and lower bound
		if(m_upper_bound == m_lower_bound) {
			m_upper_bound = *std::max_element(data, data + size);
			m_lower_bound = *std::min_element(data, data + size);
		}

		std::transform(data, data + size, keep_alive.begin(), [&](const value_type& number) { return compress(number); });

		return std::move(keep_alive);
	}

  private:
	value_type m_upper_bound;
	value_type m_lower_bound;
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
	    : base(compression.compress_data(data, range.size()).data(), range), m_compression(compression) {}

	buffer(range<Dims> range, compressed<compression>& compression) : base(range), m_compression(compression) {}

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
	compressed<celerity::compression::quantization<Intype, DataT>>& m_compression;
};


template <typename DataT, int Dims, typename Intype, access_mode Mode, target Target>
class accessor<DataT, Dims, Mode, Target, compressed<celerity::compression::quantization<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, Target, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, Target, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;
	using quant_type = typename compression::quant_type;
	using value_type = typename compression::value_type;
	using retval = std::conditional_t<detail::access::mode_traits::is_producer(Mode), uncompressed_wrapper<Intype, DataT>,
	    const uncompressed_wrapper_const<Intype, DataT>>;

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

	template <target T = Target, std::enable_if_t<T == target::host_task, int> = 0>
	inline const std::vector<value_type> get_pointer(range<Dims> new_range) const {
		auto* new_buff = base::get_pointer();

		std::vector<value_type> uncompressed_data(new_range.size());
		std::transform(
		    new_buff, new_buff + new_range.size(), uncompressed_data.begin(), [&](const quant_type& number) { return m_compression.decompress(number); });

		return uncompressed_data;
	}

  private:
	compressed<compression> m_compression;
};


template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<Mode, ModeNoInit, Target> tag)
    -> accessor<DataT, Dims, Mode, Target, compressed<celerity::compression::quantization<Intype, DataT>>>;

template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode TagMode, target Target>
accessor(const buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<TagMode, Mode, Target> tag, const property::no_init& prop)
    -> accessor<DataT, Dims, Mode, Target, compressed<celerity::compression::quantization<Intype, DataT>>>;


template <typename DataT, int Dims, typename Intype, access_mode TagMode, access_mode TagModeNoInit, target Target>
accessor(buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>>>& buff, handler& cgh,
    const detail::access_tag<TagMode, TagModeNoInit, Target> tag, const property_list& prop_list)
    -> accessor<DataT, Dims, TagModeNoInit, Target, compressed<celerity::compression::quantization<Intype, DataT>>>;

} // namespace celerity