#pragma once

#include <celerity.h>

#include <algorithm>

namespace celerity {
template <typename T, typename C, template <typename, typename> typename SelectedCompression, compression_category Category>
struct uncompressed_wrapper_const {
  public:
	uncompressed_wrapper_const(const C& compressed_ref, const compressed<SelectedCompression<T, C>, Category>& compression)
	    : m_compressed_ref(compressed_ref), m_compression(compression) {}

	operator T() const { return m_compression.decompress(m_compressed_ref); }

  private:
	const C& m_compressed_ref;
	const compressed<SelectedCompression<T, C>, Category>& m_compression;
};

template <typename T, typename C, template <typename, typename> typename SelectedCompression, compression_category Category>
struct uncompressed_wrapper {
  public:
	uncompressed_wrapper(C& compressed_ref, const compressed<SelectedCompression<T, C>, Category>& compression)
	    : m_compressed_ref(compressed_ref), m_compression(compression) {}

	uncompressed_wrapper& operator=(T value) {
		m_compressed_ref = m_compression.compress(value);
		return *this;
	}

	operator T() const { return m_compression.decompress(m_compressed_ref); }
	explicit operator C() const { return m_compressed_ref; }

  private:
	C& m_compressed_ref;
	const compressed<SelectedCompression<T, C>, Category>& m_compression;
};


template <typename DataT, int Dims, typename Intype, template <typename, typename> typename SelectedCompression, compression_category Category>
class buffer<Intype, Dims, compressed<SelectedCompression<Intype, DataT>, Category>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = SelectedCompression<Intype, DataT>;

	buffer(const Intype* data, range<Dims> range, compressed<compression, Category>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression) {}

	buffer(range<Dims> range, compressed<compression, Category>& compression) : base(range), m_compression(compression) {}

	// buffer(buffer&& other) noexcept : base(std::move(other)), m_compression(other.m_compression) {}

	// buffer(const buffer& other) : base(other), m_compression(other.m_compression) {}

	// buffer& operator=(buffer&& other) noexcept {
	// 	base::operator=(std::move(other));
	// 	m_compression = other.m_compression;
	// 	return *this;
	// }

	// buffer& operator=(const buffer& other) {
	// 	base::operator=(other);
	// 	m_compression = other.m_compression;
	// 	return *this;
	// }

	const compressed<compression, Category>& get_compression() const { return m_compression; }

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression, Category>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression) {}

	std::vector<DataT> m_data;

	compressed<SelectedCompression<Intype, DataT>, Category> m_compression;
};


template <typename DataT, int Dims, typename Intype, access_mode Mode, target Target, template <typename, typename> typename SelectedCompression, compression_category Category>
class accessor<DataT, Dims, Mode, Target, compressed<SelectedCompression<Intype, DataT>, Category>>
    : public accessor<DataT, Dims, Mode, Target, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, Target, compression::uncompressed>;
	using compression = SelectedCompression<Intype, DataT>;
	using compressed_type = typename compression::quant_type;
	using value_type = typename compression::value_type;
	using retval = std::conditional_t<detail::is_producer_mode(Mode), uncompressed_wrapper<Intype, DataT, SelectedCompression, Category>,
	    const uncompressed_wrapper_const<Intype, DataT, SelectedCompression, Category>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, Target> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, Target> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}


	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression, Category>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, Target> tag,
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
	inline std::vector<value_type> get_pointer(range<Dims> new_range) const {
		auto* new_buff = base::get_pointer();

		std::vector<value_type> uncompressed_data(new_range.size());
		std::transform(
		    new_buff, new_buff + new_range.size(), uncompressed_data.begin(), [&](const compressed_type& number) { return m_compression.decompress(number); });

		return std::move(uncompressed_data);
	}

  private:
	compressed<compression, Category> m_compression;
};
} // namespace celerity