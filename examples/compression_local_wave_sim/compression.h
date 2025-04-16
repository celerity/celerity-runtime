#pragma once

#include <celerity.h>

#include <algorithm>

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

namespace compression {
	template <typename T, typename Q>
	struct conversion {
		using value_type = T;
		using quant_type = Q;
	};
} // namespace compression

template <typename T, typename Q>
class compressed<celerity::compression::quantization<T, Q>> {
	using compression_type = typename celerity::compression::quantization<T, Q>::quant_type;
	using quant_type = typename celerity::compression::quantization<T, Q>::quant_type;
	using value_type = typename celerity::compression::quantization<T, Q>::value_type;

	using vec_value_type = typename vec_element_type<value_type>::type;
	using vec_quant_type = typename vec_element_type<quant_type>::type;
	using vec_compression_type = typename vec_element_type<quant_type>::type;

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
		std::transform(data, data + size, keep_alive.begin(), [&](const value_type& number) { return compress(number); });
		return std::move(keep_alive);
	}

	template <typename CompressedPoints, typename UncompressedData>
	void compress(
	    celerity::nd_item<2> item, const UncompressedData& local_tile_point_acc, CompressedPoints& tile_point_acc, range<2> range, bool neighborhood) const {
		celerity::group_barrier(item.get_group());
		if(item.get_global_id(0) < range[0] && item.get_global_id(1) < range[1]) {
			tile_point_acc[item.get_global_id()] = compress(local_tile_point_acc[celerity::detail::get_linear_index(
			    {item.get_local_range(0) + 2, item.get_local_range(1) + 2}, {item.get_local_id(0) + 1, item.get_local_id(1) + 1})]);
		}

		celerity::group_barrier(item.get_group());
	}

	template <typename GlobalPoints, typename LocalPoints>
	void decompress_memory_chunk(
	    celerity::nd_item<2>& item, GlobalPoints& tile_point_acc_last_dim, LocalPoints& local_tile_point_acc, bool neighborhood) const {
		celerity::id<2> local_id = {item.get_local_id(0) + 1, item.get_local_id(1) + 1};
		celerity::range<2> local_range = {item.get_local_range(1) + 2, item.get_local_range(1) + 2};
		auto global_id = item.get_global_id();

		if(neighborhood) {
			auto global_range = item.get_global_range();

			if(item.get_local_id(0) == item.get_local_range(0) - 1) {
				size_t py = global_id[0] < global_range[0] - 1 ? global_id[0] + 1 : global_id[0];
				local_tile_point_acc[celerity::detail::get_linear_index(local_range, {local_id[0] + 1, local_id[1]})] =
				    decompress(tile_point_acc_last_dim[{py, global_id[1]}]);
			}

			if(item.get_local_id(0) == 0) {
				size_t my = global_id[0] > 0 ? global_id[0] - 1 : global_id[0];
				local_tile_point_acc[celerity::detail::get_linear_index(local_range, {local_id[0] - 1, local_id[1]})] =
				    decompress(tile_point_acc_last_dim[{my, global_id[1]}]);
			}

			if(item.get_local_id(1) == item.get_local_range(1) - 1) {
				size_t px = global_id[1] < global_range[1] - 1 ? global_id[1] + 1 : global_id[1];
				local_tile_point_acc[celerity::detail::get_linear_index(local_range, {local_id[0], local_id[1] + 1})] =
				    decompress(tile_point_acc_last_dim[{global_id[0], px}]);
			}

			if(item.get_local_id(1) == 0) {
				size_t mx = global_id[1] > 0 ? global_id[1] - 1 : global_id[1];
				local_tile_point_acc[celerity::detail::get_linear_index(local_range, {local_id[0], local_id[1] - 1})] =
				    decompress(tile_point_acc_last_dim[{global_id[0], mx}]);
			}
		}

		local_tile_point_acc[celerity::detail::get_linear_index(local_range, local_id)] = decompress(tile_point_acc_last_dim[global_id]);
	}


	template <typename CompressedData, typename UncompressedData>
	void decompress(
	    celerity::nd_item<2> item, CompressedData& compressed_data_acc, UncompressedData& uncompressed_data_acc, range<2> range, bool neighborhood) const {
		celerity::group_barrier(item.get_group());
		if(item.get_global_id(0) < range[0] && item.get_global_id(1) < range[1]) {
			decompress_memory_chunk(item, compressed_data_acc, uncompressed_data_acc, neighborhood);
		}
		celerity::group_barrier(item.get_group());
	}

	template <typename CompressedData, typename UncompressedData>
	void decompress(CompressedData& compressed_data, UncompressedData& uncompressed_data, const size_t width, const size_t height) const {
		for(size_t i = 0; i < width; i++) {
			for(size_t j = 0; j < height; j++) {
				uncompressed_data[i * height + j] = decompress(compressed_data[{i, j}]);
			}
		}
	}

  private:
	vec_value_type m_lower_bound;
	vec_value_type m_upper_bound;
};

template <int TargetDims, typename Target, int SubscriptDim = 0>
class subscript_proxy_compressed;

template <int TargetDims, typename Target, int SubscriptDim>
inline decltype(auto) subscript_compressed(Target& tgt, id<TargetDims> id, const size_t index, nd_item<TargetDims> item, const int const_offset) {
	static_assert(SubscriptDim < TargetDims);
	id[SubscriptDim] = index - item.get_global_id().get(SubscriptDim) + const_offset;
	if constexpr(SubscriptDim == TargetDims - 1) {
		return tgt[std::as_const(id[2])];
	} else {
		return subscript_proxy_compressed<TargetDims, Target, SubscriptDim + 1>{tgt, id, item, const_offset};
	}
}

template <int TargetDims, typename Target>
inline decltype(auto) subscript_compressed(Target& tgt, const size_t index, nd_item<TargetDims> item, const int const_offset) {
	return subscript_compressed<TargetDims, Target, 0>(tgt, id<TargetDims>{}, index);
}

template <int TargetDims, typename Target, int SubscriptDim>
class subscript_proxy_compressed {
  public:
	subscript_proxy_compressed(Target& tgt, const id<TargetDims> id, nd_item<TargetDims> item, const int const_offset)
	    : m_tgt(tgt), m_id(id), m_item(item), m_const_offset(const_offset) {}

	inline decltype(auto) operator[](const size_t index) const { //
		return subscript_compressed<TargetDims, Target, SubscriptDim>(m_tgt, m_id, index, m_item, m_const_offset);
	}

  private:
	Target& m_tgt;
	id<TargetDims> m_id{};
	nd_item<TargetDims> m_item;
	const int m_const_offset;
};

template <access_mode AccessMode, typename DataT, int Dim, typename Compression, typename CompressedData, typename UncompressedData>
struct local_accessor_compressor {
	local_accessor_compressor(const Compression& compression, CompressedData& compressed_data_acc, UncompressedData uncompressed_data_acc, nd_item<Dim> item,
	    range<2> range, bool neighborhood)
	    : m_item(item), m_compression(compression), m_local_tile(uncompressed_data_acc), m_compressed_data(compressed_data_acc), m_range(range),
	      m_neighborhood(neighborhood) {
		if constexpr(detail::is_consumer_mode(AccessMode)) { m_compression.decompress(item, compressed_data_acc, uncompressed_data_acc, range, neighborhood); }
	}

	~local_accessor_compressor() {
		if constexpr(detail::is_producer_mode(AccessMode)) { m_compression.compress(m_item, m_local_tile, m_compressed_data, m_range, m_neighborhood); }
	}

	local_accessor_compressor& operator=(const local_accessor_compressor&) = delete;
	local_accessor_compressor& operator=(local_accessor_compressor&&) = delete;


	// template <access_mode M = Mode>
	inline DataT& operator[](const id<Dim>& index) const {
		return m_local_tile[celerity::detail::get_linear_index({m_item.get_local_range(0) + 2, m_item.get_local_range(1) + 2},
		    {(index[0] - m_item.get_group(0) * m_item.get_local_range(0)) + 1, (index[1] - m_item.get_group(1) * m_item.get_local_range(1)) + 1})];
	}

	// template <int D = Dim, std::enable_if_t<(D > 0), int> = 0>
	// inline decltype(auto) operator[](const size_t dim0) const {
	// 	return subscript_compressed(m_local_tile, dim0, m_item, 0);
	// }

	// default copy constructor
	local_accessor_compressor(const local_accessor_compressor&) = default;
	// default move constructor
	local_accessor_compressor(local_accessor_compressor&&) = default;

  private:
	celerity::nd_item<Dim> m_item;
	const Compression& m_compression;
	UncompressedData m_local_tile;
	CompressedData& m_compressed_data;
	range<2> m_range;
	bool m_neighborhood = false;
};

// buffer specialization compressed buffer initialization
template <typename DataT, int Dims, typename Intype>
class buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;

	buffer(const Intype* data, range<Dims> range, compressed<compression>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression) {}

	buffer(range<Dims> range, compressed<compression>& compression) : base(range), m_compression(compression) {}

	const compressed<compression>& get_compression() const { return m_compression; }

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression) {}

	std::vector<DataT> m_data;

	compressed<compression> m_compression;
};

template <typename Memory, typename DataT>
struct alloc_chunk {
	alloc_chunk(const Memory& memory, const size_t size, const size_t start, int& current)
	    : m_memory(memory), m_size(size), m_start(start), m_current(current) {}

	~alloc_chunk() {
		assert(m_current == m_start + m_size && "Something went wrong memory lost");

		if(m_current == m_start + m_size) { m_current = m_start; }
	}

	DataT& operator[](const size_t index) const {
		assert(index < m_size && "Index out of bounds");
		return m_memory[m_start + index];
	}

  private:
	const Memory& m_memory;
	const size_t m_size;
	const size_t m_start;
	int& m_current;
};

template <typename Memory, typename DataT>
struct allocator {
	allocator(const size_t size, handler& cgh) : m_memory(size, cgh), m_size(size), m_current(0) {}

	alloc_chunk<Memory, DataT> allocate(const size_t size) const {
		assert(m_current + size < m_size && "Out of memory");

		size_t start = m_current;
		m_current += size;

		return {m_memory, size, start, m_current};
	}

  private:
	const Memory m_memory;
	const size_t m_size;
	mutable int m_current;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::quantization<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;
	using quant_type = typename compression::quant_type;
	using value_type = typename compression::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()), m_all(LOCAL_MEMORY_SIZE, cgh) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::device> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()), m_all(LOCAL_MEMORY_SIZE, cgh) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::device> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()), m_all(LOCAL_MEMORY_SIZE, cgh) {}

	inline auto decompress_data(celerity::nd_item<Dims> item, celerity::range<2> range, bool neighborhood = false) const {
		return local_accessor_compressor<Mode, Intype, Dims, decltype(m_compression), decltype(*this), alloc_chunk<local_accessor<Intype, 1>, Intype>>(
		    m_compression, *this, m_all.allocate((item.get_local_range().get(0) + 2) * (item.get_local_range().get(1) + 2)), item, range, neighborhood);
	}

	// inline auto decompress_data(celerity::nd_item<2> item, celerity::range<Dims> size) const { return decompress_data(item,size); }

  private:
	compressed<compression> m_compression;
	allocator<local_accessor<Intype, 1>, Intype> m_all;

	static constexpr int LOCAL_MEMORY_SIZE = 400;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::host_task, compressed<celerity::compression::quantization<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;
	using quant_type = typename compression::quant_type;
	using value_type = typename compression::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, target::host_task> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::host_task> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	inline auto decompress_data(size_t width, size_t height) const {
		std::vector<Intype> uncompressed_data(width * height);
		m_compression.decompress(*this, uncompressed_data, width, height);
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