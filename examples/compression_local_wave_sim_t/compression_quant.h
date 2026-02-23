#pragma once

#include <celerity.h>
#include <compression.h>

#include <algorithm>


namespace celerity {
template <typename T, template <typename...> class Template>
struct is_specialization_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {};

template <typename, template <auto...> class>
struct is_specialization_of_nttp : std::false_type {};

template <template <auto...> class Template, auto... Args>
struct is_specialization_of_nttp<Template<Args...>, Template> : std::true_type {};

template <typename T>
concept specialization_of_item = is_specialization_of_nttp<T, celerity::item>::value || is_specialization_of_nttp<T, celerity::nd_item>::value;

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
		using compressed_type = Q;
	};
} // namespace compression


namespace compression {
	template <typename T, typename Q>
	struct quantization {
		using value_type = T;
		using compressed_type = Q;
	};

	template <typename T, typename Q>
	struct point_cloud {
		using value_type = T;
		using compression_type = Q;
	};
} // namespace compression

constexpr uint32_t LOCAL_RANGE_X = 8;
constexpr uint32_t LOCAL_RANGE_Y = 8;

template <typename T, typename Q, compression_category Category>
class compressed<celerity::compression::quantization<T, Q>, Category>
    : public compression_tags<compression_category::local_memory | compression_category::global_memory> {
	using compression_type = typename celerity::compression::quantization<T, Q>::compressed_type;
	using compressed_type = typename celerity::compression::quantization<T, Q>::compressed_type;
	using value_type = typename celerity::compression::quantization<T, Q>::value_type;

	using vec_value_type = typename vec_element_type<value_type>::type;
	using vec_compressed_type = typename vec_element_type<compressed_type>::type;
	using vec_compression_type = typename vec_element_type<compressed_type>::type;

  public:
	compressed() : m_lower_bound(0), m_upper_bound(1) {}

	template <typename ValueT>
	compressed(ValueT lower_bound, ValueT upper_bound)
	    : m_lower_bound(lower_bound), m_upper_bound(upper_bound),
	      m_decompression_factor(1 / static_cast<value_type>(std::numeric_limits<compressed_type>::max()) * (m_upper_bound - m_lower_bound)),
	      m_compression_factor((1 / (m_upper_bound - m_lower_bound)) * std::numeric_limits<compressed_type>::max()) {
		static_assert(std::is_same<ValueT, vec_value_type>(), "Value type isn't the same");
		static_assert(is_vec<value_type>::value == is_vec<compressed_type>::value, "Value and Quant type must be both either sycl::vec or fundamental");
	}

	vec_value_type get_upper_bound() const { return m_upper_bound; }
	vec_value_type get_lower_bound() const { return m_lower_bound; }

	void set_upper_bound(vec_value_type upper_bound) { m_upper_bound = upper_bound; }
	void set_lower_bound(vec_value_type lower_bound) { m_lower_bound = lower_bound; }

	compressed_type compress(const value_type number) const {
		if constexpr(is_vec<value_type>::value) {
			compressed_type result;
			for(int i = 0; i < vec_size<compressed_type>::value; ++i) {
				result[i] = static_cast<vec_compressed_type>(
				    ((number[i] - m_lower_bound) / (m_upper_bound - m_lower_bound) * std::numeric_limits<vec_compressed_type>::max()) + 0.5f);
			}

			return result;
		} else {
			return static_cast<compressed_type>(((number - m_lower_bound) * m_compression_factor) + 0.5f);
		}
	}

	value_type decompress(const compressed_type number) const {
		if constexpr(is_vec<value_type>::value) {
			value_type result;

			for(int i = 0; i < vec_size<value_type>::value; ++i) {
				result[i] = static_cast<vec_value_type>(number[i]) / static_cast<vec_value_type>(std::numeric_limits<vec_compressed_type>::max())
				                * (m_upper_bound - m_lower_bound)
				            + m_lower_bound;
			}

			return result;
		} else {
			return (static_cast<value_type>(number) * m_decompression_factor) + m_lower_bound;
		}
	}

	std::vector<compressed_type> compress_data(const value_type* data, const size_t size) {
		std::vector<compressed_type> keep_alive(size);
		std::transform(data, data + size, keep_alive.begin(), [&](const value_type& number) { return compress(number); });
		return std::move(keep_alive);
	}

	template <typename CompressedData, typename UncompressedData>
	void compress_memory_chunk(celerity::nd_item<2> item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc) const {
		// TODO: make this more general for probable different sizes
		auto global_id = item.get_global_id();

		compressed_data_acc[global_id] = compress(uncompressed_data_acc[global_id]);
	}

	template <typename CompressedData, typename UncompressedData>
	void decompress_memory_chunk(celerity::nd_item<2> item, CompressedData& compressed_data_acc, UncompressedData& uncompressed_data_acc) const {
		auto global_id = item.get_global_id();
		auto chunk_size = uncompressed_data_acc.get_range();
		auto allocation_range = uncompressed_data_acc.get_allocation_range();

		// printf("Decompressing chunk at global id: (%ld, %ld), chunk size: (%ld, %ld), allocation range: (%ld, %ld)\n", global_id[0], global_id[1],
		//     chunk_size[0], chunk_size[1], allocation_range[0], allocation_range[1]);

		// TODO: Here we have chunk size as the size to decompress and determine if we need to decompress halo regions according to chunk size. This
		// is probably not the best way to determine this, but it works for now in this specific case. Lets not overthink it right now, but come up with a
		// better solution later.

		if(item.get_local_range(0) < chunk_size[0] || item.get_local_range(1) < chunk_size[1]) {
			auto global_range = item.get_global_range();

			if(item.get_local_id(0) == item.get_local_range(0) - 1) {
				size_t py = global_id[0] < global_range[0] - 1 ? global_id[0] + 1 : global_id[0];
				uncompressed_data_acc[{py, global_id[1]}] = decompress(compressed_data_acc[{py, global_id[1]}]);
			}

			if(item.get_local_id(0) == 0) {
				size_t my = global_id[0] > 0 ? global_id[0] - 1 : global_id[0];
				uncompressed_data_acc[{my, global_id[1]}] = decompress(compressed_data_acc[{my, global_id[1]}]);
			}

			if(item.get_local_id(1) == item.get_local_range(1) - 1) {
				size_t px = global_id[1] < global_range[1] - 1 ? global_id[1] + 1 : global_id[1];
				uncompressed_data_acc[{global_id[0], px}] = decompress(compressed_data_acc[{global_id[0], px}]);
			}

			if(item.get_local_id(1) == 0) {
				size_t mx = global_id[1] > 0 ? global_id[1] - 1 : global_id[1];
				uncompressed_data_acc[{global_id[0], mx}] = decompress(compressed_data_acc[{global_id[0], mx}]);
			}
		}

		uncompressed_data_acc[global_id] = decompress(compressed_data_acc[global_id]);
	}

	template <typename CompressedData, typename UncompressedData>
	void decompress(CompressedData& compressed_data, UncompressedData& uncompressed_data, const size_t width, const size_t height) const {
		for(size_t i = 0; i < width; i++) {
			for(size_t j = 0; j < height; j++) {
				uncompressed_data[(i * height) + j] = decompress(compressed_data[{i, j}]);
			}
		}
	}

  private:
	vec_value_type m_lower_bound;
	vec_value_type m_upper_bound;
	vec_value_type m_decompression_factor;
	vec_value_type m_compression_factor;
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

struct empty {};

template <typename ChunkDataState, typename WorkgroupCompressionState>
struct global_compression_state_tracker {
  public:
	global_compression_state_tracker(ChunkDataState& chunk_acc, WorkgroupCompressionState& local_acc, size_t device_offset)
	    : m_chunk_data_state_counter_global_acc(chunk_acc), m_compression_state_local_acc(local_acc), m_device_offset(device_offset) {}

	template <specialization_of_item Item>
	void try_get_decompression_lock(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& atomic_ref_count) {
			// printf("d");
			while(status == compressing) {
				// if(item.get_local_id(0) == 0 && item.get_local_id(1) == 0) { printf("dead lock item %ld %ld\n", item.get_global_id(0),
				// item.get_global_id(1)); }
				int32_t combined_status = atomic_ref_count.load();

				auto [separate_status, separate_count] = separate_status_atomic(combined_status);

				status = separate_status;
				count = separate_count;
			}

			count++;

			if(status == is_compressed && count == 1) {
				status = decompressing;
				m_compression_state_local_acc[0] = decompressing;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_is_decompressed(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			// printf("e");
			if(status == decompressing) {
				// printf("es");
				status = is_decompressed;
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_decompressed_no_consumer(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& atomic_ref_count) {
			// printf("a");
			while(status == compressing) {
				int32_t combined_status = atomic_ref_count.load();

				auto [separate_status, separate_count] = separate_status_atomic(combined_status);

				status = separate_status;
				count = separate_count;
				// printf("item %ld, %ld, status: %d, count: %d, compressing %d\n", item.get_global_id(0), item.get_global_id(1), status, count,
				// compressing);
			}

			count++;

			if(status == is_compressed) {
				status = is_decompressed;
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_get_compression_lock(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			// printf("b");
			count--;

			if(count == 0 && status == is_decompressed) {
				status = compressing;
				m_compression_state_local_acc[0] = compressing;
				// printf("bs");
			} else if(count > 0) {
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_is_compressed(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			// printf("c");
			if(status == compressing) {
				// printf("cs");
				status = is_compressed;
				count = 0;
				m_compression_state_local_acc[0] = is_compressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_compressed_no_producer(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			// printf("f");
			count--;

			if(count == 0 && status == is_decompressed) {
				// printf("fs");
				status = is_compressed;
				m_compression_state_local_acc[0] = is_compressed;
			} else if(count > 0) {
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	bool have_decompressing_lock() const { return m_compression_state_local_acc[0] == decompressing; }
	bool have_compressing_lock() const { return m_compression_state_local_acc[0] == compressing; }

	auto get_local_state() const { return m_compression_state_local_acc[0]; }

  private:
	// TODO: THIS IS NOT CORRECT
	template <specialization_of_item Item, typename Lambda>
	inline void compare_exchange_run(Item& item, std::array<int32_t, 3> offset, Lambda&& func) const {
		auto linearized_idx = celerity::detail::get_linear_index(item.get_global_range() / item.get_local_range(),
		    (item.get_global_id() / item.get_local_range()) + id<2>{static_cast<size_t>(offset[0]), static_cast<size_t>(offset[1])});

		sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
		    m_chunk_data_state_counter_global_acc[{m_device_offset, linearized_idx}]};

		int32_t status = 0;
		int32_t count = 0;

		if(item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
			int32_t combined_status = atomic_ref_count.load();
			int32_t old_m_compression_state = m_compression_state_local_acc[0];

			do {
				auto [separate_status, separate_count] = separate_status_atomic(combined_status);
				m_compression_state_local_acc[0] = old_m_compression_state; // restore local state

				status = separate_status;
				count = separate_count;

				std::forward<Lambda>(func)(status, count, atomic_ref_count);

			} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
		}
	}


	ChunkDataState& m_chunk_data_state_counter_global_acc;
	WorkgroupCompressionState& m_compression_state_local_acc;
	size_t m_device_offset = 0;

	// bit manipulation for status and count combined atomic
	static constexpr int32_t bit_mask = 0b11;
	static constexpr int32_t shift = 2;

	static constexpr int is_compressed = 0;
	static constexpr int decompressing = 1;
	static constexpr int is_decompressed = 2;
	static constexpr int compressing = 3;

	std::pair<int32_t, int32_t> separate_status_atomic(int32_t status_atomic) const { return {status_atomic & bit_mask, status_atomic >> shift}; }
	int32_t combine_status_atomic(int32_t status, int32_t count) const { return (count << shift) | (status & bit_mask); }
};


// template <compression_category Category>
// using StateTracker = std::conditional_t<Category == compression_category::global_memory,
//     global_compression_state_tracker<celerity::accessor<int32_t, 2, celerity::access_mode::discard_read_write, target::device>,
//         celerity::local_accessor<int32_t, 1>>,
//     empty>;

template <typename DataT, int Dim, typename UncompressedData, compression_category Category>
struct uncompressed_container;

template <typename DataT, int Dim, typename UncompressedData>
struct uncompressed_container<DataT, Dim, UncompressedData, compression_category::local_memory> {
  public:
	uncompressed_container(celerity::nd_item<Dim> item, celerity::range<Dim> range, id<Dim> offset, UncompressedData&& data)
	    : m_item(item), m_range(range), m_offset(offset), m_data(std::move(data)) {}

	inline DataT& operator[](const id<Dim>& index) const {
		return m_data[celerity::detail::get_linear_index(m_range, {(index[0] - (m_item.get_group(0) * m_item.get_local_range(0))) + m_offset[0],
		                                                              (index[1] - (m_item.get_group(1) * m_item.get_local_range(1))) + m_offset[1]})];
	}

	celerity::range<Dim> get_range() const { return m_range; }

  private:
	// TODO: Think about making these accessible in compression/decompression object
	celerity::nd_item<Dim> m_item;
	celerity::range<Dim> m_range;
	id<Dim> m_offset;

	UncompressedData m_data;
};

template <typename DataT, int Dim, typename UncompressedData>
struct uncompressed_container<DataT, Dim, UncompressedData, compression_category::global_memory> {
  public:
	uncompressed_container(celerity::nd_item<Dim> item, celerity::range<Dim> range, id<Dim> offset, UncompressedData& data, size_t device_offset = 0)
	    : m_item(item), m_range(range), m_offset(offset), m_device_offset(device_offset), m_data(data) {}

	inline DataT& operator[](const id<Dim>& index) const {
		// TODO: Might work check later again
		auto linear_index = celerity::detail::get_linear_index(m_range, index);
		// linear_index -= m_device_offset * 512; // TODO: Fix hardcoded 128
		// printf("Accessing global memory at linear index: %ld (device offset: %ld)\n", linear_index, m_device_offset);

		return m_data[{m_device_offset, linear_index}];
	}

	celerity::range<Dim> get_range() const { return m_range; }
	celerity::range<Dim> get_allocation_range() const { return m_data.get_allocation_range(); } // TODO: DELETE THIS!!!!

  private:
	// TODO: Think about making these accessible in compression/decompression object
	celerity::nd_item<Dim> m_item;
	celerity::range<Dim> m_range;
	id<Dim> m_offset;
	size_t m_device_offset = 0;

	UncompressedData& m_data;
};

template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData, compression_category Category>
struct decompressed_data_accessor;


template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData>
struct decompressed_data_accessor<AccessMode, DataT, Dim, Compression, Algorithm, CompressedData, UncompressedData, compression_category::local_memory> {
  public:
	decompressed_data_accessor(const Compression<Algorithm, compression_category::local_memory>& compression, CompressedData& compressed_data_acc,
	    UncompressedData&& uncompressed_data_acc, nd_item<Dim> item, celerity::range<Dim> range, id<Dim> offset)
	    : m_item(item), m_compression(compression), m_uncompressed_data_acc(item, range, offset, std::move(uncompressed_data_acc)),
	      m_compressed_data_acc(compressed_data_acc) {
		if constexpr(detail::is_consumer_mode(AccessMode)) {
			celerity::group_barrier(m_item.get_group());
			m_compression.decompress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc);
			celerity::group_barrier(m_item.get_group());
		}
	}

	~decompressed_data_accessor() {
		if(m_is_moved) return;
		if constexpr(detail::is_producer_mode(AccessMode)) {
			celerity::group_barrier(m_item.get_group());
			m_compression.compress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc);
			celerity::group_barrier(m_item.get_group());
		}
	}

	decompressed_data_accessor& operator=(const decompressed_data_accessor&) = delete;
	decompressed_data_accessor& operator=(decompressed_data_accessor&&) = delete;


	// template <access_mode M = Mode>
	inline DataT& operator[](const id<Dim>& index) const { return m_uncompressed_data_acc[index]; }

	// template <int D = Dim, std::enable_if_t<(D > 0), int> = 0>
	// inline decltype(auto) operator[](const size_t dim0) const {
	// 	if(m_item.get_global_id(1) >= m_item.get_global_range(1)) { return; }
	// 	return subscript_compressed(m_uncompressed_data_acc, dim0, m_item, 0);
	// }

	// default copy constructor
	decompressed_data_accessor(const decompressed_data_accessor&) = delete;

	// make move constructor not compress again
	decompressed_data_accessor(decompressed_data_accessor&& other) noexcept
	    : m_item(other.m_item), m_compression(other.m_compression), m_uncompressed_data_acc(other.m_uncompressed_data_acc),
	      m_compressed_data_acc(other.m_compressed_data_acc) {
		other.m_is_moved = true;
	}

  private:
	celerity::nd_item<Dim> m_item;
	const Compression<Algorithm, compression_category::local_memory>& m_compression;
	uncompressed_container<DataT, Dim, UncompressedData, compression_category::local_memory> m_uncompressed_data_acc;
	CompressedData& m_compressed_data_acc;
	bool m_is_moved = false; // Avoid compression in the moved-from object
};

template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData>
struct decompressed_data_accessor<AccessMode, DataT, Dim, Compression, Algorithm, CompressedData, UncompressedData, compression_category::global_memory> {
  public:
	decompressed_data_accessor(const Compression<Algorithm, compression_category::global_memory>& compression, CompressedData& compressed_data_acc,
	    UncompressedData& uncompressed_data_acc, int device_offset, nd_item<Dim> item, celerity::range<Dim> range, id<Dim> offset,
	    global_compression_state_tracker<celerity::accessor<int32_t, Dim, celerity::access_mode::discard_read_write, target::device>,
	        celerity::local_accessor<int32_t, 1>>
	        state_tracker)
	    : m_item(item), m_compression(compression), m_uncompressed_data_acc(item, range, offset, uncompressed_data_acc, device_offset),
	      m_compressed_data_acc(compressed_data_acc), m_state_tracker(state_tracker) {
		if constexpr(detail::is_consumer_mode(AccessMode)) {
			// printf("GLOBAL MEMORY CONSUMER CONSTRUCTOR\n");
			// if(item.get_global_id(0) == 0 && item.get_global_id(1) == 0 && item.get_global_id(2) == 0) {
			// 	printf("Decompressing tile (%d, %d) in global memory\n", item.get_global_id(0) + m_x, item.get_global_id(1) + m_y);
			// }
			if(item.get_global_id(0) + 0 < item.get_global_range(0) && item.get_global_id(1) + 0 < item.get_global_range(1)) {
				state_tracker.try_get_decompression_lock(item, {0, 0});
			}

			celerity::group_barrier(item.get_group());

			if(state_tracker.have_decompressing_lock()) {
				// TODO: items
				m_compression.decompress_memory_chunk(item, m_compressed_data_acc, m_uncompressed_data_acc);
			}

			celerity::group_barrier(item.get_group());

			if(item.get_global_id(0) + 0 < item.get_global_range(0) && item.get_global_id(1) + 0 < item.get_global_range(1)) {
				state_tracker.try_set_is_decompressed(item, {0, 0});
			}
		} else {
			// printf("GLOBAL MEMORY PRODUCER CONSTRUCTOR\n");
			celerity::group_barrier(item.get_group());
			if(item.get_global_id(0) + 0 < item.get_global_range(0) && item.get_global_id(1) + 0 < item.get_global_range(1)) {
				state_tracker.try_set_decompressed_no_consumer(item, {0, 0});
			}
			celerity::group_barrier(item.get_group());
		}
	}

	~decompressed_data_accessor() {
		if(m_is_moved) return;
		if constexpr(detail::is_producer_mode(AccessMode)) {
			// printf("GLOBAL MEMORY PRODUCER DESTRUCTOR\n");
			// if(m_item.get_global_id(0) == 0 && m_item.get_global_id(1) == 0 && m_item.get_global_id(2) == 0) {
			// 	printf("Compressing tile (%d, %d) in global memory\n", m_item.get_global_id(0) + m_x, m_item.get_global_id(1) + m_y);
			// }
			if(m_item.get_global_id(0) + 0 < m_item.get_global_range(0) && m_item.get_global_id(1) + 0 < m_item.get_global_range(1)) {
				m_state_tracker.try_get_compression_lock(m_item, {0, 0});
			}

			celerity::group_barrier(m_item.get_group());

			if(m_state_tracker.have_compressing_lock()) {
				// TODO: items
				m_compression.compress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc);
			}

			celerity::group_barrier(m_item.get_group());

			if(m_item.get_global_id(0) + 0 < m_item.get_global_range(0) && m_item.get_global_id(1) + 0 < m_item.get_global_range(1)) {
				m_state_tracker.try_set_is_compressed(m_item, {0, 0});
			}
			celerity::group_barrier(m_item.get_group());
		} else {
			// printf("GLOBAL MEMORY CONSUMER DESTRUCTOR\n");
			celerity::group_barrier(m_item.get_group());
			if(m_item.get_global_id(0) + 0 < m_item.get_global_range(0) && m_item.get_global_id(1) + 0 < m_item.get_global_range(1)) {
				m_state_tracker.try_set_compressed_no_producer(m_item, {0, 0});
			}
			celerity::group_barrier(m_item.get_group());
		}

		// print every global_compression_state_tracker_state which is compressing

		// if(m_state_tracker.get_local_state()) {
		// 	printf("Final state tile (%d, %d): %d\n", m_item.get_global_id(0), m_item.get_global_id(1), m_state_tracker.get_local_state());
		// }
	}

	decompressed_data_accessor& operator=(const decompressed_data_accessor&) = delete;
	decompressed_data_accessor& operator=(decompressed_data_accessor&&) = delete;


	// template <access_mode M = Mode>
	inline DataT& operator[](const id<Dim>& index) const {
		// TODO: Might work check later again
		// return m_uncompressed_data_acc[celerity::detail::get_linear_index({m_item.get_global_range(0), m_item.get_global_range(1)}, index)];
		// auto linear_index = celerity::detail::get_linear_index({m_item.get_global_range(0), m_item.get_global_range(1)}, index);
		// if(linear_index >= m_range[0] * m_range[1]) {
		// 	printf("Error: Accessing out of bounds index %ld (max %ld)\n", linear_index, m_range[0] * m_range[1] - 1);
		// 	printf("accessing index (%ld, %ld) to linear index %ld\n", index[0], index[1], linear_index);
		// }
		// return m_uncompressed_data_acc[linear_index];
		return m_uncompressed_data_acc[index];
	}

	// template <int D = Dim, std::enable_if_t<(D > 0), int> = 0>
	// inline decltype(auto) operator[](const size_t dim0) const {
	// 	if(m_item.get_global_id(1) >= m_item.get_global_range(1)) { return; }
	// 	return subscript_compressed(m_uncompressed_data_acc, dim0, m_item, 0);
	// }

	// default copy constructor
	decompressed_data_accessor(const decompressed_data_accessor&) = delete;

	// make move constructor not compress again
	decompressed_data_accessor(decompressed_data_accessor&& other) noexcept
	    : m_item(other.m_item), m_compression(other.m_compression), m_uncompressed_data_acc(other.m_uncompressed_data_acc),
	      m_compressed_data_acc(other.m_compressed_data_acc), m_state_tracker(other.m_state_tracker) {
		other.m_is_moved = true;
	}

  private:
	celerity::nd_item<Dim> m_item;
	const Compression<Algorithm, compression_category::global_memory>& m_compression;
	uncompressed_container<DataT, Dim, UncompressedData, compression_category::global_memory> m_uncompressed_data_acc;
	CompressedData& m_compressed_data_acc;

	bool m_is_moved = false; // Avoid compression in the moved-from object

	// range<Dim> m_range;
	// id<Dim> m_offset;

	global_compression_state_tracker<celerity::accessor<int32_t, 2, celerity::access_mode::discard_read_write, target::device>,
	    celerity::local_accessor<int32_t, 1>>
	    m_state_tracker;
};

template <typename DataT, int Dims, typename Intype>
class buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>, compression_category::element_wise>>
    : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;

	buffer(const Intype* data, range<Dims> range, compressed<compression, compression_category::element_wise>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression) {
		std::cout << "Element-wise compressed buffer created with range " << std::endl;
	}

	buffer(range<Dims> range, compressed<compression, compression_category::element_wise>& compression) : base(range), m_compression(compression) {
		std::cout << "Element-wise compressed buffer created with range " << std::endl;
	}

	const compressed<compression, compression_category::element_wise>& get_compression() const { return m_compression; }

	void init(auto& queue) {} // no init needed for element-wise compression

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression, compression_category::element_wise>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression) {}

	std::vector<DataT> m_data;

	compressed<compression, compression_category::element_wise> m_compression;
};

// TMP 2 x 2

// buffer specialization compressed buffer initialization
template <typename DataT, int Dims, typename Intype, compression_category Category>
class buffer<Intype, Dims, compressed<celerity::compression::quantization<Intype, DataT>, Category>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;

	static_assert(
	    check_provided_type_with_algorithm<compressed<compression, Category>>, "Compression algorithm does not satisfy compression_algorithm concept");

	// TODO: Set size according to compression, and change 4 to something dynamic (based on compression...)
	buffer(const Intype* data, range<Dims> range, compressed<compression, Category>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression),
	      m_uncompressed_buffer({2, (range[0]) * (range[1])}),            // FIXME: hardcoded 2 for GPUs
	      m_state_and_count_tracking_buffer({2, (range[0] * range[1])}) { // FIXME: hardcoded 2 for GPUs
		celerity::debug::set_buffer_name(m_uncompressed_buffer, "uncompressed");
		celerity::debug::set_buffer_name(m_state_and_count_tracking_buffer, "tracker");
	}

	buffer(range<Dims> range, compressed<compression, Category>& compression)
	    : base(range), m_compression(compression), m_uncompressed_buffer({2, (range[0]) * (range[1])}), // FIXME: hardcoded 2 for GPUs
	      m_state_and_count_tracking_buffer({2, (range[0] * range[1])}) {                               // FIXME: hardcoded 2 for GPUs
		celerity::debug::set_buffer_name(m_uncompressed_buffer, "uncompressed");
		celerity::debug::set_buffer_name(m_state_and_count_tracking_buffer, "tracker");
	}

	void init(auto& queue) {
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor acc{m_state_and_count_tracking_buffer, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			auto range = m_state_and_count_tracking_buffer.get_range();

			cgh.parallel_for<class init_decompression_interface>(range, [=](celerity::item<2> item) { acc[item] = 0; });
		});

		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor device_local_acc{m_uncompressed_buffer, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			auto range = m_uncompressed_buffer.get_range();

			cgh.parallel_for<class init_uncompressed_buffer>(range, [=](celerity::item<2> item) { device_local_acc[item] = 0; });
		});
	}

	compressed<compression, Category>& get_compression() { return m_compression; }
	celerity::buffer<Intype, 2>& get_uncompressed_buffer() { return m_uncompressed_buffer; }
	celerity::buffer<int32_t, 2>& get_state_and_count_tracking_buffer() { return m_state_and_count_tracking_buffer; }

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression, Category>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression) {}

	std::vector<DataT> m_data;
	compressed<compression, Category> m_compression;

	celerity::buffer<Intype, 2> m_uncompressed_buffer;              // Amount of GPUs, Linearized ND buffer
	celerity::buffer<int32_t, 2> m_state_and_count_tracking_buffer; // Amount of GPUs, Linearized ND buffer
};

template <typename Memory, typename DataT>
struct alloc_chunk {
	alloc_chunk(const Memory& memory, const uint32_t size, const int start, int& current)
	    : m_memory(memory), m_size(size), m_start(start), m_current(current) {}

	// move constructor
	alloc_chunk(alloc_chunk&& other) noexcept : m_memory(other.m_memory), m_size(other.m_size), m_start(other.m_start), m_current(other.m_current) {
		// printf("Move constructor of alloc_chunk\n");
		other.m_start = -1; // invalidate the other chunk to prevent double free
	}

	// move assignment
	alloc_chunk& operator=(alloc_chunk&& other) noexcept {
		if(this != &other) {
			// printf("Move assignment of alloc_chunk\n");
			m_memory = other.m_memory;
			m_size = other.m_size;
			m_start = other.m_start;
			m_current = other.m_current;

			other.m_start = -1; // invalidate the other chunk to prevent double free
		}
		return *this;
	}

	// delete copy constructor
	alloc_chunk(const alloc_chunk&) = delete;
	alloc_chunk& operator=(const alloc_chunk&) = delete;

	~alloc_chunk() {
		if(m_start == -1) { return; } // already moved
		assert(m_current == m_start + m_size && "Something went wrong memory lost");

		if(m_current == m_start + m_size) { m_current = m_start; }
	}

	int get_start() const { return m_start; }

	DataT& operator[](const uint32_t index) const {
		if(m_start == -1) { printf("Accessing moved-from alloc_chunk %d\n", index); }
		assert(m_start != -1 && "Accessing moved-from alloc_chunk");
		assert(index < m_size && "Index out of bounds");
		assert(index >= 0 && "Index out of bounds");
		return m_memory[m_start + index];
	}

  private:
	const Memory& m_memory;
	uint32_t m_size;
	int m_start;
	int& m_current; // we need to modify the current position when the chunk is destroyed
};

template <typename Memory, typename DataT>
struct allocator {
	allocator(const uint32_t size, const Memory& accessor) : m_memory(accessor), m_size(size) {}

	alloc_chunk<Memory, DataT> allocate(const uint32_t size) {
		assert(m_current + size < m_size && "Out of memory");

		int start = m_current;
		m_current += size;

		return {m_memory, size, start, m_current};
	}

  private:
	const Memory& m_memory;
	uint32_t m_size;
	int m_current = 0;
};

template <typename DataT>
auto make_local_allocator(const uint32_t size, const local_accessor<DataT, 1>& accessor) {
	return allocator<local_accessor<DataT, 1>, DataT>(size, accessor);
}

// template <typename DataT, int Dims, typename Intype, access_mode Mode, compression_category Category>
// class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::quantization<Intype, DataT>, Category>>
//     : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
//   public:
// 	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
// 	using compression = celerity::compression::quantization<Intype, DataT>;
// 	using compressed_type = typename compression::compressed_type;
// 	using value_type = typename compression::value_type;

// 	template <typename T, int D, typename Functor, access_mode ModeNoInit>
// 	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn,
// 	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
// 	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()), m_all(LOCAL_MEMORY_SIZE, cgh) {}

// 	template <typename T, int D, typename Functor, access_mode TagMode>
// 	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn,
// 	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
// 	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()), m_all(LOCAL_MEMORY_SIZE, cgh) {}

// 	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
// 	accessor(buffer<DataT, Dims, compressed<compression, Category>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::device> tag,
// 	    const property_list& prop_list)
// 	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()), m_all(LOCAL_MEMORY_SIZE, cgh) {}

// 	inline auto decompress_data(celerity::nd_item<Dims> item, celerity::range<2> range, bool neighborhood = false) const {
// 		// printf("Decompressing data for item (%d, %d) in range (%d, %d) %d\n", item.get_local_id().get(0), item.get_local_id().get(1), range.get(0),
// 		// range.get(1), (item.get_local_range().get(0) + 2) * (item.get_local_range().get(1) + 2));
// 		return local_accessor_compressor<Mode, Intype, Dims, decltype(m_compression), decltype(*this), alloc_chunk<local_accessor<Intype, 1>, Intype>>(
// 		    m_compression, *this, std::move(m_all.allocate((item.get_local_range().get(0) + 2) * (item.get_local_range().get(1) + 2))), item, range,
// 		    neighborhood);
// 	}

// 	// inline auto decompress_data(celerity::nd_item<2> item, celerity::range<Dims> size) const { return decompress_data(item,size); }

//   private:
// 	compressed<compression, Category> m_compression;
// 	allocator<local_accessor<Intype, 1>, Intype> m_all;

// 	static constexpr int LOCAL_MEMORY_SIZE = 264;
// };


// TODO: adapt this to new accessor
template <int MainBufferDims>
struct first_range_mapper_to_ranges {
	first_range_mapper_to_ranges(const celerity::range<MainBufferDims>& buffer_size) : m_buffer_size(buffer_size) {}

	template <int KernelDims, int BufferDims>
	celerity::detail::region<BufferDims> operator()(
	    const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size, const celerity::detail::region<MainBufferDims>& sbr) const {
		celerity::detail::region_builder<BufferDims> builder;

		auto f = sbr.get_boxes();
		for(const auto& box : f) {
			auto min = box.get_min();
			auto max = box.get_max();
			if constexpr(MainBufferDims == 3) {
				const auto start_index = celerity::detail::get_linear_index(m_buffer_size, min);
				const auto end_index = celerity::detail::get_linear_index(m_buffer_size, {max[0] - 1, max[1] - 1, max[2]});

				builder.add({start_index, end_index});
			} else if constexpr(MainBufferDims == 2) {
				const auto start_index = celerity::detail::get_linear_index({m_buffer_size[0], m_buffer_size[1]}, {min[0], min[1]});
				const auto end_index = celerity::detail::get_linear_index({m_buffer_size[0], m_buffer_size[1]}, {max[0] - 1, max[1]});

				builder.add({start_index, end_index});
			} else {
				builder.add({min[0], max[0]});
			}
		}

		return std::move(builder).into_region();
	}

  private:
	const celerity::range<MainBufferDims> m_buffer_size;
};

struct downscale_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(
	    const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size, const celerity::detail::region<MainBufferDims>& sbr) const {
		celerity::detail::region_builder<BufferDims> builder;
		// printf("AAA\n");

		auto f = sbr.get_boxes();
		for(const auto& box : f) {
			auto min = box.get_min();
			auto max = box.get_max();
			for(int d = 0; d < BufferDims; ++d) {
				size_t divisor = 1;
				// In Celerity, id[0] is typically Y (rows), id[1] is X (cols)
				if(d == 0) {
					divisor = LOCAL_RANGE_Y;
				} else if(d == 1) {
					divisor = LOCAL_RANGE_X;
				}

				min[d] /= divisor;
				max[d] = (max[d] + divisor - 1) / divisor;
			}
			builder.add({min, max});
		}

		auto d = std::move(builder).into_region();

		// for(const auto& box : d.get_boxes()) {
		// 	auto min = box.get_min();
		// 	auto max = box.get_max();
		// 	printf("a box: (");
		// 	for(int d = 0; d < MainBufferDims; ++d) {
		// 		printf("%ld", min[d]);
		// 		if(d < MainBufferDims - 1) { printf(", "); }
		// 	}
		// 	printf(") to (");
		// 	for(int d = 0; d < MainBufferDims; ++d) {
		// 		printf("%ld", max[d]);
		// 		if(d < MainBufferDims - 1) { printf(", "); }
		// 	}
		// 	printf(")\n");
		// }


		return d;
	}
};

// range mapper mapping range to device and range mapper (direct)
struct device_specific_range_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(
	    const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size, const celerity::detail::region<MainBufferDims>& sbr) const {
		// calculate chnk ID
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);
		// CELERITY_CRITICAL("Chunk global size {}, range {}, offset {} => linearized id {}", chnk.global_size, chnk.range, chnk.offset, linearized_id);

		auto f = sbr.get_boxes();

		// print boxes
		// for(const auto& box : f) {
		// 	auto min = box.get_min();
		// 	auto max = box.get_max();
		// 	printf("Box: (");
		// 	for(int d = 0; d < MainBufferDims; ++d) {
		// 		printf("%ld", min[d]);
		// 		if(d < MainBufferDims - 1) { printf(", "); }
		// 	}
		// 	printf(") to (");
		// 	for(int d = 0; d < MainBufferDims; ++d) {
		// 		printf("%ld", max[d]);
		// 		if(d < MainBufferDims - 1) { printf(", "); }
		// 	}
		// 	printf(")\n");
		// }

		auto bounding = celerity::detail::bounding_box(f.begin(), f.end());

		// printf bounding
		// printf("Bounding box: (");
		// for(int d = 0; d < MainBufferDims; ++d) {
		// 	printf("%ld", bounding.get_min()[d]);
		// 	if(d < MainBufferDims - 1) { printf(", "); }
		// }
		// printf(") to (");
		// for(int d = 0; d < MainBufferDims; ++d) {
		// 	printf("%ld", bounding.get_max()[d]);
		// 	if(d < MainBufferDims - 1) { printf(", "); }
		// }
		// printf(")\n");

		// auto linearized_offset = linearized_id * chnk.range.size();
		auto linearized_offset = celerity::detail::get_linear_index({chnk.global_size[0], chnk.global_size[1]}, {bounding.get_min()[0], bounding.get_min()[1]});
		auto linearized_offset_max =
		    celerity::detail::get_linear_index({chnk.global_size[0], chnk.global_size[1]}, {bounding.get_max()[0] - 1, bounding.get_max()[1] - 1});
		// print chnk size global and linear offset
		// printf("Chunk global size: (%ld, %ld), linearized offset: %ld to %ld\n", chnk.global_size[0], chnk.global_size[1], linearized_offset,
		//     linearized_offset_max);


		//(bounding.get_min()[0] * chnk.global_size[1]) + bounding.get_min()[1];

		size_t items = bounding.get_area();

		// print area
		// printf("Bounding box area: %ld\n", items);

		// clamp to buffer size
		if(items > buffer_size.size()) { items = buffer_size.size(); }

		auto my_region = celerity::detail::region_builder<BufferDims>();
		my_region.add({{linearized_id, linearized_offset}, {linearized_id + 1, linearized_offset_max + 1}});

		return std::move(my_region).into_region();
	}
};

struct downscale_device_specific_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(
	    const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size, const celerity::detail::region<MainBufferDims>& sbr) const {
		// calculate chnk ID
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);
		// CELERITY_CRITICAL("Chunk global size {}, range {}, offset {} => linearized id {}", chnk.global_size, chnk.range, chnk.offset, linearized_id);

		auto linearized_offset = linearized_id * chnk.range.size() / (LOCAL_RANGE_X * LOCAL_RANGE_Y);

		auto f = sbr.get_boxes();
		auto bounding = celerity::detail::bounding_box(f.begin(), f.end());

		size_t items = bounding.get_area() / (LOCAL_RANGE_X * LOCAL_RANGE_Y);

		// clamp to buffer size
		if(items > buffer_size.size()) { items = buffer_size.size(); }

		auto my_region = celerity::detail::region_builder<BufferDims>();
		my_region.add({{linearized_id, linearized_offset}, {linearized_id + 1, linearized_offset + items}});

		return std::move(my_region).into_region();
	}
};


template <typename RangeMapper, typename RangeToRangeMapper, int MainBufferDims>
struct range_mapper_to_map_ranges {
	range_mapper_to_map_ranges(const RangeMapper& rm, const RangeToRangeMapper& rtm, const celerity::range<MainBufferDims>& buffer_size, bool i)
	    : m_range_mapper(rm), m_range_to_mapper(rtm), m_buffer_size(buffer_size), m_i(i) {}


	// template <int KernelDims, int BufferDims>
	// celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk) const {
	// 	auto intermediate = m_range_mapper(chnk);
	// 	return m_range_to_mapper(chnk, m_buffer_size, intermediate);
	// }

	// TODO: could be more efficient with Reflections
	template <int KernelDims, int BufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		auto intermediate = detail::invoke_range_mapper(m_range_mapper, chnk, m_buffer_size);
		// // print intermediate boxes
		// if(m_i) {
		// auto boxes = intermediate.get_boxes();

		// for(const auto& box : boxes) {
		// 	auto min = box.get_min();
		// 	auto max = box.get_max();
		// 	printf("Intermediate box: (");
		// 	for(int d = 0; d < MainBufferDims; ++d) {
		// 		printf("%ld", min[d]);
		// 		if(d < MainBufferDims - 1) { printf(", "); }
		// 	}
		// 	printf(") to (");
		// 	for(int d = 0; d < MainBufferDims; ++d) {
		// 		printf("%ld", max[d]);
		// 		if(d < MainBufferDims - 1) { printf(", "); }
		// 	}
		// 	printf(")\n");
		// }
		// }
		auto t = m_range_to_mapper(chnk, buffer_size, intermediate);

		auto boxes = t.get_boxes();

		if(m_i) {
			for(const auto& box : boxes) {
				auto min = box.get_min();
				auto max = box.get_max();
				printf("New Intermediate box: (");
				for(int d = 0; d < MainBufferDims; ++d) {
					printf("%ld", min[d]);
					if(d < MainBufferDims - 1) { printf(", "); }
				}
				printf(") to (");
				for(int d = 0; d < MainBufferDims; ++d) {
					printf("%ld", max[d]);
					if(d < MainBufferDims - 1) { printf(", "); }
				}
				printf(")\n");
			}
		}

		return t;
	}

  private:
	const RangeMapper m_range_mapper;
	const RangeToRangeMapper m_range_to_mapper;
	const celerity::range<MainBufferDims> m_buffer_size;
	bool m_i = false;
};

// new accessor divide since it is much more readable
template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::quantization<Intype, DataT>, compression_category::local_memory>>
    : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression_algorithm = celerity::compression::quantization<Intype, DataT>;
	using compressed_type = typename compression_algorithm::compressed_type;
	using value_type = typename compression_algorithm::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression_algorithm, compression_category::local_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression_algorithm, compression_category::local_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression_algorithm, compression_category::local_memory>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	// TODO: Add range mapper to the extend here to get local range + 2 for neighborhood
	inline auto decompress_data(celerity::nd_item<Dims> item, celerity::range<Dims> range, celerity::range<Dims> local_range,
	    allocator<local_accessor<Intype, 1>, Intype>& allocator, celerity::id<Dims> offset) const {
		// if(local_range[0] > item.get_local_range(0) && local_range[1] > item.get_local_range(1)) {
		// 	printf("ff m_range: (%ld, %ld) larger than local range (%ld, %ld)\n", local_range[0], local_range[1], item.get_local_range(0),
		// 	    item.get_local_range(1));
		// }
		// printf("Decompressing data for item (%d, %d) in local range (%d, %d)\n", item.get_local_id().get(0), item.get_local_id().get(1), local_range.get(0),
		//     local_range.get(1));

		return decompressed_data_accessor<Mode, Intype, 2, compressed, compression_algorithm, decltype(*this), alloc_chunk<local_accessor<Intype, 1>, Intype>,
		    compression_category::local_memory>(m_compression, *this, allocator.allocate(local_range[0] * local_range[1]), item, local_range, offset);
	}

	// template <typename DataAccess>
	// inline auto decompress_data(celerity::nd_item<3> item, DataAccess& data_point_available, const Intype min, const int tile_size) const {
	// 	return decompress_data(item, 0, 0, data_point_available, min, tile_size);
	// }

  private:
	compressed<compression_algorithm, compression_category::local_memory> m_compression;
};


template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::quantization<Intype, DataT>, compression_category::global_memory>>
    : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression_algorithm = celerity::compression::quantization<Intype, DataT>;
	using compressed_type = typename compression_algorithm::compressed_type;
	using value_type = typename compression_algorithm::value_type;

	// TODO: ADD range_mapper_to_map_ranges HERE
	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression_algorithm, compression_category::global_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()),
	      m_uncompressed_data_acc(buff.get_uncompressed_buffer(), cgh, range_mapper_to_map_ranges{rmfn, device_specific_range_mapper{}, buff.get_range(), true},
	          celerity::read_write, celerity::no_init),
	      m_state_and_count_data_acc(buff.get_state_and_count_tracking_buffer(), cgh,
	          range_mapper_to_map_ranges{rmfn, downscale_device_specific_mapper{}, buff.get_range(), false}, celerity::read_write, celerity::no_init),
	      m_workgroup_local_flag_acquiring_acc({1}, cgh) {}

	// TODO: ADD range_mapper_to_map_ranges HERE
	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression_algorithm, compression_category::global_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()),
	      m_uncompressed_data_acc(buff.get_uncompressed_buffer(), cgh, range_mapper_to_map_ranges{rmfn, device_specific_range_mapper{}, buff.get_range(), true},
	          celerity::read_write, celerity::no_init),
	      m_state_and_count_data_acc(buff.get_state_and_count_tracking_buffer(), cgh,
	          range_mapper_to_map_ranges{rmfn, downscale_device_specific_mapper{}, buff.get_range(), false}, celerity::read_write, celerity::no_init),
	      m_workgroup_local_flag_acquiring_acc({1}, cgh) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression_algorithm, compression_category::global_memory>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()),
	      m_uncompressed_data_acc(buff.get_uncompressed_buffer(), cgh, celerity::access::all(), celerity::read_write, celerity::no_init),
	      m_state_and_count_data_acc(buff.get_state_and_count_tracking_buffer(), cgh, celerity::access::all(), celerity::read_write, celerity::no_init),
	      m_workgroup_local_flag_acquiring_acc({1}, cgh) {}

	// TODO: FIX THIS
	inline auto decompress_data(celerity::nd_item<2> item, celerity::range<2> range, celerity::range<2> local_range, celerity::id<2> offset) const {
		using UncompressedAccessorType = decltype(m_uncompressed_data_acc);
		// printf(range, local_range, item.get_global_range());
		// printf("Range: (%ld, %ld), Local Range: (%ld, %ld), Global Range: (%ld, %ld)\n", range[0], range[1], local_range[0], local_range[1],
		//     item.get_global_range(0), item.get_global_range(1));
		// CELERITY_CRITICAL("Decompressing data for item ({}, {}) in local range ({}, {})", item.get_local_id().get(0), item.get_local_id().get(1),
		//     local_range.get(0), local_range.get(1));

		// printf("item range: (%ld, %ld), item local_range (%ld, %ld), local range: (%ld, %ld)\n", item.get_global_range(0), item.get_global_range(1),
		//     item.get_local_range(0), item.get_local_range(1), local_range[0], local_range[1]);

		// auto compressed_data_range = this->get_allocation_range();
		auto uncompressed_data_range = m_uncompressed_data_acc.get_allocation_range();

		// if(compressed_data_range[0] < range[0] || compressed_data_range[1] < range[1]) {
		// 	printf("compressed_data_range: (%ld, %ld)\n", compressed_data_range[0], compressed_data_range[1]);
		// }

		// printf("m_uncompressed_data_acc range: (%ld, %ld)\n", uncompressed_data_range[0], uncompressed_data_range[1]);

		// printf("m_uncompressed_data_acc allocation offset: (%ld, %ld)\n", m_uncompressed_data_acc.get_allocation_offset()[0],
		//     m_uncompressed_data_acc.get_allocation_offset()[1]);
		return decompressed_data_accessor<Mode, Intype, 2, compressed, compression_algorithm, decltype(*this), UncompressedAccessorType,
		    compression_category::global_memory>(m_compression, *this, m_uncompressed_data_acc, m_uncompressed_data_acc.get_allocation_offset()[0], item,
		    local_range, offset, {m_state_and_count_data_acc, m_workgroup_local_flag_acquiring_acc, m_uncompressed_data_acc.get_allocation_offset()[0]});
	}
	// inline auto decompress_data(celerity::nd_item<2> item, celerity::range<2> range, celerity::range<2> local_range, celerity::id<2> offset) const {
	// 	return decompressed_data_accessor<Mode, Intype, 2, compressed, compression_algorithm, decltype(*this), decltype(m_members.m_local_accessor),
	// 	    compression_category::global_memory>(m_compression, *this, m_members.m_local_accessor, item, local_range, {0, 0},
	// 	    {m_members.m_local_decompression_interface, m_members.m_local_decompression_interface_local});
	// }

	// template <typename DataAccess>
	// inline auto decompress_data(celerity::nd_item<3> item, DataAccess& data_point_available, const Intype min, const int tile_size) const {
	// 	return decompress_data(item, 0, 0, data_point_available, min, tile_size);
	// }

  private:
	compressed<compression_algorithm, compression_category::global_memory> m_compression;


	mutable celerity::accessor<Intype, 2, celerity::access_mode::discard_read_write, target::device> m_uncompressed_data_acc;
	mutable celerity::accessor<int32_t, 2, celerity::access_mode::discard_read_write, target::device> m_state_and_count_data_acc;
	mutable celerity::local_accessor<int32_t, 1> m_workgroup_local_flag_acquiring_acc;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode, compression_category Category>
class accessor<DataT, Dims, Mode, target::host_task, compressed<celerity::compression::quantization<Intype, DataT>, Category>>
    : public accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed>;
	using compression = celerity::compression::quantization<Intype, DataT>;
	using compressed_type = typename compression::compressed_type;
	using value_type = typename compression::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::host_task> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::host_task> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression, Category>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag, const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	inline auto decompress_data(size_t width, size_t height) const {
		std::vector<Intype> uncompressed_data(width * height);
		m_compression.decompress(*this, uncompressed_data, width, height);
		return std::move(uncompressed_data);
	}

  private:
	compressed<compression, Category> m_compression;
};
} // namespace celerity