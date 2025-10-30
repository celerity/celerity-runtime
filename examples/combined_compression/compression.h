#pragma once

#include "umuguc_types.hpp"
#include <celerity.h>

#include <algorithm>


// *****  IDEA section: ****

// When we want to decompress a tile in global memory, we don't need the whole buffer decompressed at once, so we can save there some memory as only a
// part of the buffer is decompressed at a time (everything currently working). As this is only a small part of the buffer, we can get away with
// a smaller allocation of the decompressed buffer and just need some memory management to handle this which was done in the local memory compression case.
// The problem is that we need to know how many workgroups are currently working at the same time such that we can allocate enough memory for the
// decompressed buffer.
// This can be done in multiple ways:
// 1) Figure out a way to know how many workgroups are running at the same time (ask the runtime? ask sycl?)
// 2) Just allocate a fixed number of tiles which should be enough for most cases (e.g. 24)
// 3) Intentionally undersize the decompressed buffer to trade memory for speed (e.g. only allocate memory for 8 tiles, if more workgroups are running at
//      the same time let them wait (similar to let them wait for the compression to finish))
// 4) Let the user decide how many tiles should be allocated (not a good idea as it is not user friendly and not very flexible)
// 5) Other ideas?

// For the local memory compression we could do something like a available local memroy tracker which tracks how much local memory is currently
// used and how much is available. If there is not enough memory available, throw an error, or make a fallback to use global memory comrpession.
// This can be done in multiple ways:
// 1) Track the memory in the runtime (needs changes in runtime)
// 2) Other ideas?

namespace celerity {

template <compression_category Category>
struct compression_tags {
	static constexpr compression_category category = Category;
};

template <typename T>
struct compression_checker {
	static constexpr bool value = false;
};

template <template <typename, typename> typename T, typename C, typename V, compression_category Category>
struct compression_checker<compressed<T<C, V>, Category>> {
	static constexpr bool value = true;
};

template <typename T>
concept compression_algorithm = requires(T a) {
	compression_checker<T>::value;
	{ T::category } -> std::convertible_to<compression_category>;
};

template <typename T>
concept element_wise_category = compression_algorithm<T> && (T::category & compression_category::element_wise) == compression_category::element_wise;

template <typename T>
concept local_memory_category = compression_algorithm<T> && (T::category & compression_category::local_memory) == compression_category::local_memory;

template <typename T>
concept global_memory_category = compression_algorithm<T> && (T::category & compression_category::global_memory) == compression_category::global_memory;

template <typename T>
concept kernel_category = compression_algorithm<T> && (T::category & compression_category::kernel) == compression_category::kernel;

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

template <typename T, typename Q, compression_category Category>
class compressed<celerity::compression::point_cloud<T, Q>, Category>
    : public compression_tags<compression_category::local_memory | compression_category::global_memory> {
  private:
	using compression_type = typename celerity::compression::point_cloud<T, Q>::compression_type;
	using value_type = typename celerity::compression::point_cloud<T, Q>::value_type;

  public:
	compressed() = default;

	inline size_t get_index(celerity::nd_item<3> item, size_t idx, int points_per_tile, celerity::range<3> offset = {0, 0, 0}) const {
		if constexpr(Category == compression_category::local_memory) {
			return idx;
		} else if constexpr(Category == compression_category::global_memory) {
			return celerity::detail::get_linear_index({item.get_global_range(0), item.get_global_range(1), static_cast<size_t>(points_per_tile)},
			    {item.get_global_id(0) + offset[0], item.get_global_id(1) + offset[1], idx});
		} else {
			static_assert(false, "Compression category not supported for this function");
		}
	}

	template <typename CompressedData, typename UncompressedData, typename Point>
	void compress_memory_chunk(celerity::nd_item<3>& item, size_t num_points_per_runner, int points_per_tile, Point& center, CompressedData& tile_point_acc,
	    const UncompressedData& local_tile_point_acc) const {
		// static assert if compressed data is const
		// static_assert(!std::is_const_v<CompressedData>, "Compressed data must not be const");


		for(size_t i = 0; i < num_points_per_runner; i++) {
			size_t idx = i + (item.get_local_id(2) * num_points_per_runner);

			if(idx >= static_cast<size_t>(points_per_tile)) { break; }

			T p = local_tile_point_acc[get_index(item, idx, points_per_tile)];
			p = p - center;

			compression_type compressed_point = {p.x(), p.y(), p.z()};
			tile_point_acc[{item.get_global_id(0), item.get_global_id(1), idx}] = compressed_point;
		}
	}


	template <typename GlobalOffset, typename CompressedPoints, typename DataAvailable, typename UncompressedData>
	void compress(celerity::nd_item<3> item, DataAvailable& data_point_available, const UncompressedData& local_tile_point_acc,
	    GlobalOffset& compression_offset_acc, CompressedPoints& tile_point_acc, int work_items_per_tile, int points_per_tile, T min, int tile_size) const {
		if(item.get_global_id(2) >= static_cast<size_t>(points_per_tile)) { return; }

		int actual_points = data_point_available[{item.get_global_id(0), item.get_global_id(1)}];

		size_t amount = actual_points / work_items_per_tile;
		size_t num_points_per_runner = (amount + 1);

		T center = {min.x() + (tile_size * item.get_global_id(0)), min.y() + (tile_size * item.get_global_id(1)), 0};
		auto center_point = center;

		compression_offset_acc[{item.get_global_id(0), item.get_global_id(1), 0}] = center_point;

		compress_memory_chunk(item, num_points_per_runner, points_per_tile, center_point, tile_point_acc, local_tile_point_acc);
	}

	template <typename CompressedData, typename UncompressedData, typename Point>
	void decompress_memory_chunk(int x, int y, celerity::nd_item<3>& item, int num_points_per_runner, int actual_points, int points_per_tile,
	    Point& centerpoint, CompressedData& tile_point_acc_last_dim, UncompressedData& local_tile_point_acc) const {
		for(int i = 0; i < num_points_per_runner; i++) {
			size_t idx = i + (item.get_global_id(2) * num_points_per_runner);

			if(static_cast<size_t>(actual_points) <= idx) { break; }

			const Q p = tile_point_acc_last_dim[{(item.get_global_id(0) + x), (item.get_global_id(1) + y), idx}];

			T new_p;
			new_p.x() = p.x() + centerpoint.x();
			new_p.y() = p.y() + centerpoint.y();
			new_p.z() = p.z() + centerpoint.z();

			local_tile_point_acc[get_index(item, idx, points_per_tile, {static_cast<size_t>(x), static_cast<size_t>(y), 0})] = new_p;
		}
	}


	template <typename GlobalOffset, typename CompressedData, typename UncompressedData, typename DataAvailable>
	void decompress(GlobalOffset& compression_offset_acc, CompressedData& compressed_data_acc, UncompressedData& uncompressed_data_acc,
	    celerity::nd_item<3> item, const int points_per_tile, const int work_items_per_tile, DataAvailable& data_point_available, const int x,
	    const int y) const {
		int amount = points_per_tile / work_items_per_tile;
		int num_points_per_runner = (amount + 1);

		if(item.get_global_id(0) + x >= 0 && item.get_global_id(0) + x < item.get_global_range(0) && item.get_global_id(1) + y >= 0
		    && item.get_global_id(1) + y < item.get_global_range(1)) {
			int actual_points = data_point_available[{item.get_global_id(0) + x, item.get_global_id(1) + y}];

			const auto& cp = compression_offset_acc[{item.get_global_id(0) + x, item.get_global_id(1) + y, 0}];
			decompress_memory_chunk(x, y, item, num_points_per_runner, actual_points, points_per_tile, cp, compressed_data_acc, uncompressed_data_acc);
		}
	}

	template <typename CompressedData, typename GlobalMetadata, typename GlobalCount, typename UncompressedData>
	void decompress(CompressedData& compressed_data, GlobalMetadata& m_global_offset, GlobalCount& data_point_available, UncompressedData& uncompressed_data,
	    const size_t width, const size_t height) const {
		for(size_t i = 0; i < width; i++) {
			for(size_t j = 0; j < height; j++) {
				int actual_points = data_point_available[{i, j}];
				// printf("Decompressing tile (%zu, %zu) with %d points\n", i, j, actual_points);
				for(size_t k = 0; k < static_cast<size_t>(actual_points); k++) {
					compression_type p = compressed_data[{i, j, k}];
					value_type compression_offset = m_global_offset[{i, j, 0}];

					value_type decompressed_p;
					decompressed_p.x() = p.x() + compression_offset.x();
					decompressed_p.y() = p.y() + compression_offset.y();
					decompressed_p.z() = p.z() + compression_offset.z();

					uncompressed_data[i][j][k] = decompressed_p;
				}
			}
		}
	}
};

template <typename T, typename Q>
class compressed<celerity::compression::point_cloud<T, Q>, compression_category::element_wise> : public compression_tags<compression_category::element_wise> {
  private:
	using compression_type = typename celerity::compression::point_cloud<T, Q>::compression_type;
	using value_type = typename celerity::compression::point_cloud<T, Q>::value_type;

  public:
	compressed(const value_type min, const int tile_size) : m_min(min), m_tile_size(tile_size) {};

	template <typename Item>
	compression_type compress(const value_type p, const Item item) const {
		value_type center = {m_min.x() + m_tile_size * item[0], m_min.y() + m_tile_size * item[1], 0};
		auto tmp = p - center;
		compression_type compressed_point = {tmp.x(), tmp.y(), tmp.z()};
		return compressed_point;
	}

	template <typename Item>
	value_type decompress(const compression_type p, const Item item) const {
		value_type center = {m_min.x() + m_tile_size * item[0], m_min.y() + m_tile_size * item[1], 0};
		value_type decompressed_p;

		decompressed_p.x() = p.x();
		decompressed_p.y() = p.y();
		decompressed_p.z() = p.z();

		decompressed_p += center;
		return decompressed_p;
	}

  private:
	const value_type m_min;
	const int m_tile_size;
};


template <typename T, typename C, int Dims, typename SelectedCompression, compression_category Category>
struct uncompressed_item_wrapper_const {
  public:
	uncompressed_item_wrapper_const(const C& compressed_ref, const id<Dims>& item, const compressed<SelectedCompression, Category>& compression)
	    : m_compressed_ref(compressed_ref), m_item(item), m_compression(compression) {}

	operator T() const { return m_compression.decompress(m_compressed_ref, m_item); }

  private:
	const C& m_compressed_ref;
	const id<Dims>& m_item;
	const compressed<SelectedCompression, Category>& m_compression;
};

template <typename T, typename C, int Dims, typename SelectedCompression, compression_category Category>
struct uncompressed_item_wrapper {
  public:
	uncompressed_item_wrapper(C& compressed_ref, const id<Dims>& item, const compressed<SelectedCompression, Category>& compression)
	    : m_compressed_ref(compressed_ref), m_item(item), m_compression(compression) {}

	uncompressed_item_wrapper& operator=(T value) {
		m_compressed_ref = m_compression.compress(value, m_item);
		return *this;
	}

	operator T() const { return m_compression.decompress(m_compressed_ref, m_item); }
	explicit operator C() const { return m_compressed_ref; }

  private:
	C& m_compressed_ref;
	const id<Dims>& m_item;
	const compressed<SelectedCompression, Category>& m_compression;
};


// TODO: Could be useful also for tracking the local memory usage, just with tracking which chunks are
// currently in local memory decompressed at the same time
// Could definitely reduce the amount of local memory needed if there is a lot of chunks decompressed at the same time.
struct empty {};

template <typename ChunkDataState, typename WorkgroupCompressionState>
struct global_compression_state_tracker {
  public:
	global_compression_state_tracker(ChunkDataState& chunk_acc, WorkgroupCompressionState& local_acc)
	    : m_chunk_data_state_counter_global_acc(chunk_acc), m_compression_state_local_acc(local_acc) {}

	template <specialization_of_item Item>
	void try_get_decompression_lock(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& atomic_ref_count) {
			if(status == compressing) {
				do {
					int32_t combined_status = atomic_ref_count.load();

					auto [separate_status, separate_count] = separate_status_atomic(combined_status);

					status = separate_status;
					count = separate_count;
				} while(status == compressing);
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
			if(status == decompressing) {
				status = is_decompressed;
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_decompressed_no_consumer(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& atomic_ref_count) {
			if(status == compressing) {
				do {
					int32_t combined_status = atomic_ref_count.load();

					auto [separate_status, separate_count] = separate_status_atomic(combined_status);

					status = separate_status;
					count = separate_count;
				} while(status == compressing);
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
			count--;

			if(count == 0 && status == is_decompressed) {
				status = compressing;
				m_compression_state_local_acc[0] = compressing;
			} else if(count > 0) {
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_is_compressed(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			if(status == compressing) {
				status = is_compressed;
				count = 0;
				m_compression_state_local_acc[0] = is_compressed;
			}
		});
	}

	template <specialization_of_item Item>
	void try_set_compressed_no_producer(Item& item, std::array<int32_t, 3> offset) const {
		compare_exchange_run(item, offset, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			count--;

			if(count == 0 && status == is_decompressed) {
				status = is_compressed;
				m_compression_state_local_acc[0] = is_compressed;
			} else if(count > 0) {
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	bool have_decompressing_lock() const { return m_compression_state_local_acc[0] == decompressing; }
	bool have_compressing_lock() const { return m_compression_state_local_acc[0] == compressing; }

  private:
	template <specialization_of_item Item, typename Lambda>
	inline void compare_exchange_run(Item& item, std::array<int32_t, 3> offset, Lambda&& func) const {
		sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
		    m_chunk_data_state_counter_global_acc[{item.get_global_id(0) + offset[0], item.get_global_id(1) + offset[1], 0}]};

		int32_t status = 0;
		int32_t count = 0;

		if(item.get_global_id(2) == 0) {
			int32_t combined_status = atomic_ref_count.load();

			do {
				auto [separate_status, separate_count] = separate_status_atomic(combined_status);

				status = separate_status;
				count = separate_count;

				std::forward<Lambda>(func)(status, count, atomic_ref_count);

			} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
		}
	}


	ChunkDataState& m_chunk_data_state_counter_global_acc;
	WorkgroupCompressionState& m_compression_state_local_acc;

	// bit manipulation for status and count combined atomic
	static constexpr int32_t bit_mask = 0b11;
	static constexpr int32_t shift = 2;

	static constexpr int is_compressed = 0;
	static constexpr int decompressing = 1;
	static constexpr int is_decompressed = 2;
	static constexpr int compressing = 3;

	// TODO: THIS IS HIGHLY EXPERIMENTAL
	std::pair<int32_t, int32_t> separate_status_atomic(int32_t status_atomic) const { return {status_atomic & bit_mask, status_atomic >> shift}; }

	int32_t combine_status_atomic(int32_t status, int32_t count) const { return (count << shift) | (status & bit_mask); }
};

// global memory compression state tracker for global compression state else empty
template <compression_category Category>
using StateTracker = std::conditional_t<Category == compression_category::global_memory,
    global_compression_state_tracker<celerity::accessor<int32_t, 3, celerity::access_mode::discard_read_write, target::device>,
        celerity::local_accessor<int32_t, 1>>,
    empty>;

template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    compression_category Category, typename GlobalOffset, typename CompressedData, typename UncompressedData, typename DataAvailable>
struct local_accessor_compressor {
	local_accessor_compressor(const Compression<Algorithm, Category>& compression, const GlobalOffset& compression_offset_acc,
	    CompressedData& compressed_data_acc, UncompressedData uncompressed_data_acc, nd_item<Dim> item, const int points_per_tile,
	    const int work_items_per_tile, DataAvailable& data_point_available, const int x, const int y, const DataT min, const int tile_size,
	    StateTracker<Category> state_tracker)
	    : m_item(item), m_compression(compression), m_data_point_available(data_point_available), m_local_tile(uncompressed_data_acc),
	      m_compression_offset_acc(compression_offset_acc), m_compressed_data(compressed_data_acc), m_work_items_per_tile(work_items_per_tile),
	      m_points_per_tile(points_per_tile), m_x(x), m_y(y), m_min(min), m_tile_size(tile_size), m_state_tracker(state_tracker) {
		if constexpr(Category == compression_category::global_memory) {
			if constexpr(detail::is_consumer_mode(AccessMode)) {
				// if(item.get_global_id(0) == 0 && item.get_global_id(1) == 0 && item.get_global_id(2) == 0) {
				// 	printf("Decompressing tile (%d, %d) in global memory\n", item.get_global_id(0) + m_x, item.get_global_id(1) + m_y);
				// }
				if(item.get_global_id(0) + m_x < item.get_global_range(0) && item.get_global_id(1) + m_y < item.get_global_range(1)) {
					state_tracker.try_get_decompression_lock(item, {m_x, m_y, 0});
				}

				celerity::group_barrier(item.get_group());

				if(state_tracker.have_decompressing_lock()) {
					m_compression.decompress(compression_offset_acc, compressed_data_acc, uncompressed_data_acc, item, points_per_tile, work_items_per_tile,
					    data_point_available, x, y);
				}

				celerity::group_barrier(item.get_group());

				if(item.get_global_id(0) + m_x < item.get_global_range(0) && item.get_global_id(1) + m_y < item.get_global_range(1)) {
					state_tracker.try_set_is_decompressed(item, {m_x, m_y, 0});
				}
			} else {
				if(item.get_global_id(0) + m_x < item.get_global_range(0) && item.get_global_id(1) + m_y < item.get_global_range(1)) {
					state_tracker.try_set_decompressed_no_consumer(item, {m_x, m_y, 0});
				}
			}
		} else if constexpr(Category == compression_category::local_memory) {
			if constexpr(detail::is_consumer_mode(AccessMode)) {
				celerity::group_barrier(item.get_group());
				m_compression.decompress(
				    compression_offset_acc, compressed_data_acc, uncompressed_data_acc, item, points_per_tile, work_items_per_tile, data_point_available, x, y);
				celerity::group_barrier(item.get_group());
			}
		}
	}

	~local_accessor_compressor() {
		if(m_is_moved) return;
		if constexpr(Category == compression_category::global_memory) {
			if constexpr(detail::is_producer_mode(AccessMode)) {
				// if(m_item.get_global_id(0) == 0 && m_item.get_global_id(1) == 0 && m_item.get_global_id(2) == 0) {
				// 	printf("Compressing tile (%d, %d) in global memory\n", m_item.get_global_id(0) + m_x, m_item.get_global_id(1) + m_y);
				// }
				if(m_item.get_global_id(0) + m_x < m_item.get_global_range(0) && m_item.get_global_id(1) + m_y < m_item.get_global_range(1)) {
					m_state_tracker.try_get_compression_lock(m_item, {m_x, m_y, 0});
				}

				celerity::group_barrier(m_item.get_group());

				if(m_state_tracker.have_compressing_lock()) {
					m_compression.compress(m_item, m_data_point_available, m_local_tile, m_compression_offset_acc, m_compressed_data, m_work_items_per_tile,
					    m_points_per_tile, m_min, m_tile_size);
				}

				celerity::group_barrier(m_item.get_group());

				if(m_item.get_global_id(0) + m_x < m_item.get_global_range(0) && m_item.get_global_id(1) + m_y < m_item.get_global_range(1)) {
					m_state_tracker.try_set_is_compressed(m_item, {m_x, m_y, 0});
				}
			} else {
				if(m_item.get_global_id(0) + m_x < m_item.get_global_range(0) && m_item.get_global_id(1) + m_y < m_item.get_global_range(1)) {
					m_state_tracker.try_set_compressed_no_producer(m_item, {m_x, m_y, 0});
				}
			}
		} else if constexpr(Category == compression_category::local_memory) {
			if constexpr(detail::is_producer_mode(AccessMode)) {
				celerity::group_barrier(m_item.get_group());
				m_compression.compress(m_item, m_data_point_available, m_local_tile, m_compression_offset_acc, m_compressed_data, m_work_items_per_tile,
				    m_points_per_tile, m_min, m_tile_size);
				celerity::group_barrier(m_item.get_group());
			}
		}
	}

	local_accessor_compressor& operator=(const local_accessor_compressor&) = delete;
	local_accessor_compressor& operator=(local_accessor_compressor&&) = delete;


	// template <access_mode M = Mode>
	inline DataT& operator[](const id<Dim>& index) const {
		if constexpr(Category == compression_category::global_memory) {
			return m_local_tile[celerity::detail::get_linear_index({m_item.get_global_range(0), m_item.get_global_range(1), m_points_per_tile}, index)];
		} else if constexpr(Category == compression_category::local_memory) {
			return m_local_tile[index[2]];
		} else {
			static_assert(false, "Compression category not supported for this function");
		}
	}

	template <int D = Dim, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t dim0) const {
		if(m_item.get_global_id(2) >= m_points_per_tile) { return; }
		return subscript_compressed(m_local_tile, dim0, m_item, 0);
	}

	// default copy constructor
	local_accessor_compressor(const local_accessor_compressor&) = delete;

	// make move constructor not compress again
	local_accessor_compressor(local_accessor_compressor&& other) noexcept
	    : m_item(other.m_item), m_compression(other.m_compression), m_data_point_available(other.m_data_point_available), m_local_tile(other.m_local_tile),
	      m_compression_offset_acc(other.m_compression_offset_acc), m_compressed_data(other.m_compressed_data),
	      m_work_items_per_tile(other.m_work_items_per_tile), m_points_per_tile(other.m_points_per_tile), m_x(other.m_x), m_y(other.m_y), m_min(other.m_min),
	      m_tile_size(other.m_tile_size), m_state_tracker(other.m_state_tracker) {
		other.m_is_moved = true;
	}

  private:
	celerity::nd_item<Dim> m_item;
	const Compression<Algorithm, Category>& m_compression;
	DataAvailable& m_data_point_available;
	UncompressedData m_local_tile;
	const GlobalOffset& m_compression_offset_acc;
	CompressedData& m_compressed_data;
	const int m_work_items_per_tile;
	const int m_points_per_tile;
	const int m_x;
	const int m_y;
	const DataT m_min;
	const int m_tile_size;
	bool m_is_moved = false;

	StateTracker<Category> m_state_tracker;
};

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
			if constexpr(KernelDims == 3) {
				const auto start_index = celerity::detail::get_linear_index(m_buffer_size, min);
				const auto end_index = celerity::detail::get_linear_index(m_buffer_size, {max[0] - 1, max[1] - 1, max[2]});

				builder.add({start_index, end_index});
			} else if constexpr(KernelDims == 2) {
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

template <typename RangeMapper, typename RangeToRangeMapper, int MainBufferDims>
struct range_mapper_to_map_ranges {
	range_mapper_to_map_ranges(const RangeMapper& rm, const RangeToRangeMapper& rtm, const celerity::range<MainBufferDims>& buffer_size)
	    : m_range_mapper(rm), m_range_to_mapper(rtm), m_buffer_size(buffer_size) {}

	// TODO: could be extremely more effective with Reflections
	template <int KernelDims, int BufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		auto intermediate = m_range_mapper(chnk, m_buffer_size);
		return m_range_to_mapper(chnk, buffer_size, intermediate);
	}

  private:
	const RangeMapper m_range_mapper;
	const RangeToRangeMapper m_range_to_mapper;
	const celerity::range<MainBufferDims> m_buffer_size;
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

// buffer specialization compressed buffer initialization
template <typename DataT, int Dims, typename Intype, compression_category Category>
class buffer<Intype, Dims, compressed<celerity::compression::point_cloud<Intype, DataT>, Category>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;

	buffer(const Intype* data, range<Dims> range, compressed<compression, Category>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression), m_global_index({range.get(0), range.get(1), 1}),
	      m_device_local_buffer(range.get(2) * range.get(0) * range.get(1)), m_decompression_interface({range.get(0), range.get(1), 1}) {
		celerity::debug::set_buffer_name(m_global_index, "global_index");
	}

	buffer(range<Dims> range, compressed<compression, Category>& compression)
	    : base(range), m_global_index({range.get(0), range.get(1), 1}), m_device_local_buffer((range.get(2) + 2) * (range.get(0) + 2) * (range.get(1) + 2)),
	      m_decompression_interface({range.get(0) + 2, range.get(1) + 2, 1}), m_compression(compression) {
		celerity::debug::set_buffer_name(m_global_index, "global_index");
	}

	// TODO: MAKE THIS GLOBAL MEMORY ONLY
	void init(auto& queue) {
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor acc{m_decompression_interface, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			celerity::accessor global_index_acc{m_global_index, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			celerity::accessor device_local_acc{m_device_local_buffer, cgh, celerity::access::all{}, celerity::write_only, celerity::no_init};
			auto range = m_decompression_interface.get_range();

			cgh.parallel_for<class init_decompression_interface>(m_decompression_interface.get_range(), [=](celerity::item<3> item) {
				acc[item] = 0;
				global_index_acc[item] = 0;
				device_local_acc[celerity::detail::get_linear_index(range, item)] = 0;
			});
		});
	}

	compressed<compression, Category>& get_compression() { return m_compression; }

	celerity::buffer<Intype, Dims>& get_global_index_buffer() { return m_global_index; }
	celerity::buffer<Intype, 1>& get_device_local_buffer() { return m_device_local_buffer; }
	celerity::buffer<int32_t, 3>& get_decompression_interface_buffer() { return m_decompression_interface; }


  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression, Category>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression), m_global_index({range[0], range[1]}) {}

	std::vector<DataT> m_data;

	celerity::buffer<Intype, Dims> m_global_index;
	celerity::buffer<Intype, 1> m_device_local_buffer;
	celerity::buffer<int32_t, 3> m_decompression_interface;

	compressed<compression, Category> m_compression;
};

template <typename DataT, int Dims, typename Intype>
class buffer<Intype, Dims, compressed<celerity::compression::point_cloud<Intype, DataT>, compression_category::element_wise>>
    : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;

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


template <typename Memory, typename DataT>
struct alloc_chunk {
	alloc_chunk(const Memory& memory, const size_t size, const size_t start, size_t& current)
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
	size_t& m_current;
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
	mutable size_t m_current;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::point_cloud<Intype, DataT>, compression_category::element_wise>>
    : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;
	using compressed_type = typename compression::compression_type;
	using value_type = typename compression::value_type;
	using retval =
	    std::conditional_t<detail::is_producer_mode(Mode), uncompressed_item_wrapper<Intype, DataT, Dims, compression, compression_category::element_wise>,
	        const uncompressed_item_wrapper_const<Intype, DataT, Dims, compression, compression_category::element_wise>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression, compression_category::element_wise>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression, compression_category::element_wise>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}


	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression, compression_category::element_wise>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	template <access_mode M = Mode>
	inline retval operator[](const id<Dims>& index) const {
		return { base::operator[](index), index, m_compression };
	}

  private:
	compressed<compression, compression_category::element_wise> m_compression;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::host_task, compressed<celerity::compression::point_cloud<Intype, DataT>, compression_category::element_wise>>
    : public accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;
	using compressed_type = typename compression::compression_type;
	using value_type = typename compression::value_type;
	using retval =
	    std::conditional_t<detail::is_producer_mode(Mode), uncompressed_item_wrapper<Intype, DataT, Dims, compression, compression_category::element_wise>,
	        const uncompressed_item_wrapper_const<Intype, DataT, Dims, compression, compression_category::element_wise>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression, compression_category::element_wise>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::host_task> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression, compression_category::element_wise>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::host_task> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}


	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression, compression_category::element_wise>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag, const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	template <access_mode M = Mode>
	inline retval operator[](const id<Dims>& index) const {
		return { base::operator[](index), index, m_compression };
	}

	inline std::vector<std::vector<std::vector<value_type>>> get_decompressed_data(range<Dims> new_range) const {
		std::vector<std::vector<std::vector<value_type>>> uncompressed_data(
		    new_range.get(0), std::vector<std::vector<value_type>>(new_range.get(1), std::vector<value_type>(new_range.get(2))));
		for(int i = 0; i < new_range.get(0); ++i) {
			for(int j = 0; j < new_range.get(1); ++j) {
				for(int k = 0; k < new_range.get(2); ++k) {
					uncompressed_data[i][j][k] =
					    m_compression.decompress(base::operator[]({static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k)}), id<3>(i, j, k));
				}
			}
		}

		return uncompressed_data;
	}

  private:
	compressed<compression, compression_category::element_wise> m_compression;
};

// TODO: make separate local and global memory specializations
template <typename DataT, int Dims, typename Intype, access_mode Mode, compression_category Category>
class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::point_cloud<Intype, DataT>, Category>>
    : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression_algorithm = celerity::compression::point_cloud<Intype, DataT>;
	using quant_type = typename compression_algorithm::compression_type;
	using value_type = typename compression_algorithm::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression_algorithm, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    requires(Category == compression_category::global_memory)
	    : base(buff, cgh, rmfn, tag), m_global_offset(buff.get_global_index_buffer(), cgh, rmfn, tag), m_compression(buff.get_compression()),
	      m_members({buff.get_device_local_buffer(), cgh, range_mapper_to_map_ranges{rmfn, first_range_mapper_to_ranges<D>{buff.get_range()}, buff.get_range()},
	                    celerity::read_write, celerity::no_init},
	          {buff.get_decompression_interface_buffer(), cgh, rmfn, celerity::read_write, celerity::no_init}, {{1}, cgh}),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression_algorithm, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    requires(Category == compression_category::global_memory)
	    : base(buff, cgh, rmfn, tag, prop), m_global_offset(buff.get_global_index_buffer(), cgh, rmfn, tag, prop), m_compression(buff.get_compression()),
	      m_members({buff.get_device_local_buffer(), cgh, range_mapper_to_map_ranges{rmfn, first_range_mapper_to_ranges<D>{buff.get_range()}, buff.get_range()},
	                    celerity::read_write, celerity::no_init},
	          {buff.get_decompression_interface_buffer(), cgh, rmfn, celerity::read_write, celerity::no_init}, {{1}, cgh}),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression_algorithm, Category>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    requires(Category == compression_category::global_memory)
	    : base(buff, cgh, access::all(), tag, prop_list), m_global_offset(buff.get_global_index_buffer(), cgh, access::all{}, tag, prop_list),
	      m_compression(buff.get_compression()), m_members({buff.get_device_local_buffer(), cgh, access::all{}, read_write, no_init},
	                                                 {buff.get_decompression_interface_buffer(), cgh, access::all{}, read_write, no_init}, {{1}, cgh}),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression_algorithm, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    requires(Category == compression_category::local_memory)
	    : base(buff, cgh, rmfn, tag), m_global_offset(buff.get_global_index_buffer(), cgh, rmfn, tag), m_compression(buff.get_compression()),
	      m_members({1000, cgh}), m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression_algorithm, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    requires(Category == compression_category::local_memory)
	    : base(buff, cgh, rmfn, tag, prop), m_global_offset(buff.get_global_index_buffer(), cgh, rmfn, tag, prop), m_compression(buff.get_compression()),
	      m_members({1000, cgh}), m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression_algorithm, Category>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    requires(Category == compression_category::local_memory)
	    : base(buff, cgh, access::all(), tag, prop_list), m_global_offset(buff.get_global_index_buffer(), cgh, access::all{}, tag, prop_list),
	      m_compression(buff.get_compression()), m_members({1000, cgh}), m_points_per_tile(buff.get_range().get(2)) {}

	template <typename DataAccess>
	inline auto decompress_data(celerity::nd_item<3> item, int x, int y, DataAccess& data_point_available, const Intype min, const int tile_size) const {
		// printf("IN DECOMPRESS DATA\n");
		if constexpr(Category == compression_category::global_memory) {
			return local_accessor_compressor<Mode, Intype, 3, compressed, compression_algorithm, Category, decltype(m_global_offset), decltype(*this),
			    decltype(m_members.m_local_accessor), decltype(data_point_available)>(m_compression, m_global_offset, *this, m_members.m_local_accessor, item,
			    m_points_per_tile, item.get_local_range().get(2), data_point_available, x, y, min, tile_size,
			    {m_members.m_local_decompression_interface, m_members.m_local_decompression_interface_local});
		} else {
			return local_accessor_compressor<Mode, Intype, 3, compressed, compression_algorithm, Category, decltype(m_global_offset), decltype(*this),
			    alloc_chunk<local_accessor<Intype, 1>, Intype>, decltype(data_point_available)>(m_compression, m_global_offset, *this,
			    m_members.m_local_allocator.allocate(data_point_available[{item.get_global_id(0), item.get_global_id(1)}]), item, m_points_per_tile,
			    item.get_local_range().get(2), data_point_available, x, y, min, tile_size, {});
		}
	}

	template <typename DataAccess>
	inline auto decompress_data(celerity::nd_item<3> item, DataAccess& data_point_available, const Intype min, const int tile_size) const {
		return decompress_data(item, 0, 0, data_point_available, min, tile_size);
	}

  private:
	struct global_memory {
		celerity::accessor<Intype, 1, celerity::access_mode::discard_read_write, target::device> m_local_accessor;
		mutable celerity::accessor<int32_t, 3, celerity::access_mode::discard_read_write, target::device> m_local_decompression_interface;
		mutable celerity::local_accessor<int32_t, 1> m_local_decompression_interface_local;
	};

	struct local_memory {
		allocator<local_accessor<Intype, 1>, Intype> m_local_allocator;
	};

	celerity::accessor<Intype, Dims, Mode, target::device> m_global_offset;
	compressed<compression_algorithm, Category> m_compression;

	using Members = std::conditional_t<Category == compression_category::local_memory, local_memory, global_memory>;
	Members m_members;

	const int m_points_per_tile;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode, compression_category Category>
class accessor<DataT, Dims, Mode, target::host_task, compressed<celerity::compression::point_cloud<Intype, DataT>, Category>>
    : public accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;
	using quant_type = typename compression::compression_type;
	using value_type = typename compression::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::host_task> tag)
	    : base(buff, cgh, rmfn, tag), m_global_offset(buff.get_global_index_buffer(), cgh, rmfn, tag), m_compression(buff.get_compression()),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::host_task> tag, const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_global_offset(buff.get_global_index_buffer(), cgh, rmfn, tag, prop), m_compression(buff.get_compression()),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression, Category>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag, const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_global_offset(buff.get_global_index_buffer(), cgh, access::all{}, tag, prop_list),
	      m_compression(buff.get_compression()), m_points_per_tile(buff.get_range().get(2)) {}

	template <typename DataAccess>
	inline auto decompress_data(size_t width, size_t height, DataAccess& data_point_available) const {
		std::vector<std::vector<std::vector<Intype>>> uncompressed_data(
		    width, std::vector<std::vector<Intype>>(height, std::vector<Intype>(m_points_per_tile)));
		m_compression.decompress(*this, m_global_offset, data_point_available, uncompressed_data, width, height);
		return std::move(uncompressed_data);
	}

  private:
	celerity::accessor<Intype, Dims, Mode, target::host_task> m_global_offset;
	compressed<compression, Category> m_compression;
	int m_points_per_tile;
};
} // namespace celerity