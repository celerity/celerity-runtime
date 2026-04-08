#pragma once

#include "ranges.h"
#include "types.h"
#include <celerity.h>
#include <concepts>
#include <optional>
#include <tuple>
#include <unordered_set>

#include <algorithm>


// Compression type implementation for multiple compression categories
// In addition to the basic compress/decompress methods, category specific methods are provided for element-wise, local and global memory compression

// Throughout the file there a bunch of TODOs left over, specific to the parts where they are marked.
// Some overarching todos are:
// [x] Removing dependency of quantization compression. (Just a template parameter)
// [x] Writing some more template code to reduce dimensionality specific code duplication
// [] General code cleanup and documentation ADD TESTs
// [] Add a way to generalize compression algorithm  dependent parameters (like extra buffers and so on)
// [x] Add a way to generalize extra user parameters (like bound values)

// Some more specific todos specific to some locations:
// [x] How to get Range mapper infos during kernel calculation
// [x] How would we get information of the amount of devices
// [x] New device local accessor (-> Design it in such a way that it is replacable by a later implementation of device local accessor?)

// Some Limitations:
// * The current design assumes that the amount of devices divides the global range evenly.
// * No oversubscription allowed.
// * Assumption that range mapper evenly divides the buffer range into partitions of equal size for all devices.
//   (No irregular partitions)
//   Not completely if eg the first x partitions are of size n and the last ones are of size n - 1 due to rounding, is currently also supported.


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

template <typename Derived, typename UncompressedBuff, int Dims, typename... Args>
concept contains_compress = requires(Derived a, UncompressedBuff uncompressed_data, Args&&... args) {
	a.compress(uncompressed_data[celerity::id<Dims>{}], celerity::id<Dims>{}, std::forward<Args>(args)...);
};

template <typename Derived, typename CompressedBuff, int Dims, typename... Args>
concept contains_decompress = requires(Derived a, CompressedBuff compressed_data, Args&&... args) {
	a.decompress(compressed_data[celerity::id<Dims>{}], celerity::id<Dims>{}, std::forward<Args>(args)...);
};

template <typename Derived, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept contains_compress_memory_chunk = requires(Derived a, CompressedBuff compressed_data, UncompressedBuff uncompressed_data, Args&&... args) {
	{ a.compress_memory_chunk(std::declval<celerity::nd_item<Dims>&>(), compressed_data, uncompressed_data, std::forward<Args>(args)...) };
};

template <typename Derived, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept contains_decompress_memory_chunk = requires(Derived a, CompressedBuff compressed_data, UncompressedBuff uncompressed_data, Args&&... args) {
	{ a.decompress_memory_chunk(std::declval<celerity::nd_item<Dims>&>(), compressed_data, uncompressed_data, std::forward<Args>(args)...) };
};

template <typename Derived, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept contains_compress_decompress =
    contains_compress<Derived, UncompressedBuff, Dims, Args...> && contains_decompress<Derived, CompressedBuff, Dims, Args...>;

template <typename Derived, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept contains_compress_decompress_memory_chunk = contains_compress_memory_chunk<Derived, CompressedBuff, UncompressedBuff, Dims, Args...>
                                                    && contains_decompress_memory_chunk<Derived, CompressedBuff, UncompressedBuff, Dims, Args...>;

// template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
// concept satisfies_compression_interface = []() {
// 	if constexpr(detail::contains_same_category<T::category, compression_category::element_wise>) {
// 		// Must have the element-wise compress function
// 		return contains_compress_decompress<T, CompressedBuff, UncompressedBuff, Dims, Args...>;
// 	} else {
// 		// Must have the memory chunk function
// 		return contains_compress_decompress_memory_chunk<T, CompressedBuff, UncompressedBuff, Dims, Args...>;
// 	}
// }();

template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept satisfies_element_wise = detail::contains_same_category<T::category, compression_category::element_wise>
                                 && contains_compress_decompress<T, CompressedBuff, UncompressedBuff, Dims, Args...>;

template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept satisfies_memory_chunk = !detail::contains_same_category<T::category, compression_category::element_wise>
                                 && contains_compress_decompress_memory_chunk<T, CompressedBuff, UncompressedBuff, Dims, Args...>;

template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept satisfies_compression_interface =
    satisfies_element_wise<T, CompressedBuff, UncompressedBuff, Dims, Args...> || satisfies_memory_chunk<T, CompressedBuff, UncompressedBuff, Dims, Args...>;

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

namespace detail {
	// Similar to chunk in ranges.h but for compression purposes.
	// Just contains the necessary information without the additional flavoring of chunk.
	template <int Dims>
	struct compression_chunk {
		static constexpr int dimensions = Dims;

		CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> offset;
		CELERITY_DETAIL_NO_UNIQUE_ADDRESS celerity::range<Dims> range = detail::zeros;
		CELERITY_DETAIL_NO_UNIQUE_ADDRESS celerity::range<Dims> global_size = detail::zeros;

		constexpr compression_chunk() = default;

		constexpr compression_chunk(const id<Dims>& offset, const celerity::range<Dims>& range, const celerity::range<Dims>& global_size)
		    : offset(offset), range(range), global_size(global_size) {}

		friend bool operator==(const compression_chunk& lhs, const compression_chunk& rhs) {
			return lhs.offset == rhs.offset && lhs.range == rhs.range && lhs.global_size == rhs.global_size;
		}
		friend bool operator!=(const compression_chunk& lhs, const compression_chunk& rhs) { return !operator==(lhs, rhs); }
	};
} // namespace detail


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

template <typename Derived, compression_category CompressionCategory>
class compressed_default : public compression_tags<CompressionCategory> {
	static_assert(CompressionCategory != compression_category::none, "Compression category cannot be none");
	// TODO: maybe add some static assert to check if the compress and decompress functions are present in the derived class and if they have the right
	// signature. This might be a bit tricky as we want to allow for different signatures depending on the compression algorithm args. Concepts?
  private:
	template <int Dims>
	inline bool all_true(const celerity::id<Dims>& condition) const {
		if constexpr(Dims == 0) {
			return true;
		} else if constexpr(Dims == 1) {
			return condition[0];
		} else if constexpr(Dims == 2) {
			return condition[0] && condition[1];
		} else if constexpr(Dims == 3) {
			return condition[0] && condition[1] && condition[2];
		} else {
			for(int i = 0; i < Dims; ++i) {
				if(!condition[i]) { return false; }
			}
			return true;
		}
	}


  public:
	// TODO: ARGS are here to be able to pass additional parameters if needed... these might not be present now
	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void compress_memory_chunk(
	    celerity::nd_item<Dims>& item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, Args&&... args) const
	    requires(celerity::detail::contains_same_category<CompressionCategory, compression_category::element_wise>
	             && celerity::detail::contains_same_category<CompressionCategory, compression_category::local_memory | compression_category::global_memory>)
	{
		// static_assert(contains_compress<Derived, UncompressedData, Dims, Args...>, "Derived class must implement compress function");

		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;


		// TODO: (done part) Most range mappers are a rectangle -> for loop (or one-to-one)
		// 		             Special casing for build in neighborhood along axis shape
		//					 -> is the special casing needed? As the memory chunk is loaded anyway (dens)
		//                      and we might introduce more complexity as we need to check for each part separately

		for(size_t i = 0; i < calculations_per_item; ++i) {
			auto idx = (i * amount) + item.get_local_linear_id();
			if(idx >= chunk_range.size()) { break; }

			celerity::id<Dims> local_index = celerity::detail::get_nd_index(idx, chunk_range);
			celerity::id<Dims> global_index = uncompressed_data_acc.get_offset() + local_index;

			// since size_t -1 = uint64_max we do not need to check for underflow
			if(!detail::all_true(global_index < item.get_global_range())) { continue; }

			compressed_data_acc[global_index] =
			    static_cast<const Derived*>(this)->compress(uncompressed_data_acc[global_index], global_index, std::forward<Args>(args)...);
		}
	}

	// TODO: ARGS are here to be able to pass additional parameters if needed... these might not be present now
	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void decompress_memory_chunk(
	    celerity::nd_item<Dims>& item, const CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, Args&&... args) const
	    requires(detail::contains_same_category<CompressionCategory, compression_category::element_wise>
	             && detail::contains_same_category<CompressionCategory, compression_category::local_memory | compression_category::global_memory>)
	{
		// static_assert(contains_decompress<Derived, CompressedData, Dims, Args...>, "Derived class must implement decompress function");

		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;

		// TODO: (done part) Most range mappers are a rectangle -> for loop (or one-to-one)
		// 		             Special casing for build in neighborhood along axis shape
		//					 -> is the special casing needed? As the memory chunk is loaded anyway (dens)
		//                      and we might introduce more complexity as we need to check for each part separately

		// static_assert(false, "This function currently assumes that the global range is evenly divisible by the amount of work items. Otherwise, some items "
		//                      "will do more work than others and we might have to check for out of bounds accesses within the loop.");

		for(size_t i = 0; i < calculations_per_item; ++i) {
			auto idx = (i * amount) + item.get_local_linear_id();
			if(idx >= chunk_range.size()) { break; }

			celerity::id<Dims> local_index = celerity::detail::get_nd_index<Dims>(idx, chunk_range);
			celerity::id<Dims> global_index = uncompressed_data_acc.get_offset() + local_index;

			// since size_t -1 = uint64_max we do not need to check for underflow
			if(!detail::all_true(global_index < item.get_global_range())) { continue; }

			uncompressed_data_acc[global_index] =
			    static_cast<const Derived*>(this)->decompress(compressed_data_acc[global_index], global_index, std::forward<Args>(args)...);

			// if(uncompressed_data_acc.get_offset()[0] == 7 && uncompressed_data_acc.get_offset()[1] == 7) {
			// 	printf("local_index: %ld %ld, global_index: %ld %ld uncompressed_data_acc %lf\n", local_index[0], local_index[1], global_index[0],
			// 	    global_index[1], uncompressed_data_acc[global_index]);
			// }
		}
	}

	// TODO: At the current stage, we only support compress_data and decompress_data for element-wise compression algorithms.
	//       This is because we implemented this for our autogenerated element wise compression.
	//       IS there a way to implement this in a general sense for memory chunk based algorithms?
	template <typename CompressedData, typename UncompressedData, typename... Args>
	void compress_data(const UncompressedData* data, CompressedData& compressed_data, size_t size, Args&&... args) const {
		// This was done with std::transform until we needed the index for compress.
		for(size_t i = 0; i < size; ++i) {
			compressed_data[i] = static_cast<const Derived*>(this)->compress(data[i], celerity::id<1>{i}, std::forward<Args>(args)...);
		}
	}

	// TODO: This is for host decompression. We might want to move this to a separate class???
	//       UNTESTED somewhat tested. No official test case yet.

	// TODO: At the current stage, we only support compress_data and decompress_data for element-wise compression algorithms.
	//       This is because we implemented this for our autogenerated element wise compression.
	//       IS there a way to implement this in a general sense for memory chunk based algorithms?
	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void decompress_data(const CompressedData& compressed_data, UncompressedData& uncompressed_data, const celerity::range<Dims> range, Args&&... args) const {
		if constexpr(Dims == 1) {
			for(size_t i = 0; i < range[0]; i++) {
				uncompressed_data[i] = static_cast<const Derived*>(this)->decompress(compressed_data[i], celerity::id<Dims>{i}, std::forward<Args>(args)...);
			}
		} else if constexpr(Dims == 2) {
			for(size_t i = 0; i < range[0]; i++) {
				for(size_t j = 0; j < range[1]; j++) {
					uncompressed_data[celerity::detail::get_linear_index(range, {i, j})] =
					    static_cast<const Derived*>(this)->decompress(compressed_data[{i, j}], celerity::id<Dims>{i, j}, std::forward<Args>(args)...);
				}
			}
		} else if constexpr(Dims == 3) {
			for(size_t i = 0; i < range[0]; i++) {
				for(size_t j = 0; j < range[1]; j++) {
					for(size_t k = 0; k < range[2]; k++) {
						uncompressed_data[celerity::detail::get_linear_index(range, {i, j, k})] =
						    static_cast<const Derived*>(this)->decompress(compressed_data[{i, j, k}], celerity::id<3>{i, j, k}, std::forward<Args>(args)...);
					}
				}
			}
		} else {
			static_assert(Dims == 1 || Dims == 2 || Dims == 3, "Decompress data is currently only implemented for 1D, 2D, and 3D data");
		}
	}

	template <specialization_of_item Item, int Dim>
	inline celerity::id<1> calculate_tile_tracking_idx(const Item& item, const id<Dim>& offset) const {
		return celerity::detail::get_linear_index(item.get_global_range() / item.get_local_range(), offset / item.get_local_range());
	}
};

struct downscale_device_specific_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		// calculate chnk ID
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);

		auto linearized_offset = linearized_id * chnk.range.size() / chnk.local_range.size(); //(chnk.local_range[0] * chnk.local_range[1]);

		auto f = sbr.get_boxes();
		auto bounding = celerity::detail::bounding_box(f.begin(), f.end());

		size_t items = bounding.get_area() / chnk.local_range.size();

		if(items > buffer_size.size()) { items = buffer_size.size(); }

		auto my_region = celerity::detail::region_builder<BufferDims>();
		my_region.add({{linearized_id, linearized_offset}, {linearized_id + 1, linearized_offset + items}});

		return std::move(my_region).into_region();
	}
};

struct downscale_point_cloud_buffer {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);
		auto linearized_range = chnk.range.size() / chnk.local_range.size();
		auto linearized_offset =
		    celerity::detail::get_linear_index(celerity::detail::range_cast<2>(chnk.global_size), celerity::detail::id_cast<2>(chnk.offset));

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
		// Extract tile coordinates from the chunk offset instead of using item's global ID
		// This ensures correct dependency storage for neighborhood accesses at grid boundaries
		auto chunk_offset = uncompressed_data_acc.get_offset();
		auto tile_x = chunk_offset[0];
		auto tile_y = chunk_offset[1];

		// Bounds check before writing to dependency buffer to avoid out-of-bounds access
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
		// Extract tile coordinates from the chunk offset instead of using item's global ID
		// This ensures correct dependency lookup for neighborhood accesses at grid boundaries
		auto chunk_offset = uncompressed_data_acc.get_offset();
		auto tile_x = chunk_offset[0];
		auto tile_y = chunk_offset[1];

		// Bounds check before accessing dependency buffer to avoid out-of-bounds access
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
		// THIS WON'T WORK as it is not implemented
		// for(size_t i = 0; i < size; ++i) {
		// 	compressed_data[i] = static_cast<const Derived*>(this)->compress(data[i], celerity::id<1>{i}, std::forward<Args>(args)...);
		// }
		(void)data;
		(void)compressed_data;
		(void)size;
		(void)dep;
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

template <typename T>
concept BufferAccessDescription = requires(T t, celerity::range<T::range_dims> r) {
	typename T::data_type;
	t.range_mapper;
};

template <typename DataT, typename RangeMapper, int Dims = 1>
struct buffer_access_description {
	using data_type = DataT;
	static constexpr int range_dims = Dims;

	// TODO: Is this required? Durante might be right in that we can get away with just the range mapper as
	//       the range mapper with applied to an all{} range mapper and the size of the main buffer should
	//       give us the size of the reachable range here. Not quite sure if this gets us in trouble with indexing though.
	std::function<celerity::range<Dims>(celerity::range<3>)> calculate_size = [](const celerity::range<3>& r) { return r.size(); };

	RangeMapper range_mapper = RangeMapper{};
};

// Helper to extract data_type from buffer_access_description
template <typename T>
struct extract_data_type {
	using type = typename T::data_type;
};

// Helper to extract range dimensions from buffer_access_description
template <typename T>
struct extract_range_dims {
	static constexpr int value = T::range_dims;
};

// Helper to unwrap compression_dependency or pass through tuples
template <typename T>
struct unwrap_tuple {
	using type = T;
};

template <typename T>
struct unwrap_tuple<detail::compression_dependency<T>> {
	using type = typename detail::compression_dependency<T>::type;
};

// Converts a tuple of buffer_access_description types to a tuple of accessor types
// Template parameters:
//   - Tuple: std::tuple of buffer_access_description types (or wrapped in compression_dependency)
//   - Mode: Access mode (read_only, write_only, read_write, etc.)
//   - Target: Target (device or host_task)
template <typename Tuple, access_mode Mode, target Target, typename Indices = std::make_index_sequence<std::tuple_size_v<typename unwrap_tuple<Tuple>::type>>>
struct accessors_from_descriptions;

// Specialization that unpacks the tuple and creates accessor types for each element
template <typename... Descriptions, access_mode Mode, target Target, std::size_t... Is>
struct accessors_from_descriptions<std::tuple<Descriptions...>, Mode, Target, std::index_sequence<Is...>> {
	using type = std::tuple<accessor<typename extract_data_type<std::tuple_element_t<Is, std::tuple<Descriptions...>>>::type,
	    extract_range_dims<std::tuple_element_t<Is, std::tuple<Descriptions...>>>::value, Mode, Target>...>;
};

// Convenience type alias
template <typename Tuple, access_mode Mode, target Target>
using accessors_from_descriptions_t = typename accessors_from_descriptions<typename unwrap_tuple<Tuple>::type, Mode, Target>::type;

// Converts a tuple of buffer_access_description types to a tuple of buffer types
// Template parameters:
//   - Tuple: std::tuple of buffer_access_description types (or wrapped in compression_dependency)
template <typename Tuple, typename Indices = std::make_index_sequence<std::tuple_size_v<typename unwrap_tuple<Tuple>::type>>>
struct buffers_from_descriptions;

// Specialization that unpacks the tuple and creates buffer types for each element
template <typename... Descriptions, std::size_t... Is>
struct buffers_from_descriptions<std::tuple<Descriptions...>, std::index_sequence<Is...>> {
	using type = std::tuple<buffer<typename extract_data_type<std::tuple_element_t<Is, std::tuple<Descriptions...>>>::type,
	    extract_range_dims<std::tuple_element_t<Is, std::tuple<Descriptions...>>>::value>...>;
};

// Convenience type alias (handles both direct tuples and compression_dependency wrappers)
template <typename Tuple>
using buffers_from_descriptions_t = typename buffers_from_descriptions<typename unwrap_tuple<Tuple>::type>::type;

template <typename RangeMapper, typename RangeToRangeMapper, int MainBufferDims>
struct range_mapper_to_map_ranges;

template <typename Tracking, typename... Dependencies>
struct dependency_bundle {
	dependency_bundle(Tracking tracking_description, Dependencies... other_dependencies)
	    : m_tracking_description(tracking_description), m_other_dependencies(other_dependencies...) {}

	dependency_bundle() : m_tracking_description(), m_other_dependencies() {}

	constexpr auto tracking() const { return m_tracking_description; }

	constexpr auto dependencies() const { return m_other_dependencies; }

	using dependency_type = std::tuple<Dependencies...>;

	template <int MainBufferDims>
	auto buffers_from_descriptions(const celerity::range<MainBufferDims>& main_buffer_range) const {
		const auto main_range_3d = celerity::detail::range_cast<3>(main_buffer_range);
		return transform_dependencies(
		    [&](const auto& dep) {
			    using dep_type = std::decay_t<decltype(dep)>;
			    return buffer<typename dep_type::data_type, dep_type::range_dims>{dep.calculate_size(main_range_3d)};
		    },
		    std::index_sequence_for<Dependencies...>{});
	}

	template <access_mode Mode, access_mode ModeNoInit, target TargetV, typename BufferTuple, typename Functor, int MainBufferDims>
	auto accessors_from_buffers(BufferTuple& buffers, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, TargetV> tag,
	    const celerity::range<MainBufferDims>& main_buffer_range) const {
		return transform_dependencies_with_buffers(
		    buffers,
		    [&](auto& dep_buffer, const auto& dep) {
			    return accessor{dep_buffer, cgh, range_mapper_to_map_ranges{rmfn, dep.range_mapper, main_buffer_range, false}, tag};
		    },
		    std::index_sequence_for<Dependencies...>{});
	}

	template <access_mode Mode, access_mode TagMode, target TargetV, typename BufferTuple, typename Functor, int MainBufferDims>
	auto accessors_from_buffers(BufferTuple& buffers, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, TargetV> tag,
	    const property::no_init& prop, const celerity::range<MainBufferDims>& main_buffer_range) const {
		return transform_dependencies_with_buffers(
		    buffers,
		    [&](auto& dep_buffer, const auto& dep) {
			    return accessor{dep_buffer, cgh, range_mapper_to_map_ranges{rmfn, dep.range_mapper, main_buffer_range, false}, tag, prop};
		    },
		    std::index_sequence_for<Dependencies...>{});
	}

	template <access_mode TagMode, access_mode TagModeNoInit, target TargetV, typename BufferTuple>
	auto accessors_from_buffers(
	    BufferTuple& buffers, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, TargetV> tag, const property_list& prop_list) const {
		return transform_dependencies_with_buffers(
		    buffers, [&](auto& dep_buffer, const auto&) { return accessor{dep_buffer, cgh, access::all(), tag, prop_list}; },
		    std::index_sequence_for<Dependencies...>{});
	}

  private:
	template <typename Fn, std::size_t... Is>
	auto transform_dependencies(Fn&& fn, std::index_sequence<Is...>) const {
		return std::make_tuple(std::forward<Fn>(fn)(std::get<Is>(m_other_dependencies))...);
	}

	template <typename BufferTuple, typename Fn, std::size_t... Is>
	auto transform_dependencies_with_buffers(BufferTuple& buffers, Fn&& fn, std::index_sequence<Is...>) const {
		return std::make_tuple(std::forward<Fn>(fn)(std::get<Is>(buffers), std::get<Is>(m_other_dependencies))...);
	}

	Tracking m_tracking_description;

	std::tuple<Dependencies...> m_other_dependencies;
};

template <typename Impl>
concept CompressionImpl = requires(const Impl& impl) {
	{ impl.get_compression_object() } -> std::same_as<typename Impl::compression_object_type>;
	{ impl.get_dependencies() };
};

template <typename Impl>
struct compression_object_skeleton {
	auto get_dependencies() const { return static_cast<const Impl*>(this)->get_dependencies(); }
	// TODO: Thought of specifying this more to make sure we get what we want but I am not sure if it is a good idea at all as it might limit developers in
	// some way.
	auto get_compression_object() const { return static_cast<const Impl*>(this)->get_compression_object(); }
};


// Fast and easy way to get this
template <typename Algorithm>
struct algorithm_skeleton {
	using compression_object_type = decltype(std::declval<Algorithm>().get_compression_object());
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
	} // TODO: remove this

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


// Element wise compressed accessor proxy for reading compressed data
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

// Element wise compressed accessor proxy for reading and writing compressed data
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
	explicit operator C() const
	    requires(!std::is_same_v<C, T>)
	{
		return m_compressed_ref;
	}

  private:
	C& m_compressed_ref;
	const id<Dims>& m_item;
	const compressed<SelectedCompression, Category>& m_compression;
};

// TODO: LEFTOVER of the first working build. Do we need full feature parity for compressed accessors?
//       Is this even needed for anything as of now?
// IDEA: We had this functionality mainly for host decompression of compressed accessors as there we decompressed
//       the buffer manually by looping through it in a user written host task. Our current build generates a
//       vector by calling decompress_data() on a host task which is way simpler and cleaner as we need this any
//       way for global and local accessor compression.
// IDEA: After team discussion we decided to leave this in for now as it might be usefull for the future. But we
//       won't invest time in cleaning this up right now nor guarantee its functionality.

// template <int TargetDims, typename Target, int SubscriptDim = 0>
// class subscript_proxy_compressed;

// template <int TargetDims, typename Target, int SubscriptDim>
// inline decltype(auto) subscript_compressed(Target& tgt, id<TargetDims> id, const size_t index, nd_item<TargetDims> item, const int const_offset) {
// 	static_assert(SubscriptDim < TargetDims);
// 	id[SubscriptDim] = index - item.get_global_id().get(SubscriptDim) + const_offset;
// 	if constexpr(SubscriptDim == TargetDims - 1) {
// 		return tgt[std::as_const(id[2])];
// 	} else {
// 		return subscript_proxy_compressed<TargetDims, Target, SubscriptDim + 1>{tgt, id, item, const_offset};
// 	}
// }

// template <int TargetDims, typename Target>
// inline decltype(auto) subscript_compressed(Target& tgt, const size_t index, nd_item<TargetDims> item, const int const_offset) {
// 	return subscript_compressed<TargetDims, Target, 0>(tgt, id<TargetDims>{}, index);
// }

// template <int TargetDims, typename Target, int SubscriptDim>
// class subscript_proxy_compressed {
//   public:
// 	subscript_proxy_compressed(Target& tgt, const id<TargetDims> id, nd_item<TargetDims> item, const int const_offset)
// 	    : m_tgt(tgt), m_id(id), m_item(item), m_const_offset(const_offset) {}

// 	inline decltype(auto) operator[](const size_t index) const { //
// 		return subscript_compressed<TargetDims, Target, SubscriptDim>(m_tgt, m_id, index, m_item, m_const_offset);
// 	}

//   private:
// 	Target& m_tgt;
// 	id<TargetDims> m_id{};
// 	nd_item<TargetDims> m_item;
// 	const int m_const_offset;
// };

template <typename ChunkDataState, typename WorkgroupCompressionState>
struct global_compression_state_tracker {
  public:
	global_compression_state_tracker(const ChunkDataState& chunk_acc, const WorkgroupCompressionState& local_acc)
	    : m_chunk_data_state_counter_global_acc(chunk_acc), m_compression_state_local_acc(local_acc) {}

	void try_get_decompression_lock(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto& atomic_ref_count) {
			while(status == compressing) {
				const int32_t combined_status = atomic_ref_count.load();
				status = extract_status(combined_status);
				count = extract_count(combined_status);
			}

			count++;

			if(status == is_compressed && count == 1) {
				status = decompressing;
				m_compression_state_local_acc[0] = decompressing;
			}
		});
	}

	void try_set_is_decompressed(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			if(status == decompressing) {
				status = is_decompressed;
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	void try_set_decompressed_no_consumer(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto& atomic_ref_count) {
			while(status == compressing) {
				const int32_t combined_status = atomic_ref_count.load();
				status = extract_status(combined_status);
				count = extract_count(combined_status);
			}

			count++;

			if(status == is_compressed) {
				status = is_decompressed;
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	void try_get_compression_lock(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			count--;

			if(count == 0 && status == is_decompressed) {
				status = compressing;
				m_compression_state_local_acc[0] = compressing;
			} else if(count > 0) {
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	void try_set_is_compressed(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
			if(status == compressing) {
				status = is_compressed;
				count = 0;
				m_compression_state_local_acc[0] = is_compressed;
			}
		});
	}

	void try_set_compressed_no_producer(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto& /*atomic_ref_count*/) {
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

	auto get_local_state() const { return m_compression_state_local_acc[0]; }

  private:
	template <typename Lambda>
	inline void compare_exchange_run(celerity::id<1> linearized_idx, bool is_leader, Lambda&& func) const {
		if(!is_leader) { return; }

		// TODO: This at the moment can only work for chunks the same size as the local range.
		//       Do we want to support others?
		//       The interesting thing is that we could calculate the linearized chunk index based on the global id,
		//       global range of the buffer, and the offset and the range of the chunk.
		//       This would allow us to support any chunk size as long as we have the right range mapper for it.
		//       This would mean a bit more overhead from the user I think as they would need to communicate somehow
		// IDEA 1) User defines a strange range mapper for their chunk size
		//         We map this range mapper according to what we have in the compression state tracker
		//         Local range

		const auto allocation_offset = m_chunk_data_state_counter_global_acc.get_allocation_offset()[0];
		sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
		    m_chunk_data_state_counter_global_acc[{allocation_offset, linearized_idx[0]}]};

		int32_t status = 0;
		int32_t count = 0;

		int32_t combined_status = atomic_ref_count.load();
		const int32_t old_local_state = m_compression_state_local_acc[0];

		do {
			status = extract_status(combined_status);
			count = extract_count(combined_status);
			m_compression_state_local_acc[0] = old_local_state;

			std::forward<Lambda>(func)(status, count, atomic_ref_count);
		} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
	}


	const ChunkDataState& m_chunk_data_state_counter_global_acc;
	const WorkgroupCompressionState& m_compression_state_local_acc;

	// bit manipulation for status and count combined atomic
	static constexpr int32_t bit_mask = 0b11;
	static constexpr int32_t shift = 2; // Maybe calculate the shift based on the bitmask? Like 0b11 -> shift 2, 0b111 -> shift 3, ...

	static constexpr int is_compressed = 0;
	static constexpr int decompressing = 1;
	static constexpr int is_decompressed = 2;
	static constexpr int compressing = 3;

	int32_t extract_status(int32_t status_atomic) const { return status_atomic & bit_mask; }
	int32_t extract_count(int32_t status_atomic) const { return status_atomic >> shift; }
	int32_t combine_status_atomic(int32_t status, int32_t count) const { return (count << shift) | (status & bit_mask); }
};

// TODO: Check if we might be able to use some sort of specialized accessor like device specific which would make this obsolete.
// IDEA: A device specific accessor knows the extend of its allocation and the offset -> No need to recalculate offsets again and again
//       With this we could also avoid passing offsets ranges ... around all the time making the code clearer
//       THis is mainly useful for global memory compression where we have a big allocation and offsets get tricky

// A container to forward uncompressed data accessor to compression algorithm such that ND access can be used inside compression/decompression
// No need to worry about offsets and such as everything is handled by this container
template <typename DataT, int Dim, typename UncompressedData, compression_category Category>
struct uncompressed_container;

template <typename DataT, int Dim, typename UncompressedData>
struct uncompressed_container<DataT, Dim, UncompressedData, compression_category::local_memory> {
  public:
	uncompressed_container(UncompressedData&& data, detail::compression_chunk<Dim>&& chunk) : m_data(std::move(data)), m_chunk(std::move(chunk)) {}

	inline DataT& operator[](const id<Dim>& index) const { return m_data[celerity::detail::get_linear_index(m_chunk.range, index - m_chunk.offset)]; }

	range<Dim> get_range() const { return m_chunk.global_size; }
	id<Dim> get_offset() const { return m_chunk.offset; }
	range<Dim> get_chunk_size() const { return m_chunk.range; }

  private:
	UncompressedData m_data;

	detail::compression_chunk<Dim> m_chunk;
};

template <typename DataT, int Dim, typename UncompressedData>
struct uncompressed_container<DataT, Dim, UncompressedData, compression_category::global_memory> {
  public:
	uncompressed_container(const UncompressedData& data, detail::compression_chunk<Dim>&& chunk) : m_data(data), m_chunk(std::move(chunk)) {}

	inline DataT& operator[](const id<Dim>& index) const {
		auto linear_index = celerity::detail::get_linear_index(m_chunk.global_size, index);

		return m_data[{m_data.get_allocation_offset()[0], linear_index}];
	}

	range<Dim> get_range() const { return m_chunk.global_size; }
	id<Dim> get_offset() const { return m_chunk.offset; }
	range<Dim> get_chunk_size() const { return m_chunk.range; }

  private:
	const UncompressedData& m_data;

	// TODO: Think about making these accessible in compression/decompression object
	detail::compression_chunk<Dim> m_chunk;
};

template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData, compression_category Category, typename... Args>
struct decompressed_data_accessor;


template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData, typename... Args>
struct decompressed_data_accessor<AccessMode, DataT, Dim, Compression, Algorithm, CompressedData, UncompressedData, compression_category::local_memory,
    Args...> {
  public:
	decompressed_data_accessor(const Compression<Algorithm, compression_category::local_memory>& compression, const CompressedData& compressed_data_acc,
	    UncompressedData&& uncompressed_data_acc, nd_item<Dim>& item, Args&&... args)
	    : m_item(item), m_compression(compression), m_uncompressed_data_acc(std::move(uncompressed_data_acc)), m_compressed_data_acc(compressed_data_acc),
	      m_args(std::forward<Args>(args)...) {
		const auto group = m_item.get_group();
		if constexpr(detail::is_consumer_mode(AccessMode)) {
			celerity::group_barrier(group);
			m_compression.decompress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc, std::forward<Args>(args)...);
			celerity::group_barrier(group);
		}
	}

	~decompressed_data_accessor() {
		if(m_is_moved) return;
		if constexpr(detail::is_producer_mode(AccessMode)) {
			const auto group = m_item.get_group();
			celerity::group_barrier(group);
			if constexpr(sizeof...(Args) > 0) {
				std::apply(
				    [&](auto&&... unpacked_args) {
					    m_compression.compress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc, unpacked_args...);
				    },
				    m_args);
			} else {
				m_compression.compress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc);
			}
			celerity::group_barrier(group);
		}
	}

	// default copy constructor
	decompressed_data_accessor(const decompressed_data_accessor&) = delete;

	decompressed_data_accessor(decompressed_data_accessor&& other) noexcept
	    : m_item(other.m_item), m_compression(other.m_compression), m_uncompressed_data_acc(other.m_uncompressed_data_acc),
	      m_compressed_data_acc(other.m_compressed_data_acc), m_args(std::move(other.m_args)) {
		other.m_is_moved = true; // Avoid double compression
	}

	decompressed_data_accessor& operator=(const decompressed_data_accessor&) = delete;
	decompressed_data_accessor& operator=(decompressed_data_accessor&& other) noexcept {
		if(this != &other) {
			m_item = other.m_item;
			m_compression = other.m_compression;
			m_uncompressed_data_acc = other.m_uncompressed_data_acc;
			m_compressed_data_acc = other.m_compressed_data_acc;
			m_args = std::move(other.m_args);
			other.m_is_moved = true; // Avoid double compression
		}

		return *this;
	}

	inline DataT& operator[](const id<Dim>& index) const { return m_uncompressed_data_acc[index]; }


  private:
	celerity::nd_item<Dim>& m_item;
	const Compression<Algorithm, compression_category::local_memory>& m_compression;
	UncompressedData m_uncompressed_data_acc;
	const CompressedData& m_compressed_data_acc;
	bool m_is_moved = false; // Avoid compression in the moved-from object

	CELERITY_DETAIL_NO_UNIQUE_ADDRESS std::tuple<Args...> m_args;
};


template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData, typename... Args>
struct decompressed_data_accessor<AccessMode, DataT, Dim, Compression, Algorithm, CompressedData, UncompressedData, compression_category::global_memory,
    Args...> {
	using GlobalStateTracker = global_compression_state_tracker<celerity::accessor<int32_t, 2, celerity::access_mode::discard_read_write, target::device>,
	    celerity::local_accessor<int32_t, 1>>;

  private:
	bool multidim_check(const id<Dim>& local_id, const id<Dim>& offset, const id<Dim>& global_range) const {
		for(int i = 0; i < Dim; ++i) {
			if(static_cast<int>(local_id[i]) + static_cast<int>(offset[i]) >= static_cast<int>(global_range[i])) { return false; }
		}
		return true;
	}

  public:
	decompressed_data_accessor(nd_item<Dim>& item, const Compression<Algorithm, compression_category::global_memory>& compression,
	    const UncompressedData&& uncompressed_data_acc, const CompressedData& compressed_data_acc, GlobalStateTracker&& state_tracker, Args&&... args)
	    : m_item(item), m_compression(compression), m_uncompressed_data_acc(std::move(uncompressed_data_acc)), m_compressed_data_acc(compressed_data_acc),
	      m_state_tracker(std::move(state_tracker)), m_args(std::forward<Args>(args)...) {
		const auto group = m_item.get_group();
		const bool is_leader = m_item.get_local_linear_id() == 0;
		const auto offset = m_uncompressed_data_acc.get_offset();
		const bool in_bounds = multidim_check(m_item.get_local_id(), offset, m_item.get_global_range());
		const celerity::id<1> linearized_idx = m_compression.calculate_tile_tracking_idx(m_item, offset);
		if constexpr(detail::is_consumer_mode(AccessMode)) {
			if(in_bounds) { m_state_tracker.try_get_decompression_lock(linearized_idx, is_leader); }

			celerity::group_barrier(group);

			if(m_state_tracker.have_decompressing_lock()) {
				m_compression.decompress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc, args...);
			}

			celerity::group_barrier(group);

			if(in_bounds) { m_state_tracker.try_set_is_decompressed(linearized_idx, is_leader); }
		} else {
			celerity::group_barrier(group);
			if(in_bounds) { m_state_tracker.try_set_decompressed_no_consumer(linearized_idx, is_leader); }
			celerity::group_barrier(group);
		}
	}

	~decompressed_data_accessor() {
		if(m_is_moved) return;

		const auto group = m_item.get_group();
		const bool is_leader = m_item.get_local_linear_id() == 0;
		const auto offset = m_uncompressed_data_acc.get_offset();
		const bool in_bounds = multidim_check(m_item.get_local_id(), offset, m_item.get_global_range());
		const celerity::id<1> linearized_idx = m_compression.calculate_tile_tracking_idx(m_item, offset);
		if constexpr(detail::is_producer_mode(AccessMode)) {
			if(in_bounds) { m_state_tracker.try_get_compression_lock(linearized_idx, is_leader); }

			celerity::group_barrier(group);

			if(m_state_tracker.have_compressing_lock()) {
				if constexpr(sizeof...(Args) > 0) {
					std::apply(
					    [&](auto&&... unpacked_args) {
						    m_compression.compress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc, unpacked_args...);
					    },
					    m_args);
				} else {
					m_compression.compress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc);
				}
			}

			celerity::group_barrier(group);

			if(in_bounds) { m_state_tracker.try_set_is_compressed(linearized_idx, is_leader); }
			celerity::group_barrier(group);
		} else {
			celerity::group_barrier(group);
			if(in_bounds) { m_state_tracker.try_set_compressed_no_producer(linearized_idx, is_leader); }
			celerity::group_barrier(group);
		}
	}

	decompressed_data_accessor(const decompressed_data_accessor&) = delete;

	decompressed_data_accessor(decompressed_data_accessor&& other) noexcept
	    : m_item(other.m_item), m_compression(other.m_compression), m_uncompressed_data_acc(other.m_uncompressed_data_acc),
	      m_compressed_data_acc(other.m_compressed_data_acc), m_state_tracker(other.m_state_tracker), m_args(std::move(other.m_args)) {
		other.m_is_moved = true; // Avoid double compression
	}

	decompressed_data_accessor& operator=(const decompressed_data_accessor&) = delete;
	decompressed_data_accessor& operator=(decompressed_data_accessor&& other) noexcept {
		if(this != &other) {
			m_item = other.m_item;
			m_compression = other.m_compression;
			m_uncompressed_data_acc = other.m_uncompressed_data_acc;
			m_compressed_data_acc = other.m_compressed_data_acc;
			m_state_tracker = other.m_state_tracker;

			other.m_is_moved = true; // Avoid double compression
		}

		return *this;
	}

	inline DataT& operator[](const id<Dim>& index) const { return m_uncompressed_data_acc[index]; }

  private:
	celerity::nd_item<Dim>& m_item;
	const Compression<Algorithm, compression_category::global_memory>& m_compression;
	const UncompressedData m_uncompressed_data_acc;
	const CompressedData& m_compressed_data_acc;

	const GlobalStateTracker m_state_tracker;

	bool m_is_moved = false; // Avoid compression in the moved-from object

	CELERITY_DETAIL_NO_UNIQUE_ADDRESS std::tuple<Args...> m_args;
};

template <typename DataT, int Dims, typename Intype, template <typename, typename> typename SelectedCompression>
class buffer<Intype, Dims, compressed<SelectedCompression<Intype, DataT>, compression_category::element_wise>>
    : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = SelectedCompression<Intype, DataT>;

	static_assert(check_provided_type_with_algorithm<compressed<compression, compression_category::element_wise>>,
	    "Compression algorithm is not compatible with the selected category");

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


// buffer specialization compressed buffer initialization
template <typename DataT, int Dims, typename Intype, template <typename, typename, compression_category> typename SelectedCompression,
    compression_category Category>
class buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type;
	// using compression = compression_cat

	static_assert(check_provided_type_with_algorithm<compression>, "Compression algorithm is not compatible with the selected category");

	buffer(const Intype* data, range<Dims> range, SelectedCompression<Intype, DataT, Category>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression),
	      m_uncompressed_buffer(
	          {celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices() * celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(),
	              range.size()}),
	      m_state_and_count_tracking_buffer(
	          {celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices() * celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(),
	              compression.get_dependencies().tracking().calculate_size(celerity::detail::range_cast<3>(range)).size()}),
	      m_buffers_from_descriptions(compression.get_dependencies().buffers_from_descriptions(range)) {
		celerity::debug::set_buffer_name(m_uncompressed_buffer, "uncompressed");
		celerity::debug::set_buffer_name(m_state_and_count_tracking_buffer, "tracker");
	}

	buffer(range<Dims> range, SelectedCompression<Intype, DataT, Category>& compression)
	    : base(range), m_compression(compression),
	      m_uncompressed_buffer(
	          {celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices() * celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(),
	              range.size()}),
	      m_state_and_count_tracking_buffer(
	          {celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices() * celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(),
	              compression.get_dependencies().tracking().calculate_size(celerity::detail::range_cast<3>(range)).size()}),
	      m_buffers_from_descriptions(compression.get_dependencies().buffers_from_descriptions(range)) {
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

	SelectedCompression<Intype, DataT, Category>& get_compression() { return m_compression; }
	celerity::buffer<Intype, 2>& get_uncompressed_buffer() { return m_uncompressed_buffer; }
	celerity::buffer<int32_t, 2>& get_state_and_count_tracking_buffer() { return m_state_and_count_tracking_buffer; }
	auto& get_buffers_from_descriptions() { return m_buffers_from_descriptions; }
	const auto& get_buffers_from_descriptions() const { return m_buffers_from_descriptions; }

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, SelectedCompression<Intype, DataT, Category>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression) {}

	std::vector<DataT> m_data;
	SelectedCompression<Intype, DataT, Category> m_compression;

	celerity::buffer<Intype, 2> m_uncompressed_buffer;              // Amount of GPUs, Linearized ND buffer
	celerity::buffer<int32_t, 2> m_state_and_count_tracking_buffer; // Amount of GPUs, Linearized ND buffer

	buffers_from_descriptions_t<detail::compression_dependency<SelectedCompression<Intype, DataT, Category>>> m_buffers_from_descriptions;
};

template <typename Memory, typename DataT>
struct alloc_chunk {
	alloc_chunk(const Memory& memory, const uint32_t size, const int start, int& current)
	    : m_memory(memory), m_size(size), m_start(start), m_current(current) {}

	// move constructor
	alloc_chunk(alloc_chunk&& other) noexcept : m_memory(other.m_memory), m_size(other.m_size), m_start(other.m_start), m_current(other.m_current) {
		other.m_start = -1; // invalidate the other chunk to prevent double free
	}

	// move assignment
	alloc_chunk& operator=(alloc_chunk&& other) noexcept {
		if(this != &other) {
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
		assert(m_current == static_cast<int>(m_start + m_size) && "Something went wrong memory lost");

		if(m_current == static_cast<int>(m_start + m_size)) { m_current = m_start; }
	}

	int get_start() const { return m_start; }

	DataT& operator[](const uint32_t index) const {
		// if(m_start == -1) { printf("Accessing moved-from alloc_chunk %d\n", index); }
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

// TODO: LOOK INTO REMOVING THIS FUNCTION SOMETIME IN THE FUTURE
// This is temporary to simplify this part for the user
// IDEA: To replace this we could wrap the kernel to automatically start with this
//       allocator creation at the begining and somehow hydrate in the local accessor
//       pointer? Might be tricky to do this automatically without compiler plugin or
//       some kind of magical type deduction.
template <typename DataT>
auto make_local_allocator(const uint32_t size, const local_accessor<DataT, 1>& accessor) {
	return allocator<local_accessor<DataT, 1>, DataT>(size, accessor);
}

// range mapper mapping range to device and range mapper (direct)
struct device_specific_range_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);

		auto f = sbr.get_boxes();
		auto bounding = celerity::detail::bounding_box(f.begin(), f.end());

		// TODO: Check if this is correct. From my standpoint it should be decent.
		auto linearized_offset_min = celerity::detail::get_linear_index(celerity::detail::range_cast<MainBufferDims>(main_buffer_size), bounding.get_min());
		auto linearized_offset_max =
		    celerity::detail::get_linear_index(celerity::detail::range_cast<MainBufferDims>(main_buffer_size), bounding.get_max() - celerity::detail::ones);

		if(linearized_offset_max >= buffer_size.size()) { linearized_offset_max = buffer_size[1] - 1; }

		auto my_region = celerity::detail::region_builder<BufferDims>();
		my_region.add({{linearized_id, linearized_offset_min}, {linearized_id + 1, linearized_offset_max + 1}});

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		auto test = std::move(my_region).into_region();

		auto box_range = test.get_boxes().front().get_range();
		auto box_offset = test.get_boxes().front().get_offset();

		CELERITY_WARN("chnk offset {} {} {} chunck range {} {} {}, local_range {} {} {}, linearized_id {} linearized_offset {} linearized_range {} "
		              "chunk global size {} {} {}, box_range {} {}, box_offset {} {}\n",
		    chnk.offset[0], chnk.offset[1], chnk.offset[2], chnk.range[0], chnk.range[1], chnk.range[2], chnk.local_range[0], chnk.local_range[1],
		    chnk.local_range[2], linearized_id, linearized_offset_min, linearized_offset_max, chnk.global_size[0], chnk.global_size[1], chnk.global_size[2],
		    box_range[0], box_range[1], box_offset[0], box_offset[1]);

		return test;
#else
		return std::move(my_region).into_region();
#endif
	}
};

// OLD code for range mapper amount, we don't use this anymore as we want to move
// template <typename RangeMapper, int MainBufferDims>
// struct range_mapper_amount { // TODO: FIND BETTER NAMING
// 	range_mapper_amount(const RangeMapper& rm, const celerity::range<MainBufferDims>& buffer_size, bool i) : m_range_mapper(rm), m_buffer_size(buffer_size) {}

// 	template <int Dims, typename Func>
// 	void for_each_local_chunk(const celerity::chunk<Dims>& device_chunk, Func&& func) const {
// 		// Calculate how many local chunks fit in each dimension
// 		celerity::range<Dims> num_local_chunks;
// 		for(int i = 0; i < Dims; ++i) {
// 			num_local_chunks[i] = (device_chunk.range[i] + device_chunk.local_range[i] - 1) / device_chunk.local_range[i];
// 		}

// 		// Helper function to iterate over all combinations
// 		if constexpr(Dims == 1) {
// 			for(size_t i = 0; i < num_local_chunks[0]; ++i) {
// 				celerity::id<Dims> local_offset;
// 				local_offset[0] = device_chunk.offset[0] + i * device_chunk.local_range[0];

// 				celerity::range<Dims> local_range;
// 				local_range[0] = std::min(device_chunk.local_range[0], device_chunk.range[0] - i * device_chunk.local_range[0]);

// 				func(local_offset, local_range);
// 			}
// 		} else if constexpr(Dims == 2) {
// 			for(size_t i = 0; i < num_local_chunks[0]; ++i) {
// 				for(size_t j = 0; j < num_local_chunks[1]; ++j) {
// 					celerity::id<Dims> local_offset;
// 					local_offset[0] = device_chunk.offset[0] + i * device_chunk.local_range[0];
// 					local_offset[1] = device_chunk.offset[1] + j * device_chunk.local_range[1];

// 					celerity::range<Dims> local_range;
// 					local_range[0] = std::min(device_chunk.local_range[0], device_chunk.range[0] - i * device_chunk.local_range[0]);
// 					local_range[1] = std::min(device_chunk.local_range[1], device_chunk.range[1] - j * device_chunk.local_range[1]);

// 					func(local_offset, local_range);
// 				}
// 			}
// 		} else if constexpr(Dims == 3) {
// 			for(size_t i = 0; i < num_local_chunks[0]; ++i) {
// 				for(size_t j = 0; j < num_local_chunks[1]; ++j) {
// 					for(size_t k = 0; k < num_local_chunks[2]; ++k) {
// 						celerity::id<Dims> local_offset;
// 						local_offset[0] = device_chunk.offset[0] + i * device_chunk.local_range[0];
// 						local_offset[1] = device_chunk.offset[1] + j * device_chunk.local_range[1];
// 						local_offset[2] = device_chunk.offset[2] + k * device_chunk.local_range[2];

// 						celerity::range<Dims> local_range;
// 						local_range[0] = std::min(device_chunk.local_range[0], device_chunk.range[0] - i * device_chunk.local_range[0]);
// 						local_range[1] = std::min(device_chunk.local_range[1], device_chunk.range[1] - j * device_chunk.local_range[1]);
// 						local_range[2] = std::min(device_chunk.local_range[2], device_chunk.range[2] - k * device_chunk.local_range[2]);

// 						func(local_offset, local_range);
// 					}
// 				}
// 			}
// 		} else {
// 			static_assert(Dims >= 1 && Dims <= 3, "Only 1D, 2D, and 3D chunks are supported");
// 		}
// 	}

// 	template <int KernelDims, int BufferDims>
// 	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
// 		auto local_range = chnk.local_range;
// 		auto range = chnk.range;
// 		auto offset = chnk.offset;

// 		std::unordered_set<celerity::range<MainBufferDims>> distinct_boxes;

// 		for_each_local_chunk(chnk, [&](const celerity::id<KernelDims>& local_offset, const celerity::range<KernelDims>& local_range) {
// 			// Create a chunk for this local work group
// 			celerity::chunk<KernelDims> local_chunk{local_offset, local_range, chnk.global_size, chnk.local_range};

// 			// Invoke the range mapper for this local chunk
// 			auto region = detail::invoke_range_mapper(m_range_mapper, local_chunk, m_buffer_size);
// 			auto region_box = region.get_boxes();
// 			auto bounding = celerity::detail::bounding_box(region_box.begin(), region_box.end());

// 			distinct_boxes.insert(bounding.get_range());
// 		});

// 		auto distinct_boxes_count = distinct_boxes.size();

// 		printf("Device chunk offset %ld %ld %ld, range %ld %ld %ld, local_range %ld %ld %ld, distinct boxes count %ld\n", chnk.offset[0], chnk.offset[1],
// 		    chnk.offset[2], chnk.range[0], chnk.range[1], chnk.range[2], chnk.local_range[0], chnk.local_range[1], chnk.local_range[2], distinct_boxes_count);

// 		// calculate new region for this device similar to downscale_device_specific_mapper but with the distinct regions we got from the range mapper to range
// 		// mapper where items is the amount of distinct regions we got
// 		auto my_region = celerity::detail::region_builder<BufferDims>();
// 		size_t items = distinct_boxes_count;
// 		if(items > buffer_size.size()) { items = buffer_size.size(); }
// 		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);

// 		my_region.add({{linearized_id, 0}, {linearized_id + 1, items}});

// 		return std::move(my_region).into_region();
// 	}

//   private:
// 	const RangeMapper m_range_mapper;
// 	const celerity::range<MainBufferDims> m_buffer_size;
// };

template <typename RangeMapper, typename RangeToRangeMapper, int MainBufferDims>
struct range_mapper_to_map_ranges {
	range_mapper_to_map_ranges(const RangeMapper& rm, const RangeToRangeMapper& rtm, const celerity::range<MainBufferDims>& buffer_size, bool i)
	    : m_range_mapper(rm), m_range_to_mapper(rtm), m_buffer_size(buffer_size) {}

	// TODO: could be more efficient with Reflections
	template <int KernelDims, int BufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size) const {
		auto intermediate = detail::invoke_range_mapper(m_range_mapper, chnk, m_buffer_size);
		return m_range_to_mapper(chnk, buffer_size, intermediate, m_buffer_size);
	}

  private:
	const RangeMapper m_range_mapper;
	const RangeToRangeMapper m_range_to_mapper;
	const celerity::range<MainBufferDims> m_buffer_size;
};

namespace detail {
	template <int Dims, celerity::detail::range_mapper_type Kind>
	struct static_range_mapper_data {
		static_range_mapper_data() = default;

		template <typename RM>
		explicit static_range_mapper_data(const RM&) {}
	};

	template <int Dims>
	struct static_range_mapper_data<Dims, celerity::detail::range_mapper_type::fixed> {
		explicit static_range_mapper_data(const celerity::access::fixed<Dims>& rm) : m_range(rm.m_range), m_id(rm.m_id) {}

		celerity::subrange<Dims> get_fixed() const { return {m_id, m_range}; }

	  private:
		celerity::range<Dims> m_range = celerity::detail::zeros;
		celerity::id<Dims> m_id = celerity::detail::zeros;
	};

	template <int Dims>
	struct static_range_mapper_data<Dims, celerity::detail::range_mapper_type::neighborhood> {
		explicit static_range_mapper_data(const celerity::access::neighborhood<Dims>& rm) : m_range(rm.m_extent) {}

		celerity::range<Dims> get_neighborhood() const { return m_range; }

	  private:
		celerity::range<Dims> m_range = celerity::detail::zeros;
	};

	template <int Dims>
	struct static_range_mapper_data<Dims, celerity::detail::range_mapper_type::slice> {
		explicit static_range_mapper_data(const celerity::access::slice<Dims>& rm) : m_size(rm.m_dim_idx) {}

		size_t get_slice_dim() const { return m_size; }

	  private:
		size_t m_size = 0;
	};
} // namespace detail

template <typename DataT, int Dims, typename Intype, access_mode Mode, template <typename, typename> typename SelectedCompression,
    detail::range_mapper_type RangeMapperType, typename ExtensionArg>
class accessor<DataT, Dims, Mode, target::device,
    celerity::detail::accessor_extension<compressed<SelectedCompression<Intype, DataT>, compression_category::local_memory>, RangeMapperType, ExtensionArg>> {
  public:
	using compression_algorithm = SelectedCompression<Intype, DataT>;
	using compressed_type = typename compression_algorithm::compressed_type;
	using value_type = typename compression_algorithm::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::local_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : m_base_data_acc(buff, cgh, rmfn, tag), m_range_mapper_data(rmfn), m_compression(buff.get_compression().get_compression_object()),
	      m_extension_accessors(
	          buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, rmfn, tag, buff.get_range())) {}

	template <typename T, int D, typename Functor, access_mode TagMode, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::local_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : m_base_data_acc(buff, cgh, rmfn, tag, prop), m_range_mapper_data(rmfn), m_compression(buff.get_compression().get_compression_object()),
	      m_extension_accessors(
	          buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, rmfn, tag, prop, buff.get_range())) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::local_memory>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : m_base_data_acc(buff, cgh, access::all(), tag, prop_list), m_range_mapper_data(), m_compression(buff.get_compression().get_compression_object()),
	      m_extension_accessors(buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, tag, prop_list)) {}


	template <int Dim, typename... Args>
	inline auto decompress_data(
	    nd_item<Dim>& item, allocator<local_accessor<Intype, 1>, Intype>& allocator, detail::compression_chunk<Dims> chunk, Args&&... args) const {
		return std::apply(
		    [&](const auto&... extension_accessors) {
			    return decompress_impl(item, allocator, std::move(chunk), extension_accessors..., std::forward<Args>(args)...);
		    },
		    m_extension_accessors);
	}


	template <int Dim, typename... Args>
	inline auto decompress_data(nd_item<Dim>& item, allocator<local_accessor<Intype, 1>, Intype>& allocator, Args&&... args) const
	    requires(RangeMapperType != detail::range_mapper_type::custom)
	{
		celerity::detail::compression_chunk<Dim> chunk{
		    item.get_group().get_group_id() * item.get_local_range(), item.get_local_range(), item.get_global_range()};

		if constexpr(RangeMapperType == detail::range_mapper_type::neighborhood) {
			auto compressed_data_range = m_range_mapper_data.get_neighborhood();
			chunk = celerity::detail::compression_chunk<Dim>{(item.get_group().get_group_id() * item.get_local_range()) - compressed_data_range,
			    item.get_local_range() + (compressed_data_range * 2), item.get_global_range()};
		} else if constexpr(RangeMapperType == detail::range_mapper_type::fixed) {
			auto fixed = m_range_mapper_data.get_fixed();
			chunk = celerity::detail::compression_chunk<Dim>{fixed.offset, fixed.range, item.get_global_range()};
		} else if constexpr(RangeMapperType == detail::range_mapper_type::slice) {
			auto slice_dim = m_range_mapper_data.get_slice_dim();
			celerity::id<Dim> offset = item.get_group().get_group_id() * item.get_local_range();
			offset[slice_dim] = 0;
			celerity::range<Dim> range = item.get_local_range();
			range[slice_dim] = item.get_global_range(slice_dim);
			chunk = celerity::detail::compression_chunk<Dim>{offset, range, item.get_global_range()};
		}

		return std::apply(
		    [&](const auto&... extension_accessors) {
			    return decompress_impl(item, allocator, std::move(chunk), extension_accessors..., std::forward<Args>(args)...);
		    },
		    m_extension_accessors);
	}


  private:
	template <int Dim, typename... Args>
	inline auto decompress_impl(
	    nd_item<Dim>& item, allocator<local_accessor<Intype, 1>, Intype>& allocator, detail::compression_chunk<Dims>&& chunk, Args&&... args) const {
		using CompressedContainer = decltype(m_base_data_acc);
		using MemoryChunk = decltype(allocator.allocate(-1));
		using UncompressedContainer = uncompressed_container<Intype, Dims, MemoryChunk, compression_category::local_memory>;

		static_assert(satisfies_compression_interface<compressed<SelectedCompression<Intype, DataT>, compression_category::local_memory>, CompressedContainer,
		                  UncompressedContainer, Dims, Args...>,
		    "Error: wrong parameters for compress_memory_chunk/decompress_memory_chunk, please provide the correct number of additional parameters and make "
		    "sure the provided parameters match the expected types. To find out the expected types, look at the definition of "
		    "compress_memory_chunk/decompress_memory_chunk or compress/decompress functions of the provided compression algorithm.");

		return decompressed_data_accessor<Mode, Intype, Dims, compressed, compression_algorithm, CompressedContainer, UncompressedContainer,
		    compression_category::local_memory, Args...>(
		    m_compression, m_base_data_acc, {allocator.allocate(chunk.range.size()), std::move(chunk)}, item, std::forward<Args>(args)...);
	}

	accessor<DataT, Dims, Mode, target::device> m_base_data_acc;

	detail::static_range_mapper_data<Dims, RangeMapperType> m_range_mapper_data;

	compressed<compression_algorithm, compression_category::local_memory> m_compression;

	accessors_from_descriptions_t<ExtensionArg, Mode, target::device> m_extension_accessors;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode, template <typename, typename> typename SelectedCompression,
    detail::range_mapper_type RangeMapperType, typename ExtensionArg>
class accessor<DataT, Dims, Mode, target::device,
    celerity::detail::accessor_extension<compressed<SelectedCompression<Intype, DataT>, compression_category::global_memory>, RangeMapperType, ExtensionArg>> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression_algorithm = SelectedCompression<Intype, DataT>;
	using compressed_type = typename compression_algorithm::compressed_type;
	using value_type = typename compression_algorithm::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::global_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : m_compression(buff.get_compression().get_compression_object()), m_range_mapper_data(rmfn), m_compressed_data_acc(buff, cgh, rmfn, tag),
	      m_uncompressed_data_acc(buff.get_uncompressed_buffer(), cgh,
	          range_mapper_to_map_ranges{rmfn, device_specific_range_mapper{}, buff.get_range(), false}, celerity::read_write, celerity::no_init),
	      m_state_and_count_data_acc(buff.get_state_and_count_tracking_buffer(), cgh,
	          range_mapper_to_map_ranges{rmfn, buff.get_compression().get_dependencies().tracking().range_mapper, buff.get_range(), false},
	          celerity::read_write, celerity::no_init),
	      m_workgroup_local_flag_acquiring_acc({1}, cgh), m_extension_accessors(buff.get_compression().get_dependencies().accessors_from_buffers(
	                                                          buff.get_buffers_from_descriptions(), cgh, rmfn, tag, buff.get_range())) {}

	template <typename T, int D, typename Functor, access_mode TagMode, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::global_memory>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : m_compression(buff.get_compression().get_compression_object()), m_range_mapper_data(rmfn), m_compressed_data_acc(buff, cgh, rmfn, tag, prop),
	      m_uncompressed_data_acc(buff.get_uncompressed_buffer(), cgh,
	          range_mapper_to_map_ranges{rmfn, device_specific_range_mapper{}, buff.get_range(), false}, celerity::read_write, celerity::no_init),
	      m_state_and_count_data_acc(buff.get_state_and_count_tracking_buffer(), cgh,
	          range_mapper_to_map_ranges{rmfn, buff.get_compression().get_dependencies().tracking().range_mapper, buff.get_range(), false},
	          celerity::read_write, celerity::no_init),
	      m_workgroup_local_flag_acquiring_acc({1}, cgh), m_extension_accessors(buff.get_compression().get_dependencies().accessors_from_buffers(
	                                                          buff.get_buffers_from_descriptions(), cgh, rmfn, tag, prop, buff.get_range())) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::global_memory>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : m_compression(buff.get_compression().get_compression_object()), m_compressed_data_acc(buff, cgh, access::all(), tag, prop_list),
	      m_range_mapper_data(), m_uncompressed_data_acc(buff.get_uncompressed_buffer(), cgh, celerity::access::all(), celerity::read_write, celerity::no_init),
	      m_state_and_count_data_acc(buff.get_state_and_count_tracking_buffer(), cgh, celerity::access::all(), celerity::read_write, celerity::no_init),
	      m_workgroup_local_flag_acquiring_acc({1}, cgh),
	      m_extension_accessors(buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, tag, prop_list)) {}

	template <int Dim, typename... Args>
	inline auto decompress_data(celerity::nd_item<Dim>& item, Args&&... args) const
	    requires(RangeMapperType != detail::range_mapper_type::custom)
	{
		celerity::detail::compression_chunk<Dim> chunk{
		    item.get_group().get_group_id() * item.get_local_range(), item.get_local_range(), item.get_global_range()};

		if constexpr(RangeMapperType == detail::range_mapper_type::neighborhood) {
			auto compressed_data_range = m_range_mapper_data.get_neighborhood();
			chunk = celerity::detail::compression_chunk<Dim>{(item.get_group().get_group_id() * item.get_local_range()) - compressed_data_range,
			    item.get_local_range() + (compressed_data_range * 2), item.get_global_range()};
		} else if constexpr(RangeMapperType == detail::range_mapper_type::fixed) {
			auto fixed = m_range_mapper_data.get_fixed();
			chunk = celerity::detail::compression_chunk<Dim>{fixed.offset, fixed.range, item.get_global_range()};
		} else if constexpr(RangeMapperType == detail::range_mapper_type::slice) {
			auto slice_dim = m_range_mapper_data.get_slice_dim();
			celerity::id<Dim> offset = item.get_group().get_group_id() * item.get_local_range();
			offset[slice_dim] = 0;
			celerity::range<Dim> range = item.get_local_range();
			range[slice_dim] = item.get_global_range(slice_dim);
			chunk = celerity::detail::compression_chunk<Dim>{offset, range, item.get_global_range()};
		}

		return std::apply(
		    [&](const auto&... extension_accessors) { return decompress_impl(item, std::move(chunk), extension_accessors..., std::forward<Args>(args)...); },
		    m_extension_accessors);
	}

	template <int Dim, typename... Args> // this dimension is needed to allow different dimensions for item and chunk (e.g. 2D kernel and 3D buffer)
	inline auto decompress_data(celerity::nd_item<Dim>& item, celerity::detail::compression_chunk<Dims> chunk, Args&&... args) const {
		return std::apply(
		    [&](const auto&... extension_accessors) { return decompress_impl(item, std::move(chunk), extension_accessors..., std::forward<Args>(args)...); },
		    m_extension_accessors);
	}

  private:
	template <int Dim, typename... Args>
	inline auto decompress_impl(celerity::nd_item<Dim>& item, celerity::detail::compression_chunk<Dims> chunk, Args&&... args) const {
		using CompressedContainer = decltype(m_compressed_data_acc);
		using UncompressedContainer = uncompressed_container<Intype, Dims, decltype(m_uncompressed_data_acc), compression_category::global_memory>;

		static_assert(contains_compress_decompress_memory_chunk<compressed<SelectedCompression<Intype, DataT>, compression_category::global_memory>,
		                  CompressedContainer, UncompressedContainer, Dims, Args...>,
		    "Error: wrong parameters for compress_memory_chunk/decompress_memory_chunk, please provide the correct number of additional parameters and make "
		    "sure the provided parameters match the expected types. To find out the expected types, look at the definition of "
		    "compress_memory_chunk/decompress_memory_chunk or compress/decompress functions of the provided compression algorithm.");

		return decompressed_data_accessor<Mode, Intype, Dims, compressed, compression_algorithm, CompressedContainer, UncompressedContainer,
		    compression_category::global_memory, Args...>(item, m_compression, {m_uncompressed_data_acc, std::move(chunk)}, m_compressed_data_acc,
		    {m_state_and_count_data_acc, m_workgroup_local_flag_acquiring_acc}, std::forward<Args>(args)...);
	}

	compressed<compression_algorithm, compression_category::global_memory> m_compression;

	detail::static_range_mapper_data<Dims, RangeMapperType> m_range_mapper_data;

	celerity::accessor<DataT, Dims, Mode, target::device> m_compressed_data_acc;
	celerity::accessor<Intype, 2, celerity::access_mode::discard_read_write, target::device> m_uncompressed_data_acc;
	celerity::accessor<int32_t, 2, celerity::access_mode::discard_read_write, target::device> m_state_and_count_data_acc;
	celerity::local_accessor<int32_t, 1> m_workgroup_local_flag_acquiring_acc;

	accessors_from_descriptions_t<ExtensionArg, Mode, target::device> m_extension_accessors;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode, template <typename, typename> typename SelectedCompression,
    detail::range_mapper_type RangeMapperType, typename ExtensionArg>
class accessor<DataT, Dims, Mode, target::device,
    celerity::detail::accessor_extension<compressed<SelectedCompression<Intype, DataT>, compression_category::element_wise>, RangeMapperType, ExtensionArg>> {
  public:
	using compression = SelectedCompression<Intype, DataT>;
	using compressed_type = typename compression::compressed_type;
	using value_type = typename compression::value_type;
	using retval =
	    std::conditional_t<detail::is_producer_mode(Mode), uncompressed_item_wrapper<Intype, DataT, Dims, compression, compression_category::element_wise>,
	        const uncompressed_item_wrapper_const<Intype, DataT, Dims, compression, compression_category::element_wise>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::element_wise>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : m_compression(buff.get_compression().get_compression_object()), m_base_data_acc(buff, cgh, rmfn, tag),
	      m_extension_accessors(
	          buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, rmfn, tag, buff.get_range())) {}

	template <typename T, int D, typename Functor, access_mode TagMode, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::element_wise>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& prop)
	    : m_compression(buff.get_compression().get_compression_object()), m_base_data_acc(buff, cgh, rmfn, tag, prop),
	      m_extension_accessors(
	          buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, rmfn, tag, prop, buff.get_range())) {}


	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, compression_category::element_wise>>& buff, handler& cgh,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : m_compression(buff.get_compression().get_compression_object()), m_base_data_acc(buff, cgh, access::all(), tag, prop_list),
	      m_extension_accessors(buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, tag, prop_list)) {}

	template <access_mode M = Mode>
	inline retval operator[](const id<Dims>& index) const {
		return {m_base_data_acc[index], index, m_compression};
	}

  private:
	compressed<compression, compression_category::element_wise> m_compression;
	accessor<DataT, Dims, Mode, target::device> m_base_data_acc;

	accessors_from_descriptions_t<ExtensionArg, Mode, target::device> m_extension_accessors;
};


template <typename DataT, int Dims, typename Intype, access_mode Mode, template <typename, typename> typename SelectedCompression,
    compression_category Category, detail::range_mapper_type RangeMapperType, typename ExtensionArg>
class accessor<DataT, Dims, Mode, target::host_task,
    celerity::detail::accessor_extension<compressed<SelectedCompression<Intype, DataT>, Category>, RangeMapperType, ExtensionArg>> {
  public:
	using compression = SelectedCompression<Intype, DataT>;
	using compressed_type = typename compression::compressed_type;
	using value_type = typename compression::value_type;

	using retval =
	    std::conditional_t<detail::is_producer_mode(Mode), uncompressed_item_wrapper<Intype, DataT, Dims, compression, compression_category::element_wise>,
	        const uncompressed_item_wrapper_const<Intype, DataT, Dims, compression, compression_category::element_wise>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<Mode, ModeNoInit, target::host_task> tag)
	    : m_compression(buff.get_compression().get_compression_object()), m_range(buff.get_range()), m_base_data_acc(buff, cgh, rmfn, tag),
	      m_extension_accessors(
	          buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, rmfn, tag, buff.get_range())) {}

	template <typename T, int D, typename Functor, access_mode TagMode, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, Category>>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::host_task> tag, const property::no_init& prop)
	    : m_compression(buff.get_compression().get_compression_object()), m_range(buff.get_range()), m_base_data_acc(buff, cgh, rmfn, tag, prop),
	      m_extension_accessors(
	          buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, rmfn, tag, prop, buff.get_range())) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit, template <typename, typename, compression_category> typename Compression>
	accessor(buffer<T, D, Compression<T, DataT, Category>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag,
	    const property_list& prop_list)
	    : m_compression(buff.get_compression().get_compression_object()), m_range(buff.get_range()), m_base_data_acc(buff, cgh, access::all(), tag, prop_list),
	      m_extension_accessors(buff.get_compression().get_dependencies().accessors_from_buffers(buff.get_buffers_from_descriptions(), cgh, tag, prop_list)) {}

	template <access_mode M = Mode>
	    requires(Category == compression_category::element_wise)
	inline retval operator[](const id<Dims>& index) const {
		return {m_base_data_acc[index], index, m_compression};
	}

	template <typename... Args>
	inline auto decompress_data(Args&&... args) const {
		std::vector<Intype> uncompressed_data(m_range.size());
		std::apply(
		    [&](const auto&... extension_accessors) {
			    return m_compression.decompress_data(m_base_data_acc, uncompressed_data, m_range, extension_accessors..., std::forward<Args>(args)...);
		    },
		    m_extension_accessors);
		return std::move(uncompressed_data);
	}

  private:
	compressed<compression, Category> m_compression;
	range<Dims> m_range;
	accessor<DataT, Dims, Mode, target::host_task> m_base_data_acc;

	accessors_from_descriptions_t<ExtensionArg, Mode, target::host_task> m_extension_accessors;
};
} // namespace celerity