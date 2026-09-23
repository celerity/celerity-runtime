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
// [x] General code cleanup and documentation ADD TESTs
// [x] Add a way to generalize compression algorithm  dependent parameters (like extra buffers and so on)
// [x] Add a way to generalize extra user parameters (like bound values)

// Some more specific todos specific to some locations:
// [x] How to get Range mapper infos during kernel calculation
// [x] How would we get information of the amount of devices
// [x] New device local accessor (-> Design it in such a way that it is replaceable by a later implementation of device local accessor?)

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

template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept satisfies_element_wise = detail::contains_same_category<T::category, compression_category::element_wise>
                                 && contains_compress_decompress<T, CompressedBuff, UncompressedBuff, Dims, Args...>;

template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept satisfies_memory_chunk = !detail::contains_same_category<T::category, compression_category::element_wise>
                                 && contains_compress_decompress_memory_chunk<T, CompressedBuff, UncompressedBuff, Dims, Args...>;

template <typename T, typename CompressedBuff, typename UncompressedBuff, int Dims, typename... Args>
concept satisfies_compression_interface =
    satisfies_element_wise<T, CompressedBuff, UncompressedBuff, Dims, Args...> || satisfies_memory_chunk<T, CompressedBuff, UncompressedBuff, Dims, Args...>;

template <typename T>
struct is_vec : std::false_type {};

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

	struct compression_testspy;
} // namespace detail

template <typename Derived, compression_category CompressionCategory>
class compressed_default : public compression_tags<CompressionCategory> {
	static_assert(CompressionCategory != compression_category::none, "Compression category cannot be none");

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
	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void compress_memory_chunk(
	    celerity::nd_item<Dims>& item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, Args&&... args) const
	    requires(celerity::detail::contains_same_category<CompressionCategory, compression_category::element_wise>
	             && celerity::detail::contains_same_category<CompressionCategory, compression_category::local_memory | compression_category::global_memory>)
	{
		auto amount = item.get_local_range().size();
		auto chunk_range = uncompressed_data_acc.get_chunk_size();
		auto calculations_per_item = (chunk_range.size() + amount - 1) / amount;

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

	template <typename CompressedData, typename UncompressedData, int Dims, typename... Args>
	void decompress_memory_chunk(
	    celerity::nd_item<Dims>& item, const CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, Args&&... args) const
	    requires(detail::contains_same_category<CompressionCategory, compression_category::element_wise>
	             && detail::contains_same_category<CompressionCategory, compression_category::local_memory | compression_category::global_memory>)
	{
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

			uncompressed_data_acc[global_index] =
			    static_cast<const Derived*>(this)->decompress(compressed_data_acc[global_index], global_index, std::forward<Args>(args)...);
		}
	}

	// NOTE: At the current stage, we only support compress_data and decompress_data for element-wise compression algorithms.
	//       This is because we implemented this for our autogenerated element wise compression.
	//       IS there a way to implement this in a general sense for memory chunk based algorithms?
	template <typename CompressedData, typename UncompressedData, typename... Args>
	void compress_data(const UncompressedData* data, CompressedData& compressed_data, size_t size, Args&&... args) const {
		for(size_t i = 0; i < size; ++i) {
			compressed_data[i] = static_cast<const Derived*>(this)->compress(data[i], celerity::id<1>{i}, std::forward<Args>(args)...);
		}
	}

	// NOTE: At the current stage, we only support compress_data and decompress_data for element-wise compression algorithms.
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

template <typename T>
concept BufferAccessDescription = requires(T t, celerity::range<T::range_dims> r) {
	typename T::data_type;
	t.range_mapper;
};

template <typename DataT, typename RangeMapper, int Dims = 1>
struct buffer_access_description {
	using data_type = DataT;
	static constexpr int range_dims = Dims;

	// NOTE: Is this required? Offline discussion: might be right in that we can get away with just the range mapper as
	//       the range mapper with applied to an all{} range mapper and the size of the main buffer should
	//       give us the size of the reachable range here. Not quite sure if this gets us in trouble with indexing though.
	std::function<celerity::range<Dims>(celerity::range<3>)> calculate_size = [](const celerity::range<3>& r) { return r.size(); };

	RangeMapper range_mapper = RangeMapper{};
};

template <typename T>
struct extract_data_type {
	using type = typename T::data_type;
};

template <typename T>
struct extract_range_dims {
	static constexpr int value = T::range_dims;
};

template <typename T>
struct unwrap_tuple {
	using type = T;
};

template <typename T>
struct unwrap_tuple<detail::compression_dependency<T>> {
	using type = typename detail::compression_dependency<T>::type;
};

template <typename Tuple, access_mode Mode, target Target, typename Indices = std::make_index_sequence<std::tuple_size_v<typename unwrap_tuple<Tuple>::type>>>
struct accessors_from_descriptions;

template <typename... Descriptions, access_mode Mode, target Target, std::size_t... Is>
struct accessors_from_descriptions<std::tuple<Descriptions...>, Mode, Target, std::index_sequence<Is...>> {
	using type = std::tuple<accessor<typename extract_data_type<std::tuple_element_t<Is, std::tuple<Descriptions...>>>::type,
	    extract_range_dims<std::tuple_element_t<Is, std::tuple<Descriptions...>>>::value, Mode, Target>...>;
};

template <typename Tuple, access_mode Mode, target Target>
using accessors_from_descriptions_t = typename accessors_from_descriptions<typename unwrap_tuple<Tuple>::type, Mode, Target>::type;

template <typename Tuple, typename Indices = std::make_index_sequence<std::tuple_size_v<typename unwrap_tuple<Tuple>::type>>>
struct buffers_from_descriptions;

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
	auto get_compression_object() const { return static_cast<const Impl*>(this)->get_compression_object(); }
};


template <typename Algorithm>
struct algorithm_skeleton {
	using compression_object_type = decltype(std::declval<Algorithm>().get_compression_object());
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

// NOTE: LEFTOVER of the first working build. Do we need full feature parity for compressed accessors?
//       Is this even needed for anything as of now?
// IDEA: We had this functionality mainly for host decompression of compressed accessors as there we decompressed
//       the buffer manually by looping through it in a user written host task. Our current build generates a
//       vector by calling decompress_data() on a host task which is way simpler and cleaner as we need this any
//       way for global and local accessor compression.
// NOTE: After team discussion we decided to leave this in for now as it might be usefull for the future. But we
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
	friend struct detail::compression_testspy;

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
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto&) {
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
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto&) {
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
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto&) {
			if(status == compressing) {
				status = is_compressed;
				count = 0;
				m_compression_state_local_acc[0] = is_compressed;
			}
		});
	}

	void try_set_compressed_no_producer(celerity::id<1> linearized_idx, bool is_leader) const {
		compare_exchange_run(linearized_idx, is_leader, [&](int32_t& status, int32_t& count, auto&) {
			count--;

			if(count == 0 && status == is_decompressed) {
				status = is_compressed;
				m_compression_state_local_acc[0] = is_compressed;
			} else if(count > 0) {
				m_compression_state_local_acc[0] = is_decompressed;
			}
		});
	}

	void wait_until_decompressed(celerity::id<1> linearized_idx, bool is_leader) const {
		if(!is_leader) { return; }
		const auto allocation_offset = m_chunk_data_state_counter_global_acc.get_allocation_offset()[0];
		sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> atomic_ref_count{
		    m_chunk_data_state_counter_global_acc[{allocation_offset, linearized_idx[0]}]};
		while(extract_status(atomic_ref_count.load(sycl::memory_order::acquire)) == decompressing) {}
	}

	bool have_decompressing_lock() const { return m_compression_state_local_acc[0] == decompressing; }
	bool have_compressing_lock() const { return m_compression_state_local_acc[0] == compressing; }

	auto get_local_state() const { return m_compression_state_local_acc[0]; }

  private:
	template <typename Lambda>
	inline void compare_exchange_run(celerity::id<1> linearized_idx, bool is_leader, Lambda&& func) const {
		if(!is_leader) { return; }

		// NOTE: This at the moment can only work for chunks the same size as the local range.
		//       Do we want to support others?
		//       The interesting thing is that we could calculate the linearized chunk index based on the global id,
		//       global range of the buffer, and the offset and the range of the chunk.
		//       This would allow us to support any chunk size as long as we have the right range mapper for it.
		//       This would mean a bit more overhead from the user I think as they would need to communicate somehow
		// IDEA 1) User defines a strange range mapper for their chunk size
		//         We map this range mapper according to what we have in the compression state tracker
		//         Local range

		const auto allocation_offset = m_chunk_data_state_counter_global_acc.get_allocation_offset()[0];
		sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device> atomic_ref_count{
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
	static constexpr int32_t shift = 2;

	static constexpr int is_compressed = 0;
	static constexpr int decompressing = 1;
	static constexpr int is_decompressed = 2;
	static constexpr int compressing = 3;

	int32_t extract_status(int32_t status_atomic) const { return status_atomic & bit_mask; }
	int32_t extract_count(int32_t status_atomic) const { return status_atomic >> shift; }
	int32_t combine_status_atomic(int32_t status, int32_t count) const { return (count << shift) | (status & bit_mask); }
};

// NOTE: Check if we might be able to use some sort of specialized accessor like device specific which would make this obsolete.
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

	detail::compression_chunk<Dim> m_chunk;
};

template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData, compression_category Category, typename... Args>
struct decompressed_data_accessor;


template <access_mode AccessMode, typename DataT, int Dim, template <typename, compression_category> typename Compression, typename Algorithm,
    typename CompressedData, typename UncompressedData, typename... Args>
struct decompressed_data_accessor<AccessMode, DataT, Dim, Compression, Algorithm, CompressedData, UncompressedData, compression_category::local_memory,
    Args...> {
	friend struct detail::compression_testspy;

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
	    : m_item(other.m_item), m_compression(other.m_compression), m_uncompressed_data_acc(std::move(other.m_uncompressed_data_acc)),
	      m_compressed_data_acc(other.m_compressed_data_acc), m_args(std::move(other.m_args)) {
		other.m_is_moved = true; // Avoid double compression
	}

	decompressed_data_accessor& operator=(const decompressed_data_accessor&) = delete;
	decompressed_data_accessor& operator=(decompressed_data_accessor&& other) noexcept {
		if(this != &other) {
			m_item = other.m_item;
			m_compression = other.m_compression;
			m_uncompressed_data_acc = std::move(other.m_uncompressed_data_acc);
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
	friend struct detail::compression_testspy;

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

			celerity::group_barrier(group, memory_scope::device);

			if(m_state_tracker.have_decompressing_lock()) {
				m_compression.decompress_memory_chunk(m_item, m_compressed_data_acc, m_uncompressed_data_acc, args...);
			}

			celerity::group_barrier(group, memory_scope::device);

			if(in_bounds && m_state_tracker.have_decompressing_lock()) { m_state_tracker.try_set_is_decompressed(linearized_idx, is_leader); }
			if(in_bounds) { m_state_tracker.wait_until_decompressed(linearized_idx, is_leader); }

			celerity::group_barrier(group, memory_scope::device);
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
		celerity::group_barrier(group, memory_scope::device);

		if constexpr(detail::is_producer_mode(AccessMode)) {
			if(in_bounds) { m_state_tracker.try_get_compression_lock(linearized_idx, is_leader); }

			celerity::group_barrier(group, memory_scope::device);

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

			celerity::group_barrier(group, memory_scope::device);

			if(in_bounds && m_state_tracker.have_compressing_lock()) { m_state_tracker.try_set_is_compressed(linearized_idx, is_leader); }
			celerity::group_barrier(group);
		} else {
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


template <typename DataT, int Dims, typename Intype, template <typename, typename, compression_category> typename SelectedCompression,
    compression_category Category>
class buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type;

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

	celerity::buffer<Intype, 2> m_uncompressed_buffer;
	celerity::buffer<int32_t, 2> m_state_and_count_tracking_buffer;

	buffers_from_descriptions_t<detail::compression_dependency<SelectedCompression<Intype, DataT, Category>>> m_buffers_from_descriptions;
};

template <typename Memory, typename DataT>
struct alloc_chunk {
	alloc_chunk(const Memory& memory, const uint32_t size, const int start, int& current)
	    : m_memory(memory), m_size(size), m_start(start), m_current(current) {}

	alloc_chunk(alloc_chunk&& other) noexcept : m_memory(other.m_memory), m_size(other.m_size), m_start(other.m_start), m_current(other.m_current) {
		other.m_start = -1; // invalidate the other chunk to prevent double free
	}

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

	alloc_chunk(const alloc_chunk&) = delete;
	alloc_chunk& operator=(const alloc_chunk&) = delete;

	~alloc_chunk() {
		if(m_start == -1) { return; } // already moved
		assert(m_current == static_cast<int>(m_start + m_size) && "Something went wrong memory lost");

		if(m_current == static_cast<int>(m_start + m_size)) { m_current = m_start; }
	}

	int get_start() const { return m_start; }

	DataT& operator[](const uint32_t index) const {
		assert(m_start != -1 && "Accessing moved-from alloc_chunk");
		assert(index < m_size && "Index out of bounds");
		assert(index >= 0 && "Index out of bounds");
		return m_memory[m_start + index];
	}

  private:
	const Memory& m_memory;
	uint32_t m_size;
	int m_start;
	int& m_current;
};

template <typename Memory, typename DataT>
struct allocator {
	allocator(const uint32_t size, const Memory& accessor) : m_memory(accessor), m_size(size) {}

	alloc_chunk<Memory, DataT> allocate(const uint32_t size) {
		assert(m_current + size <= m_size && "Out of memory");

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

struct device_specific_range_mapper {
	template <int KernelDims, int BufferDims, int MainBufferDims>
	celerity::detail::region<BufferDims> operator()(const celerity::chunk<KernelDims>& chnk, const celerity::range<BufferDims>& buffer_size,
	    const celerity::detail::region<MainBufferDims>& sbr, const celerity::range<MainBufferDims>& main_buffer_size) const {
		auto linearized_id = celerity::detail::get_linear_index(chnk.global_size / chnk.range, chnk.offset / chnk.range);

		auto f = sbr.get_boxes();
		auto bounding = celerity::detail::bounding_box(f.begin(), f.end());

		auto linearized_offset_min = celerity::detail::get_linear_index(celerity::detail::range_cast<MainBufferDims>(main_buffer_size), bounding.get_min());
		auto linearized_offset_max =
		    celerity::detail::get_linear_index(celerity::detail::range_cast<MainBufferDims>(main_buffer_size), bounding.get_max() - celerity::detail::ones);

		if(linearized_offset_max >= buffer_size.size()) { linearized_offset_max = buffer_size[1] - 1; }

		auto my_region = celerity::detail::region_builder<BufferDims>();
		my_region.add({{linearized_id, linearized_offset_min}, {linearized_id + 1, linearized_offset_max + 1}});

		return std::move(my_region).into_region();
	}
};

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
	friend struct detail::compression_testspy;

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
	friend struct detail::compression_testspy;

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
	friend struct detail::compression_testspy;

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
	friend struct detail::compression_testspy;

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
