#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "compression.h"
#include "compression_algorithms.h"
#include "compression_impl.h"
#include "compression_wrapper.h"
#include "test_utils.h"
#include "testspy/compression_testspy.h"

namespace celerity::test_detail {
template <typename, typename>
struct stub_algorithm {};

template <typename T, typename C>
struct stub_transform {
	using value_type = T;
	using compressed_type = C;
};

template <typename T, typename C>
struct stub_transform_with_dep {
	using value_type = T;
	using compressed_type = C;
};


template <typename Algorithm, typename CompressedData, typename UncompressedData, int Dims, typename... Args>
static void simple_compress_memory_chunk(const Algorithm& alg, const celerity::nd_item<Dims>& item, CompressedData& compressed_data_acc,
    const UncompressedData& uncompressed_data_acc, const Args&... args) {
	if(item.get_local_linear_id() != 0) return;
	const auto chunk_range = uncompressed_data_acc.get_chunk_size();
	for(size_t idx = 0; idx < chunk_range.size(); ++idx) {
		const auto global_index = uncompressed_data_acc.get_offset() + celerity::detail::get_nd_index(idx, chunk_range);
		compressed_data_acc[global_index] = alg.compress(uncompressed_data_acc[global_index], global_index, args...);
	}
}

template <typename Algorithm, typename CompressedData, typename UncompressedData, int Dims, typename... Args>
static void simple_decompress_memory_chunk(const Algorithm& alg, const celerity::nd_item<Dims>& item, const CompressedData& compressed_data_acc,
    const UncompressedData& uncompressed_data_acc, const Args&... args) {
	if(item.get_local_linear_id() != 0) return;
	const auto chunk_range = uncompressed_data_acc.get_chunk_size();
	for(size_t idx = 0; idx < chunk_range.size(); ++idx) {
		const auto global_index = uncompressed_data_acc.get_offset() + celerity::detail::get_nd_index(idx, chunk_range);
		uncompressed_data_acc[global_index] = alg.decompress(compressed_data_acc[global_index], global_index, args...);
	}
}

struct fake_tile_offset {
	size_t value;
	size_t operator[](int) const { return value; }
};

struct fake_chunk_data_state {
	int32_t* data;
	size_t stride;
	fake_tile_offset get_allocation_offset() const { return {0}; }
	int32_t& operator[](const celerity::id<2>& idx) const { return data[idx[0] * stride + idx[1]]; }
};

struct fake_local_flag {
	int32_t* value;
	int32_t& operator[](size_t) const { return *value; }
};
} // namespace celerity::test_detail

namespace celerity {
template <compression_category Category>
class compressed<test_detail::stub_algorithm<int, float>, Category>
    : public compression_tags<compression_category::element_wise | compression_category::local_memory> {};

template <typename T, typename C, compression_category Category>
class compressed<test_detail::stub_transform<T, C>, Category>
    : public compressed_default<compressed<test_detail::stub_transform<T, C>, Category>,
          compression_category::element_wise | compression_category::local_memory | compression_category::global_memory> {
  public:
	template <int Dims>
	C compress(const T number, const id<Dims>&) const {
		return static_cast<C>(-number);
	}
	template <int Dims>
	T decompress(const C number, const id<Dims>&) const {
		return static_cast<T>(-number);
	}
};

template <typename T, typename C, compression_category Category>
class stub_transform_compression : public compression_object_skeleton<stub_transform_compression<T, C, Category>> {
  public:
	stub_transform_compression() : m_dependencies({[](const range<3>& r) { return range<2>{r[0], r[1]}.size(); }, downscale_device_specific_mapper{}}) {}

	auto get_compression_object() const { return m_compression_object; }
	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<test_detail::stub_transform<T, C>, Category> m_compression_object;
	dependency_bundle<buffer_access_description<T, downscale_device_specific_mapper>> m_dependencies;
};

template <typename T, typename C, compression_category Category>
class compressed<test_detail::stub_transform_with_dep<T, C>, Category>
    : public compressed_default<compressed<test_detail::stub_transform_with_dep<T, C>, Category>,
          compression_category::local_memory | compression_category::global_memory> {
  public:
	template <int Dims>
	C compress(const T number, const id<Dims>&, const T& dep) const {
		return static_cast<C>(-(number - dep));
	}
	template <int Dims>
	T decompress(const C number, const id<Dims>&, const T& dep) const {
		return static_cast<T>(-number) + dep;
	}

	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void compress_memory_chunk(
	    celerity::nd_item<Dims>& item, CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, const Dependency& dep) const {
		test_detail::simple_compress_memory_chunk(*this, item, compressed_data_acc, uncompressed_data_acc, dep);
	}
	template <typename CompressedData, typename UncompressedData, int Dims, typename Dependency>
	void decompress_memory_chunk(
	    celerity::nd_item<Dims>& item, const CompressedData& compressed_data_acc, const UncompressedData& uncompressed_data_acc, const Dependency& dep) const {
		test_detail::simple_decompress_memory_chunk(*this, item, compressed_data_acc, uncompressed_data_acc, dep);
	}
};

template <typename T, typename C, compression_category Category>
class stub_transform_with_dep_compression : public compression_object_skeleton<stub_transform_with_dep_compression<T, C, Category>> {
  public:
	stub_transform_with_dep_compression()
	    : m_dependencies({[](const range<3>& r) { return range<2>{r[0], r[1]}.size(); }, downscale_device_specific_mapper{}},
	          {[](const range<3>& r) { return range<2>{r[0], r[1]}; }, point_cloud_dep_buffer_mapper{}}) {}

	auto get_compression_object() const { return m_compression_object; }
	auto get_dependencies() const { return m_dependencies; }

  private:
	compressed<test_detail::stub_transform_with_dep<T, C>, Category> m_compression_object;
	dependency_bundle<buffer_access_description<T, downscale_device_specific_mapper>, buffer_access_description<T, point_cloud_dep_buffer_mapper, 2>>
	    m_dependencies;
};
} // namespace celerity

namespace {
using namespace celerity;
using namespace celerity::detail;
} // namespace

TEST_CASE("compression_category supports bitwise operators", "[compression]") {
	CHECK((compression_category::element_wise | compression_category::local_memory) != compression_category::element_wise);
	CHECK((compression_category::element_wise | compression_category::none) == compression_category::element_wise);

	const auto combined = compression_category::element_wise | compression_category::local_memory | compression_category::global_memory;
	CHECK((combined & compression_category::element_wise) == compression_category::element_wise);
	CHECK((combined & compression_category::local_memory) == compression_category::local_memory);
	CHECK((combined & compression_category::global_memory) == compression_category::global_memory);
	CHECK((combined & compression_category::kernel) == compression_category::none);
	CHECK((combined & compression_category::automatic) == compression_category::none);
}

TEST_CASE("contains_same_category identifies matching categories", "[compression]") {
	using detail::contains_same_category;
	constexpr auto mask = compression_category::element_wise | compression_category::local_memory | compression_category::global_memory;
	STATIC_REQUIRE(contains_same_category<mask, compression_category::element_wise>);
	STATIC_REQUIRE(contains_same_category<mask, compression_category::local_memory>);
	STATIC_REQUIRE(contains_same_category<mask, compression_category::element_wise | compression_category::local_memory>);
	STATIC_REQUIRE_FALSE(contains_same_category<mask, compression_category::kernel>);
	STATIC_REQUIRE_FALSE(contains_same_category<mask, compression_category::automatic>);
}

TEST_CASE("compression_checker identifies compression classes", "[compression]") {
	using detail::compression_checker;

	STATIC_REQUIRE(compression_checker<compressed<test_detail::stub_algorithm<int, float>, compression_category::element_wise>>::value);
	STATIC_REQUIRE(compression_checker<compressed<test_detail::stub_algorithm<int, float>, compression_category::local_memory>>::value);
	STATIC_REQUIRE(compression_checker<compressed<test_detail::stub_algorithm<int, float>, compression_category::global_memory>>::value);

	STATIC_REQUIRE_FALSE(compression_checker<int>::value);
	STATIC_REQUIRE_FALSE(compression_checker<compression::uncompressed>::value);
}

TEST_CASE("compression_category_extractor extracts category from compressed<>", "[compression]") {
	using detail::compression_category_extractor;
	STATIC_REQUIRE(compression_category_extractor<compressed<test_detail::stub_algorithm<int, float>, compression_category::element_wise>>::value
	               == compression_category::element_wise);
	STATIC_REQUIRE(compression_category_extractor<compressed<test_detail::stub_algorithm<int, float>, compression_category::local_memory>>::value
	               == compression_category::local_memory);
	STATIC_REQUIRE(compression_category_extractor<compressed<test_detail::stub_algorithm<int, float>, compression_category::global_memory>>::value
	               == compression_category::global_memory);

	STATIC_REQUIRE_FALSE(compression_category_extractor<compressed<test_detail::stub_algorithm<int, float>, compression_category::element_wise>>::value
	                     == compression_category::global_memory);

	STATIC_REQUIRE(compression_category_extractor<int>::value == compression_category::none);
}

TEST_CASE("check_provided_type_with_algorithm accepts/rejects compression categories", "[compression]") {
	STATIC_REQUIRE(check_provided_type_with_algorithm<compressed<test_detail::stub_algorithm<int, float>, compression_category::element_wise>>);
	STATIC_REQUIRE(check_provided_type_with_algorithm<compressed<test_detail::stub_algorithm<int, float>, compression_category::local_memory>>);
	STATIC_REQUIRE(check_provided_type_with_algorithm<
	    compressed<test_detail::stub_algorithm<int, float>, compression_category::element_wise | compression_category::local_memory>>);

	STATIC_REQUIRE_FALSE(check_provided_type_with_algorithm<compressed<test_detail::stub_algorithm<int, float>, compression_category::global_memory>>);
	STATIC_REQUIRE_FALSE(check_provided_type_with_algorithm<compressed<test_detail::stub_algorithm<int, float>, compression_category::kernel>>);
	STATIC_REQUIRE_FALSE(check_provided_type_with_algorithm<compressed<test_detail::stub_algorithm<int, float>, compression_category::automatic>>);

	STATIC_REQUIRE_FALSE(check_provided_type_with_algorithm<int>);
}

TEST_CASE("is_vec, vec_element_type and vec_size identify sycl::vec", "[compression]") {
	struct test_element_type {};

	STATIC_REQUIRE_FALSE(is_vec<float>::value);
	STATIC_REQUIRE(is_vec<sycl::vec<float, 3>>::value);

	STATIC_REQUIRE(std::is_same_v<vec_element_type<float>::type, float>);
	STATIC_REQUIRE(std::is_same_v<vec_element_type<sycl::vec<int, 3>>::type, int>);
	STATIC_REQUIRE(std::is_same_v<vec_element_type<sycl::vec<test_element_type, 3>>::type, test_element_type>);

	STATIC_REQUIRE(vec_size<float>::value == 1);
	STATIC_REQUIRE(vec_size<sycl::vec<int, 1>>::value == 1);
	STATIC_REQUIRE(vec_size<sycl::vec<float, 2>>::value == 2);
	STATIC_REQUIRE(vec_size<sycl::vec<uint8_t, 3>>::value == 3);
}

namespace celerity::test_detail {
struct fake_float_buf {
	float operator[](const celerity::id<1>&) const { return 0.0f; }
};
struct fake_int_buf {
	int32_t operator[](const celerity::id<1>&) const { return 0; }
};
} // namespace celerity::test_detail


TEST_CASE("contains_compress/decompress detect member functions", "[compression]") {
	using ElementWiseAlgorithm = compressed<test_detail::stub_transform<float, int32_t>, compression_category::element_wise>;
	STATIC_REQUIRE(contains_compress<ElementWiseAlgorithm, test_detail::fake_float_buf, 1>);
	STATIC_REQUIRE(contains_decompress<ElementWiseAlgorithm, test_detail::fake_int_buf, 1>);
	STATIC_REQUIRE(contains_compress_decompress<ElementWiseAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1>);

	using MemoryChunkAlgorithm = compressed<test_detail::stub_transform_with_dep<float, int32_t>, compression_category::local_memory>;
	using float_dep_buf = std::reference_wrapper<float>;
	STATIC_REQUIRE(contains_compress_memory_chunk<MemoryChunkAlgorithm, test_detail::fake_float_buf, test_detail::fake_int_buf, 1, float_dep_buf>);
	STATIC_REQUIRE(contains_decompress_memory_chunk<MemoryChunkAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1, float_dep_buf>);
	STATIC_REQUIRE(contains_compress_decompress_memory_chunk<MemoryChunkAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1, float_dep_buf>);

	STATIC_REQUIRE_FALSE(contains_compress<int, test_detail::fake_float_buf, 1>);
}

TEST_CASE("satisfies_* concepts select matching access pattern", "[compression]") {
	using ElementWiseAlgorithm = compressed<test_detail::stub_transform<float, int32_t>, compression_category::element_wise>;
	STATIC_REQUIRE(satisfies_element_wise<ElementWiseAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1>);
	STATIC_REQUIRE_FALSE(satisfies_memory_chunk<ElementWiseAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1>);

	using MemoryChunkAlgorithm = compressed<test_detail::stub_transform_with_dep<float, int32_t>, compression_category::local_memory>;
	STATIC_REQUIRE(satisfies_memory_chunk<MemoryChunkAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1, float>);
	STATIC_REQUIRE(satisfies_compression_interface<MemoryChunkAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1, float>);

	STATIC_REQUIRE_FALSE(satisfies_element_wise<MemoryChunkAlgorithm, test_detail::fake_int_buf, test_detail::fake_float_buf, 1, float>);
	STATIC_REQUIRE_FALSE(satisfies_compression_interface<int, test_detail::fake_int_buf, test_detail::fake_float_buf, 1>);
}

TEMPLATE_TEST_CASE_SIG("accessor deduction guides resolve compressed accessor type", "[compression][accessor]", ((compression_category Category), Category),
    compression_category::element_wise, compression_category::local_memory, compression_category::global_memory) {
	using CompressionAlgorithm = stub_transform_compression<float, int32_t, Category>;
	using CompressedBuffer = buffer<float, 1, CompressionAlgorithm>;

	using acc_t = decltype(accessor{std::declval<CompressedBuffer&>(), std::declval<handler&>(), access::one_to_one{}, celerity::read_only});
	STATIC_REQUIRE(std::is_same_v<accessor<int32_t, 1, access_mode::read, target::device,
	                                  detail::accessor_extension<compressed<test_detail::stub_transform<float, int32_t>, Category>,
	                                      detail::range_mapper_type::one_to_one, typename detail::compression_dependency<CompressionAlgorithm>::type>>,
	    acc_t>);
}

TEST_CASE("compress_data/decompress_data round-trip host array", "[compression]") {
	const compressed<test_detail::stub_transform<float, int32_t>, compression_category::element_wise> stub{};

	const std::vector<float> input{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
	std::vector<int32_t> compressed_data(input.size());
	stub.compress_data(input.data(), compressed_data, input.size());
	CHECK(compressed_data == std::vector<int32_t>{0, -1, -2, -3, -4});

	std::vector<float> output(input.size());
	stub.decompress_data(compressed_data, output, range<1>{input.size()});
	CHECK(output == input);
}

TEST_CASE("downscale_device_specific_mapper maps chunk to tracking slot", "[compression]") {
	const chunk<1> chnk{id<1>{0}, range<1>{8}, range<1>{8}, range<1>{8}};
	const range<2> tracking_buffer_size{1, 10}; // 1 device slot, capacity for 10 tiles
	const detail::region<1> touched_main_range{detail::box<1>(id<1>{0}, id<1>{8})};
	const range<1> main_buffer_size{8};

	const auto result = downscale_device_specific_mapper{}(chnk, tracking_buffer_size, touched_main_range, main_buffer_size);

	CHECK(result == detail::region<2>(detail::box<2>(id<2>{0, 0}, id<2>{1, 1})));
}

TEST_CASE("downscale_device_specific_mapper clamps tile count to capacity", "[compression]") {
	const chunk<1> chnk{id<1>{0}, range<1>{16}, range<1>{16}, range<1>{4}};
	const range<2> tracking_buffer_size{1, 2};
	const detail::region<1> touched_main_range{detail::box<1>(id<1>{0}, id<1>{16})};
	const range<1> main_buffer_size{16};

	const auto result = downscale_device_specific_mapper{}(chnk, tracking_buffer_size, touched_main_range, main_buffer_size);

	CHECK(result == detail::region<2>(detail::box<2>(id<2>{0, 0}, id<2>{1, 2})));
}

namespace celerity::test_detail {
struct stub_same_dims_projection {
	celerity::detail::region<1> operator()(
	    const celerity::chunk<1>&, const celerity::range<1>&, const celerity::detail::region<1>& sbr, const celerity::range<1>&) const {
		return sbr;
	}
};

struct stub_leading_slot_projection {
	celerity::detail::region<2> operator()(
	    const celerity::chunk<1>& chnk, const celerity::range<2>&, const celerity::detail::region<1>& sbr, const celerity::range<1>&) const {
		const auto box = *sbr.get_boxes().begin();
		return celerity::detail::region<2>(
		    celerity::detail::box<2>(id<2>{0, box.get_offset()[0]}, id<2>{chnk.range[0], box.get_offset()[0] + box.get_range()[0]}));
	}
};
} // namespace celerity::test_detail

TEST_CASE("range_mapper_to_map_ranges maps chunks to projected regions", "[compression]") {
	SECTION("same-dims: the projected region is the chunk's own subrange, untouched") {
		const range<1> main_buffer_size{10};
		const range_mapper_to_map_ranges<celerity::access::one_to_one, test_detail::stub_same_dims_projection, 1> mapper{
		    celerity::access::one_to_one{}, test_detail::stub_same_dims_projection{}, main_buffer_size, false};

		const chunk<1> chnk{id<1>{2}, range<1>{4}, range<1>{10}, range<1>{4}};
		const auto result = mapper(chnk, range<1>{10});

		CHECK(result == detail::region<1>(detail::box<1>(id<1>{2}, id<1>{6})));
	}

	SECTION("dims-changing: the chunk and its projected subrange both reach the outer projection") {
		const range<1> main_buffer_size{8};
		const range_mapper_to_map_ranges<celerity::access::one_to_one, test_detail::stub_leading_slot_projection, 1> mapper{
		    celerity::access::one_to_one{}, test_detail::stub_leading_slot_projection{}, main_buffer_size, false};

		const chunk<1> chnk{id<1>{0}, range<1>{8}, range<1>{8}, range<1>{8}};
		const auto result = mapper(chnk, range<2>{1, 10});

		CHECK(result == detail::region<2>(detail::box<2>(id<2>{0, 0}, id<2>{8, 8})));
	}

	SECTION("dims-changing: the chunk's offset threads through into the outer projection's output") {
		const range<1> main_buffer_size{8};
		const range_mapper_to_map_ranges<celerity::access::one_to_one, test_detail::stub_leading_slot_projection, 1> mapper{
		    celerity::access::one_to_one{}, test_detail::stub_leading_slot_projection{}, main_buffer_size, false};

		const chunk<1> chnk{id<1>{4}, range<1>{4}, range<1>{8}, range<1>{4}}; // second half of the buffer
		const auto result = mapper(chnk, range<2>{1, 10});

		CHECK(result == detail::region<2>(detail::box<2>(id<2>{0, 4}, id<2>{4, 8})));
	}
}

TEST_CASE("dependency_bundle declares the defined buffer count", "[compression]") {
	STATIC_REQUIRE(
	    std::tuple_size_v<typename detail::compression_dependency<stub_transform_compression<float, int32_t, compression_category::local_memory>>::type> == 0);

	STATIC_REQUIRE(std::tuple_size_v<
	                   typename detail::compression_dependency<stub_transform_with_dep_compression<float, int32_t, compression_category::local_memory>>::type>
	               == 1);
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "buffers_from_descriptions() creates correctly-sized dependency buffer", "[compression]") {
	stub_transform_with_dep_compression<float, int32_t, compression_category::local_memory> alg;
	const auto buffers = alg.get_dependencies().buffers_from_descriptions(range<3>{4, 6, 100});
	STATIC_REQUIRE(std::tuple_size_v<std::decay_t<decltype(buffers)>> == 1);
	CHECK(std::get<0>(buffers).get_range() == range<2>{4, 6});
}

TEST_CASE("dependency_bundle exposes tracking calculate_size", "[compression]") {
	const stub_transform_compression<float, int32_t, compression_category::local_memory> alg;
	const auto tracking = alg.get_dependencies().tracking();
	CHECK(tracking.calculate_size(range<3>{4, 6, 100}) == range<1>{24});
}

TEST_CASE("buffer_access_description's calculate_size returns total element count", "[compression]") {
	struct dummy_mapper {};
	const buffer_access_description<int, dummy_mapper> desc{};
	CHECK(desc.calculate_size(range<3>{2, 3, 4}) == range<1>{24});
}

namespace celerity::test_detail {
struct mutable_array_view {
	float* data;
	float& operator[](size_t i) const { return data[i]; }
};
struct mutable_2d_view {
	float* data;
	size_t stride;
	struct fake_offset {
		size_t value;
		size_t operator[](int) const { return value; }
	};
	fake_offset get_allocation_offset() const { return {0}; }
	float& operator[](const celerity::id<2>& idx) const { return data[idx[0] * stride + idx[1]]; }
};
} // namespace celerity::test_detail

TEST_CASE("uncompressed_container maps global indices to storage", "[compression]") {
	SECTION("local_memory: indices are relative to the chunk's offset, linearized within the chunk's own range") {
		std::vector<float> storage(4, 0.0f);
		test_detail::mutable_array_view view{storage.data()};
		detail::compression_chunk<1> chunk{id<1>{2}, range<1>{4}, range<1>{10}};

		uncompressed_container<float, 1, test_detail::mutable_array_view, compression_category::local_memory> container(std::move(view), std::move(chunk));

		CHECK(container.get_range() == range<1>{10});
		CHECK(container.get_offset() == id<1>{2});
		CHECK(container.get_chunk_size() == range<1>{4});

		container[id<1>{3}] = 42.0f;
		CHECK(storage[1] == 42.0f);
	}

	SECTION("global_memory: indices are linearized directly, then looked up at [device slot, linear index]") {
		std::vector<float> storage(1 * 6, 0.0f); // 1 device slot, 6 linearized global entries
		test_detail::mutable_2d_view view{storage.data(), 6};
		detail::compression_chunk<1> chunk{id<1>{0}, range<1>{6}, range<1>{6}};

		uncompressed_container<float, 1, test_detail::mutable_2d_view, compression_category::global_memory> container(view, std::move(chunk));

		container[id<1>{4}] = 7.0f;
		CHECK(storage[4] == 7.0f);
	}
}

TEST_CASE("global_compression_state_tracker tracks lock state", "[compression]") {
	using Tracker = global_compression_state_tracker<test_detail::fake_chunk_data_state, test_detail::fake_local_flag>;

	std::vector<int32_t> tile_status(2, 0); // two tiles, both starting at "is_compressed" (0), reader count 0
	int32_t local_flag = 0;
	test_detail::fake_chunk_data_state chunk_state{tile_status.data(), tile_status.size()};
	test_detail::fake_local_flag local_flag_state{&local_flag};
	const Tracker tracker(chunk_state, local_flag_state);
	const celerity::id<1> tile{0};

	SECTION("a single consumer acquires the decompression lock, then the compression lock, and returns to the initial state") {
		tracker.try_get_decompression_lock(tile, true);
		CHECK(tracker.have_decompressing_lock());
		CHECK_FALSE(tracker.have_compressing_lock());

		tracker.try_set_is_decompressed(tile, true);
		CHECK(tracker.get_local_state() == detail::compression_testspy::status_is_decompressed(tracker));

		tracker.try_get_compression_lock(tile, true);
		CHECK(tracker.have_compressing_lock());

		tracker.try_set_is_compressed(tile, true);
		CHECK(tracker.get_local_state() == detail::compression_testspy::status_is_compressed(tracker));

		const auto raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_compressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 0);
	}

	SECTION("two overlapping consumers share the decompression lock via a reader count") {
		tracker.try_get_decompression_lock(tile, true);
		CHECK(tracker.have_decompressing_lock());
		tracker.try_set_is_decompressed(tile, true);

		tracker.try_get_decompression_lock(tile, true);
		CHECK_FALSE(tracker.have_decompressing_lock());
		auto raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 2);

		tracker.try_get_compression_lock(tile, true);
		CHECK_FALSE(tracker.have_compressing_lock());
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 1);

		tracker.try_get_compression_lock(tile, true);
		CHECK(tracker.have_compressing_lock());
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 0);
	}

	SECTION("a producer-only pass (no consumer) still transitions to the compressed state") {
		tracker.try_set_decompressed_no_consumer(tile, true);
		CHECK(tracker.get_local_state() == detail::compression_testspy::status_is_decompressed(tracker));

		tracker.try_set_compressed_no_producer(tile, true);
		CHECK(tracker.get_local_state() == detail::compression_testspy::status_is_compressed(tracker));
	}

	SECTION("three overlapping consumers share the decompression lock via a reader count") {
		tracker.try_get_decompression_lock(tile, true);
		CHECK(tracker.have_decompressing_lock());
		tracker.try_set_is_decompressed(tile, true);

		tracker.try_get_decompression_lock(tile, true);
		CHECK_FALSE(tracker.have_decompressing_lock());
		tracker.try_get_decompression_lock(tile, true);
		CHECK_FALSE(tracker.have_decompressing_lock());

		auto raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_decompressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 3);

		tracker.try_get_compression_lock(tile, true);
		CHECK_FALSE(tracker.have_compressing_lock());
		tracker.try_get_compression_lock(tile, true);
		CHECK_FALSE(tracker.have_compressing_lock());
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 1);

		tracker.try_get_compression_lock(tile, true);
		CHECK(tracker.have_compressing_lock());
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 0);
	}

	SECTION("the tracker returns to the exact rest state after many repeated compress/decompress cycles") {
		for(int cycle = 0; cycle < 50; ++cycle) {
			CAPTURE(cycle);
			tracker.try_get_decompression_lock(tile, true);
			CHECK(tracker.have_decompressing_lock());

			tracker.try_set_is_decompressed(tile, true);
			tracker.try_get_compression_lock(tile, true);
			CHECK(tracker.have_compressing_lock());

			tracker.try_set_is_compressed(tile, true);

			const auto raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
			CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_compressed(tracker));
			CHECK(detail::compression_testspy::extract_count(tracker, raw) == 0);
		}
	}

	SECTION("mixing consumer cycles and producer-only cycles in sequence still returns to the rest state") {
		tracker.try_get_decompression_lock(tile, true);
		tracker.try_set_is_decompressed(tile, true);
		tracker.try_get_compression_lock(tile, true);
		tracker.try_set_is_compressed(tile, true);

		tracker.try_set_decompressed_no_consumer(tile, true);
		tracker.try_set_compressed_no_producer(tile, true);

		tracker.try_get_decompression_lock(tile, true);
		tracker.try_set_is_decompressed(tile, true);
		tracker.try_get_compression_lock(tile, true);
		CHECK(tracker.have_compressing_lock());
		tracker.try_set_is_compressed(tile, true);

		const auto raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_compressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 0);
	}

	SECTION("a producer-only pass with several overlapping writers serializes via the reader count, like consumers do") {
		tracker.try_set_decompressed_no_consumer(tile, true);
		tracker.try_set_decompressed_no_consumer(tile, true);
		tracker.try_set_decompressed_no_consumer(tile, true);

		auto raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_decompressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 3);

		tracker.try_set_compressed_no_producer(tile, true);
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_decompressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 2);

		tracker.try_set_compressed_no_producer(tile, true);
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 1);

		tracker.try_set_compressed_no_producer(tile, true);
		raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, raw) == detail::compression_testspy::status_is_compressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, raw) == 0);

		CHECK_FALSE(tracker.have_decompressing_lock());
		CHECK_FALSE(tracker.have_compressing_lock());
	}

	SECTION("a non-leader call never touches global or local state") {
		const auto before_raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		const auto before_local = tracker.get_local_state();

		tracker.try_get_decompression_lock(tile, false);
		tracker.try_set_is_decompressed(tile, false);
		tracker.try_get_compression_lock(tile, false);
		tracker.try_set_is_compressed(tile, false);
		tracker.try_set_decompressed_no_consumer(tile, false);
		tracker.try_set_compressed_no_producer(tile, false);

		const auto after_raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(after_raw == before_raw);
		CHECK(tracker.get_local_state() == before_local);
	}

	SECTION("two tiles managed by the same tracker maintain fully independent lock state") {
		const celerity::id<1> other_tile{1};

		tracker.try_get_decompression_lock(tile, true);
		CHECK(tracker.have_decompressing_lock());

		const auto other_raw = detail::compression_testspy::get_raw_status_and_count(tracker, other_tile);
		CHECK(detail::compression_testspy::extract_status(tracker, other_raw) == detail::compression_testspy::status_is_compressed(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, other_raw) == 0);

		tracker.try_get_decompression_lock(other_tile, true);
		CHECK(tracker.have_decompressing_lock());

		const auto first_raw = detail::compression_testspy::get_raw_status_and_count(tracker, tile);
		CHECK(detail::compression_testspy::extract_status(tracker, first_raw) == detail::compression_testspy::status_decompressing(tracker));
		CHECK(detail::compression_testspy::extract_count(tracker, first_raw) == 1);
	}
}

TEST_CASE("only the lock holder is allowed to finish decompression or compression of a tile", "[compression]") {
	using Tracker = global_compression_state_tracker<test_detail::fake_chunk_data_state, test_detail::fake_local_flag>;

	std::vector<int32_t> tile_status(1, 0);
	test_detail::fake_chunk_data_state chunk_state{tile_status.data(), tile_status.size()};
	const celerity::id<1> tile{0};

	int32_t a_flag = 0;
	int32_t b_flag = 0;
	test_detail::fake_local_flag a_local{&a_flag};
	test_detail::fake_local_flag b_local{&b_flag};
	const Tracker a(chunk_state, a_local);
	const Tracker b(chunk_state, b_local);

	const auto status = [&] { return detail::compression_testspy::extract_status(a, detail::compression_testspy::get_raw_status_and_count(a, tile)); };
	const auto count = [&] { return detail::compression_testspy::extract_count(a, detail::compression_testspy::get_raw_status_and_count(a, tile)); };

	SECTION("a reader that joined while the tile was decompressing cannot mark it decompressed") {
		a.try_get_decompression_lock(tile, true);
		REQUIRE(a.have_decompressing_lock());

		b.try_get_decompression_lock(tile, true);
		REQUIRE_FALSE(b.have_decompressing_lock());

		if(b.have_decompressing_lock()) { b.try_set_is_decompressed(tile, true); }
		CHECK(status() == detail::compression_testspy::status_decompressing(a));

		if(a.have_decompressing_lock()) { a.try_set_is_decompressed(tile, true); }
		CHECK(status() == detail::compression_testspy::status_is_decompressed(a));
		CHECK(count() == 2);
	}

	SECTION("a workgroup that released its reference cannot mark the tile compressed while another one compresses it") {
		a.try_get_decompression_lock(tile, true);
		a.try_set_is_decompressed(tile, true);
		b.try_get_decompression_lock(tile, true);
		REQUIRE(count() == 2);

		a.try_get_compression_lock(tile, true);
		REQUIRE_FALSE(a.have_compressing_lock());
		b.try_get_compression_lock(tile, true);
		REQUIRE(b.have_compressing_lock());

		if(a.have_compressing_lock()) { a.try_set_is_compressed(tile, true); }
		CHECK(status() == detail::compression_testspy::status_compressing(a));

		if(b.have_compressing_lock()) { b.try_set_is_compressed(tile, true); }
		CHECK(status() == detail::compression_testspy::status_is_compressed(a));
		CHECK(count() == 0);
	}

	SECTION("waiting for a decompressed tile returns immediately when nobody is decompressing it") {
		a.try_get_decompression_lock(tile, true);
		a.try_set_is_decompressed(tile, true);
		b.try_get_decompression_lock(tile, true);
		b.wait_until_decompressed(tile, true); // must not block
		a.wait_until_decompressed(tile, true);
		CHECK(status() == detail::compression_testspy::status_is_decompressed(a));
	}
}

#if !CELERITY_SYCL_IS_SIMSYCL

TEST_CASE("a reader that finds a tile decompressing waits until the lock holder published it", "[compression][thread]") {
	using Tracker = global_compression_state_tracker<test_detail::fake_chunk_data_state, test_detail::fake_local_flag>;

	std::vector<int32_t> tile_status(1, 0);
	test_detail::fake_chunk_data_state chunk_state{tile_status.data(), tile_status.size()};
	const celerity::id<1> tile{0};

	int32_t holder_flag = 0;
	test_detail::fake_local_flag holder_local{&holder_flag};
	const Tracker holder(chunk_state, holder_local);

	int32_t reader_flag = 0;
	test_detail::fake_local_flag reader_local{&reader_flag};
	const Tracker reader(chunk_state, reader_local);

	int payload = 0; // stands in for the uncompressed tile, published by the state change only

	holder.try_get_decompression_lock(tile, true);
	REQUIRE(holder.have_decompressing_lock());

	std::atomic<bool> reader_done{false};
	int payload_seen_by_reader = -1;
	std::thread reader_thread([&] {
		reader.try_get_decompression_lock(tile, true);
		if(reader.have_decompressing_lock()) { reader.try_set_is_decompressed(tile, true); }
		reader.wait_until_decompressed(tile, true);
		payload_seen_by_reader = payload;
		reader_done.store(true, std::memory_order_release);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	CHECK_FALSE(reader_done.load(std::memory_order_acquire));

	payload = 42;
	holder.try_set_is_decompressed(tile, true);

	// bounded poll instead of a raw join(): fails fast if the reader never sees the tile as decompressed
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while(!reader_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	REQUIRE(reader_done.load(std::memory_order_acquire));
	reader_thread.join();

	CHECK(payload_seen_by_reader == 42);
	const auto raw = detail::compression_testspy::get_raw_status_and_count(holder, tile);
	CHECK(detail::compression_testspy::extract_status(holder, raw) == detail::compression_testspy::status_is_decompressed(holder));
	CHECK(detail::compression_testspy::extract_count(holder, raw) == 2);
}

TEST_CASE("global_compression_state_tracker elects one leader under contention", "[compression][thread]") {
	using Tracker = global_compression_state_tracker<test_detail::fake_chunk_data_state, test_detail::fake_local_flag>;

	constexpr int num_threads = 16;
	std::vector<int32_t> tile_status(1, 0); // starts at "is_compressed", reader count 0
	test_detail::fake_chunk_data_state chunk_state{tile_status.data(), tile_status.size()};
	const celerity::id<1> tile{0};

	std::vector<int32_t> local_flags(num_threads, 0);
	std::vector<test_detail::fake_local_flag> local_flag_states;
	local_flag_states.reserve(num_threads);
	for(auto& flag : local_flags) {
		local_flag_states.push_back(test_detail::fake_local_flag{&flag});
	}

	std::vector<Tracker> trackers;
	trackers.reserve(num_threads);
	for(int i = 0; i < num_threads; ++i) {
		trackers.emplace_back(chunk_state, local_flag_states[i]);
	}

	const auto run_phase = [&](auto&& phase_fn) {
		std::atomic<bool> start{false};
		std::vector<std::thread> threads;
		threads.reserve(num_threads);
		for(int i = 0; i < num_threads; ++i) {
			threads.emplace_back([&, i] {
				while(!start.load(std::memory_order_acquire)) {
					std::this_thread::yield();
				}
				phase_fn(trackers[i]);
			});
		}
		start.store(true, std::memory_order_release);
		for(auto& t : threads) {
			t.join();
		}
	};

	std::atomic<int> decompressing_leaders{0};
	run_phase([&](const Tracker& t) {
		t.try_get_decompression_lock(tile, true);
		if(t.have_decompressing_lock()) { decompressing_leaders.fetch_add(1, std::memory_order_relaxed); }
	});

	CHECK(decompressing_leaders.load() == 1);
	auto raw = detail::compression_testspy::get_raw_status_and_count(trackers[0], tile);
	CHECK(detail::compression_testspy::extract_status(trackers[0], raw) == detail::compression_testspy::status_decompressing(trackers[0]));
	CHECK(detail::compression_testspy::extract_count(trackers[0], raw) == num_threads);

	run_phase([&](const Tracker& t) { t.try_set_is_decompressed(tile, true); });

	raw = detail::compression_testspy::get_raw_status_and_count(trackers[0], tile);
	CHECK(detail::compression_testspy::extract_status(trackers[0], raw) == detail::compression_testspy::status_is_decompressed(trackers[0]));
	CHECK(detail::compression_testspy::extract_count(trackers[0], raw) == num_threads);

	std::atomic<int> compressing_leaders{0};
	run_phase([&](const Tracker& t) {
		t.try_get_compression_lock(tile, true);
		if(t.have_compressing_lock()) { compressing_leaders.fetch_add(1, std::memory_order_relaxed); }
	});

	CHECK(compressing_leaders.load() == 1);
	raw = detail::compression_testspy::get_raw_status_and_count(trackers[0], tile);
	CHECK(detail::compression_testspy::extract_status(trackers[0], raw) == detail::compression_testspy::status_compressing(trackers[0]));
	CHECK(detail::compression_testspy::extract_count(trackers[0], raw) == 0);

	run_phase([&](const Tracker& t) { t.try_set_is_compressed(tile, true); });

	raw = detail::compression_testspy::get_raw_status_and_count(trackers[0], tile);
	CHECK(detail::compression_testspy::extract_status(trackers[0], raw) == detail::compression_testspy::status_is_compressed(trackers[0]));
	CHECK(detail::compression_testspy::extract_count(trackers[0], raw) == 0);
}

TEST_CASE("pending compression blocks concurrent decompression, no deadlock", "[compression][thread]") {
	using Tracker = global_compression_state_tracker<test_detail::fake_chunk_data_state, test_detail::fake_local_flag>;

	std::vector<int32_t> tile_status(1, 0);
	test_detail::fake_chunk_data_state chunk_state{tile_status.data(), tile_status.size()};
	const celerity::id<1> tile{0};

	int32_t compressor_local_flag = 0;
	test_detail::fake_local_flag compressor_local_state{&compressor_local_flag};
	const Tracker compressor(chunk_state, compressor_local_state);

	int32_t waiter_local_flag = 0;
	test_detail::fake_local_flag waiter_local_state{&waiter_local_flag};
	const Tracker waiter(chunk_state, waiter_local_state);

	compressor.try_get_decompression_lock(tile, true);
	compressor.try_set_is_decompressed(tile, true);
	compressor.try_get_compression_lock(tile, true);
	REQUIRE(compressor.have_compressing_lock());

	std::atomic<bool> waiter_done{false};
	std::thread waiter_thread([&] {
		waiter.try_get_decompression_lock(tile, true);
		waiter_done.store(true, std::memory_order_release);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	CHECK_FALSE(waiter_done.load(std::memory_order_acquire));

	compressor.try_set_is_compressed(tile, true);

	// bounded poll instead of a raw join(): fails fast if the tracker is deadlocked, instead of hanging
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while(!waiter_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	CHECK(waiter_done.load(std::memory_order_acquire));

	waiter_thread.join();

	CHECK(waiter.have_decompressing_lock());
	const auto raw = detail::compression_testspy::get_raw_status_and_count(compressor, tile);
	CHECK(detail::compression_testspy::extract_status(compressor, raw) == detail::compression_testspy::status_decompressing(compressor));
	CHECK(detail::compression_testspy::extract_count(compressor, raw) == 1);
}

TEST_CASE("many threads cycling many tiles settle at the rest state", "[compression][thread]") {
	using Tracker = global_compression_state_tracker<test_detail::fake_chunk_data_state, test_detail::fake_local_flag>;

	constexpr int num_tiles = 8;
	constexpr int num_threads = 16;
	constexpr int iterations_per_thread = 200;

	std::vector<int32_t> tile_status(num_tiles, 0);
	test_detail::fake_chunk_data_state chunk_state{tile_status.data(), tile_status.size()};

	std::vector<std::thread> threads;
	threads.reserve(num_threads);
	for(int worker = 0; worker < num_threads; ++worker) {
		threads.emplace_back([&, worker] {
			int32_t local_flag = 0;
			test_detail::fake_local_flag local_flag_state{&local_flag};
			const Tracker tracker(chunk_state, local_flag_state);

			for(int i = 0; i < iterations_per_thread; ++i) {
				const celerity::id<1> tile{static_cast<size_t>((worker + i) % num_tiles)};

				tracker.try_get_decompression_lock(tile, true);
				std::this_thread::yield();
				tracker.try_set_is_decompressed(tile, true);
				std::this_thread::yield();
				tracker.try_get_compression_lock(tile, true);
				std::this_thread::yield();
				tracker.try_set_is_compressed(tile, true);
			}
		});
	}
	for(auto& t : threads) {
		t.join();
	}

	int32_t observer_local_flag = 0;
	test_detail::fake_local_flag observer_local_state{&observer_local_flag};
	const Tracker observer(chunk_state, observer_local_state);

	for(int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
		CAPTURE(tile_idx);
		const auto raw = detail::compression_testspy::get_raw_status_and_count(observer, celerity::id<1>{static_cast<size_t>(tile_idx)});
		CHECK(detail::compression_testspy::extract_status(observer, raw) == detail::compression_testspy::status_is_compressed(observer));
		CHECK(detail::compression_testspy::extract_count(observer, raw) == 0);
	}
}
#endif // !CELERITY_SYCL_IS_SIMSYCL

TEST_CASE("uncompressed_item_wrapper compresses on write, decompresses on read", "[compression]") {
	using Algorithm = test_detail::stub_transform<float, int32_t>;
	const compressed<Algorithm, compression_category::element_wise> compression{};
	const celerity::id<1> idx{0};

	SECTION("uncompressed_item_wrapper (producer accessors: assignment compresses, conversion decompresses)") {
		int32_t storage = 0;
		uncompressed_item_wrapper<float, int32_t, 1, Algorithm, compression_category::element_wise> wrapper{storage, idx, compression};

		wrapper = 7.0f;
		CHECK(storage == -7);

		const float decompressed = wrapper;
		CHECK(decompressed == 7.0f);

		CHECK(static_cast<int32_t>(wrapper) == storage);
	}

	SECTION("uncompressed_item_wrapper_const (consumer-only accessors only ever decompress)") {
		const int32_t storage = -9;
		const uncompressed_item_wrapper_const<float, int32_t, 1, Algorithm, compression_category::element_wise> wrapper{storage, idx, compression};

		const float decompressed = wrapper;
		CHECK(decompressed == 9.0f);
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "element-wise device accessors wire compression at runtime", "[compression][accessor]") {
	using CompressionAlgorithm = stub_transform_compression<float, int32_t, compression_category::element_wise>;
	CompressionAlgorithm compression;

	constexpr size_t n = 4;
	buffer<float, 1, CompressionAlgorithm> buf(range<1>{n}, compression);
	buffer<float, 1> decompressed_out(range<1>{n});

	queue q;
	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		accessor out_acc{decompressed_out, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(buf.get_range(), [=](item<1> it) {
			acc[it] = static_cast<float>(it.get_linear_id());
			out_acc[it] = acc[it];
		});
	});

	std::array<float, n> decompressed_result{};
	q.submit([&](handler& cgh) {
		accessor out_acc{decompressed_out, cgh, access::all{}, celerity::read_only_host_task};
		cgh.host_task(on_master_node, [=, &decompressed_result] {
			for(size_t i = 0; i < n; ++i) {
				decompressed_result[i] = out_acc[i];
			}
		});
	});
	q.wait();

	for(size_t i = 0; i < n; ++i) {
		CAPTURE(i);
		CHECK(decompressed_result[i] == static_cast<float>(i));
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "element-wise host_task accessor wires compression and decompress_data()", "[compression][accessor]") {
	using CompressionAlgorithm = stub_transform_compression<float, int32_t, compression_category::element_wise>;
	CompressionAlgorithm compression;

	constexpr size_t n = 5;
	buffer<float, 1, CompressionAlgorithm> buf(range<1>{n}, compression);

	queue q;
	std::vector<float> bulk_result;
	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::all{}, celerity::read_write_host_task, celerity::no_init};
		cgh.host_task(on_master_node, [=, &bulk_result] {
			for(size_t i = 0; i < n; ++i) {
				acc[celerity::id<1>{i}] = static_cast<float>(i);
			}
			bulk_result = acc.decompress_data();
		});
	});
	q.wait();

	REQUIRE(bulk_result.size() == n);
	for(size_t i = 0; i < n; ++i) {
		CAPTURE(i);
		CHECK(bulk_result[i] == static_cast<float>(i));
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "local-memory accessors stage via local_accessor", "[compression][accessor]") {
	using CompressionAlgorithm = stub_transform_compression<float, int32_t, compression_category::local_memory>;
	CompressionAlgorithm compression;

	constexpr size_t n = 8;
	buffer<float, 1, CompressionAlgorithm> buf(range<1>{n}, compression);

	queue q;
	buf.init(q);

	buffer<int, 1> move_flags(range<1>{2});
	buffer<int32_t, 1> raw_bytes(range<1>{n});
	buffer<float, 1> decompressed_out(range<1>{n});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		accessor flags_acc{move_flags, cgh, access::all{}, celerity::write_only, celerity::no_init};
		local_accessor<float> local_mem({n}, cgh);

		cgh.parallel_for(nd_range<1>(buf.get_range(), buf.get_range()), [=](nd_item<1> item) {
			auto alloc = make_local_allocator<float>(n, local_mem);
			auto staged = acc.decompress_data(item, alloc);
			auto moved_staged = std::move(staged);

			if(item.get_local_linear_id() == 0) {
				flags_acc[0] = detail::compression_testspy::is_moved(staged) ? 1 : 0;
				flags_acc[1] = detail::compression_testspy::is_moved(moved_staged) ? 1 : 0;
			}

			moved_staged[item.get_global_id()] = static_cast<float>(item.get_global_linear_id());
		});
	});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::read_only};
		auto raw_acc = detail::compression_testspy::get_raw_storage_accessor(acc);
		accessor raw_out_acc{raw_bytes, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(buf.get_range(), [=](item<1> it) { raw_out_acc[it] = raw_acc[it]; });
	});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::read_only};
		accessor out_acc{decompressed_out, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		local_accessor<float> local_mem({n}, cgh);

		cgh.parallel_for(nd_range<1>(buf.get_range(), buf.get_range()), [=](nd_item<1> item) {
			auto alloc = make_local_allocator<float>(n, local_mem);
			auto staged = acc.decompress_data(item, alloc);
			out_acc[item.get_global_id()] = staged[item.get_global_id()];
		});
	});

	std::array<int, 2> flags_result{};
	std::array<int32_t, n> raw_result{};
	std::array<float, n> decompressed_result{};
	q.submit([&](handler& cgh) {
		accessor flags_acc{move_flags, cgh, access::all{}, celerity::read_only_host_task};
		accessor raw_acc{raw_bytes, cgh, access::all{}, celerity::read_only_host_task};
		accessor out_acc{decompressed_out, cgh, access::all{}, celerity::read_only_host_task};
		cgh.host_task(on_master_node, [=, &flags_result, &raw_result, &decompressed_result] {
			flags_result[0] = flags_acc[0];
			flags_result[1] = flags_acc[1];
			for(size_t i = 0; i < n; ++i) {
				raw_result[i] = raw_acc[i];
				decompressed_result[i] = out_acc[i];
			}
		});
	});
	q.wait();

	CHECK(flags_result[0] == 1);
	CHECK(flags_result[1] == 0);

	for(size_t i = 0; i < n; ++i) {
		CAPTURE(i);
		CHECK(raw_result[i] == -static_cast<int32_t>(i));
		CHECK(decompressed_result[i] == static_cast<float>(i));
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "global-memory accessors round-trip via decompressed cache", "[compression][accessor]") {
	// device_specific_range_mapper logs a diagnostic warning on every invocation under CELERITY_ACCESSOR_BOUNDARY_CHECK; expected here
	test_utils::allow_max_log_level(detail::log_level::warn);

	using CompressionAlgorithm = stub_transform_compression<float, int32_t, compression_category::global_memory>;
	CompressionAlgorithm compression;

	constexpr size_t n = 8;
	buffer<float, 1, CompressionAlgorithm> buf(range<1>{n}, compression);

	queue q;
	buf.init(q);

	buffer<int32_t, 1> raw_bytes(range<1>{n});
	buffer<float, 1> decompressed_out(range<1>{n});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(nd_range<1>(buf.get_range(), buf.get_range()), [=](nd_item<1> item) {
			auto staged = acc.decompress_data(item);
			staged[item.get_global_id()] = static_cast<float>(item.get_global_linear_id());
		});
	});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::read_only};
		auto raw_acc = detail::compression_testspy::get_compressed_storage_accessor(acc);
		accessor raw_out_acc{raw_bytes, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(buf.get_range(), [=](item<1> it) {
			(void)acc;
			raw_out_acc[it] = raw_acc[it];
		});
	});

	q.submit([&](handler& cgh) {
		accessor acc{buf, cgh, access::one_to_one{}, celerity::read_only};
		accessor out_acc{decompressed_out, cgh, access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(nd_range<1>(buf.get_range(), buf.get_range()), [=](nd_item<1> item) {
			auto staged = acc.decompress_data(item);
			out_acc[item.get_global_id()] = staged[item.get_global_id()];
		});
	});

	std::array<int32_t, n> raw_result{};
	std::array<float, n> decompressed_result{};
	q.submit([&](handler& cgh) {
		accessor raw_acc{raw_bytes, cgh, access::all{}, celerity::read_only_host_task};
		accessor out_acc{decompressed_out, cgh, access::all{}, celerity::read_only_host_task};
		cgh.host_task(on_master_node, [=, &raw_result, &decompressed_result] {
			for(size_t i = 0; i < n; ++i) {
				raw_result[i] = raw_acc[i];
				decompressed_result[i] = out_acc[i];
			}
		});
	});
	q.wait();

	for(size_t i = 0; i < n; ++i) {
		CAPTURE(i);
		CHECK(raw_result[i] == -static_cast<int32_t>(i));
		CHECK(decompressed_result[i] == static_cast<float>(i));
	}
}
