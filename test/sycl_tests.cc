#include "test_utils.h"

namespace celerity::detail {

template <int Dims>
class dim_device_queue_fixture : public test_utils::device_queue_fixture {};

template <int Dims>
class accessor_range_test_kernel;

TEMPLATE_TEST_CASE_METHOD_SIG(dim_device_queue_fixture, "ranged_sycl_access works around differences in offset computation for device accessors",
    "[sycl][accessor]", ((int Dims), Dims), 1, 2, 3) {
	constexpr static size_t tile_width = 2;
	constexpr static size_t num_tiles[] = {2, 3, 4};

	const auto tile_range = range_cast<Dims>(sycl::range<3>{tile_width, tile_width, tile_width});
	const auto buffer_range = range_cast<Dims>(sycl::range<3>{2 * tile_width, 3 * tile_width, 4 * tile_width});

	// GENERATE macros cannot be executed in a loop
	sycl::id<Dims> offset;
	if constexpr(Dims >= 1) { offset[0] = GENERATE(Catch::Generators::range(size_t{0}, num_tiles[0])) * tile_width; }
	if constexpr(Dims >= 2) { offset[1] = GENERATE(Catch::Generators::range(size_t{0}, num_tiles[1])) * tile_width; }
	if constexpr(Dims >= 3) { offset[2] = GENERATE(Catch::Generators::range(size_t{0}, num_tiles[2])) * tile_width; }

	CAPTURE(buffer_range);
	CAPTURE(tile_range);
	CAPTURE(offset);

	auto& q = dim_device_queue_fixture<Dims>::get_device_queue();

	sycl::buffer<int, Dims> buf{buffer_range};
	q.submit([&](sycl::handler& cgh) { cgh.fill(buf.template get_access<sycl::access_mode::discard_write>(cgh), -1); });

	q.submit([&](sycl::handler& cgh) {
		const auto acc = buf.template get_access<sycl::access_mode::read_write>(cgh, tile_range, offset);
		const auto buf_range = buf.get_range();
		cgh.parallel_for<bind_kernel_name<accessor_range_test_kernel<Dims>>>(tile_range, [=](const sycl::id<Dims> rel_index) {
			const auto abs_index = offset + rel_index;
			int value = 0;
			for(int d = 0; d < Dims; ++d) {
				value = 10 * value + static_cast<int>(abs_index[d]);
			}
			ranged_sycl_access(acc, buf_range, rel_index) += 1 + value;
		});
	});

	std::vector<int> expected;
	expected.reserve(buffer_range.size());
	test_utils::for_each_in_range(buffer_range, [&](const id<Dims> index) {
		bool inside_tile = true;
		int value = 0;
		for(int d = 0; d < Dims; ++d) {
			inside_tile &= index[d] >= offset[d] && index[d] < offset[d] + tile_width;
			value = 10 * value + static_cast<int>(index[d]);
		}
		expected.push_back(inside_tile ? value : -1);
	});

	std::vector<int> actual(buf.get_range().size());
	q.submit([&](sycl::handler& cgh) { cgh.copy(buf.template get_access<sycl::access::mode::read>(cgh), actual.data()); }).wait_and_throw();

	CHECK(actual == expected);
}

} // namespace celerity::detail
