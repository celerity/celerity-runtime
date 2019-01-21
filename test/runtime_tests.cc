#define COMPILE_SYCL_KERNELS 1
#include <algorithm>
#include <memory>
#include <random>

#include <CL/sycl.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "region_map.h"

#include "test_utils.h"

GridBox<3> make_grid_box(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
	const auto end = cl::sycl::range<3>(offset) + range;
	return {celerity::detail::sycl_range_to_grid_point(cl::sycl::range<3>(offset)), celerity::detail::sycl_range_to_grid_point(end)};
}

GridRegion<3> make_grid_region(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
	return GridRegion<3>(make_grid_box(range, offset));
}

struct test_context {
	std::unique_ptr<celerity::distr_queue> queue = nullptr;
	test_context() {
		celerity::runtime::init_for_testing();
		queue = std::make_unique<celerity::distr_queue>();
	}
};

namespace celerity {

TEST_CASE("region_map correctly handles region updates", "[region_map]") {
	detail::region_map<std::string> rm(cl::sycl::range<3>(256, 128, 1));

	rm.update_region(make_grid_region({256, 1, 1}), "foo");
	{
		const auto rvs = rm.get_region_values(make_grid_region({32, 1, 1}, {32, 0, 0}));
		REQUIRE(rvs.size() == 1);
		REQUIRE(rvs[0].first == make_grid_box({32, 1, 1}, {32, 0, 0}));
		REQUIRE(rvs[0].second == "foo");
	}

	rm.update_region(make_grid_region({64, 1, 1}), "baz");
	{
		const auto rvs = rm.get_region_values(make_grid_region({256, 1, 1}));
		REQUIRE(rvs.size() == 2);
		REQUIRE(rvs[1].first == make_grid_box({64, 1, 1}));
		REQUIRE(rvs[1].second == "baz");
		REQUIRE(rvs[0].first == make_grid_box({192, 1, 1}, {64, 0, 0}));
		REQUIRE(rvs[0].second == "foo");
	}
}

TEST_CASE("region_map collapses stored regions with the same values", "[region_map]") {
	// We test region_map<>::collapse_regions by observing the order of the
	// returned boxes. This somewhat relies on implementation details of
	// region_map<>::get_region_values.
	// TODO: We may want to test this directly instead
	detail::region_map<std::unordered_set<size_t>> rm(cl::sycl::range<3>(256, 1, 1));
	rm.update_region(make_grid_region({64, 1, 1}, {64, 0, 0}), {1});
	rm.update_region(make_grid_region({64, 1, 1}, {192, 0, 0}), {1});

	auto rvs = rm.get_region_values(make_grid_region({192, 1, 1}, {64, 0, 0}));
	REQUIRE(rvs.size() == 3);
	REQUIRE(rvs[0].first == make_grid_box({64, 1, 1}, {64, 0, 0}));
	REQUIRE(rvs[0].second.size() == 1);
	REQUIRE(rvs[0].second.count(1) == 1);

	// Since this one is returned before the [128,192) box,
	// the {[64,128), [192,256)} region must exist internally.
	REQUIRE(rvs[1].first == make_grid_box({64, 1, 1}, {192, 0, 0}));
	REQUIRE(rvs[1].second.size() == 1);
	REQUIRE(rvs[1].second.count(1) == 1);

	REQUIRE(rvs[2].first == make_grid_box({64, 1, 1}, {128, 0, 0}));
	// This is the default initialized region that was never updated
	REQUIRE(rvs[2].second.empty());
}

TEST_CASE("region_map correctly merges with other instance", "[region_map]") {
	detail::region_map<size_t> rm1(cl::sycl::range<3>(128, 64, 32));
	detail::region_map<size_t> rm2(cl::sycl::range<3>(128, 64, 32));
	rm1.update_region(make_grid_region({128, 64, 32}, {0, 0, 0}), 5);
	rm2.update_region(make_grid_region({128, 8, 1}, {0, 24, 0}), 1);
	rm2.update_region(make_grid_region({128, 24, 1}, {0, 0, 0}), 2);
	rm1.merge(rm2);

	const auto rvs = rm1.get_region_values(make_grid_region({128, 64, 32}));
	REQUIRE(rvs.size() == 4);
	REQUIRE(rvs[0].first == make_grid_box({128, 32, 31}, {0, 0, 1}));
	REQUIRE(rvs[0].second == 5);

	REQUIRE(rvs[1].first == make_grid_box({128, 32, 32}, {0, 32, 0}));
	REQUIRE(rvs[1].second == 5);

	REQUIRE(rvs[2].first == make_grid_box({128, 24, 1}, {0, 0, 0}));
	REQUIRE(rvs[2].second == 2);

	REQUIRE(rvs[3].first == make_grid_box({128, 8, 1}, {0, 24, 0}));
	REQUIRE(rvs[3].second == 1);

	// Attempting to merge region maps with incompatible extents should throw
	const detail::region_map<size_t> rm_incompat(cl::sycl::range<3>(128, 64, 30));
	REQUIRE_THROWS_WITH(rm1.merge(rm_incompat), Catch::Equals("Incompatible region map"));
}

TEST_CASE("host_accessor 1D indexing behaves the same way as a SYCL host-accessor", "[host_accessor]") {
	test_context ctx;
	auto cel_host_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::HOST_BUFFER, cl::sycl::range<1>(15));
	auto cel_device_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::DEVICE_BUFFER, cl::sycl::range<1>(15));
	auto sycl_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::DEVICE_BUFFER, cl::sycl::range<1>(15));

	float test_values[15];
	std::mt19937 gen(1337);
	std::uniform_real_distribution<float> dis(0.f, 10.f);
	for(int i = 0; i < 15; ++i) {
		test_values[i] = dis(gen);
	}
	{
		host_accessor<float, 1, cl::sycl::access::mode::write> cel_host_acc(cel_host_buffer, cl::sycl::range<1>(6), cl::sycl::id<1>(8));
		host_accessor<float, 1, cl::sycl::access::mode::write> cel_device_acc(cel_device_buffer, cl::sycl::range<1>(6), cl::sycl::id<1>(8));
		auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::write>(cl::sycl::range<1>(6), cl::sycl::id<1>(8));
		for(auto i = 8u; i < 14; ++i) {
			cel_host_acc[i] = test_values[i];
			cel_device_acc[i] = test_values[i];
			sycl_acc[i] = test_values[i];
		}
	}
	host_accessor<float, 1, cl::sycl::access::mode::read> cel_host_acc(cel_host_buffer, cl::sycl::range<1>(15));
	host_accessor<float, 1, cl::sycl::access::mode::read> cel_device_acc(cel_device_buffer, cl::sycl::range<1>(15));
	auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(15));
	for(auto i = 8u; i < 14; ++i) {
		REQUIRE(cel_host_acc[i] == test_values[i]);
		REQUIRE(cel_device_acc[i] == test_values[i]);
		REQUIRE(sycl_acc[i] == test_values[i]);

		// Also test pointer access.
		// TODO: Move into separate test, add offsets (we likely don't handle this correctly, see 2D case below)
		REQUIRE(*(cel_host_acc.get_pointer() + i) == test_values[i]);
		REQUIRE(*(cel_device_acc.get_pointer() + i) == test_values[i]);
		REQUIRE(*(sycl_acc.get_pointer() + i) == test_values[i]);
	}
}

TEST_CASE("host_accessor 2D indexing behaves the same way as a SYCL host-accessor", "[host_accessor]") {
	test_context ctx;
	auto cel_host_buffer = std::make_shared<detail::buffer_storage<float, 2>>(detail::buffer_type::HOST_BUFFER, cl::sycl::range<2>(37, 22));
	auto cel_device_buffer = std::make_shared<detail::buffer_storage<float, 2>>(detail::buffer_type::DEVICE_BUFFER, cl::sycl::range<2>(37, 22));
	auto sycl_buffer = std::make_shared<detail::buffer_storage<float, 2>>(detail::buffer_type::DEVICE_BUFFER, cl::sycl::range<2>(37, 22));

	float test_values[7 * 6];
	std::mt19937 gen(1337);
	std::uniform_real_distribution<float> dis(0.f, 10.f);
	for(int i = 0; i < 7 * 6; ++i) {
		test_values[i] = dis(gen);
	}
	{
		host_accessor<float, 2, cl::sycl::access::mode::write> cel_host_acc(cel_host_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
		host_accessor<float, 2, cl::sycl::access::mode::write> cel_device_acc(cel_device_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
		auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::write>(cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
		for(auto i = 30u; i < 37; ++i) {
			for(auto j = 16u; j < 22; ++j) {
				const float value = test_values[(i - 30) * 6 + j - 16];
				cel_host_acc[{i, j}] = value;
				cel_device_acc[{i, j}] = value;
				sycl_acc[{i, j}] = value;
			}
		}
	}
	host_accessor<float, 2, cl::sycl::access::mode::read> cel_host_acc(cel_host_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
	host_accessor<float, 2, cl::sycl::access::mode::read> cel_device_acc(cel_device_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
	auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::read>(cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
	for(auto i = 30u; i < 37; ++i) {
		for(auto j = 16u; j < 22; ++j) {
			const auto subrange_offset = (i - 30) * 6 + j - 16;
			const float expected_value = test_values[subrange_offset];
			REQUIRE(cel_host_acc[{i, j}] == expected_value);
			REQUIRE(cel_device_acc[{i, j}] == expected_value);
			REQUIRE(sycl_acc[{i, j}] == expected_value);

			// Also test pointer access.
			// FIXME: SYCL appears to set the pointer to the global first item regardless of offset. We currently don't!
			// Figure this out: Either it points into an unrelated memory region, or it always allocates the full buffer size (which seems wasteful...)
			REQUIRE(*(cel_host_acc.get_pointer() + subrange_offset) == expected_value);
			REQUIRE(*(cel_device_acc.get_pointer() + subrange_offset) == expected_value);
			const auto full_size_offset = i * 22 + j;
			REQUIRE(*(sycl_acc.get_pointer() + full_size_offset) == expected_value);
		}
	}
}

TEST_CASE("host_accessor 3D indexing behaves the same way as a SYCL host-accessor", "[host_accessor]") {
	// TODO
}

TEST_CASE("host_accessor can be written if captured by value", "[host_accessor]") {
	test_context ctx;
	auto host_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::HOST_BUFFER, cl::sycl::range<1>(10));

	{
		host_accessor<float, 1, cl::sycl::access::mode::write> host_acc(host_buffer, cl::sycl::range<1>(10));
		[=]() { host_acc[5] = 22.f; }();
	}

	{
		host_accessor<float, 1, cl::sycl::access::mode::read> host_acc(host_buffer, cl::sycl::range<1>(10));
		REQUIRE(host_acc[5] == 22.f);
	}
}

// NOTE: We assume buffer to be of size 10 here
template <cl::sycl::access::mode ProducerMode, cl::sycl::access::mode ConsumerMode = cl::sycl::access::mode::read, typename DataT, int Dims>
void test_host_accessor_mode(std::shared_ptr<detail::buffer_storage<DataT, Dims>> bs) {
	{
		host_accessor<DataT, Dims, ProducerMode> host_acc(bs, cl::sycl::range<1>(10));
		host_acc[5] = 33.f;
	}
	{
		host_accessor<DataT, Dims, ConsumerMode> host_acc(bs, cl::sycl::range<1>(10));
		REQUIRE(host_acc[5] == 33.f);
	}
}

TEST_CASE("host_accessor handles all producer modes correctly", "[host_accessor]") {
	using namespace cl::sycl::access;
	test_context ctx;
	auto host_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::HOST_BUFFER, cl::sycl::range<1>(10));
	auto device_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::DEVICE_BUFFER, cl::sycl::range<1>(10));

	SECTION("discard_read_write") {
		test_host_accessor_mode<mode::discard_read_write>(host_buffer);
		test_host_accessor_mode<mode::discard_read_write>(device_buffer);
	}

	SECTION("discard_write") {
		test_host_accessor_mode<mode::discard_write>(host_buffer);
		test_host_accessor_mode<mode::discard_write>(device_buffer);
	}

	SECTION("read_write") {
		test_host_accessor_mode<mode::read_write>(host_buffer);
		test_host_accessor_mode<mode::read_write>(device_buffer);
	}

	SECTION("write") {
		test_host_accessor_mode<mode::write>(host_buffer);
		test_host_accessor_mode<mode::write>(device_buffer);
	}
}

TEST_CASE("host_accessor handles all consumer modes correctly", "[host_accessor]") {
	using namespace cl::sycl::access;
	test_context ctx;
	auto host_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::HOST_BUFFER, cl::sycl::range<1>(10));
	auto device_buffer = std::make_shared<detail::buffer_storage<float, 1>>(detail::buffer_type::DEVICE_BUFFER, cl::sycl::range<1>(10));

	SECTION("read") {
		test_host_accessor_mode<mode::discard_write, mode::read>(host_buffer);
		test_host_accessor_mode<mode::discard_write, mode::read>(device_buffer);
	}

	SECTION("read_write") {
		test_host_accessor_mode<mode::discard_write, mode::read_write>(host_buffer);
		test_host_accessor_mode<mode::discard_write, mode::read_write>(device_buffer);
	}

	// While not semantically correct, reading from "write" access is perfectly fine on host buffers
	SECTION("write") {
		test_host_accessor_mode<mode::discard_write, mode::write>(host_buffer);
		test_host_accessor_mode<mode::discard_write, mode::write>(device_buffer);
	}
}

TEST_CASE("range mapper results are clamped to buffer range", "[range-mapper]") {
	const auto rmfn = [](chunk<3>) { return subrange<3>{{0, 100, 127}, {256, 64, 32}}; };
	detail::range_mapper<3, 3> rm(rmfn, cl::sycl::access::mode::read, {128, 128, 128});
	auto sr = rm.map_3({});
	REQUIRE(sr.offset == cl::sycl::id<3>{0, 100, 127});
	REQUIRE(sr.range == cl::sycl::range<3>{128, 28, 1});
}

TEST_CASE("one_to_one built-in range mapper behaves as expected", "[range-mapper]") {
	detail::range_mapper<2, 2> rm(access::one_to_one<2>(), cl::sycl::access::mode::read, {128, 128});
	auto sr = rm.map_2({{64, 32}, {32, 4}, {128, 128}});
	REQUIRE(sr.offset == cl::sycl::id<2>{64, 32});
	REQUIRE(sr.range == cl::sycl::range<2>{32, 4});
}

TEST_CASE("fixed built-in range mapper behaves as expected", "[range-mapper]") {
	detail::range_mapper<2, 1> rm(access::fixed<2, 1>({{3}, {97}}), cl::sycl::access::mode::read, {128});
	auto sr = rm.map_1({{64, 32}, {32, 4}, {128, 128}});
	REQUIRE(sr.offset == cl::sycl::id<1>{3});
	REQUIRE(sr.range == cl::sycl::range<1>{97});
}

TEST_CASE("slice built-in range mapper behaves as expected", "[range-mapper]") {
	{
		detail::range_mapper<3, 3> rm(access::slice<3>(0), cl::sycl::access::mode::read, {128, 128, 128});
		auto sr = rm.map_3({{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<3>{0, 32, 32});
		REQUIRE(sr.range == cl::sycl::range<3>{128, 32, 32});
	}
	{
		detail::range_mapper<3, 3> rm(access::slice<3>(1), cl::sycl::access::mode::read, {128, 128, 128});
		auto sr = rm.map_3({{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<3>{32, 0, 32});
		REQUIRE(sr.range == cl::sycl::range<3>{32, 128, 32});
	}
	{
		detail::range_mapper<3, 3> rm(access::slice<3>(2), cl::sycl::access::mode::read, {128, 128, 128});
		auto sr = rm.map_3({{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<3>{32, 32, 0});
		REQUIRE(sr.range == cl::sycl::range<3>{32, 32, 128});
	}
}

TEST_CASE("neighborhood built-in range mapper behaves as expected", "[range-mapper]") {
	{
		detail::range_mapper<1, 1> rm(access::neighborhood<1>(10), cl::sycl::access::mode::read, {128});
		auto sr = rm.map_1({{15}, {10}, {128}});
		REQUIRE(sr.offset == cl::sycl::id<1>{5});
		REQUIRE(sr.range == cl::sycl::range<1>{30});
	}
	{
		detail::range_mapper<2, 2> rm(access::neighborhood<2>(10, 10), cl::sycl::access::mode::read, {128, 128});
		auto sr = rm.map_2({{5, 100}, {10, 20}, {128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<2>{0, 90});
		REQUIRE(sr.range == cl::sycl::range<2>{25, 38});
	}
	{
		detail::range_mapper<3, 3> rm(access::neighborhood<3>(3, 4, 5), cl::sycl::access::mode::read, {128, 128, 128});
		auto sr = rm.map_3({{3, 4, 5}, {1, 1, 1}, {128, 128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
		REQUIRE(sr.range == cl::sycl::range<3>{7, 9, 11});
	}
}

TEST_CASE("task_manager invokes callback upon task creation", "[task_manager]") {
	detail::task_manager tm;
	size_t call_counter = 0;
	tm.register_task_callback([&call_counter]() { call_counter++; });
	tm.create_compute_task([](auto& cgh) {});
	REQUIRE(call_counter == 1);
	tm.create_master_access_task([](auto& mah) {});
	REQUIRE(call_counter == 2);
}

// NOTE: Making assertions on the task graph isn't great as it really is an implementation detail, but it works for now
TEST_CASE("task_manager keeps track of task processing status", "[task_manager]") {
	detail::task_manager tm;
	test_utils::mock_buffer_factory mbf(&tm);
	auto buf = mbf.create_buffer(cl::sycl::range<1>(32));

	const auto tid_a = test_utils::add_master_access_task(tm, [&](auto& mah) { buf.get_access<cl::sycl::access::mode::discard_write>(mah, 32); });
	REQUIRE((*tm.get_task_graph())[tid_a].num_unsatisfied == 0);
	REQUIRE_FALSE((*tm.get_task_graph())[tid_a].processed);

	// Having a dependency on an unprocessed task should increase the number of unsatisfied dependencies
	const auto tid_b = test_utils::add_master_access_task(tm, [&](auto& mah) { buf.get_access<cl::sycl::access::mode::read>(mah, 32); });
	REQUIRE((*tm.get_task_graph())[tid_b].num_unsatisfied == 1);
	REQUIRE_FALSE((*tm.get_task_graph())[tid_b].processed);

	tm.mark_task_as_processed(tid_a);
	REQUIRE((*tm.get_task_graph())[tid_a].processed);
	REQUIRE((*tm.get_task_graph())[tid_b].num_unsatisfied == 0);

	// Having a dependency onto an already processed task shouldn't increase the number of unsatisfied dependencies
	const auto tid_c = test_utils::add_master_access_task(tm, [&](auto& mah) { buf.get_access<cl::sycl::access::mode::read>(mah, 32); });
	REQUIRE((*tm.get_task_graph())[tid_c].num_unsatisfied == 0);
	REQUIRE_FALSE((*tm.get_task_graph())[tid_c].processed);

	// Anti-dependencies behave in the same way
	const auto tid_d = test_utils::add_master_access_task(tm, [&](auto& mah) { buf.get_access<cl::sycl::access::mode::discard_write>(mah, 32); });
	REQUIRE((*tm.get_task_graph())[tid_d].num_unsatisfied == 2);
	REQUIRE_FALSE((*tm.get_task_graph())[tid_d].processed);

	tm.mark_task_as_processed(tid_b);
	tm.mark_task_as_processed(tid_c);
	tm.mark_task_as_processed(tid_d);

	const auto tid_e = test_utils::add_master_access_task(tm, [&](auto& mah) { buf.get_access<cl::sycl::access::mode::discard_write>(mah, 32); });
	REQUIRE((*tm.get_task_graph())[tid_e].num_unsatisfied == 0);
	REQUIRE_FALSE((*tm.get_task_graph())[tid_e].processed);
}

TEST_CASE("task_manager correctly records compute task information", "[task_manager][task][compute_task]") {
	detail::task_manager tm;
	test_utils::mock_buffer_factory mbf(&tm);
	auto buf_a = mbf.create_buffer(cl::sycl::range<2>(64, 152));
	auto buf_b = mbf.create_buffer(cl::sycl::range<3>(7, 21, 99));
	const auto tid = test_utils::add_compute_task(tm,
	    [&](auto& cgh) {
		    buf_a.get_access<cl::sycl::access::mode::read>(cgh, access::one_to_one<2>());
		    buf_b.get_access<cl::sycl::access::mode::discard_read_write>(cgh, access::fixed<2, 3>(subrange<3>({}, {5, 18, 74})));
	    },
	    cl::sycl::range<2>{32, 128}, cl::sycl::id<2>{32, 24});
	const auto tsk = tm.get_task(tid);
	REQUIRE(tsk->get_type() == task_type::COMPUTE);
	const auto ctsk = dynamic_cast<const detail::compute_task*>(tsk.get());
	REQUIRE(ctsk->get_dimensions() == 2);
	REQUIRE(ctsk->get_global_size() == cl::sycl::range<3>{32, 128, 1});
	REQUIRE(ctsk->get_global_offset() == cl::sycl::id<3>{32, 24, 0});

	const auto bufs = ctsk->get_accessed_buffers();
	REQUIRE(bufs.size() == 2);
	REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_a.get_id()) != bufs.cend());
	REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_b.get_id()) != bufs.cend());
	REQUIRE(ctsk->get_access_modes(buf_a.get_id()).count(cl::sycl::access::mode::read) == 1);
	REQUIRE(ctsk->get_access_modes(buf_b.get_id()).count(cl::sycl::access::mode::discard_read_write) == 1);
	const auto reqs_a = ctsk->get_requirements(buf_a.get_id(), cl::sycl::access::mode::read, {ctsk->get_global_offset(), ctsk->get_global_size()});
	REQUIRE(reqs_a == detail::subrange_to_grid_region(subrange<3>({32, 24, 0}, {32, 128, 1})));
	const auto reqs_b =
	    ctsk->get_requirements(buf_b.get_id(), cl::sycl::access::mode::discard_read_write, {ctsk->get_global_offset(), ctsk->get_global_size()});
	REQUIRE(reqs_b == detail::subrange_to_grid_region(subrange<3>({}, {5, 18, 74})));
}

TEST_CASE("compute_task merges multiple accesses with the same mode", "[task][compute_task]") {
	auto ctsk = std::make_unique<detail::compute_task>(0, nullptr);
	ctsk->set_dimensions(2);
	ctsk->add_range_mapper(0, std::make_unique<detail::range_mapper<2, 2>>(
	                              access::fixed<2, 2>(subrange<2>({3, 0}, {10, 20})), cl::sycl::access::mode::read, cl::sycl::range<2>{30, 30}));
	ctsk->add_range_mapper(0, std::make_unique<detail::range_mapper<2, 2>>(
	                              access::fixed<2, 2>(subrange<2>({10, 0}, {7, 20})), cl::sycl::access::mode::read, cl::sycl::range<2>{30, 30}));
	const auto req = ctsk->get_requirements(0, cl::sycl::access::mode::read, subrange<3>({0, 0, 0}, {100, 100, 1}));
	REQUIRE(req == detail::subrange_to_grid_region(subrange<3>({3, 0, 0}, {14, 20, 1})));
}

TEST_CASE("task_manager correctly records master access task information", "[task_manager][task][master_access_task]") {
	detail::task_manager tm;
	test_utils::mock_buffer_factory mbf(&tm);
	auto buf_a = mbf.create_buffer(cl::sycl::range<3>(45, 90, 160));
	auto buf_b = mbf.create_buffer(cl::sycl::range<3>(5, 30, 100));

	size_t call_counter = 0;
	const auto ma_functor = [&call_counter]() { call_counter++; };
	const auto tid = test_utils::add_master_access_task(tm, [&](auto& mah) {
		buf_a.get_access<cl::sycl::access::mode::read>(mah, cl::sycl::range<3>{32, 48, 96}, cl::sycl::id<3>{4, 8, 16});
		buf_a.get_access<cl::sycl::access::mode::write>(mah, cl::sycl::range<3>{21, 84, 75}, cl::sycl::id<3>{9, 2, 44});
		buf_b.get_access<cl::sycl::access::mode::discard_write>(mah, cl::sycl::range<3>{1, 7, 19}, cl::sycl::id<3>{0, 3, 8});
		mah.run(ma_functor);
	});
	const auto tsk = tm.get_task(tid);
	REQUIRE(tsk->get_type() == task_type::MASTER_ACCESS);
	const auto matsk = dynamic_cast<const detail::master_access_task*>(tsk.get());
	master_access_livepass_handler mah;
	matsk->get_functor()(mah);
	REQUIRE(call_counter == 1);

	const auto bufs = matsk->get_accessed_buffers();
	REQUIRE(bufs.size() == 2);
	REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_a.get_id()) != bufs.cend());
	REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_b.get_id()) != bufs.cend());
	REQUIRE(matsk->get_access_modes(buf_a.get_id()).count(cl::sycl::access::mode::read) == 1);
	REQUIRE(matsk->get_access_modes(buf_a.get_id()).count(cl::sycl::access::mode::write) == 1);
	REQUIRE(matsk->get_access_modes(buf_b.get_id()).count(cl::sycl::access::mode::discard_write) == 1);

	const auto reqs_a_r = matsk->get_requirements(buf_a.get_id(), cl::sycl::access::mode::read);
	REQUIRE(reqs_a_r == detail::subrange_to_grid_region(subrange<3>({4, 8, 16}, {32, 48, 96})));
	const auto reqs_a_w = matsk->get_requirements(buf_a.get_id(), cl::sycl::access::mode::write);
	REQUIRE(reqs_a_w == detail::subrange_to_grid_region(subrange<3>({9, 2, 44}, {21, 84, 75})));
	const auto reqs_b_dw = matsk->get_requirements(buf_b.get_id(), cl::sycl::access::mode::discard_write);
	REQUIRE(reqs_b_dw == detail::subrange_to_grid_region(subrange<3>({0, 3, 8}, {1, 7, 19})));
}

TEST_CASE("master_access_task merges multiple accesses with the same mode", "[task][master_access_task]") {
	auto matsk = std::make_unique<detail::master_access_task>(0, nullptr);
	matsk->add_buffer_access(0, cl::sycl::access::mode::read, subrange<2>({3, 0}, {10, 20}));
	matsk->add_buffer_access(0, cl::sycl::access::mode::read, subrange<2>({10, 0}, {7, 20}));
	const auto req = matsk->get_requirements(0, cl::sycl::access::mode::read);
	REQUIRE(req == detail::subrange_to_grid_region(subrange<3>({3, 0, 0}, {14, 20, 1})));
}

} // namespace celerity
