#include "unit_test_suite_celerity.h"

#include <algorithm>
#include <memory>
#include <random>

#include <catch2/catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "ranges.h"
#include "region_map.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::neighborhood;
	using celerity::access::one_to_one;
	using celerity::access::slice;
	using celerity::experimental::access::even_split;

	GridBox<3> make_grid_box(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
		const auto end = celerity::detail::range_cast<3>(offset) + range;
		return {sycl_id_to_grid_point(celerity::detail::range_cast<3>(offset)), sycl_id_to_grid_point(end)};
	}

	GridRegion<3> make_grid_region(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) { return GridRegion<3>(make_grid_box(range, offset)); }

	TEST_CASE("only a single distr_queue can be created", "[distr_queue][lifetime][dx]") {
		distr_queue q1;
		auto q2{q1}; // Copying is allowed
		REQUIRE_THROWS_WITH(distr_queue{}, "Only one celerity::distr_queue can be created per process (but it can be copied!)");
	}

	TEST_CASE("distr_queue implicitly initializes the runtime", "[distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::is_initialized());
		distr_queue queue;
		REQUIRE(runtime::is_initialized());
	}

	TEST_CASE("an explicit device can only be provided to distr_queue if runtime has not been initialized", "[distr_queue][lifetime]") {
		cl::sycl::default_selector selector;
		auto device = selector.select_device();
		{
			REQUIRE_FALSE(runtime::is_initialized());
			REQUIRE_NOTHROW(distr_queue{device});
		}
		runtime::teardown();
		{
			REQUIRE_FALSE(runtime::is_initialized());
			runtime::init(nullptr, nullptr);
			REQUIRE_THROWS_WITH(distr_queue{device}, "Passing explicit device not possible, runtime has already been initialized.");
		}
	}

	TEST_CASE("buffer implicitly initializes the runtime", "[distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::is_initialized());
		buffer<float, 1> buf(cl::sycl::range<1>{1});
		REQUIRE(runtime::is_initialized());
	}

	TEST_CASE("buffer can be copied", "[distr_queue][lifetime]") {
		buffer<float, 1> buf_a{cl::sycl::range<1>{10}};
		buffer<float, 1> buf_b{cl::sycl::range<1>{10}};
		auto buf_c{buf_a};
		buf_b = buf_c;
	}

	TEST_CASE("get_access can be called on const buffer", "[buffer]") {
		buffer<float, 2> buf_a{cl::sycl::range<2>{32, 64}};
		auto& tm = runtime::get_instance().get_task_manager();
		const auto tid = test_utils::add_compute_task<class get_access_const>(
		    tm, [buf_a /* capture by value */](handler& cgh) { buf_a.get_access<cl::sycl::access::mode::read>(cgh, one_to_one<2>()); }, buf_a.get_range());
		const auto tsk = tm.get_task(tid);
		const auto bufs = tsk->get_buffer_access_map().get_accessed_buffers();
		REQUIRE(bufs.size() == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(0).count(cl::sycl::access::mode::read) == 1);
	}

	TEST_CASE("region_map correctly handles region updates", "[region_map]") {
		region_map<std::string> rm(cl::sycl::range<3>(256, 128, 1));

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
		region_map<std::unordered_set<size_t>> rm(cl::sycl::range<3>(256, 1, 1));
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
		region_map<size_t> rm1(cl::sycl::range<3>(128, 64, 32));
		region_map<size_t> rm2(cl::sycl::range<3>(128, 64, 32));
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
		const region_map<size_t> rm_incompat(cl::sycl::range<3>(128, 64, 30));
		REQUIRE_THROWS_WITH(rm1.merge(rm_incompat), Catch::Equals("Incompatible region map"));
	}

	TEST_CASE("range mapper results are clamped to buffer range", "[range-mapper]") {
		const auto rmfn = [](chunk<3>) { return subrange<3>{{0, 100, 127}, {256, 64, 32}}; };
		range_mapper<3, 3> rm(rmfn, cl::sycl::access::mode::read, {128, 128, 128});
		auto sr = rm.map_3({});
		REQUIRE(sr.offset == cl::sycl::id<3>{0, 100, 127});
		REQUIRE(sr.range == cl::sycl::range<3>{128, 28, 1});
	}

	TEST_CASE("one_to_one built-in range mapper behaves as expected", "[range-mapper]") {
		range_mapper<2, 2> rm(one_to_one<2>(), cl::sycl::access::mode::read, {128, 128});
		auto sr = rm.map_2({{64, 32}, {32, 4}, {128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<2>{64, 32});
		REQUIRE(sr.range == cl::sycl::range<2>{32, 4});
	}

	TEST_CASE("fixed built-in range mapper behaves as expected", "[range-mapper]") {
		range_mapper<2, 1> rm(fixed<2, 1>({{3}, {97}}), cl::sycl::access::mode::read, {128});
		auto sr = rm.map_1({{64, 32}, {32, 4}, {128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<1>{3});
		REQUIRE(sr.range == cl::sycl::range<1>{97});
	}

	TEST_CASE("slice built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper<3, 3> rm(slice<3>(0), cl::sycl::access::mode::read, {128, 128, 128});
			auto sr = rm.map_3({{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 32, 32});
			REQUIRE(sr.range == cl::sycl::range<3>{128, 32, 32});
		}
		{
			range_mapper<3, 3> rm(slice<3>(1), cl::sycl::access::mode::read, {128, 128, 128});
			auto sr = rm.map_3({{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{32, 0, 32});
			REQUIRE(sr.range == cl::sycl::range<3>{32, 128, 32});
		}
		{
			range_mapper<3, 3> rm(slice<3>(2), cl::sycl::access::mode::read, {128, 128, 128});
			auto sr = rm.map_3({{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{32, 32, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{32, 32, 128});
		}
	}

	TEST_CASE("all built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper<1, 1> rm(all<1, 1>(), cl::sycl::access::mode::read, {128});
			auto sr = rm.map_1({{}, {}, {}});
			REQUIRE(sr.offset == cl::sycl::id<1>{0});
			REQUIRE(sr.range == cl::sycl::range<1>{128});
		}
		{
			range_mapper<1, 2> rm(all<1, 2>(), cl::sycl::access::mode::read, {128, 64});
			auto sr = rm.map_2({{}, {}, {}});
			REQUIRE(sr.offset == cl::sycl::id<2>{0, 0});
			REQUIRE(sr.range == cl::sycl::range<2>{128, 64});
		}
		{
			range_mapper<1, 3> rm(all<1, 3>(), cl::sycl::access::mode::read, {128, 64, 32});
			auto sr = rm.map_3({{}, {}, {}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{128, 64, 32});
		}
	}

	TEST_CASE("neighborhood built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper<1, 1> rm(neighborhood<1>(10), cl::sycl::access::mode::read, {128});
			auto sr = rm.map_1({{15}, {10}, {128}});
			REQUIRE(sr.offset == cl::sycl::id<1>{5});
			REQUIRE(sr.range == cl::sycl::range<1>{30});
		}
		{
			range_mapper<2, 2> rm(neighborhood<2>(10, 10), cl::sycl::access::mode::read, {128, 128});
			auto sr = rm.map_2({{5, 100}, {10, 20}, {128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<2>{0, 90});
			REQUIRE(sr.range == cl::sycl::range<2>{25, 38});
		}
		{
			range_mapper<3, 3> rm(neighborhood<3>(3, 4, 5), cl::sycl::access::mode::read, {128, 128, 128});
			auto sr = rm.map_3({{3, 4, 5}, {1, 1, 1}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{7, 9, 11});
		}
	}

	TEST_CASE("even_split built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper<1, 3> rm(even_split<3>(), cl::sycl::access::mode::read, {128, 345, 678});
			auto sr = rm.map_3({{0}, {1}, {8}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{16, 345, 678});
		}
		{
			range_mapper<1, 3> rm(even_split<3>(), cl::sycl::access::mode::read, {128, 345, 678});
			auto sr = rm.map_3({{4}, {2}, {8}});
			REQUIRE(sr.offset == cl::sycl::id<3>{64, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{32, 345, 678});
		}
		{
			range_mapper<1, 3> rm(even_split<3>(), cl::sycl::access::mode::read, {131, 992, 613});
			auto sr = rm.map_3({{5}, {2}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{95, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{36, 992, 613});
		}
		{
			range_mapper<1, 3> rm(even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, {128, 345, 678});
			auto sr = rm.map_3({{0}, {1}, {8}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{20, 345, 678});
		}
		{
			range_mapper<1, 3> rm(even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, {131, 992, 613});
			auto sr = rm.map_3({{0}, {1}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{20, 992, 613});
		}
		{
			range_mapper<1, 3> rm(even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, {131, 992, 613});
			auto sr = rm.map_3({{5}, {2}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{100, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{31, 992, 613});
		}
		{
			range_mapper<1, 3> rm(even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, {236, 992, 613});
			auto sr = rm.map_3({{6}, {1}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{200, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{36, 992, 613});
		}
	}

	TEST_CASE("task_manager invokes callback upon task creation", "[task_manager]") {
		task_manager tm{1, nullptr, true};
		size_t call_counter = 0;
		tm.register_task_callback([&call_counter](task_id) { call_counter++; });
		cl::sycl::range<2> gs = {1, 1};
		cl::sycl::id<2> go = {};
		tm.create_task([=](handler& cgh) { cgh.parallel_for<class kernel>(gs, go, [](auto) {}); });
		REQUIRE(call_counter == 1);
		tm.create_task([](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
		REQUIRE(call_counter == 2);
	}

	TEST_CASE("task_manager correctly records compute task information", "[task_manager][task][device_compute_task]") {
		task_manager tm{1, nullptr, true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<2>(64, 152));
		auto buf_b = mbf.create_buffer(cl::sycl::range<3>(7, 21, 99));
		const auto tid = test_utils::add_compute_task(
		    tm,
		    [&](handler& cgh) {
			    buf_a.get_access<cl::sycl::access::mode::read>(cgh, one_to_one<2>());
			    buf_b.get_access<cl::sycl::access::mode::discard_read_write>(cgh, fixed<2, 3>(subrange<3>({}, {5, 18, 74})));
		    },
		    cl::sycl::range<2>{32, 128}, cl::sycl::id<2>{32, 24});
		const auto tsk = tm.get_task(tid);
		REQUIRE(tsk->get_type() == task_type::DEVICE_COMPUTE);
		REQUIRE(tsk->get_dimensions() == 2);
		REQUIRE(tsk->get_global_size() == cl::sycl::range<3>{32, 128, 1});
		REQUIRE(tsk->get_global_offset() == cl::sycl::id<3>{32, 24, 0});

		auto& bam = tsk->get_buffer_access_map();
		const auto bufs = bam.get_accessed_buffers();
		REQUIRE(bufs.size() == 2);
		REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_a.get_id()) != bufs.cend());
		REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_b.get_id()) != bufs.cend());
		REQUIRE(bam.get_access_modes(buf_a.get_id()).count(cl::sycl::access::mode::read) == 1);
		REQUIRE(bam.get_access_modes(buf_b.get_id()).count(cl::sycl::access::mode::discard_read_write) == 1);
		const auto reqs_a = bam.get_requirements_for_access(
		    buf_a.get_id(), cl::sycl::access::mode::read, {tsk->get_global_offset(), tsk->get_global_size()}, tsk->get_global_size());
		REQUIRE(reqs_a == subrange_to_grid_box(subrange<3>({32, 24, 0}, {32, 128, 1})));
		const auto reqs_b = bam.get_requirements_for_access(
		    buf_b.get_id(), cl::sycl::access::mode::discard_read_write, {tsk->get_global_offset(), tsk->get_global_size()}, tsk->get_global_size());
		REQUIRE(reqs_b == subrange_to_grid_box(subrange<3>({}, {5, 18, 74})));
	}

	TEST_CASE("buffer_access_map merges multiple accesses with the same mode", "[task][device_compute_task]") {
		buffer_access_map bam;
		bam.add_access(
		    0, std::make_unique<range_mapper<2, 2>>(fixed<2, 2>(subrange<2>({3, 0}, {10, 20})), cl::sycl::access::mode::read, cl::sycl::range<2>{30, 30}));
		bam.add_access(
		    0, std::make_unique<range_mapper<2, 2>>(fixed<2, 2>(subrange<2>({10, 0}, {7, 20})), cl::sycl::access::mode::read, cl::sycl::range<2>{30, 30}));
		const auto req = bam.get_requirements_for_access(0, cl::sycl::access::mode::read, subrange<3>({0, 0, 0}, {100, 100, 1}), {100, 100, 1});
		REQUIRE(req == subrange_to_grid_box(subrange<3>({3, 0, 0}, {14, 20, 1})));
	}

	TEST_CASE("tasks gracefully handle get_requirements() calls for buffers they don't access", "[task]") {
		buffer_access_map bam;
		const auto req = bam.get_requirements_for_access(0, cl::sycl::access::mode::read, subrange<3>({0, 0, 0}, {100, 1, 1}), {100, 1, 1});
		REQUIRE(req == subrange_to_grid_box(subrange<3>({0, 0, 0}, {0, 0, 0})));
	}

	TEST_CASE("safe command group functions must not capture by reference", "[lifetime][dx]") {
		int value = 123;
		const auto unsafe = [&]() { return value + 1; };
		REQUIRE_FALSE(is_safe_cgf<decltype(unsafe)>);
		const auto safe = [=]() { return value + 1; };
		REQUIRE(is_safe_cgf<decltype(safe)>);
	}

	TEST_CASE("basic SYNC command functionality", "[distr_queue][sync][control-flow]") {
		constexpr int N = 10;

		distr_queue q;
		buffer<int, 1> buff(N);
		std::vector<int> host_buff(N);

		q.submit([=](handler& cgh) {
			auto b = buff.get_access<cl::sycl::access::mode::discard_write>(cgh, one_to_one<1>());
			cgh.parallel_for<class sync_test>(cl::sycl::range<1>(N), [=](cl::sycl::item<1> item) { b[item] = item.get_linear_id(); });
		});

		q.submit(allow_by_ref, [&](handler& cgh) {
			auto b =
			    buff.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::fixed<1>{{{}, buff.get_range()}});
			cgh.host_task(on_master_node, [=, &host_buff] {
				std::this_thread::sleep_for(std::chrono::milliseconds(10)); // give the synchronization more time to fail
				for(int i = 0; i < N; i++) {
					host_buff[i] = b[i];
				}
			});
		});

		q.slow_full_sync();

		for(int i = 0; i < N; i++) {
			CHECK(host_buff[i] == i);
		}
	}

	TEST_CASE("memcpy_strided correctly copies") {
		SECTION("strided 1D data") {
			const cl::sycl::range<1> source_range{128};
			const cl::sycl::id<1> source_offset{32};
			const cl::sycl::range<1> target_range{64};
			const cl::sycl::id<1> target_offset{16};
			const cl::sycl::range<1> copy_range{32};
			const auto source_buffer = std::make_unique<size_t[]>(source_range.size());
			const auto target_buffer = std::make_unique<size_t[]>(target_range.size());
			for(size_t i = 0; i < copy_range[0]; ++i) {
				source_buffer[source_offset[0] + i] = source_offset[0] + i;
			}
			memcpy_strided(source_buffer.get(), target_buffer.get(), sizeof(size_t), source_range, source_offset, target_range, target_offset, copy_range);
			bool valid = true;
			for(size_t i = 0; i < copy_range[0]; ++i) {
				valid &= target_buffer[target_offset[0] + i] == source_offset[0] + i;
			}
			REQUIRE(valid);
		}

		SECTION("strided 2D data") {
			const cl::sycl::range<2> source_range{128, 96};
			const cl::sycl::id<2> source_offset{32, 24};
			const cl::sycl::range<2> target_range{64, 48};
			const cl::sycl::id<2> target_offset{16, 32};
			const cl::sycl::range<2> copy_range{32, 8};
			const auto source_buffer = std::make_unique<size_t[]>(source_range.size());
			const auto target_buffer = std::make_unique<size_t[]>(target_range.size());
			for(size_t i = 0; i < copy_range[0]; ++i) {
				for(size_t j = 0; j < copy_range[1]; ++j) {
					const auto id = source_offset + cl::sycl::id<2>{i, j};
					source_buffer[get_linear_index(source_range, id)] = id[0] * 10000 + id[1];
				}
			}
			memcpy_strided(source_buffer.get(), target_buffer.get(), sizeof(size_t), source_range, source_offset, target_range, target_offset, copy_range);
			bool valid = true;
			for(size_t i = 0; i < copy_range[0]; ++i) {
				for(size_t j = 0; j < copy_range[1]; ++j) {
					const auto id = target_offset + cl::sycl::id<2>{i, j};
					const auto source_id = source_offset + cl::sycl::id<2>{i, j};
					valid &= target_buffer[get_linear_index(target_range, id)] == source_id[0] * 10000 + source_id[1];
				}
			}
			REQUIRE(valid);
		}

		SECTION("strided 3D data") {
			const cl::sycl::range<3> source_range{128, 96, 48};
			const cl::sycl::id<3> source_offset{32, 24, 16};
			const cl::sycl::range<3> target_range{64, 48, 24};
			const cl::sycl::id<3> target_offset{16, 32, 4};
			const cl::sycl::range<3> copy_range{32, 8, 16};
			const auto source_buffer = std::make_unique<size_t[]>(source_range.size());
			const auto target_buffer = std::make_unique<size_t[]>(target_range.size());
			for(size_t i = 0; i < copy_range[0]; ++i) {
				for(size_t j = 0; j < copy_range[1]; ++j) {
					for(size_t k = 0; k < copy_range[2]; ++k) {
						const auto id = source_offset + cl::sycl::id<3>{i, j, k};
						source_buffer[get_linear_index(source_range, id)] = id[0] * 10000 + id[1] * 100 + id[2];
					}
				}
			}
			memcpy_strided(source_buffer.get(), target_buffer.get(), sizeof(size_t), source_range, source_offset, target_range, target_offset, copy_range);
			bool valid = true;
			for(size_t i = 0; i < copy_range[0]; ++i) {
				for(size_t j = 0; j < copy_range[1]; ++j) {
					for(size_t k = 0; k < copy_range[2]; ++k) {
						const auto id = target_offset + cl::sycl::id<3>{i, j, k};
						const auto source_id = source_offset + cl::sycl::id<3>{i, j, k};
						valid &= target_buffer[get_linear_index(target_range, id)] == source_id[0] * 10000 + source_id[1] * 100 + source_id[2];
						if(!valid) {
							printf("Unexpected value at %lu %lu %lu: %lu != %lu\n", id[0], id[1], id[2], target_buffer[get_linear_index(target_range, id)],
							    source_id[0] * 10000 + source_id[1] * 100 + source_id[2]);
							REQUIRE(false);
						}
					}
				}
			}
			REQUIRE(valid);
		}
	}

	TEST_CASE("raw_buffer_data works as expected") {
		const cl::sycl::range<3> data1_range{3, 5, 7};
		raw_buffer_data data1{sizeof(size_t), data1_range};
		REQUIRE(data1.get_range() == data1_range);
		REQUIRE(data1.get_pointer() != nullptr);
		REQUIRE(data1.get_size() == sizeof(size_t) * data1_range.size());

		for(size_t i = 0; i < data1_range[0]; ++i) {
			for(size_t j = 0; j < data1_range[1]; ++j) {
				for(size_t k = 0; k < data1_range[2]; ++k) {
					reinterpret_cast<size_t*>(data1.get_pointer())[i * data1_range[1] * data1_range[2] + j * data1_range[2] + k] = i * 100 + j * 10 + k;
				}
			}
		}

		const cl::sycl::range<3> data2_range{2, 2, 4};
		const cl::sycl::id<3> data2_offset{1, 2, 2};
		auto data2 = data1.copy(data2_offset, data2_range);
		REQUIRE(data2.get_range() == data2_range);
		REQUIRE(data2.get_pointer() != nullptr);
		REQUIRE(data2.get_pointer() != data1.get_pointer());
		REQUIRE(data2.get_size() == sizeof(size_t) * data2_range.size());

		bool valid = true;
		for(size_t i = 0; i < 2; ++i) {
			for(size_t j = 0; j < 2; ++j) {
				for(size_t k = 0; k < 4; ++k) {
					valid &= reinterpret_cast<size_t*>(data2.get_pointer())[i * data2_range[1] * data2_range[2] + j * data2_range[2] + k]
					         == (i + data2_offset[0]) * 100 + (j + data2_offset[1]) * 10 + (k + data2_offset[2]);
				}
			}
		}
		REQUIRE(valid);

		const auto data2_ptr = data2.get_pointer();
		auto data3 = std::move(data2);
		REQUIRE(data2.get_pointer() == nullptr);
		REQUIRE(data3.get_range() == data2_range);
		REQUIRE(data3.get_pointer() == data2_ptr);
		REQUIRE(data3.get_size() == sizeof(size_t) * data2_range.size());

		raw_buffer_data data4{sizeof(uint64_t), {16, 8, 4}};
		data4.reinterpret(sizeof(uint32_t), {32, 8, 4});
		REQUIRE(data4.get_range() == cl::sycl::range<3>{32, 8, 4});
		REQUIRE(data4.get_size() == sizeof(uint64_t) * 16 * 8 * 4);
	}

	class buffer_manager_fixture {
	  public:
		enum class access_target { HOST, DEVICE };

		~buffer_manager_fixture() { get_device_queue().get_sycl_queue().wait_and_throw(); }

		void initialize(buffer_manager::buffer_lifecycle_callback cb = [](buffer_manager::buffer_lifecycle_event, buffer_id) {}) {
			l = std::make_unique<logger>("test", log_level::warn);
			cfg = std::make_unique<config>(nullptr, nullptr, *l);
			dq = std::make_unique<device_queue>(*l);
			dq->init(*cfg, nullptr);
			bm = std::make_unique<buffer_manager>(*dq, cb);
			initialized = true;
		}

		buffer_manager& get_buffer_manager() {
			if(!initialized) initialize();
			return *bm;
		}

		device_queue& get_device_queue() {
			if(!initialized) initialize();
			return *dq;
		}

		static access_target get_other_target(access_target tgt) {
			if(tgt == access_target::HOST) return access_target::DEVICE;
			return access_target::HOST;
		}

		template <typename DataT, int Dims>
		cl::sycl::range<Dims> get_backing_buffer_range(buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset) {
			if(tgt == access_target::HOST) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range_cast<3>(range), id_cast<3>(offset));
				return info.buffer.get_range();
			}
			auto info = bm->get_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range_cast<3>(range), id_cast<3>(offset));
			return info.buffer.get_range();
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename KernelName = class buffer_for_each, typename Callback>
		void buffer_for_each(buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, Callback cb) {
			const auto range3 = range_cast<3>(range);
			const auto offset3 = id_cast<3>(offset);

			if(tgt == access_target::HOST) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_range = range_cast<3>(info.buffer.get_range());
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - id_cast<3>(info.offset);
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							cb(id_cast<Dims>(global_idx), info.buffer.get_pointer()[linear_idx]);
						}
					}
				}
			}

			if(tgt == access_target::DEVICE) {
				auto info = bm->get_device_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_offset = info.offset;
				dq->get_sycl_queue()
				    .submit([&](cl::sycl::handler& cgh) {
					    auto acc = info.buffer.template get_access<Mode>(cgh);
					    cgh.parallel_for<KernelName>(range, [=](cl::sycl::id<Dims> idx) {
						    // Add offset manually to work around ComputeCpp PTX offset bug (still present as of 1.1.5)
						    const auto global_idx = idx + offset;
						    const auto local_idx = global_idx - buf_offset;
						    cb(global_idx, acc[local_idx]);
					    });
				    })
				    .wait();
			}
		}

		template <typename DataT, int Dims, typename KernelName = class buffer_reduce, typename ReduceT, typename Operation>
		ReduceT buffer_reduce(buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, ReduceT init, Operation op) {
			const auto range3 = range_cast<3>(range);
			const auto offset3 = id_cast<3>(offset);

			if(tgt == access_target::HOST) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
				const auto buf_range = range_cast<3>(info.buffer.get_range());
				ReduceT result = init;
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - id_cast<3>(info.offset);
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							result = op(id_cast<Dims>(global_idx), result, info.buffer.get_pointer()[linear_idx]);
						}
					}
				}
				return result;
			}

			auto info = bm->get_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
			const auto buf_offset = info.offset;
			cl::sycl::buffer<ReduceT, 1> result_buf(1); // Use 1-dimensional instead of 0-dimensional since it's NYI in hipSYCL as of 0.8.1
			// Simply do a serial reduction on the device as well
			dq->get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = info.buffer.template get_access<cl::sycl::access::mode::read>(cgh);
				    auto result_acc = result_buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
				    cgh.single_task<KernelName>([=]() {
					    result_acc[0] = init;
					    for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
						    for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
							    for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
								    const auto global_idx = cl::sycl::id<3>(i, j, k);
								    const cl::sycl::id<3> local_idx = global_idx - id_cast<3>(buf_offset);
								    result_acc[0] = op(id_cast<Dims>(global_idx), result_acc[0], acc[id_cast<Dims>(local_idx)]);
							    }
						    }
					    }
				    });
			    })
			    .wait();

			ReduceT result;
			dq->get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = result_buf.template get_access<cl::sycl::access::mode::read>(cgh);
				    cgh.copy(acc, &result);
			    })
			    .wait();

			return result;
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode>
		accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_device_accessor(
		    live_pass_device_handler& cgh, buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = bm->get_device_buffer<DataT, Dims>(bid, Mode, range_cast<3>(range), id_cast<3>(offset));
			return detail::make_device_accessor<DataT, Dims, Mode>(cgh.get_eventual_sycl_cgh(), buf_info.buffer, range, offset);
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode>
		accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> get_host_accessor(
		    buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = bm->get_host_buffer<DataT, Dims>(bid, Mode, range_cast<3>(range), id_cast<3>(offset));
			return detail::make_host_accessor<DataT, Dims, Mode>(
			    subrange<Dims>(offset, range), buf_info.buffer, buf_info.offset, range_cast<Dims>(bm->get_buffer_info(bid).range));
		}

	  private:
		bool initialized = false;
		std::unique_ptr<logger> l;
		std::unique_ptr<config> cfg;
		std::unique_ptr<device_queue> dq;
		std::unique_ptr<buffer_manager> bm;
	};

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager keeps track of buffers", "[buffer_manager]") {
		std::vector<std::pair<buffer_manager::buffer_lifecycle_event, buffer_id>> cb_calls;
		initialize([&](buffer_manager::buffer_lifecycle_event e, buffer_id bid) { cb_calls.push_back({e, bid}); });
		auto& bm = get_buffer_manager();

		REQUIRE_FALSE(bm.has_active_buffers());

		REQUIRE_FALSE(bm.has_buffer(0));
		bm.register_buffer<float, 1>({1024, 1, 1});
		REQUIRE(bm.has_buffer(0));
		REQUIRE(bm.has_active_buffers());
		REQUIRE(bm.get_buffer_info(0).range == cl::sycl::range<3>{1024, 1, 1});
		REQUIRE(bm.get_buffer_info(0).is_host_initialized == false);
		REQUIRE(cb_calls.size() == 1);
		REQUIRE(cb_calls[0] == std::make_pair(buffer_manager::buffer_lifecycle_event::REGISTERED, buffer_id(0)));

		std::vector<float> host_buf(5 * 6 * 7);
		bm.register_buffer<float, 3>({5, 6, 7}, host_buf.data());
		REQUIRE(bm.has_buffer(1));
		REQUIRE(bm.get_buffer_info(1).range == cl::sycl::range<3>{5, 6, 7});
		REQUIRE(bm.get_buffer_info(1).is_host_initialized == true);
		REQUIRE(cb_calls.size() == 2);
		REQUIRE(cb_calls[1] == std::make_pair(buffer_manager::buffer_lifecycle_event::REGISTERED, buffer_id(1)));

		bm.unregister_buffer(0);
		REQUIRE(cb_calls.size() == 3);
		REQUIRE(cb_calls[2] == std::make_pair(buffer_manager::buffer_lifecycle_event::UNREGISTERED, buffer_id(0)));
		REQUIRE(bm.has_active_buffers());

		bm.unregister_buffer(1);
		REQUIRE_FALSE(bm.has_active_buffers());
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager creates appropriately sized buffers as needed", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<float, 1>(cl::sycl::range<3>(3072, 1, 1));

		auto run_test = [&](auto access_buffer) {
			auto buf_info = access_buffer(1024, 0);

			// Even though we registered the buffer with a size of 3072, the actual backing buffer is only 1024
			REQUIRE(buf_info.buffer.get_range() == cl::sycl::range<1>(1024));

			// Requesting smaller portions of the buffer will re-use the existing backing buffer
			for(auto s = 512; s > 2; s >>= 2) {
				auto smallbuf_info = access_buffer(s, 0);
				REQUIRE(smallbuf_info.buffer == buf_info.buffer);
			}

			// As long as we're not exceeding the original 1024 items, changing the offset will also re-use the backing buffer
			for(auto o = 512; o > 2; o >>= 2) {
				auto smallbuf_info = access_buffer(512, o);
				REQUIRE(smallbuf_info.buffer == buf_info.buffer);
			}

			// If we however exceed the original 1024 by passing an offset, the backing buffer will be resized
			{
				auto buf_info = access_buffer(1024, 512);
				// Since the BM cannot discard the previous contents at offset 0, the new buffer includes them as well
				REQUIRE(buf_info.buffer.get_range() == cl::sycl::range<1>(1024 + 512));
			}

			// Likewise, requesting a larger range will cause the backing buffer to be resized
			{
				auto buf_info = access_buffer(2048, 0);
				REQUIRE(buf_info.buffer.get_range() == cl::sycl::range<1>(2048));
			}

			// Lastly, requesting a totally different (non-overlapping) sub-range will require the buffer to be resized
			// such that it contains both the previous and the new ranges.
			{
				auto buf_info = access_buffer(512, 2560);
				REQUIRE(buf_info.buffer.get_range() == cl::sycl::range<1>(3072));
			}
		};

		SECTION("when using device buffers") {
			run_test([&bm, bid](size_t range, size_t offset) {
				return bm.get_device_buffer<float, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(range, 1, 1), cl::sycl::id<3>(offset, 0, 0));
			});
		}

		SECTION("when using host buffers") {
			run_test([&bm, bid](size_t range, size_t offset) {
				return bm.get_host_buffer<float, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(range, 1, 1), cl::sycl::id<3>(offset, 0, 0));
			});
		}
	}

	TEST_CASE_METHOD(
	    buffer_manager_fixture, "buffer_manager returns correct access offset for backing buffers larger than the requested range", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<float, 1>(cl::sycl::range<3>(2048, 1, 1));

		auto run_test = [&](auto access_buffer) {
			// The returned offset indicates where the backing buffer starts, relative to the virtual buffer.
			REQUIRE(access_buffer(1024, 1024).offset == cl::sycl::id<1>(1024));
			REQUIRE(access_buffer(1024, 512).offset == cl::sycl::id<1>(512));
			REQUIRE(access_buffer(1024, 1024).offset == cl::sycl::id<1>(512));
			REQUIRE(access_buffer(256, 1024).offset == cl::sycl::id<1>(512));
			REQUIRE(access_buffer(1024, 0).offset == cl::sycl::id<1>(0));
		};

		SECTION("when using device buffers") {
			run_test([&bm, bid](size_t range, size_t offset) {
				return bm.get_device_buffer<float, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(range, 1, 1), cl::sycl::id<3>(offset, 0, 0));
			});
		}

		SECTION("when using host buffers") {
			run_test([&bm, bid](size_t range, size_t offset) {
				return bm.get_host_buffer<float, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(range, 1, 1), cl::sycl::id<3>(offset, 0, 0));
			});
		}
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager retains existing data when resizing buffers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		auto run_1D_test = [&](access_target tgt) {
			auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(160, 1, 1));

			// Request a 64 element buffer at offset 32 and initialize it with known values.
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(partial_init)>(
			    bid, tgt, {64}, {32}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Now request a 128 element buffer at offset 32, requiring the backing device buffer to be resized.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {128}, {32}, true, [](cl::sycl::id<1> idx, bool current, size_t value) {
					if(idx[0] < 96) return current && value == idx[0];
					return current;
				});
				REQUIRE(valid);
			}

			// Finally, request 128 elements at offset 0, again requiring the backing device buffer to be resized.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {128}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) {
					if(idx[0] >= 32 && idx[0] < 96) return current && value == idx[0];
					return current;
				});
				REQUIRE(valid);
			}
		};

		SECTION("when using 1D device buffers") { run_1D_test(access_target::DEVICE); }
		SECTION("when using 1D host buffers") { run_1D_test(access_target::HOST); }

		auto run_2D_test = [&](access_target tgt) {
			auto bid = bm.register_buffer<size_t, 2>(cl::sycl::range<3>(128, 128, 1));

			// Request a set of columns and initialize it with known values.
			buffer_for_each<size_t, 2, cl::sycl::access::mode::discard_write, class UKN(partial_init)>(
			    bid, tgt, {128, 64}, {0, 64}, [](cl::sycl::id<2> idx, size_t& value) { value = idx[0] * 100 + idx[1]; });

			// Now request a set of rows that partially intersect the columns from before, requiring the backing device buffer to be resized.
			{
				bool valid =
				    buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {64, 128}, {64, 0}, true, [](cl::sycl::id<2> idx, bool current, size_t value) {
					    if(idx[1] >= 64) return current && value == idx[0] * 100 + idx[1];
					    return current;
				    });
				REQUIRE(valid);
			}
		};

		SECTION("when using 2D device buffers") { run_2D_test(access_target::DEVICE); }
		SECTION("when using 2D host buffers") { run_2D_test(access_target::HOST); }

		// While the fix for bug that warranted adding a 2D test *should* also cover 3D buffers, it would be good to have a 3D test here as well.
		// TODO: Can we also come up with a good 3D case?
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager does not retain existing data when resizing buffer using a pure producer access mode",
	    "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt, bool partial_overwrite) {
			// Initialize 64 element buffer at offset 0
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(partial_init)>(
			    bid, tgt, {64}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = 1337 + idx[0]; });

			// Resize it to 128 elements using a pure producer mode
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(faux_overwrite)>(bid, tgt,
			    {partial_overwrite == false ? size_t(128) : size_t(96)}, {partial_overwrite == false ? size_t(0) : size_t(32)},
			    [](cl::sycl::id<1> idx, size_t& value) { /* NOP */ });

			// Verify that the original 64 elements have not been retained during the resizing (unless we did a partial overwrite)
			// NOTE: We're reading uninitialized memory here, so anything is possible, technically (but highly unlikely).
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {128}, {0}, true, [=](cl::sycl::id<1> idx, bool current, size_t value) {
					if(partial_overwrite) {
						// If we did a partial overwrite, the first 32 elements should have been retained
						if(idx[0] < 32) return current && value == 1337 + idx[0];
					}
					if(idx[0] < 64) return current && value != 1337 + idx[0];
					return current;
				});
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::DEVICE, false); }
		SECTION("when using host buffers") { run_test(access_target::HOST, false); }

		SECTION("unless accessed range does not fully cover previous buffer size (using device buffers)") { run_test(access_target::DEVICE, true); }
		SECTION("unless accessed range does not fully cover previous buffer size (using host buffers)") { run_test(access_target::HOST, true); }
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager ensures coherence between device and host buffers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(512, 1, 1));

		auto run_test1 = [&](access_target tgt) {
			// Initialize first half of buffer on this side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init_first_half)>(
			    bid, tgt, {256}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Initialize second half of buffer on other side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init_second_half)>(
			    bid, get_other_target(tgt), {256}, {256}, [](cl::sycl::id<1> idx, size_t& value) { value = (512 - idx[0]) * 2; });

			// Verify coherent full buffer is available on this side
			bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {512}, {0}, true,
			    [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value == (idx[0] < 256 ? idx[0] : (512 - idx[0]) * 2); });
			REQUIRE(valid);
		};

		SECTION("when writing separate parts on host and device, verifying on device") { run_test1(access_target::DEVICE); }
		SECTION("when writing separate parts on host and device, verifying on host") { run_test1(access_target::HOST); }

		// This test can be run in two slightly different variations, as overwriting a larger range incurs
		// a resize operation internally, which then leads to a somewhat different code path during the coherency update.
		auto run_test2 = [&](access_target tgt, size_t overwrite_range) {
			// Initialize on this side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, tgt, {256}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Update (potentially larger portion, depending on `overwrite_range`) on other side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::read_write, class UKN(update)>(
			    bid, get_other_target(tgt), {overwrite_range}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = (idx[0] < 256 ? value * 2 : 33); });

			// Verify result on this side
			bool valid =
			    buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {overwrite_range}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) {
				    if(idx[0] < 256) return current && value == idx[0] * 2;
				    return current && value == 33;
			    });
			REQUIRE(valid);
		};

		SECTION("when initializing on device, updating on host, verifying on device") { run_test2(access_target::DEVICE, 256); }
		SECTION("when initializing on host, updating on device, verifying on host") { run_test2(access_target::HOST, 256); }

		SECTION("when initializing on device, partially updating larger portion on host, verifying on device") { run_test2(access_target::DEVICE, 512); }
		SECTION("when initializing on host, partially updating larger portion on device, verifying on host") { run_test2(access_target::HOST, 512); }
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager does not ensure coherence when access mode is pure producer", "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize on other side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = 1337 + idx[0]; });

			// Overwrite on this side (but not really) using a pure producer mode
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(faux_overwrite)>(
			    bid, tgt, {128}, {0}, [](cl::sycl::id<1> idx, size_t& value) { /* NOP */ });

			// Verify that buffer does not have initialized contents
			// NOTE: We're reading uninitialized memory here, so anything is possible, technically (but highly unlikely).
			bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
			    bid, tgt, {128}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value != 1337 + idx[0]; });
			REQUIRE(valid);
		};

		SECTION("when initializing on host, verifying on device") { run_test(access_target::DEVICE); }
		SECTION("when initializing on device, verifying on host") { run_test(access_target::HOST); }
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager correctly updates buffer versioning for pure producer accesses that do not require a resize",
	    "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize on other side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Read buffer on this side at both ends (but not in between), forcing a resize without full replication
			buffer_for_each<size_t, 1, cl::sycl::access::mode::read, class UKN(force_copy1)>(
			    bid, tgt, {1}, {0}, [](cl::sycl::id<1> idx, size_t value) { /* NOP */ });
			buffer_for_each<size_t, 1, cl::sycl::access::mode::read, class UKN(force_copy2)>(
			    bid, tgt, {1}, {127}, [](cl::sycl::id<1> idx, size_t value) { /* NOP */ });

			// Overwrite on this side using a pure producer mode, without requiring a resize
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(overwrite)>(
			    bid, tgt, {128}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = 33; });

			// Verify that buffer contains new values
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {128}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value == 33; });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::DEVICE); }
		SECTION("when using host buffers") { run_test(access_target::HOST); }
	}

	/**
	 * This test ensures that the BM doesn't generate superfluous H <-> D data transfers after coherence has already been established
	 * by a previous access. For this to work, we have to cheat a bit and update a buffer after a second access call, which is technically not allowed.
	 */
	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager remembers coherency replications between consecutive accesses", "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();

		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(32, 1, 1));

		SECTION("when using device buffers") {
			host_buffer<size_t, 1>* host_buf;

			// Remember host buffer for later.
			{
				auto info = bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::discard_write, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));
				host_buf = &info.buffer;
			}

			// Initialize buffer on host.
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, access_target::HOST, {32}, {}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Read buffer on device. This makes the device buffer coherent with the host buffer.
			bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));

			// Here we cheat: We override the host data using the pointer we kept from before, without telling the BM (which is not allowed).
			for(size_t i = 0; i < 32; ++i) {
				host_buf->get_pointer()[i] = 33;
			}

			// Now access the buffer on device again for reading and writing. The buffer manager should realize that the newest version is already on the
			// device. After this, the device holds the newest version of the buffer.
			bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::read_write, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));

			// Verify that the data is still what we expect.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, access_target::HOST, {32}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				REQUIRE(valid);
			}

			// Finally, also check the other way round: Accessing the device buffer now doesn't generate a superfluous H -> D transfer.
			// First, we cheat again.
			for(size_t i = 0; i < 32; ++i) {
				host_buf->get_pointer()[i] = 34;
			}

			// Access device buffer. This should still contain the original data.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, access_target::DEVICE, {32}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				REQUIRE(valid);
			}
		}

		SECTION("when using host buffers") {
			cl::sycl::buffer<size_t, 1>* device_buf;

			// Remember device buffer for later.
			{
				auto info = bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::discard_write, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));
				device_buf = &info.buffer;
			}

			// Initialize buffer on device.
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, access_target::DEVICE, {32}, {}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Read buffer on host. This makes the host buffer coherent with the device buffer.
			bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));

			// Here we cheat: We override the device data using the pointer we kept from before, without telling the BM (which is not allowed).
			dq.get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = device_buf->get_access<cl::sycl::access::mode::discard_write>(cgh);
				    cgh.parallel_for<class UKN(overwrite_buf)>(cl::sycl::range<1>(32), [=](cl::sycl::item<1> item) { acc[item] = 33; });
			    })
			    .wait();

			// Now access the buffer on host again for reading and writing. The buffer manager should realize that the newest version is already on the
			// host. After this, the host holds the newest version of the buffer.
			bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::read_write, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));

			// Verify that the data is still what we expect.
			{
				auto info = bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));
				auto acc = info.buffer.get_access<cl::sycl::access::mode::read>();
				bool valid = true;
				for(size_t i = 0; i < 32; ++i) {
					valid &= acc[i] == i;
				}
				REQUIRE(valid);
			}

			// Finally, also check the other way round: Accessing the host buffer now doesn't generate a superfluous D -> H transfer.
			// First, we cheat again.
			dq.get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = device_buf->get_access<cl::sycl::access::mode::discard_write>(cgh);
				    cgh.parallel_for<class UKN(overwrite_buf)>(cl::sycl::range<1>(32), [=](cl::sycl::item<1> item) { acc[item] = 34; });
			    })
			    .wait();

			// Access host buffer. This should still contain the original data.
			{
				auto info = bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(32, 1, 1), cl::sycl::id<3>(0, 0, 0));
				bool valid = true;
				for(size_t i = 0; i < 32; ++i) {
					valid &= info.buffer.get_pointer()[i] == i;
				}
				REQUIRE(valid);
			}
		}
	}

	TEST_CASE_METHOD(
	    buffer_manager_fixture, "buffer_manager retains data that exists on both host and device when resizing buffers", "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize full buffer on other side.
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Request the first half on this side for reading, so that after this, the first half will exist on both sides.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {64}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				CHECK(valid);
				CHECK(get_backing_buffer_range<size_t, 1>(bid, tgt, {64}, {0}) == cl::sycl::range<1>{64});
			}

			// Now request the second half on this side for reading.
			// Since the first half exists on both sides, technically there is no need to retain the previous buffer's contents.
			// While this causes the buffer to be larger than necessary, it saves us an extra transfer (and re-allocation) in the future,
			// in case we ever need the first half again on this side.
			// TODO: This is a time-memory tradeoff and something we might want to change at some point.
			//		 => In particular, if we won't need the first half ever again, this wastes both time and memory!
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {64}, {64}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				CHECK(valid);
				// Check that the buffer has been resized to accomodate both halves.
				REQUIRE(get_backing_buffer_range<size_t, 1>(bid, tgt, {64}, {64}) == cl::sycl::range<1>{128});
			}
		};

		SECTION("when using device buffers") { run_test(access_target::DEVICE); }
		SECTION("when using host buffers") { run_test(access_target::HOST); }
	}

	// This test is in response to a bug that was caused by computing the region to be retained upon buffer resizing as the bounding box of the coherence
	// subrange as well as the old buffer range. While that works fine in 1D, in 2D (and 3D) it can introduce unnecessary H<->D coherence updates.
	// TODO: Ideally we'd also test this for 3D buffers
	TEST_CASE_METHOD(
	    buffer_manager_fixture, "buffer_manager does not introduce superfluous coherence updates when retaining 2D buffers", "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 2>(cl::sycl::range<3>(128, 128, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize whole buffer to known value on other side.
			buffer_for_each<size_t, 2, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128, 128}, {0, 0}, [](cl::sycl::id<2> idx, size_t& value) { value = 1337 * idx[0] + 42 + idx[1]; });

			// Request a set of columns on this side, causing a coherence update.
			{
				bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {128, 64}, {0, 64}, true,
				    [](cl::sycl::id<2> idx, bool current, size_t value) { return current && value == 1337 * idx[0] + 42 + idx[1]; });
				CHECK(valid);
			}

			// Now request a set of rows that partially intersect the columns from before, requiring the backing device buffer to be resized.
			// This resizing should retain the columns, but not introduce an additional coherence update in the empty area not covered by
			// either the columns or rows (i.e., [0,0]-[64,64]).
			{
				bool valid =
				    buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {64, 128}, {64, 0}, true, [](cl::sycl::id<2> idx, bool current, size_t value) {
					    if(idx[1] >= 64) return current && value == 1337 * idx[0] + 42 + idx[1];
					    return current;
				    });
				CHECK(valid);
			}

			// Do a faux overwrite on this side to ensure that no coherence update will be done for the next call to buffer_reduce.
			buffer_for_each<size_t, 2, cl::sycl::access::mode::discard_write, class UKN(faux_overwrite)>(
			    bid, tgt, {128, 128}, {0, 0}, [](cl::sycl::id<2> idx, size_t& value) { /* NOP */ });

			// While the backing buffer also includes the [0,0]-[64,64] region, this part should still be uninitialized.
			// NOTE: We're reading uninitialized memory here, so anything is possible, technically (but highly unlikely).
			{
				bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {64, 64}, {0, 0}, true,
				    [=](cl::sycl::id<2> idx, bool current, size_t value) { return current && value != 1337 * idx[0] + 42 + idx[1]; });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::DEVICE); }
		SECTION("when using host buffers") { run_test(access_target::HOST); }
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager correctly updates buffer versioning for queued transfers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(64, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize buffer on the other side
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(initialize)>(
			    bid, get_other_target(tgt), {64}, {0}, [](cl::sycl::id<1>, size_t& value) { value = 33; });

			// Add transfer for second half on this side
			std::vector<size_t> data(32, 77);
			auto transfer = raw_buffer_data{sizeof(size_t), cl::sycl::range<3>(32, 1, 1)};
			std::memcpy(transfer.get_pointer(), data.data(), sizeof(size_t) * data.size());
			bm.set_buffer_data(bid, cl::sycl::id<3>(32, 0, 0), std::move(transfer));

			// Check that transfer has been correctly ingested
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check_second_half)>(
				    bid, tgt, {64}, {0}, true, [](cl::sycl::id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? 33 : 77)); });
				REQUIRE(valid);
			}

			// Finally, check that accessing the other side now copies the transfer data as well
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check_second_half)>(bid, get_other_target(tgt), {64}, {0}, true,
				    [](cl::sycl::id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? 33 : 77)); });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::DEVICE); }
		SECTION("when using host buffers") { run_test(access_target::HOST); }
	}

	TEST_CASE_METHOD(
	    buffer_manager_fixture, "buffer_manager prioritizes queued transfers over resize/coherency copies for the same ranges", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt, bool resize, bool coherency) {
			// Write first half of buffer.
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(write_buf)>(
			    bid, coherency ? get_other_target(tgt) : tgt, {resize ? size_t(64) : size_t(128)}, {0}, [](cl::sycl::id<1>, size_t& value) { value = 33; });

			// Set full range to new value.
			{
				std::vector<size_t> other(128, 77);
				auto data = raw_buffer_data{sizeof(size_t), cl::sycl::range<3>(128, 1, 1)};
				std::memcpy(data.get_pointer(), other.data(), sizeof(size_t) * other.size());
				bm.set_buffer_data(bid, cl::sycl::id<3>(0, 0, 0), std::move(data));
			}

			// Now read full range.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {96}, {0}, true, [](cl::sycl::id<1>, bool current, size_t value) { return current && value == 77; });
				REQUIRE(valid);
			}
		};

		SECTION("when initializing, resizing and verifying on device") { run_test(access_target::DEVICE, true, false); }
		SECTION("when initializing, resizing and verifying on host") { run_test(access_target::HOST, true, false); }

		SECTION("when initializing on host, verifying on device") { run_test(access_target::DEVICE, false, true); }
		SECTION("when initializing on device, verifying on host") { run_test(access_target::HOST, false, true); }
	}

	TEST_CASE_METHOD(
	    buffer_manager_fixture, "buffer_manager correctly handles transfers that partially overlap with requested buffer range", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize first half of buffer with linear index.
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, tgt, {64}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });

			// Set data that only partially overlaps with currently allocated range.
			{
				std::vector<size_t> init(64, 99);
				auto data = raw_buffer_data{sizeof(size_t), cl::sycl::range<3>(64, 1, 1)};
				std::memcpy(data.get_pointer(), init.data(), sizeof(size_t) * init.size());
				bm.set_buffer_data(bid, cl::sycl::id<3>(32, 0, 0), std::move(data));
			}

			// Check that second half of buffer has been updated...
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {64}, {0}, true,
				    [](cl::sycl::id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? idx[0] : 99)); });
				REQUIRE(valid);
			}

			// ...without changing its original size.
			REQUIRE(get_backing_buffer_range<size_t, 1>(bid, tgt, {64}, {0})[0] == 64);

			// Check that remainder of buffer has been updated as well.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {96}, {0}, true,
				    [](cl::sycl::id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? idx[0] : 99)); });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::DEVICE); }
		SECTION("when using host buffers") { run_test(access_target::HOST); }
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager returns the newest raw buffer data when requested", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(32, 1, 1));

		auto run_test = [&](access_target tgt) {
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(write_buffer)>(
			    bid, get_other_target(tgt), {32}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });
			buffer_for_each<size_t, 1, cl::sycl::access::mode::read_write, class UKN(update_buffer)>(
			    bid, tgt, {32}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value += 1; });
			auto data = bm.get_buffer_data(bid, {0, 0, 0}, {32, 1, 1});
			bool valid = true;
			for(size_t i = 0; i < 32; ++i) {
				valid &= reinterpret_cast<size_t*>(data.get_pointer())[i] == i + 1;
			}
			REQUIRE(valid);
		};

		SECTION("when newest data is on device") { run_test(access_target::DEVICE); }
		SECTION("when newest data is on host") { run_test(access_target::HOST); }

		SECTION("when newest data is split across host and device") {
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(write_first_half)>(
			    bid, access_target::DEVICE, {16}, {0}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0]; });
			buffer_for_each<size_t, 1, cl::sycl::access::mode::discard_write, class UKN(write_second_half)>(
			    bid, access_target::HOST, {16}, {16}, [](cl::sycl::id<1> idx, size_t& value) { value = idx[0] * 2; });
			auto data = bm.get_buffer_data(bid, {0, 0, 0}, {32, 1, 1});
			bool valid = true;
			for(size_t i = 0; i < 32; ++i) {
				valid &= reinterpret_cast<size_t*>(data.get_pointer())[i] == (i < 16 ? i : 2 * i);
			}
			REQUIRE(valid);
		}
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager correctly handles host-initialized buffers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		constexpr size_t SIZE = 64;
		std::vector<size_t> host_buf(SIZE * SIZE);
		for(size_t i = 0; i < 7; ++i) {
			for(size_t j = 0; j < 5; ++j) {
				host_buf[i * SIZE + j] = i * 5 + j;
			}
		}

		auto bid = bm.register_buffer<size_t, 2>(cl::sycl::range<3>(SIZE, SIZE, 1), host_buf.data());

		SECTION("when accessed on host") {
			// Host buffers need to accomodate the full host-initialized data range.
			REQUIRE(get_backing_buffer_range<size_t, 2>(bid, access_target::HOST, {7, 5}, {0, 0}) == cl::sycl::range<2>{SIZE, SIZE});

			bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, access_target::HOST, {7, 5}, {0, 0}, true,
			    [](cl::sycl::id<2> idx, bool current, size_t value) { return current && (value == idx[0] * 5 + idx[1]); });
			REQUIRE(valid);
		}

		SECTION("when accessed on device") {
			// Device buffers still are only as large as required.
			REQUIRE(get_backing_buffer_range<size_t, 2>(bid, access_target::DEVICE, {7, 5}, {0, 0}) == cl::sycl::range<2>{7, 5});

			bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, access_target::DEVICE, {7, 5}, {0, 0}, true,
			    [](cl::sycl::id<2> idx, bool current, size_t value) { return current && (value == idx[0] * 5 + idx[1]); });
			REQUIRE(valid);
		}
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager correctly handles locking", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		// Check that basic functionality works.
		REQUIRE(bm.try_lock(0, {}));
		bm.unlock(0);

		// Lock buffers 1 - 3.
		CHECK(bm.try_lock(1, {1, 2, 3}));

		// No combination of these buffers can be locked.
		REQUIRE(!bm.try_lock(2, {1}));
		REQUIRE(!bm.try_lock(2, {1, 2}));
		REQUIRE(!bm.try_lock(2, {1, 3}));
		REQUIRE(!bm.try_lock(2, {2}));
		REQUIRE(!bm.try_lock(2, {2, 3}));
		REQUIRE(!bm.try_lock(2, {3}));

		// However another buffer can be locked.
		REQUIRE(bm.try_lock(2, {4}));

		// A single locked buffer prevents an entire set of otherwise unlocked buffers from being locked.
		REQUIRE(!bm.try_lock(3, {4, 5, 6}));

		// After unlocking buffer 4 can be locked again.
		bm.unlock(2);
		REQUIRE(bm.try_lock(3, {4, 5, 6}));

		// But 1 - 3 are still locked.
		REQUIRE(!bm.try_lock(4, {1}));
		REQUIRE(!bm.try_lock(4, {2}));
		REQUIRE(!bm.try_lock(4, {3}));

		// Now they can be locked again as well.
		bm.unlock(1);
		REQUIRE(bm.try_lock(4, {3}));
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "buffer_manager throws if accessing locked buffers in unsupported order", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));

		task_id tid = 0;
		auto run_test = [&](auto test_fn) {
			CHECK(bm.try_lock(tid, {bid}));
			test_fn();
			bm.unlock(tid);
			tid++;
		};

		const std::string resize_error_msg = "You are requesting multiple accessors for the same buffer, with later ones requiring a larger part of the "
		                                     "buffer, causing a backing buffer reallocation. "
		                                     "This is currently unsupported. Try changing the order of your calls to buffer::get_access.";
		const std::string discard_error_msg =
		    "You are requesting multiple accessors for the same buffer, using a discarding access mode first, followed by a non-discarding mode. "
		    "This is currently unsupported. Try changing the order of your calls to buffer::get_access.";

		SECTION("when running on device, requiring resize on second access") {
			run_test([&]() {
				bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, {64, 1, 1}, {0, 0, 0});
				REQUIRE_THROWS_WITH((bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, {128, 1, 1}, {0, 0, 0})), resize_error_msg);
			});
		}

		SECTION("when running on host, requiring resize on second access") {
			run_test([&]() {
				bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, {64, 1, 1}, {0, 0, 0});
				REQUIRE_THROWS_WITH((bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, {128, 1, 1}, {0, 0, 0})), resize_error_msg);
			});
		}

		SECTION("when running on device, using consumer after discard access") {
			run_test([&]() {
				bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::discard_write, {64, 1, 1}, {0, 0, 0});
				REQUIRE_THROWS_WITH((bm.get_device_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, {64, 1, 1}, {0, 0, 0})), discard_error_msg);
			});
		}

		SECTION("when running on host, using consumer after discard access") {
			run_test([&]() {
				bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::discard_write, {64, 1, 1}, {0, 0, 0});
				REQUIRE_THROWS_WITH((bm.get_host_buffer<size_t, 1>(bid, cl::sycl::access::mode::read, {64, 1, 1}, {0, 0, 0})), discard_error_msg);
			});
		}
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "accessor correctly handles backing buffer offsets", "[accessor][buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();
		auto bid = bm.register_buffer<size_t, 2>(cl::sycl::range<3>(64, 32, 1));

		SECTION("when using device buffers") {
			auto range = cl::sycl::range<2>(32, 32);
			auto sr = subrange<3>({}, range_cast<3>(range));
			live_pass_device_handler cgh(nullptr, sr, dq);

			auto acc = get_device_accessor<size_t, 2, cl::sycl::access::mode::discard_write>(cgh, bid, {32, 32}, {32, 0});
			// NOTE: Add offset manually to work around ComputeCpp PTX bug (still present as of version 1.1.5)
			// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-98 (psalz).
			cgh.parallel_for<class UKN(write_buf)>(range, [=](cl::sycl::id<2> id) { acc[id + cl::sycl::id<2>(32, 0)] = (id[0] + 32) + id[1]; });
			cgh.get_submission_event().wait();

			auto buf_info = bm.get_host_buffer<size_t, 2>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(32, 32, 1), cl::sycl::id<3>(32, 0, 0));
			bool valid = true;
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					valid &= (buf_info.buffer.get_pointer()[(i - 32) * 32 + j] == i + j);
				}
			}
			REQUIRE(valid);
		}

		SECTION("when using host buffers") {
			auto acc = get_host_accessor<size_t, 2, cl::sycl::access::mode::discard_write>(bid, {32, 32}, {32, 0});
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					acc[{i, j}] = i + j;
				}
			}
			auto buf_info = bm.get_host_buffer<size_t, 2>(bid, cl::sycl::access::mode::read, cl::sycl::range<3>(32, 32, 1), cl::sycl::id<3>(32, 0, 0));
			bool valid = true;
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					valid &= (buf_info.buffer.get_pointer()[(i - 32) * 32 + j] == i + j);
				}
			}
			REQUIRE(valid);
		}
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "accessor supports SYCL special member and hidden friend functions", "[accessor]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();

		auto bid_a = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(32, 1, 1));
		auto bid_b = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(32, 1, 1));
		auto bid_c = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(32, 1, 1));
		auto bid_d = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(32, 1, 1));

		SECTION("when using device buffers") {
			auto range = cl::sycl::range<1>(32);
			auto sr = subrange<3>({}, range_cast<3>(range));
			live_pass_device_handler cgh(nullptr, sr, dq);

			// For device accessors we test this both on host and device

			// Copy ctor
			auto device_acc_a = get_device_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(cgh, bid_a, {32}, {0});
			decltype(device_acc_a) device_acc_a1(device_acc_a);

			// Move ctor
			auto device_acc_b = get_device_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(cgh, bid_b, {32}, {0});
			decltype(device_acc_b) device_acc_b1(std::move(device_acc_b));

			// Copy assignment
			auto device_acc_c = get_device_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(cgh, bid_c, {32}, {0});
			auto device_acc_c1 = get_device_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(cgh, bid_a, {32}, {0});
			device_acc_c1 = device_acc_c;

			// Move assignment
			auto device_acc_d = get_device_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(cgh, bid_d, {32}, {0});
			auto device_acc_d1 = get_device_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(cgh, bid_a, {32}, {0});
			device_acc_d1 = std::move(device_acc_d);

			// Hidden friends (equality operators)
			REQUIRE(device_acc_a == device_acc_a1);
			REQUIRE(device_acc_a1 != device_acc_b1);

			cgh.parallel_for<class UKN(member_fn_test)>(range, [=](cl::sycl::id<1> id) {
				// Copy ctor
				decltype(device_acc_a1) device_acc_a2(device_acc_a1);
				device_acc_a2[id] = 1 * id[0];

				// Move ctor
				decltype(device_acc_b1) device_acc_b2(std::move(device_acc_b1));
				device_acc_b2[id] = 2 * id[0];

				// Copy assignment
				auto device_acc_c2 = device_acc_a1;
				device_acc_c2 = device_acc_c1;
				device_acc_c2[id] = 3 * id[0];

				// Move assignment
				auto device_acc_d2 = device_acc_a1;
				device_acc_d2 = std::move(device_acc_d1);
				device_acc_d2[id] = 4 * id[0];

				// Hidden friends (equality operators) are only required to be defined on the host
			});

			cgh.get_submission_event().wait();

			auto host_acc_a = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_a, {32}, {0});
			auto host_acc_b = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_b, {32}, {0});
			auto host_acc_c = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_c, {32}, {0});
			auto host_acc_d = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_d, {32}, {0});
			bool valid = true;
			for(size_t i = 0; i < 32; ++i) {
				valid &= host_acc_a[i] == 1 * i;
				valid &= host_acc_b[i] == 2 * i;
				valid &= host_acc_c[i] == 3 * i;
				valid &= host_acc_d[i] == 4 * i;
			}
			REQUIRE(valid);
		}

		SECTION("when using host buffers") {
			{
				// Copy ctor
				auto acc_a = get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_a, {32}, {0});
				decltype(acc_a) acc_a1(acc_a);

				// Move ctor
				auto acc_b = get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_b, {32}, {0});
				decltype(acc_b) acc_b1(std::move(acc_b));

				// Copy assignment
				auto acc_c = get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_c, {32}, {0});
				auto acc_c1 = get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_a, {32}, {0});
				acc_c1 = acc_c;

				// Move assignment
				auto acc_d = get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_d, {32}, {0});
				auto acc_d1 = get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_a, {32}, {0});
				acc_d1 = std::move(acc_d);

				// Hidden friends (equality operators)
				REQUIRE(acc_a == acc_a1);
				REQUIRE(acc_a1 != acc_b1);

				for(size_t i = 0; i < 32; ++i) {
					acc_a1[i] = 1 * i;
					acc_b1[i] = 2 * i;
					acc_c1[i] = 3 * i;
					acc_d1[i] = 4 * i;
				}
			}

			auto acc_a = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_a, {32}, {0});
			auto acc_b = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_b, {32}, {0});
			auto acc_c = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_c, {32}, {0});
			auto acc_d = get_host_accessor<size_t, 1, cl::sycl::access::mode::read>(bid_d, {32}, {0});
			bool valid = true;
			for(size_t i = 0; i < 32; ++i) {
				valid &= acc_a[i] == 1 * i;
				valid &= acc_b[i] == 2 * i;
				valid &= acc_c[i] == 3 * i;
				valid &= acc_d[i] == 4 * i;
			}
			REQUIRE(valid);
		}
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "device accessor supports atomic access", "[accessor]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();
		int host_data = 0;
		auto bid = bm.register_buffer<int, 1>(cl::sycl::range<3>(1, 1, 1), &host_data);

		auto range = cl::sycl::range<1>(2048);
		auto sr = subrange<3>({}, range_cast<3>(range));
		live_pass_device_handler cgh(nullptr, sr, dq);


		auto device_acc = get_device_accessor<int, 1, cl::sycl::access::mode::atomic>(cgh, bid, {1}, {0});
		cgh.parallel_for<class UKN(atomic_increment)>(range, [=](cl::sycl::id<1> id) { device_acc[0].fetch_add(2); });
		cgh.get_submission_event().wait();

		auto host_acc = get_host_accessor<int, 1, cl::sycl::access::mode::read>(bid, {1}, {0});
		REQUIRE(host_acc[0] == 4096);
	}

	TEST_CASE_METHOD(buffer_manager_fixture, "host accessor supports get_pointer", "[accessor]") {
		auto& bm = get_buffer_manager();

		auto check_values = [&](const cl::sycl::id<3>* ptr, cl::sycl::range<3> range) {
			bool valid = true;
			for(size_t i = 0; i < range[0]; ++i) {
				for(size_t j = 0; j < range[1]; ++j) {
					for(size_t k = 0; k < range[2]; ++k) {
						const auto offset = i * range[1] * range[2] + j * range[2] + k;
						valid &= ptr[offset] == cl::sycl::id<3>(i, j, k);
					}
				}
			}
			REQUIRE(valid);
		};

		SECTION("for 1D buffers") {
			auto bid = bm.register_buffer<cl::sycl::id<3>, 1>(cl::sycl::range<3>(8, 1, 1));
			buffer_for_each<cl::sycl::id<3>, 1, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, access_target::DEVICE, {8}, {0}, [](cl::sycl::id<1> idx, cl::sycl::id<3>& value) { value = id_cast<3>(idx); });
			auto acc = get_host_accessor<cl::sycl::id<3>, 1, cl::sycl::access::mode::read>(bid, {8}, {0});
			check_values(acc.get_pointer(), {8, 1, 1});
		}

		SECTION("for 2D buffers") {
			auto bid = bm.register_buffer<cl::sycl::id<3>, 2>(cl::sycl::range<3>(8, 8, 1));
			buffer_for_each<cl::sycl::id<3>, 2, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, access_target::DEVICE, {8, 8}, {0, 0}, [](cl::sycl::id<2> idx, cl::sycl::id<3>& value) { value = id_cast<3>(idx); });
			auto acc = get_host_accessor<cl::sycl::id<3>, 2, cl::sycl::access::mode::read>(bid, {8, 8}, {0, 0});
			check_values(acc.get_pointer(), {8, 8, 1});
		}

		SECTION("for 3D buffers") {
			auto bid = bm.register_buffer<cl::sycl::id<3>, 3>(cl::sycl::range<3>(8, 8, 8));
			buffer_for_each<cl::sycl::id<3>, 3, cl::sycl::access::mode::discard_write, class UKN(init)>(
			    bid, access_target::DEVICE, {8, 8, 8}, {0, 0, 0}, [](cl::sycl::id<3> idx, cl::sycl::id<3>& value) { value = id_cast<3>(idx); });
			auto acc = get_host_accessor<cl::sycl::id<3>, 3, cl::sycl::access::mode::read>(bid, {8, 8, 8}, {0, 0, 0});
			check_values(acc.get_pointer(), {8, 8, 8});
		}
	}

	TEST_CASE_METHOD(
	    buffer_manager_fixture, "host accessor throws when calling get_pointer for a backing buffer with different stride or nonzero offset", "[accessor]") {
		auto& bm = get_buffer_manager();
		auto bid_a = bm.register_buffer<size_t, 1>(cl::sycl::range<3>(128, 1, 1));
		auto bid_b = bm.register_buffer<size_t, 2>(cl::sycl::range<3>(128, 128, 1));

		const std::string error_msg = "Buffer cannot be accessed with expected stride";

		// This is not allowed, as the backing buffer hasn't been allocated from offset 0, which means the pointer would point to offset 32.
		REQUIRE_THROWS_WITH((get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_a, {32}, {32}).get_pointer()), error_msg);

		// This is fine, as the backing buffer has been resized to start from 0 now.
		REQUIRE_NOTHROW(get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_a, {64}, {0}).get_pointer());

		// This is now also okay, as the backing buffer starts at 0, and the pointer points to offset 0.
		// (Same semantics as SYCL accessor with offset, i.e., UB outside of requested range).
		REQUIRE_NOTHROW(get_host_accessor<size_t, 1, cl::sycl::access::mode::discard_write>(bid_a, {32}, {32}).get_pointer());

		// In 2D (and 3D) it's trickier, as the stride of the backing buffer must also match what the user expects.
		// This is not allowed, even though the offset is 0.
		REQUIRE_THROWS_WITH((get_host_accessor<size_t, 2, cl::sycl::access::mode::discard_write>(bid_b, {64, 64}, {0, 0}).get_pointer()), error_msg);

		// This is allowed, as we request the full buffer.
		REQUIRE_NOTHROW(get_host_accessor<size_t, 2, cl::sycl::access::mode::discard_write>(bid_b, {128, 128}, {0, 0}).get_pointer());

		// This is now allowed, as the backing buffer has the expected stride.
		REQUIRE_NOTHROW(get_host_accessor<size_t, 2, cl::sycl::access::mode::discard_write>(bid_b, {64, 64}, {0, 0}).get_pointer());

		// Passing an offset is now also possible.
		REQUIRE_NOTHROW(get_host_accessor<size_t, 2, cl::sycl::access::mode::discard_write>(bid_b, {64, 64}, {32, 32}).get_pointer());
	}

	TEST_CASE("host accessor get_host_memory produces the correct memory layout", "[task]") {
		distr_queue q;

		std::vector<char> memory1d(10);
		buffer<char, 1> buf1d(memory1d.data(), cl::sycl::range<1>(10));

		q.submit([=](handler& cgh) {
			auto b = buf1d.get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>(cgh, all<1>());
			cgh.host_task(on_master_node, [=](partition<0> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 1);
				CHECK(dims[0].get_global_offset() == 0);
				CHECK(dims[0].get_local_offset() == 0);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_size() >= 10);
				CHECK(dims[0].get_extent() == 10);
			});
		});

		q.submit([=](handler& cgh) {
			auto b = buf1d.get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>(cgh, one_to_one<1>());
			cgh.host_task(cl::sycl::range<1>(6), cl::sycl::id<1>(2), [=](partition<1> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 1);
				CHECK(dims[0].get_global_offset() == 2);
				CHECK(dims[0].get_local_offset() <= 2);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_size() >= 6);
				CHECK(dims[0].get_local_size() <= 10);
				CHECK(dims[0].get_extent() == 6);
			});
		});

		std::vector<char> memory2d(10 * 10);
		buffer<char, 2> buf2d(memory2d.data(), cl::sycl::range<2>(10, 10));

		q.submit([=](handler& cgh) {
			auto b = buf2d.get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>(cgh, one_to_one<2>());
			cgh.host_task(cl::sycl::range<2>(5, 6), cl::sycl::id<2>(1, 2), [=](partition<2> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 2);
				CHECK(dims[0].get_global_offset() == 1);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_offset() <= 1);
				CHECK(dims[0].get_local_size() >= 6);
				CHECK(dims[0].get_local_size() <= 10);
				CHECK(dims[0].get_extent() == 5);
				CHECK(dims[1].get_global_offset() == 2);
				CHECK(dims[1].get_global_size() == 10);
				CHECK(dims[1].get_extent() == 6);
			});
		});

		std::vector<char> memory3d(10 * 10 * 10);
		buffer<char, 3> buf3d(memory3d.data(), cl::sycl::range<3>(10, 10, 10));

		q.submit([=](handler& cgh) {
			auto b = buf3d.get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::host_buffer>(cgh, one_to_one<3>());
			cgh.host_task(cl::sycl::range<3>(5, 6, 7), cl::sycl::id<3>(1, 2, 3), [=](partition<3> part) {
				auto [ptr, layout] = b.get_host_memory(part);
				auto& dims = layout.get_dimensions();
				REQUIRE(dims.size() == 3);
				CHECK(dims[0].get_global_offset() == 1);
				CHECK(dims[0].get_local_offset() <= 1);
				CHECK(dims[0].get_global_size() == 10);
				CHECK(dims[0].get_local_size() >= 5);
				CHECK(dims[0].get_local_size() <= 10);
				CHECK(dims[0].get_extent() == 5);
				CHECK(dims[1].get_global_offset() == 2);
				CHECK(dims[1].get_local_offset() <= 2);
				CHECK(dims[1].get_global_size() == 10);
				CHECK(dims[1].get_local_size() >= 6);
				CHECK(dims[1].get_local_size() <= 10);
				CHECK(dims[1].get_extent() == 6);
				CHECK(dims[2].get_global_offset() == 3);
				CHECK(dims[2].get_global_size() == 10);
				CHECK(dims[2].get_extent() == 7);
			});
		});
	}

	TEST_CASE("collective host_task produces one item per rank", "[task]") {
		distr_queue{}.submit([=](handler& cgh) {
			cgh.host_task(experimental::collective, [=](experimental::collective_partition part) {
				CHECK(part.get_global_size().size() == runtime::get_instance().get_num_nodes());
				CHECK_NOTHROW(part.get_collective_mpi_comm());
			});
		});
	}

	TEST_CASE("collective host_task share MPI communicator & thread iff they are on the same collective_group", "[task]") {
		std::thread::id default1_thread, default2_thread, primary1_thread, primary2_thread, secondary1_thread, secondary2_thread;
		MPI_Comm default1_comm, default2_comm, primary1_comm, primary2_comm, secondary1_comm, secondary2_comm;

		{
			distr_queue q;
			experimental::collective_group primary_group;
			experimental::collective_group secondary_group;

			q.submit(celerity::allow_by_ref, [&](handler& cgh) {
				cgh.host_task(experimental::collective, [&](experimental::collective_partition part) {
					default1_thread = std::this_thread::get_id();
					default1_comm = part.get_collective_mpi_comm();
				});
			});
			q.submit(celerity::allow_by_ref, [&](handler& cgh) {
				cgh.host_task(experimental::collective(primary_group), [&](experimental::collective_partition part) {
					primary1_thread = std::this_thread::get_id();
					primary1_comm = part.get_collective_mpi_comm();
				});
			});
			q.submit(celerity::allow_by_ref, [&](handler& cgh) {
				cgh.host_task(experimental::collective(secondary_group), [&](experimental::collective_partition part) {
					secondary1_thread = std::this_thread::get_id();
					secondary1_comm = part.get_collective_mpi_comm();
				});
			});
			q.submit(celerity::allow_by_ref, [&](handler& cgh) {
				cgh.host_task(experimental::collective, [&](experimental::collective_partition part) {
					default2_thread = std::this_thread::get_id();
					default2_comm = part.get_collective_mpi_comm();
				});
			});
			q.submit(celerity::allow_by_ref, [&](handler& cgh) {
				cgh.host_task(experimental::collective(primary_group), [&](experimental::collective_partition part) {
					primary2_thread = std::this_thread::get_id();
					primary2_comm = part.get_collective_mpi_comm();
				});
			});
			q.submit(celerity::allow_by_ref, [&](handler& cgh) {
				cgh.host_task(experimental::collective(secondary_group), [&](experimental::collective_partition part) {
					secondary2_thread = std::this_thread::get_id();
					secondary2_comm = part.get_collective_mpi_comm();
				});
			});
		}

		CHECK(default1_thread == default2_thread);
		CHECK(primary1_thread == primary2_thread);
		CHECK(primary1_thread != default1_thread);
		CHECK(secondary1_thread == secondary2_thread);
		CHECK(secondary1_thread != default1_thread);
		CHECK(secondary1_thread != primary1_thread);
		CHECK(default1_comm == default2_comm);
		CHECK(primary1_comm == primary2_comm);
		CHECK(primary1_comm != default1_comm);
		CHECK(secondary1_comm == secondary2_comm);
		CHECK(secondary1_comm != default1_comm);
		CHECK(secondary1_comm != primary1_comm);
	}

	TEST_CASE("accessors behave correctly for 0-dimensional master node kernels", "[accessor]") {
		distr_queue q;
		std::vector mem_a{42};
		buffer<int, 1> buf_a(mem_a.data(), cl::sycl::range<1>{1});
		q.submit([=](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=] { ++a[{0}]; });
		});
		int out = 0;
		q.submit(celerity::allow_by_ref, [=, &out](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=, &out] { out = a[0]; });
		});
		q.slow_full_sync();
		CHECK(out == 43);
	}

} // namespace detail
} // namespace celerity
