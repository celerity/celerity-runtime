#include "sycl_wrappers.h"

#include <random>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <celerity.h>

#include "affinity.h"
#include "executor.h"
#include "named_threads.h"
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

	struct scheduler_testspy {
		static std::thread& get_worker_thread(scheduler& schdlr) { return schdlr.m_worker_thread; }
	};

	struct executor_testspy {
		static std::thread& get_exec_thrd(executor& exec) { return exec.m_exec_thrd; }
	};

	TEST_CASE_METHOD(test_utils::runtime_fixture, "only a single distr_queue can be created", "[distr_queue][lifetime][dx]") {
		distr_queue q1;
		auto q2{q1}; // Copying is allowed
		REQUIRE_THROWS_WITH(distr_queue{}, "Only one celerity::distr_queue can be created per process (but it can be copied!)");
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "distr_queue implicitly initializes the runtime", "[distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::is_initialized());
		distr_queue queue;
		REQUIRE(runtime::is_initialized());
	}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	TEST_CASE_METHOD(test_utils::runtime_fixture, "an explicit device can be provided to distr_queue", "[distr_queue][lifetime]") {
		cl::sycl::default_selector selector;
		cl::sycl::device device{selector};

		SECTION("before the runtime is initialized") {
			REQUIRE_FALSE(runtime::is_initialized());
			REQUIRE_NOTHROW(distr_queue{device});
		}

		SECTION("but not once the runtime has been initialized") {
			REQUIRE_FALSE(runtime::is_initialized());
			runtime::init(nullptr, nullptr);
			REQUIRE_THROWS_WITH(distr_queue{device}, "Passing explicit device not possible, runtime has already been initialized.");
		}
	}
#pragma GCC diagnostic pop

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer implicitly initializes the runtime", "[distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::is_initialized());
		buffer<float, 1> buf(cl::sycl::range<1>{1});
		REQUIRE(runtime::is_initialized());
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer can be copied", "[distr_queue][lifetime]") {
		buffer<float, 1> buf_a{cl::sycl::range<1>{10}};
		buffer<float, 1> buf_b{cl::sycl::range<1>{10}};
		auto buf_c{buf_a};
		buf_b = buf_c;
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "get_access can be called on const buffer", "[buffer]") {
		buffer<float, 2> buf_a{cl::sycl::range<2>{32, 64}};
		auto& tm = runtime::get_instance().get_task_manager();
		const auto tid = test_utils::add_compute_task<class get_access_const>(
		    tm, [buf_a /* capture by value */](handler& cgh) { buf_a.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{}); }, buf_a.get_range());
		const auto tsk = tm.get_task(tid);
		const auto bufs = tsk->get_buffer_access_map().get_accessed_buffers();
		REQUIRE(bufs.size() == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(0).count(cl::sycl::access::mode::read) == 1);
	}

	TEST_CASE("range mapper results are clamped to buffer range", "[range-mapper]") {
		const auto rmfn = [](chunk<3>) { return subrange<3>{{0, 100, 127}, {256, 64, 32}}; };
		range_mapper rm{rmfn, cl::sycl::access::mode::read, cl::sycl::range<3>{128, 128, 128}};
		auto sr = rm.map_3(chunk<3>{});
		REQUIRE(sr.offset == cl::sycl::id<3>{0, 100, 127});
		REQUIRE(sr.range == cl::sycl::range<3>{128, 28, 1});
	}

	TEST_CASE("one_to_one built-in range mapper behaves as expected", "[range-mapper]") {
		range_mapper rm{one_to_one{}, cl::sycl::access::mode::read, cl::sycl::range<2>{128, 128}};
		auto sr = rm.map_2(chunk<2>{{64, 32}, {32, 4}, {128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<2>{64, 32});
		REQUIRE(sr.range == cl::sycl::range<2>{32, 4});
	}

	TEST_CASE("fixed built-in range mapper behaves as expected", "[range-mapper]") {
		range_mapper rm{fixed<1>({{3}, {97}}), cl::sycl::access::mode::read, cl::sycl::range<1>{128}};
		auto sr = rm.map_1(chunk<2>{{64, 32}, {32, 4}, {128, 128}});
		REQUIRE(sr.offset == cl::sycl::id<1>{3});
		REQUIRE(sr.range == cl::sycl::range<1>{97});
	}

	TEST_CASE("slice built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{slice<3>(0), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 32, 32});
			REQUIRE(sr.range == cl::sycl::range<3>{128, 32, 32});
		}
		{
			range_mapper rm{slice<3>(1), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{32, 0, 32});
			REQUIRE(sr.range == cl::sycl::range<3>{32, 128, 32});
		}
		{
			range_mapper rm{slice<3>(2), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{32, 32, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{32, 32, 128});
		}
	}

	TEST_CASE("all built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{all{}, cl::sycl::access::mode::read, cl::sycl::range<1>{128}};
			auto sr = rm.map_1(chunk<1>{});
			REQUIRE(sr.offset == cl::sycl::id<1>{0});
			REQUIRE(sr.range == cl::sycl::range<1>{128});
		}
		{
			range_mapper rm{all{}, cl::sycl::access::mode::read, cl::sycl::range<2>{128, 64}};
			auto sr = rm.map_2(chunk<1>{});
			REQUIRE(sr.offset == cl::sycl::id<2>{0, 0});
			REQUIRE(sr.range == cl::sycl::range<2>{128, 64});
		}
		{
			range_mapper rm{all{}, cl::sycl::access::mode::read, cl::sycl::range<3>{128, 64, 32}};
			auto sr = rm.map_3(chunk<1>{});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{128, 64, 32});
		}
	}

	TEST_CASE("neighborhood built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{neighborhood<1>(10), cl::sycl::access::mode::read, cl::sycl::range<1>{128}};
			auto sr = rm.map_1(chunk<1>{{15}, {10}, {128}});
			REQUIRE(sr.offset == cl::sycl::id<1>{5});
			REQUIRE(sr.range == cl::sycl::range<1>{30});
		}
		{
			range_mapper rm{neighborhood<2>(10, 10), cl::sycl::access::mode::read, cl::sycl::range<2>{128, 128}};
			auto sr = rm.map_2(chunk<2>{{5, 100}, {10, 20}, {128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<2>{0, 90});
			REQUIRE(sr.range == cl::sycl::range<2>{25, 38});
		}
		{
			range_mapper rm{neighborhood<3>(3, 4, 5), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{3, 4, 5}, {1, 1, 1}, {128, 128, 128}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{7, 9, 11});
		}
	}

	TEST_CASE("even_split built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{even_split<3>(), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 345, 678}};
			auto sr = rm.map_3(chunk<1>{{0}, {1}, {8}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{16, 345, 678});
		}
		{
			range_mapper rm{even_split<3>(), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 345, 678}};
			auto sr = rm.map_3(chunk<1>{{4}, {2}, {8}});
			REQUIRE(sr.offset == cl::sycl::id<3>{64, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{32, 345, 678});
		}
		{
			range_mapper rm{even_split<3>(), cl::sycl::access::mode::read, cl::sycl::range<3>{131, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{5}, {2}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{95, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{36, 992, 613});
		}
		{
			range_mapper rm{even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, cl::sycl::range<3>{128, 345, 678}};
			auto sr = rm.map_3(chunk<1>{{0}, {1}, {8}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{20, 345, 678});
		}
		{
			range_mapper rm{even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, cl::sycl::range<3>{131, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{0}, {1}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{0, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{20, 992, 613});
		}
		{
			range_mapper rm{even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, cl::sycl::range<3>{131, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{5}, {2}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{100, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{31, 992, 613});
		}
		{
			range_mapper rm{even_split<3>(cl::sycl::range<3>(10, 1, 1)), cl::sycl::access::mode::read, cl::sycl::range<3>{236, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{6}, {1}, {7}});
			REQUIRE(sr.offset == cl::sycl::id<3>{200, 0, 0});
			REQUIRE(sr.range == cl::sycl::range<3>{36, 992, 613});
		}
	}

	TEST_CASE("task_manager invokes callback upon task creation", "[task_manager]") {
		task_manager tm{1, nullptr};
		size_t call_counter = 0;
		tm.register_task_callback([&call_counter](const task*) { call_counter++; });
		cl::sycl::range<2> gs = {1, 1};
		cl::sycl::id<2> go = {};
		tm.submit_command_group([=](handler& cgh) { cgh.parallel_for<class kernel>(gs, go, [](auto) {}); });
		REQUIRE(call_counter == 1);
		tm.submit_command_group([](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
		REQUIRE(call_counter == 2);
	}

	TEST_CASE("task_manager correctly records compute task information", "[task_manager][task][device_compute_task]") {
		task_manager tm{1, nullptr};
		test_utils::mock_buffer_factory mbf(tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<2>(64, 152));
		auto buf_b = mbf.create_buffer(cl::sycl::range<3>(7, 21, 99));
		const auto tid = test_utils::add_compute_task(
		    tm,
		    [&](handler& cgh) {
			    buf_a.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{});
			    buf_b.get_access<cl::sycl::access::mode::discard_read_write>(cgh, fixed{subrange<3>{{}, {5, 18, 74}}});
		    },
		    cl::sycl::range<2>{32, 128}, cl::sycl::id<2>{32, 24});
		const auto tsk = tm.get_task(tid);
		REQUIRE(tsk->get_type() == task_type::device_compute);
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
		    buf_a.get_id(), cl::sycl::access::mode::read, tsk->get_dimensions(), {tsk->get_global_offset(), tsk->get_global_size()}, tsk->get_global_size());
		REQUIRE(reqs_a == subrange_to_grid_box(subrange<3>({32, 24, 0}, {32, 128, 1})));
		const auto reqs_b = bam.get_requirements_for_access(buf_b.get_id(), cl::sycl::access::mode::discard_read_write, tsk->get_dimensions(),
		    {tsk->get_global_offset(), tsk->get_global_size()}, tsk->get_global_size());
		REQUIRE(reqs_b == subrange_to_grid_box(subrange<3>({}, {5, 18, 74})));
	}

	TEST_CASE("buffer_access_map merges multiple accesses with the same mode", "[task][device_compute_task]") {
		buffer_access_map bam;
		bam.add_access(0, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{3, 0}, {10, 20}}, cl::sycl::access::mode::read, cl::sycl::range<2>{30, 30}));
		bam.add_access(0, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{10, 0}, {7, 20}}, cl::sycl::access::mode::read, cl::sycl::range<2>{30, 30}));
		const auto req = bam.get_requirements_for_access(0, cl::sycl::access::mode::read, 2, subrange<3>({0, 0, 0}, {100, 100, 1}), {100, 100, 1});
		REQUIRE(req == subrange_to_grid_box(subrange<3>({3, 0, 0}, {14, 20, 1})));
	}

	TEST_CASE("tasks gracefully handle get_requirements() calls for buffers they don't access", "[task]") {
		buffer_access_map bam;
		const auto req = bam.get_requirements_for_access(0, cl::sycl::access::mode::read, 3, subrange<3>({0, 0, 0}, {100, 1, 1}), {100, 1, 1});
		REQUIRE(req == subrange_to_grid_box(subrange<3>({0, 0, 0}, {0, 0, 0})));
	}

	TEST_CASE("safe command group functions must not capture by reference", "[lifetime][dx]") {
		int value = 123;
		const auto unsafe = [&]() { return value + 1; };
		REQUIRE_FALSE(is_safe_cgf<decltype(unsafe)>);
		const auto safe = [=]() { return value + 1; };
		REQUIRE(is_safe_cgf<decltype(safe)>);
	}

	namespace foo {
		class MySecondKernel;
	}

	template <typename T>
	class MyThirdKernel;

	TEST_CASE("device_compute tasks derive debug name from kernel name", "[task][!mayfail]") {
		auto tm = std::make_unique<detail::task_manager>(1, nullptr);
		auto t1 =
		    tm->get_task(tm->submit_command_group([](handler& cgh) { cgh.parallel_for<class MyFirstKernel>(cl::sycl::range<1>{1}, [](cl::sycl::id<1>) {}); }));
		auto t2 =
		    tm->get_task(tm->submit_command_group([](handler& cgh) { cgh.parallel_for<foo::MySecondKernel>(cl::sycl::range<1>{1}, [](cl::sycl::id<1>) {}); }));
		auto t3 =
		    tm->get_task(tm->submit_command_group([](handler& cgh) { cgh.parallel_for<MyThirdKernel<int>>(cl::sycl::range<1>{1}, [](cl::sycl::id<1>) {}); }));
		REQUIRE(t1->get_debug_name() == "MyFirstKernel");
		REQUIRE(t2->get_debug_name() == "MySecondKernel");
		REQUIRE(t3->get_debug_name() == "MyThirdKernel<int>");
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "basic SYNC command functionality", "[distr_queue][sync][control-flow]") {
		constexpr int N = 10;

		distr_queue q;
		buffer<int, 1> buff(N);
		std::vector<int> host_buff(N);

		q.submit([=](handler& cgh) {
			auto b = buff.get_access<cl::sycl::access::mode::discard_write>(cgh, one_to_one{});
			cgh.parallel_for<class sync_test>(cl::sycl::range<1>(N), [=](celerity::item<1> item) { b[item] = item.get_linear_id(); });
		});

		q.submit(allow_by_ref, [&](handler& cgh) {
			auto b = buff.get_access<cl::sycl::access::mode::read, target::host_task>(cgh, celerity::access::fixed<1>{{{}, buff.get_range()}});
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
			for(size_t i = 0; i < copy_range[0]; ++i) {
				REQUIRE_LOOP(target_buffer[target_offset[0] + i] == source_offset[0] + i);
			}
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
			for(size_t i = 0; i < copy_range[0]; ++i) {
				for(size_t j = 0; j < copy_range[1]; ++j) {
					const auto id = target_offset + cl::sycl::id<2>{i, j};
					const auto source_id = source_offset + cl::sycl::id<2>{i, j};
					REQUIRE_LOOP(target_buffer[get_linear_index(target_range, id)] == source_id[0] * 10000 + source_id[1]);
				}
			}
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
			for(size_t i = 0; i < copy_range[0]; ++i) {
				for(size_t j = 0; j < copy_range[1]; ++j) {
					for(size_t k = 0; k < copy_range[2]; ++k) {
						const auto id = target_offset + cl::sycl::id<3>{i, j, k};
						const auto source_id = source_offset + cl::sycl::id<3>{i, j, k};
						CAPTURE(
						    id[0], id[1], id[2], target_buffer[get_linear_index(target_range, id)], source_id[0] * 10000 + source_id[1] * 100 + source_id[2]);
						REQUIRE_LOOP(target_buffer[get_linear_index(target_range, id)] == source_id[0] * 10000 + source_id[1] * 100 + source_id[2]);
					}
				}
			}
		}
	}

	TEST_CASE("linearize_subrange works as expected") {
		const cl::sycl::range<3> data1_range{3, 5, 7};
		std::vector<size_t> data1(data1_range.size());

		for(size_t i = 0; i < data1_range[0]; ++i) {
			for(size_t j = 0; j < data1_range[1]; ++j) {
				for(size_t k = 0; k < data1_range[2]; ++k) {
					data1[i * data1_range[1] * data1_range[2] + j * data1_range[2] + k] = i * 100 + j * 10 + k;
				}
			}
		}

		const cl::sycl::range<3> data2_range{2, 2, 4};
		const cl::sycl::id<3> data2_offset{1, 2, 2};
		std::vector<size_t> data2(data2_range.size());
		linearize_subrange(data1.data(), data2.data(), sizeof(size_t), data1_range, {data2_offset, data2_range});

		for(size_t i = 0; i < 2; ++i) {
			for(size_t j = 0; j < 2; ++j) {
				for(size_t k = 0; k < 4; ++k) {
					REQUIRE_LOOP(data2[i * data2_range[1] * data2_range[2] + j * data2_range[2] + k]
					             == (i + data2_offset[0]) * 100 + (j + data2_offset[1]) * 10 + (k + data2_offset[2]));
				}
			}
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "collective host_task produces one item per rank", "[task]") {
		distr_queue{}.submit([=](handler& cgh) {
			cgh.host_task(experimental::collective, [=](experimental::collective_partition part) {
				CHECK(part.get_global_size().size() == runtime::get_instance().get_num_nodes());
				CHECK_NOTHROW(part.get_collective_mpi_comm());
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "collective host_task share MPI communicator & thread iff they are on the same collective_group", "[task]") {
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

	template <typename T>
	extern const int range_dims;
	template <int N>
	constexpr inline int range_dims<cl::sycl::range<N>> = N;

	TEST_CASE_METHOD(test_utils::runtime_fixture, "range mappers are only invocable with correctly-dimensioned chunks", "[range-mapper]") {
		auto rmfn1 = [](chunk<2> chnk) -> subrange<3> { return {}; };
		using rmfn1_t = decltype(rmfn1);
		static_assert(!is_range_mapper_invocable<rmfn1_t, 1>);
		static_assert(!is_range_mapper_invocable<rmfn1_t, 2>);
		static_assert(is_range_mapper_invocable<rmfn1_t, 3>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 1, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 2, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 3, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 1, 2>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 2, 2>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn1_t, 3, 2>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 1, 3>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 2, 3>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn1_t, 3, 3>);

		auto rmfn2 = [](auto chnk, cl::sycl::range<2>) -> subrange<2> { return {}; };
		using rmfn2_t = decltype(rmfn2);
		static_assert(!is_range_mapper_invocable<rmfn2_t, 1>);
		static_assert(is_range_mapper_invocable<rmfn2_t, 2>);
		static_assert(!is_range_mapper_invocable<rmfn2_t, 3>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 1, 1>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn2_t, 2, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 3, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 1, 2>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn2_t, 2, 2>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 3, 2>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 1, 3>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn2_t, 2, 3>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn2_t, 3, 3>);

		auto rmfn3 = [](chunk<3> chnk, auto range) -> subrange<range_dims<decltype(range)>> { return {}; };
		using rmfn3_t = decltype(rmfn3);
		static_assert(is_range_mapper_invocable<rmfn3_t, 1>);
		static_assert(is_range_mapper_invocable<rmfn3_t, 2>);
		static_assert(is_range_mapper_invocable<rmfn3_t, 3>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 1, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 2, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 3, 1>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 1, 2>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 2, 2>);
		static_assert(!is_range_mapper_invocable_for_kernel<rmfn3_t, 3, 2>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn3_t, 1, 3>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn3_t, 2, 3>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn3_t, 3, 3>);

		auto rmfn4 = [](auto chnk, auto range) -> subrange<range_dims<decltype(range)>> { return {}; };
		using rmfn4_t = decltype(rmfn4);
		static_assert(is_range_mapper_invocable<rmfn4_t, 1>);
		static_assert(is_range_mapper_invocable<rmfn4_t, 2>);
		static_assert(is_range_mapper_invocable<rmfn4_t, 3>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 1, 1>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 2, 1>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 3, 1>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 1, 2>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 2, 2>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 3, 2>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 1, 3>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 2, 3>);
		static_assert(is_range_mapper_invocable_for_kernel<rmfn4_t, 3, 3>);

		distr_queue q;
		buffer<int, 2> buf{{10, 10}};

		CHECK_THROWS_WITH(q.submit([=](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{});
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<1>{10}, [=](celerity::item<1>) {});
		}),
		    "Invalid range mapper dimensionality: 1-dimensional kernel submitted with a requirement whose range mapper is neither invocable for chunk<1> nor "
		    "(chunk<1>, range<2>) to produce subrange<2>");

		CHECK_NOTHROW(q.submit([=](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{});
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<2>{10, 10}, [=](celerity::item<2>) {});
		}));

		CHECK_THROWS_WITH(q.submit([=](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{});
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<3>{10, 10, 10}, [=](celerity::item<3>) {});
		}),
		    "Invalid range mapper dimensionality: 3-dimensional kernel submitted with a requirement whose range mapper is neither invocable for chunk<3> nor "
		    "(chunk<3>, range<2>) to produce subrange<2>");

		CHECK_NOTHROW(q.submit([=](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, all{});
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<3>{10, 10, 10}, [=](celerity::item<3>) {});
		}));

		CHECK_NOTHROW(q.submit([=](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, all{});
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range<3>{10, 10, 10}, [=](celerity::item<3>) {});
		}));
	}

	template <int Dims>
	class linear_id_kernel;

	template <int Dims>
	class dimension_runtime_fixture : public test_utils::runtime_fixture {};

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    dimension_runtime_fixture, "item::get_id() includes global offset, item::get_linear_id() does not", "[item]", ((int Dims), Dims), 1, 2, 3) {
		distr_queue q;

		const int n = 3;
		const auto global_offset = detail::id_cast<Dims>(cl::sycl::id<3>{4, 5, 6});

		buffer<size_t, 2> linear_id{{n, Dims + 1}};
		q.submit([=](handler& cgh) {
			accessor a{linear_id, cgh, celerity::access::all{}, write_only, no_init}; // all RM is sane because runtime_tests runs single-node
			cgh.parallel_for<linear_id_kernel<Dims>>(detail::range_cast<Dims>(cl::sycl::range<3>{n, 1, 1}), global_offset, [=](celerity::item<Dims> item) {
				auto i = (item.get_id() - item.get_offset())[0];
				for(int d = 0; d < Dims; ++d) {
					a[i][d] = item[d];
				}
				a[i][Dims] = item.get_linear_id();
			});
		});
		q.submit([=](handler& cgh) {
			accessor a{linear_id, cgh, celerity::access::all{}, read_only_host_task};
			cgh.host_task(on_master_node, [=] {
				for(int i = 0; i < n; ++i) {
					CHECK(a[i][0] == global_offset[0] + i);
					for(int d = 1; d < Dims; ++d) {
						CHECK(a[i][d] == global_offset[d]);
					}
					CHECK(a[i][Dims] == static_cast<size_t>(i));
				}
			});
		});
	}

#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS

	TEST_CASE_METHOD(test_utils::runtime_fixture, "attempting a reduction on buffers with size != 1 throws", "[task-manager]") {
		runtime::init(nullptr, nullptr);
		auto& tm = runtime::get_instance().get_task_manager();

		buffer<float, 1> buf_1{cl::sycl::range<1>{2}};
		CHECK_THROWS(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(wrong_size_1)>(cl::sycl::range<1>{1}, reduction(buf_1, cgh, cl::sycl::plus<float>{}), [=](celerity::item<1>, auto&) {});
		}));

		buffer<float, 1> buf_4{cl::sycl::range<1>{1}};
		CHECK_NOTHROW(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(ok_size_1)>(cl::sycl::range<1>{1}, reduction(buf_4, cgh, cl::sycl::plus<float>{}), [=](celerity::item<1>, auto&) {});
		}));

#if CELERITY_FEATURE_SCALAR_REDUCTIONS

		buffer<float, 2> buf_2{cl::sycl::range<2>{1, 2}};
		CHECK_THROWS(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(wrong_size_2)>(
			    cl::sycl::range<2>{1, 1}, reduction(buf_2, cgh, cl::sycl::plus<float>{}), [=](celerity::item<2>, auto&) {});
		}));

		buffer<float, 3> buf_3{cl::sycl::range<3>{1, 2, 1}};
		CHECK_THROWS(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(wrong_size_3)>(
			    cl::sycl::range<3>{1, 1, 1}, reduction(buf_3, cgh, cl::sycl::plus<float>{}), [=](celerity::item<3>, auto&) {});
		}));

		buffer<float, 2> buf_5{cl::sycl::range<2>{1, 1}};
		CHECK_NOTHROW(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(ok_size_2)>(cl::sycl::range<2>{1, 1}, reduction(buf_5, cgh, cl::sycl::plus<float>{}), [=](celerity::item<2>, auto&) {});
		}));

		buffer<float, 3> buf_6{cl::sycl::range<3>{1, 1, 1}};
		CHECK_NOTHROW(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(ok_size_3)>(
			    cl::sycl::range<3>{1, 1, 1}, reduction(buf_6, cgh, cl::sycl::plus<float>{}), [=](celerity::item<3>, auto&) {});
		}));
#endif
	}

#endif

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler::parallel_for accepts nd_range", "[handler]") {
		distr_queue q;

		// Note: We assume a local range size of 64 here, this should be supported by most devices.

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_1)>(celerity::nd_range<1>{{256}, {64}}, [](nd_item<1> item) {
				group_barrier(item.get_group());
#if !CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 9) // no group primitives
				group_broadcast(item.get_group(), 42);
#endif
			});
		}));

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_2)>(celerity::nd_range<2>{{64, 64}, {8, 8}}, [](nd_item<2> item) {
				group_barrier(item.get_group());
#if !CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 9) // no group primitives
				group_broadcast(item.get_group(), 42, 25);
#endif
			});
		}));

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_3)>(celerity::nd_range<3>{{16, 16, 16}, {4, 4, 4}}, [](nd_item<3> item) {
				group_barrier(item.get_group());
#if !CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 9) // no group primitives
				group_broadcast(item.get_group(), 42, {1, 2, 3});
#endif
			});
		}));
	}

	TEST_CASE("nd_range throws on global_range indivisible by local_range", "[types]") {
		CHECK_THROWS_WITH((celerity::nd_range<1>{{256}, {19}}), "global_range is not divisible by local_range");
		CHECK_THROWS_WITH((celerity::nd_range<1>{{256}, {0}}), "global_range is not divisible by local_range");
		CHECK_THROWS_WITH((celerity::nd_range<2>{{256, 256}, {64, 63}}), "global_range is not divisible by local_range");
		CHECK_THROWS_WITH((celerity::nd_range<2>{{256, 256}, {64, 0}}), "global_range is not divisible by local_range");
		CHECK_THROWS_WITH((celerity::nd_range<3>{{256, 256, 256}, {2, 64, 9}}), "global_range is not divisible by local_range");
		CHECK_THROWS_WITH((celerity::nd_range<3>{{256, 256, 256}, {2, 1, 0}}), "global_range is not divisible by local_range");
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "nd_range kernels support local memory", "[handler]") {
		distr_queue q;
		buffer<int, 1> out{64};

		// Note: We assume a local range size of 32 here, this should be supported by most devices.

		q.submit([=](handler& cgh) {
			local_accessor<int> la{32, cgh};
			accessor ga{out, cgh, celerity::access::one_to_one{}, write_only};
			cgh.parallel_for<class UKN(device_kernel)>(celerity::nd_range<1>{64, 32}, [=](nd_item<1> item) {
				la[item.get_local_id()] = static_cast<int>(item.get_global_linear_id());
				group_barrier(item.get_group());
				ga[item.get_global_id()] = la[item.get_local_range(0) - 1 - item.get_local_id(0)];
			});
		});

		q.submit([=](handler& cgh) {
			accessor ga{out, cgh, celerity::access::all{}, read_only_host_task};
			cgh.host_task(on_master_node, [=] {
				for(size_t i = 0; i < 64; ++i) {
					CHECK(ga[i] == i / 32 * 32 + (32 - 1 - i % 32));
				}
			});
		});
	}

#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS

	TEST_CASE_METHOD(test_utils::runtime_fixture, "reductions can be passed into nd_range kernels", "[handler]") {
		// Note: We assume a local range size of 16 here, this should be supported by most devices.

		buffer<int, 1> b{cl::sycl::range<1>{1}};
		distr_queue{}.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(celerity::nd_range{cl::sycl::range<2>{8, 8}, cl::sycl::range<2>{4, 4}}, reduction(b, cgh, cl::sycl::plus<>{}),
			    [](nd_item<2> item, auto& sum) { sum += item.get_global_linear_id(); });
		});
	}

#endif

#if CELERITY_FEATURE_UNNAMED_KERNELS

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler::parallel_for kernel names are optional", "[handler]") {
		distr_queue q;

		// Note: We assume a local range size of 32 here, this should be supported by most devices.

		// without name
		q.submit([](handler& cgh) { cgh.parallel_for(cl::sycl::range<1>{64}, [](item<1> item) {}); });
		q.submit([=](handler& cgh) { cgh.parallel_for(celerity::nd_range<1>{64, 32}, [](nd_item<1> item) {}); });
#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
		buffer<int> b{{1}};
		q.submit([=](handler& cgh) {
			cgh.parallel_for(
			    cl::sycl::range<1>{64}, reduction(b, cgh, cl::sycl::plus<int>{}), [=](item<1> item, auto& r) { r += static_cast<int>(item.get_linear_id()); });
		});
		q.submit([=](handler& cgh) {
			cgh.parallel_for(celerity::nd_range<1>{64, 32}, reduction(b, cgh, cl::sycl::plus<int>{}),
			    [=](nd_item<1> item, auto& r) { r += static_cast<int>(item.get_global_linear_id()); });
		});
#endif

		// with name
		q.submit([=](handler& cgh) { cgh.parallel_for<class UKN(simple_kernel_with_name)>(cl::sycl::range<1>{64}, [=](item<1> item) {}); });
		q.submit([=](handler& cgh) { cgh.parallel_for<class UKN(nd_range_kernel_with_name)>(celerity::nd_range<1>{64, 32}, [=](nd_item<1> item) {}); });
#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(simple_kernel_with_name_and_reductions)>(
			    cl::sycl::range<1>{64}, reduction(b, cgh, cl::sycl::plus<int>{}), [=](item<1> item, auto& r) { r += static_cast<int>(item.get_linear_id()); });
		});
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_kernel_with_name_and_reductions)>(celerity::nd_range<1>{64, 32}, reduction(b, cgh, cl::sycl::plus<int>{}),
			    [=](nd_item<1> item, auto& r) { r += static_cast<int>(item.get_global_linear_id()); });
		});
#endif
	}

#endif

	// This test case requires actual command execution, which is why it is not in graph_compaction_tests
	TEST_CASE_METHOD(test_utils::runtime_fixture, "tasks behind the deletion horizon are deleted", "[task_manager][task-graph][task-horizon]") {
		using namespace cl::sycl::access;

		distr_queue q;
		auto& tm = runtime::get_instance().get_task_manager();
		tm.set_horizon_step(2);

		constexpr int extents = 16;

		buffer<int, 1> buf_a(extents);
		q.submit([=](handler& cgh) {
			accessor acc{buf_a, cgh, celerity::access::all{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(on_master_node, [] {});
		});

		SECTION("in a simple linear chain of tasks") {
			constexpr int chain_length = 1000;
			constexpr int task_limit = 15;

			for(int i = 0; i < chain_length; ++i) {
				q.submit([=](handler& cgh) {
					accessor acc{buf_a, cgh, celerity::access::all{}, celerity::read_write_host_task};
					cgh.host_task(on_master_node, [] {});
				});

				// we need to wait in each iteration, so that tasks are still generated after some have already been executed
				// (and after they therefore triggered their horizons)
				q.slow_full_sync();
			}

			// need to wait for commands to actually be executed, otherwise no tasks are deleted
			q.slow_full_sync();
			CHECK(tm.get_current_task_count() < task_limit);
		}
	}

#ifndef __APPLE__
	class restore_process_affinity_fixture {
		restore_process_affinity_fixture(const restore_process_affinity_fixture&) = delete;
		restore_process_affinity_fixture(restore_process_affinity_fixture&&) = delete;
		restore_process_affinity_fixture& operator=(const restore_process_affinity_fixture&) = delete;
		restore_process_affinity_fixture& operator=(restore_process_affinity_fixture&&) = delete;

	  public:
#ifdef _WIN32
		restore_process_affinity_fixture() {
			[[maybe_unused]] DWORD_PTR system_mask;
			const auto ret = GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask);
			REQUIRE(ret != FALSE);
		}

		~restore_process_affinity_fixture() {
			const auto ret = SetProcessAffinityMask(GetCurrentProcess(), process_mask);
			REQUIRE(ret != FALSE);
		}

	  private:
		DWORD_PTR process_mask;
#else
		restore_process_affinity_fixture() {
			const auto ret = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &m_process_mask);
			REQUIRE(ret == 0);
		}

		~restore_process_affinity_fixture() {
			const auto ret = pthread_setaffinity_np(pthread_self(), sizeof(m_process_mask), &m_process_mask);
			REQUIRE(ret == 0);
		}

	  private:
		cpu_set_t m_process_mask;
#endif
	};

	TEST_CASE_METHOD(restore_process_affinity_fixture, "affinity_cores_available works as expected", "[affinity]") {
#ifdef _WIN32
		SECTION("in Windows") {
			DWORD_PTR cpu_mask = 1;
			const auto ret = SetProcessAffinityMask(GetCurrentProcess(), cpu_mask);
			REQUIRE(ret != FALSE);
		}
#else
		SECTION("in Posix") {
			cpu_set_t cpu_mask;
			CPU_ZERO(&cpu_mask);
			CPU_SET(0, &cpu_mask);
			const auto ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_mask), &cpu_mask);
			REQUIRE(ret == 0);
		}
#endif
		const auto cores = affinity_cores_available();
		REQUIRE(cores == 1);
	}
#endif

	TEST_CASE_METHOD(test_utils::runtime_fixture, "side_effect API works as expected on a single node", "[side-effect]") {
		distr_queue q;

		experimental::host_object owned_ho{std::vector<int>{}};
		std::vector<int> exterior;
		experimental::host_object ref_ho{std::ref(exterior)};
		experimental::host_object void_ho;

		q.submit([=](handler& cgh) {
			experimental::side_effect append_owned{owned_ho, cgh};
			experimental::side_effect append_ref{ref_ho, cgh};
			experimental::side_effect track_void{void_ho, cgh};
			cgh.host_task(on_master_node, [=] {
				(*append_owned).push_back(1);
				(*append_ref).push_back(1);
			});
		});

		q.submit([=](handler& cgh) {
			experimental::side_effect append_owned{owned_ho, cgh};
			experimental::side_effect append_ref{ref_ho, cgh};
			experimental::side_effect track_void{void_ho, cgh};
			cgh.host_task(on_master_node, [=] {
				append_owned->push_back(2);
				append_ref->push_back(2);
			});
		});

		q.submit([=](handler& cgh) {
			experimental::side_effect check_owned{owned_ho, cgh};
			cgh.host_task(on_master_node, [=] { CHECK(*check_owned == std::vector{1, 2}); });
		});

		q.slow_full_sync();

		CHECK(exterior == std::vector{1, 2});
	}

#if CELERITY_DETAIL_HAS_NAMED_THREADS

	TEST_CASE_METHOD(test_utils::runtime_fixture, "thread names are set", "[threads]") {
		distr_queue q;

		auto& rt = runtime::get_instance();
		auto& schdlr = runtime_testspy::get_schdlr(rt);
		auto& exec = runtime_testspy::get_exec(rt);

		if(rt.is_master_node()) {
			const auto scheduler_thread_name = get_thread_name(scheduler_testspy::get_worker_thread(schdlr).native_handle());
			CHECK(scheduler_thread_name == "cy-scheduler");
		}

		const auto executor_thread_name = get_thread_name(executor_testspy::get_exec_thrd(exec).native_handle());
		CHECK(executor_thread_name == "cy-executor");

		q.submit([](handler& cgh) {
			cgh.host_task(experimental::collective, [&](experimental::collective_partition) {
				const auto base_name = std::string("cy-worker-");
				const auto worker_thread_name = get_thread_name(get_current_thread_handle());
				CHECK_THAT(worker_thread_name, Catch::Matchers::StartsWith(base_name));
			});
		});
	}

#endif

	void dry_run_with_nodes(const size_t nodes) {
		const std::string dryrun_envvar_name = "CELERITY_DRY_RUN_NODES";
		const auto ste = test_utils::set_test_env(dryrun_envvar_name, std::to_string(nodes));

		distr_queue q;

		auto& rt = runtime::get_instance();
		auto& tm = rt.get_task_manager();
		tm.set_horizon_step(2);

		REQUIRE(rt.is_dry_run());

		q.submit([=](handler& cgh) { cgh.host_task(range<1>{nodes * 2}, [](partition<1>) {}); });
		q.slow_full_sync();

		// (intial epoch + task + sync epoch) per node.
		CHECK(runtime_testspy::get_command_count(rt) == 3 * nodes);
		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "Dry run generates commands for an arbitrary number of simulated worker nodes", "[dryrun]") {
		const size_t nodes = GENERATE(values({4, 8, 16}));
		dry_run_with_nodes(nodes);
	}

} // namespace detail
} // namespace celerity
