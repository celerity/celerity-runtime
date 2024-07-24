#include "sycl_wrappers.h"

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

#include <libenvpp/env.hpp>

#include <celerity.h>

#include "affinity.h"
#include "live_executor.h"
#include "named_threads.h"
#include "ranges.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::neighborhood;
	using celerity::access::one_to_one;
	using celerity::access::slice;
	using celerity::experimental::access::even_split;

	struct executor_testspy {
		static std::thread& get_thread(live_executor& exec) { return exec.m_thread; }
	};

	TEST_CASE_METHOD(test_utils::runtime_fixture, "only a single distr_queue can be created", "[distr_queue][lifetime][dx]") {
		distr_queue q1;
		auto q2{q1}; // Copying is allowed
		REQUIRE_THROWS_WITH(distr_queue{}, "Only one celerity::distr_queue can be created per process (but it can be copied!)");
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "distr_queue implicitly initializes the runtime", "[distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::has_instance());
		distr_queue queue;
		REQUIRE(runtime::has_instance());
	}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	TEST_CASE_METHOD(test_utils::runtime_fixture, "an explicit device can be provided to distr_queue", "[distr_queue][lifetime]") {
		cl::sycl::default_selector selector;
		cl::sycl::device device{selector};

		SECTION("before the runtime is initialized") {
			REQUIRE_FALSE(runtime::has_instance());
			REQUIRE_NOTHROW(distr_queue{std::vector{device}});
		}

		SECTION("but not once the runtime has been initialized") {
			REQUIRE_FALSE(runtime::has_instance());
			runtime::init(nullptr, nullptr);
			REQUIRE_THROWS_WITH(distr_queue{std::vector{device}}, "Passing explicit device list not possible, runtime has already been initialized.");
		}
	}
#pragma GCC diagnostic pop

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer implicitly initializes the runtime", "[distr_queue][lifetime]") {
		REQUIRE_FALSE(runtime::has_instance());
		buffer<float, 1> buf(range<1>{1});
		REQUIRE(runtime::has_instance());
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer can be copied", "[distr_queue][lifetime]") {
		buffer<float, 1> buf_a{range<1>{10}};
		buffer<float, 1> buf_b{range<1>{10}};
		auto buf_c{buf_a};
		buf_b = buf_c;
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "get_access can be called on const buffer", "[buffer]") {
		const range<2> range{32, 64};
		std::vector<float> init(range.size());
		buffer<float, 2> buf_a{init.data(), range};
		auto& tm = runtime::get_instance().get_task_manager();
		const auto tid = test_utils::add_compute_task<class get_access_const>(
		    tm, [&](handler& cgh) { buf_a.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{}); }, range);
		const auto tsk = tm.get_task(tid);
		const auto bufs = tsk->get_buffer_access_map().get_accessed_buffers();
		REQUIRE(bufs.size() == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(0).count(cl::sycl::access::mode::read) == 1);
	}

	TEST_CASE("range mapper results are clamped to buffer range", "[range-mapper]") {
		const auto rmfn = [](chunk<3>) { return subrange<3>{{0, 100, 127}, {256, 64, 32}}; };
		range_mapper rm{rmfn, cl::sycl::access::mode::read, range<3>{128, 128, 128}};
		auto sr = rm.map_3(chunk<3>{});
		REQUIRE(sr.offset == id<3>{0, 100, 127});
		REQUIRE(sr.range == range<3>{128, 28, 1});
	}

	TEST_CASE("one_to_one built-in range mapper behaves as expected", "[range-mapper]") {
		range_mapper rm{one_to_one{}, cl::sycl::access::mode::read, range<2>{128, 128}};
		auto sr = rm.map_2(chunk<2>{{64, 32}, {32, 4}, {128, 128}});
		REQUIRE(sr.offset == id<2>{64, 32});
		REQUIRE(sr.range == range<2>{32, 4});
	}

	TEST_CASE("fixed built-in range mapper behaves as expected", "[range-mapper]") {
		range_mapper rm{fixed<1>({{3}, {97}}), cl::sycl::access::mode::read, range<1>{128}};
		auto sr = rm.map_1(chunk<2>{{64, 32}, {32, 4}, {128, 128}});
		REQUIRE(sr.offset == id<1>{3});
		REQUIRE(sr.range == range<1>{97});
	}

	TEST_CASE("slice built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{slice<3>(0), cl::sycl::access::mode::read, range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == id<3>{0, 32, 32});
			REQUIRE(sr.range == range<3>{128, 32, 32});
		}
		{
			range_mapper rm{slice<3>(1), cl::sycl::access::mode::read, range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == id<3>{32, 0, 32});
			REQUIRE(sr.range == range<3>{32, 128, 32});
		}
		{
			range_mapper rm{slice<3>(2), cl::sycl::access::mode::read, range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{32, 32, 32}, {32, 32, 32}, {128, 128, 128}});
			REQUIRE(sr.offset == id<3>{32, 32, 0});
			REQUIRE(sr.range == range<3>{32, 32, 128});
		}
	}

	TEST_CASE("all built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{all{}, cl::sycl::access::mode::read, range<1>{128}};
			auto sr = rm.map_1(chunk<1>{});
			REQUIRE(sr.offset == id<1>{0});
			REQUIRE(sr.range == range<1>{128});
		}
		{
			range_mapper rm{all{}, cl::sycl::access::mode::read, range<2>{128, 64}};
			auto sr = rm.map_2(chunk<1>{});
			REQUIRE(sr.offset == id<2>{0, 0});
			REQUIRE(sr.range == range<2>{128, 64});
		}
		{
			range_mapper rm{all{}, cl::sycl::access::mode::read, range<3>{128, 64, 32}};
			auto sr = rm.map_3(chunk<1>{});
			REQUIRE(sr.offset == id<3>{0, 0, 0});
			REQUIRE(sr.range == range<3>{128, 64, 32});
		}
	}

	TEST_CASE("neighborhood built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{neighborhood<1>(10), cl::sycl::access::mode::read, range<1>{128}};
			auto sr = rm.map_1(chunk<1>{{15}, {10}, {128}});
			REQUIRE(sr.offset == id<1>{5});
			REQUIRE(sr.range == range<1>{30});
		}
		{
			range_mapper rm{neighborhood<2>(10, 10), cl::sycl::access::mode::read, range<2>{128, 128}};
			auto sr = rm.map_2(chunk<2>{{5, 100}, {10, 20}, {128, 128}});
			REQUIRE(sr.offset == id<2>{0, 90});
			REQUIRE(sr.range == range<2>{25, 38});
		}
		{
			range_mapper rm{neighborhood<3>(3, 4, 5), cl::sycl::access::mode::read, range<3>{128, 128, 128}};
			auto sr = rm.map_3(chunk<3>{{3, 4, 5}, {1, 1, 1}, {128, 128, 128}});
			REQUIRE(sr.offset == id<3>{0, 0, 0});
			REQUIRE(sr.range == range<3>{7, 9, 11});
		}
	}

	TEST_CASE("even_split built-in range mapper behaves as expected", "[range-mapper]") {
		{
			range_mapper rm{even_split<3>(), cl::sycl::access::mode::read, range<3>{128, 345, 678}};
			auto sr = rm.map_3(chunk<1>{{0}, {1}, {8}});
			REQUIRE(sr.offset == id<3>{0, 0, 0});
			REQUIRE(sr.range == range<3>{16, 345, 678});
		}
		{
			range_mapper rm{even_split<3>(), cl::sycl::access::mode::read, range<3>{128, 345, 678}};
			auto sr = rm.map_3(chunk<1>{{4}, {2}, {8}});
			REQUIRE(sr.offset == id<3>{64, 0, 0});
			REQUIRE(sr.range == range<3>{32, 345, 678});
		}
		{
			range_mapper rm{even_split<3>(), cl::sycl::access::mode::read, range<3>{131, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{5}, {2}, {7}});
			REQUIRE(sr.offset == id<3>{95, 0, 0});
			REQUIRE(sr.range == range<3>{36, 992, 613});
		}
		{
			range_mapper rm{even_split<3>(range<3>(10, 1, 1)), cl::sycl::access::mode::read, range<3>{128, 345, 678}};
			auto sr = rm.map_3(chunk<1>{{0}, {1}, {8}});
			REQUIRE(sr.offset == id<3>{0, 0, 0});
			REQUIRE(sr.range == range<3>{20, 345, 678});
		}
		{
			range_mapper rm{even_split<3>(range<3>(10, 1, 1)), cl::sycl::access::mode::read, range<3>{131, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{0}, {1}, {7}});
			REQUIRE(sr.offset == id<3>{0, 0, 0});
			REQUIRE(sr.range == range<3>{20, 992, 613});
		}
		{
			range_mapper rm{even_split<3>(range<3>(10, 1, 1)), cl::sycl::access::mode::read, range<3>{131, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{5}, {2}, {7}});
			REQUIRE(sr.offset == id<3>{100, 0, 0});
			REQUIRE(sr.range == range<3>{31, 992, 613});
		}
		{
			range_mapper rm{even_split<3>(range<3>(10, 1, 1)), cl::sycl::access::mode::read, range<3>{236, 992, 613}};
			auto sr = rm.map_3(chunk<1>{{6}, {1}, {7}});
			REQUIRE(sr.offset == id<3>{200, 0, 0});
			REQUIRE(sr.range == range<3>{36, 992, 613});
		}
	}

	TEST_CASE("task_manager invokes callback upon task creation", "[task_manager]") {
		task_manager tm{1, nullptr};
		size_t call_counter = 0;
		tm.register_task_callback([&call_counter](const task*) { call_counter++; });
		range<2> gs = {1, 1};
		id<2> go = {};
		tm.submit_command_group([=](handler& cgh) { cgh.parallel_for<class kernel>(gs, go, [](auto) {}); });
		REQUIRE(call_counter == 1);
		tm.submit_command_group([](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
		REQUIRE(call_counter == 2);
	}

	TEST_CASE("task_manager correctly records compute task information", "[task_manager][task][device_compute_task]") {
		task_manager tm{1, nullptr};
		test_utils::mock_buffer_factory mbf(tm);
		auto buf_a = mbf.create_buffer(range<2>(64, 152), true /* host_initialized */);
		auto buf_b = mbf.create_buffer(range<3>(7, 21, 99));
		const auto tid = test_utils::add_compute_task(
		    tm,
		    [&](handler& cgh) {
			    buf_a.get_access<cl::sycl::access::mode::read>(cgh, one_to_one{});
			    buf_b.get_access<cl::sycl::access::mode::discard_read_write>(cgh, fixed{subrange<3>{{}, {5, 18, 74}}});
		    },
		    range<2>{32, 128}, id<2>{32, 24});
		const auto tsk = tm.get_task(tid);
		REQUIRE(tsk->get_type() == task_type::device_compute);
		REQUIRE(tsk->get_dimensions() == 2);
		REQUIRE(tsk->get_global_size() == range<3>{32, 128, 1});
		REQUIRE(tsk->get_global_offset() == id<3>{32, 24, 0});

		auto& bam = tsk->get_buffer_access_map();
		const auto bufs = bam.get_accessed_buffers();
		REQUIRE(bufs.size() == 2);
		REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_a.get_id()) != bufs.cend());
		REQUIRE(std::find(bufs.cbegin(), bufs.cend(), buf_b.get_id()) != bufs.cend());
		REQUIRE(bam.get_access_modes(buf_a.get_id()).count(cl::sycl::access::mode::read) == 1);
		REQUIRE(bam.get_access_modes(buf_b.get_id()).count(cl::sycl::access::mode::discard_read_write) == 1);
		const auto reqs_a = bam.get_mode_requirements(
		    buf_a.get_id(), cl::sycl::access::mode::read, tsk->get_dimensions(), {tsk->get_global_offset(), tsk->get_global_size()}, tsk->get_global_size());
		REQUIRE(reqs_a == box(subrange<3>({32, 24, 0}, {32, 128, 1})));
		const auto reqs_b = bam.get_mode_requirements(buf_b.get_id(), cl::sycl::access::mode::discard_read_write, tsk->get_dimensions(),
		    {tsk->get_global_offset(), tsk->get_global_size()}, tsk->get_global_size());
		REQUIRE(reqs_b == box(subrange<3>({}, {5, 18, 74})));
	}

	TEST_CASE("buffer_access_map merges multiple accesses with the same mode", "[task][device_compute_task]") {
		buffer_access_map bam;
		bam.add_access(0, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{3, 0}, {10, 20}}, cl::sycl::access::mode::read, range<2>{30, 30}));
		bam.add_access(0, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{10, 0}, {7, 20}}, cl::sycl::access::mode::read, range<2>{30, 30}));
		const auto req = bam.get_mode_requirements(0, cl::sycl::access::mode::read, 2, subrange<3>({0, 0, 0}, {100, 100, 1}), {100, 100, 1});
		REQUIRE(req == box(subrange<3>({3, 0, 0}, {14, 20, 1})));
	}

	TEST_CASE("tasks gracefully handle get_requirements() calls for buffers they don't access", "[task]") {
		buffer_access_map bam;
		const auto req = bam.get_mode_requirements(0, cl::sycl::access::mode::read, 3, subrange<3>({0, 0, 0}, {100, 1, 1}), {100, 1, 1});
		REQUIRE(req == box<3>());
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "basic SYNC command functionality", "[distr_queue][sync][control-flow]") {
		constexpr int N = 10;

		distr_queue q;
		buffer<int, 1> buff(N);
		std::vector<int> host_buff(N);

		q.submit([&](handler& cgh) {
			auto b = buff.get_access<cl::sycl::access::mode::discard_write>(cgh, one_to_one{});
			cgh.parallel_for<class sync_test>(range<1>(N), [=](celerity::item<1> item) { b[item] = item.get_linear_id(); });
		});

		q.submit([&](handler& cgh) {
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

	TEST_CASE_METHOD(test_utils::runtime_fixture, "collective host_task produces one item per rank", "[task]") {
		distr_queue q;
		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance()); // capture here since runtime destructor will run before the host_task
		q.submit([=](handler& cgh) {
			cgh.host_task(experimental::collective, [=](experimental::collective_partition part) {
				CHECK(part.get_global_size().size() == num_nodes);
				CHECK_NOTHROW(part.get_collective_mpi_comm());
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "collective host_task share MPI a communicator iff they are on the same collective_group", "[task]") {
		MPI_Comm default1_comm, default2_comm, primary1_comm, primary2_comm, secondary1_comm, secondary2_comm;

		{
			distr_queue q;
			experimental::collective_group primary_group;
			experimental::collective_group secondary_group;

			q.submit([&](handler& cgh) {
				cgh.host_task(experimental::collective, [&](experimental::collective_partition part) {
					default1_comm = part.get_collective_mpi_comm(); //
				});
			});
			q.submit([&](handler& cgh) {
				cgh.host_task(experimental::collective(primary_group), [&](experimental::collective_partition part) {
					primary1_comm = part.get_collective_mpi_comm(); //
				});
			});
			q.submit([&](handler& cgh) {
				cgh.host_task(experimental::collective(secondary_group), [&](experimental::collective_partition part) {
					secondary1_comm = part.get_collective_mpi_comm(); //
				});
			});
			q.submit([&](handler& cgh) {
				cgh.host_task(experimental::collective, [&](experimental::collective_partition part) {
					default2_comm = part.get_collective_mpi_comm(); //
				});
			});
			q.submit([&](handler& cgh) {
				cgh.host_task(experimental::collective(primary_group), [&](experimental::collective_partition part) {
					primary2_comm = part.get_collective_mpi_comm(); //
				});
			});
			q.submit([&](handler& cgh) {
				cgh.host_task(experimental::collective(secondary_group), [&](experimental::collective_partition part) {
					secondary2_comm = part.get_collective_mpi_comm(); //
				});
			});
		}

		CHECK(default1_comm == default2_comm);
		CHECK(primary1_comm == primary2_comm);
		CHECK(primary1_comm != default1_comm);
		CHECK(secondary1_comm == secondary2_comm);
		CHECK(secondary1_comm != default1_comm);
		CHECK(secondary1_comm != primary1_comm);

		// Celerity must also ensure that no non-deterministic, artificial dependency chains are introduced by submitting independent tasks from multiple
		// collective groups onto the same backend thread in-order. If that were to happen, an inter-node mismatch between execution orders nodes would cause
		// deadlocks in the user-provided collective operations. An earlier version of the runtime ensured this by spawning a separate thread per collective
		// group, whereas currently, we allow unbounded concurrency between host instructions to provide the same guarantees.
	}

	template <typename T>
	extern const int range_dims;
	template <int N>
	constexpr inline int range_dims<range<N>> = N;

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

		auto rmfn2 = [](auto chnk, range<2>) -> subrange<2> { return {}; };
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

		CHECK_THROWS_WITH(q.submit([&](handler& cgh) {
			auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, one_to_one{});
			cgh.parallel_for<class UKN(kernel)>(range<1>{10}, [=](celerity::item<1>) { (void)acc; });
		}),
		    "Invalid range mapper dimensionality: 1-dimensional kernel submitted with a requirement whose range mapper is neither invocable for chunk<1> nor "
		    "(chunk<1>, range<2>) to produce subrange<2>");

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, one_to_one{});
			cgh.parallel_for<class UKN(kernel)>(range<2>{10, 10}, [=](celerity::item<2>) { (void)acc; });
		}));

		CHECK_THROWS_WITH(q.submit([&](handler& cgh) {
			auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, one_to_one{});
			cgh.parallel_for<class UKN(kernel)>(range<3>{10, 10, 10}, [=](celerity::item<3>) { (void)acc; });
		}),
		    "Invalid range mapper dimensionality: 3-dimensional kernel submitted with a requirement whose range mapper is neither invocable for chunk<3> nor "
		    "(chunk<3>, range<2>) to produce subrange<2>");

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			auto acc = buf.get_access<cl::sycl::access::mode::read>(cgh, all{});
			cgh.parallel_for<class UKN(kernel)>(range<3>{10, 10, 10}, [=](celerity::item<3>) { (void)acc; });
		}));

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			auto acc = buf.get_access<cl::sycl::access::mode::read>(cgh, all{});
			cgh.parallel_for<class UKN(kernel)>(range<3>{10, 10, 10}, [=](celerity::item<3>) { (void)acc; });
		}));
	}

	template <int Dims>
	class linear_id_kernel;

	template <int Dims>
	class dimension_runtime_fixture : public test_utils::runtime_fixture {};

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    dimension_runtime_fixture, "item::get_id() includes global offset, item::get_linear_id() does not", "[item]", ((int Dims), Dims), 1, 2, 3) {
		distr_queue q{std::vector{sycl::device{sycl::default_selector_v}}}; // Initialize runtime with a single device so we don't get multiple chunks

		const int n = 3;
		const auto global_offset = test_utils::truncate_id<Dims>({4, 5, 6});

		buffer<size_t, 2> linear_id{{n, Dims + 1}};
		q.submit([&](handler& cgh) {
			accessor a{linear_id, cgh, celerity::access::all{}, write_only, no_init}; // all RM is sane because runtime_tests runs single-node
			cgh.parallel_for<linear_id_kernel<Dims>>(detail::range_cast<Dims>(range<1>{n}), global_offset, [=](celerity::item<Dims> item) {
				auto i = (item.get_id() - item.get_offset())[0];
				for(int d = 0; d < Dims; ++d) {
					a[i][d] = item[d];
				}
				a[i][Dims] = item.get_linear_id();
			});
		});
		q.submit([&](handler& cgh) {
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

	TEST_CASE_METHOD(test_utils::runtime_fixture, "attempting a reduction on buffers with size != 1 throws", "[task-manager][reduction]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation

		runtime::init(nullptr, nullptr);
		auto& tm = runtime::get_instance().get_task_manager();

		buffer<float, 1> buf_1{range<1>{2}};
		CHECK_THROWS(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(wrong_size_1)>(
			    range<1>{1}, reduction(buf_1, cgh, cl::sycl::plus<float>{}, property::reduction::initialize_to_identity()), [=](celerity::item<1>, auto&) {});
		}));

		buffer<float, 1> buf_4{range<1>{1}};
		CHECK_NOTHROW(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(ok_size_1)>(
			    range<1>{1}, reduction(buf_4, cgh, cl::sycl::plus<float>{}, property::reduction::initialize_to_identity()), [=](celerity::item<1>, auto&) {});
		}));

		buffer<float, 2> buf_2{range<2>{1, 2}};
		CHECK_THROWS(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(wrong_size_2)>(range<2>{1, 1},
			    reduction(buf_2, cgh, cl::sycl::plus<float>{}, property::reduction::initialize_to_identity()), [=](celerity::item<2>, auto&) {});
		}));

		buffer<float, 3> buf_3{range<3>{1, 2, 1}};
		CHECK_THROWS(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(wrong_size_3)>(range<3>{1, 1, 1},
			    reduction(buf_3, cgh, cl::sycl::plus<float>{}, property::reduction::initialize_to_identity()), [=](celerity::item<3>, auto&) {});
		}));

		buffer<float, 2> buf_5{range<2>{1, 1}};
		CHECK_NOTHROW(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(ok_size_2)>(range<2>{1, 1},
			    reduction(buf_5, cgh, cl::sycl::plus<float>{}, property::reduction::initialize_to_identity()), [=](celerity::item<2>, auto&) {});
		}));

		buffer<float, 3> buf_6{range<3>{1, 1, 1}};
		CHECK_NOTHROW(tm.submit_command_group([&](handler& cgh) { //
			cgh.parallel_for<class UKN(ok_size_3)>(range<3>{1, 1, 1},
			    reduction(buf_6, cgh, cl::sycl::plus<float>{}, property::reduction::initialize_to_identity()), [=](celerity::item<3>, auto&) {});
		}));
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler::parallel_for accepts nd_range", "[handler]") {
		distr_queue q;

		// Note: We assume a local range size of 64 here, this should be supported by most devices.

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_1)>(celerity::nd_range<1>{{256}, {64}}, [](nd_item<1> item) {
				group_barrier(item.get_group());
				group_broadcast(item.get_group(), 42);
			});
		}));

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_2)>(celerity::nd_range<2>{{64, 64}, {8, 8}}, [](nd_item<2> item) {
				group_barrier(item.get_group());
				group_broadcast(item.get_group(), 42, 25);
			});
		}));

		CHECK_NOTHROW(q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_3)>(celerity::nd_range<3>{{16, 16, 16}, {4, 4, 4}}, [](nd_item<3> item) {
				group_barrier(item.get_group());
				group_broadcast(item.get_group(), 42, {1, 2, 3});
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

		q.submit([&](handler& cgh) {
			local_accessor<int> la{32, cgh};
			accessor ga{out, cgh, celerity::access::one_to_one{}, write_only, no_init};
			cgh.parallel_for<class UKN(device_kernel)>(celerity::nd_range<1>{64, 32}, [=](nd_item<1> item) {
				la[item.get_local_id()] = static_cast<int>(item.get_global_linear_id());
				group_barrier(item.get_group());
				ga[item.get_global_id()] = la[item.get_local_range(0) - 1 - item.get_local_id(0)];
			});
		});

		q.submit([&](handler& cgh) {
			accessor ga{out, cgh, celerity::access::all{}, read_only_host_task};
			cgh.host_task(on_master_node, [=] {
				for(int i = 0; i < 64; ++i) {
					CHECK(ga[static_cast<size_t>(i)] == i / 32 * 32 + (32 - 1 - i % 32));
				}
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "reductions can be passed into nd_range kernels", "[handler][reduction]") {
		// Note: We assume a local range size of 16 here, this should be supported by most devices.

		buffer<int, 1> b{range<1>{1}};
		distr_queue{}.submit([&](handler& cgh) {
			cgh.parallel_for(celerity::nd_range{range<2>{8, 8}, range<2>{4, 4}},
			    reduction(b, cgh, sycl::plus<int>(), property::reduction::initialize_to_identity()),
			    [](nd_item<2> item, auto& sum) { sum += static_cast<int>(item.get_global_linear_id()); });
		});
	}

#if CELERITY_FEATURE_UNNAMED_KERNELS

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler::parallel_for kernel names are optional", "[handler][reduction]") {
		distr_queue q;

		// Note: We assume a local range size of 32 here, this should be supported by most devices.

		// without name
		q.submit([](handler& cgh) { cgh.parallel_for(range<1>{64}, [](item<1> item) {}); });
		q.submit([=](handler& cgh) { cgh.parallel_for(celerity::nd_range<1>{64, 32}, [](nd_item<1> item) {}); });
		buffer<int> b{{1}};
		q.submit([&](handler& cgh) {
			cgh.parallel_for(range<1>{64}, reduction(b, cgh, cl::sycl::plus<int>{}, property::reduction::initialize_to_identity()),
			    [=](item<1> item, auto& r) { r += static_cast<int>(item.get_linear_id()); });
		});
		q.submit([&](handler& cgh) {
			cgh.parallel_for(celerity::nd_range<1>{64, 32}, reduction(b, cgh, cl::sycl::plus<int>{}),
			    [=](nd_item<1> item, auto& r) { r += static_cast<int>(item.get_global_linear_id()); });
		});

		// with name
		q.submit([=](handler& cgh) { cgh.parallel_for<class UKN(simple_kernel_with_name)>(range<1>{64}, [=](item<1> item) {}); });
		q.submit([=](handler& cgh) { cgh.parallel_for<class UKN(nd_range_kernel_with_name)>(celerity::nd_range<1>{64, 32}, [=](nd_item<1> item) {}); });
		q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(simple_kernel_with_name_and_reductions)>(
			    range<1>{64}, reduction(b, cgh, cl::sycl::plus<int>{}), [=](item<1> item, auto& r) { r += static_cast<int>(item.get_linear_id()); });
		});
		q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(nd_range_kernel_with_name_and_reductions)>(celerity::nd_range<1>{64, 32}, reduction(b, cgh, cl::sycl::plus<int>{}),
			    [=](nd_item<1> item, auto& r) { r += static_cast<int>(item.get_global_linear_id()); });
		});
	}

#endif

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler throws if effective split constraint does not evenly divide global size", "[handler]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation
		distr_queue q;

		const auto submit = [&q](auto range, auto constraint) {
			q.submit([&](handler& cgh) {
				experimental::constrain_split(cgh, constraint);
				cgh.parallel_for(range, [=](auto) {});
			});
		};

		CHECK_THROWS_WITH(submit(range<1>{10}, range<1>{0}), "Split constraint cannot be 0");
		CHECK_THROWS_WITH(submit(range<2>{10, 10}, range<2>{2, 0}), "Split constraint cannot be 0");
		CHECK_THROWS_WITH(submit(range<3>{10, 10, 10}, range<3>{2, 2, 0}), "Split constraint cannot be 0");

		CHECK_NOTHROW(submit(range<1>{10}, range<1>{2}));
		CHECK_NOTHROW(submit(range<2>{10, 8}, range<2>{2, 4}));
		CHECK_NOTHROW(submit(range<3>{10, 8, 16}, range<3>{2, 4, 8}));

		CHECK_THROWS_WITH(submit(range<1>{10}, range<1>{3}), "The split constraint [3] does not evenly divide the kernel global size [10]");
		CHECK_THROWS_WITH(submit(range<2>{10, 8}, range<2>{2, 5}), "The split constraint [2,5] does not evenly divide the kernel global size [10,8]");
		CHECK_THROWS_WITH(
		    submit(range<3>{10, 8, 16}, range<3>{2, 4, 9}), "The split constraint [2,4,9] does not evenly divide the kernel global size [10,8,16]");

		CHECK_THROWS_WITH(submit(range<1>{10}, range<1>{20}), "The split constraint [20] does not evenly divide the kernel global size [10]");

		CHECK_NOTHROW(submit(nd_range<1>{100, 10}, range<1>{2}));
		CHECK_NOTHROW(submit(nd_range<2>{{100, 80}, {10, 20}}, range<2>{2, 4}));
		CHECK_NOTHROW(submit(nd_range<3>{{100, 80, 60}, {1, 2, 30}}, range<3>{2, 4, 20}));

		CHECK_THROWS_WITH(submit(nd_range<1>{100, 10}, range<1>{3}), "The effective split constraint [30] does not evenly divide the kernel global size [100]");
		CHECK_THROWS_WITH(submit(nd_range<2>{{100, 80}, {10, 20}}, range<2>{2, 3}),
		    "The effective split constraint [10,60] does not evenly divide the kernel global size [100,80]");
		CHECK_THROWS_WITH(submit(nd_range<3>{{100, 80, 60}, {1, 2, 30}}, range<3>{1, 2, 40}),
		    "The effective split constraint [1,2,120] does not evenly divide the kernel global size [100,80,60]");
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler throws when accessor target does not match command type", "[handler]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation

		distr_queue q;
		buffer<size_t, 1> buf0{1};
		buffer<size_t, 1> buf1{1};

		SECTION("capturing host accessor into device kernel") {
#if !defined(__SYCL_COMPILER_VERSION) // TODO: This may break when using hipSYCL w/ DPC++ as compiler
			CHECK_THROWS_WITH(([&] {
				q.submit([&](handler& cgh) {
					accessor acc0{buf1, cgh, one_to_one{}, write_only_host_task};
					accessor acc1{buf0, cgh, one_to_one{}, write_only};
					cgh.parallel_for(range<1>(1), [=](item<1>) {
						(void)acc0;
						(void)acc1;
					});
				});
			})(),
			    "Accessor 0 for buffer 1 has wrong target ('host_task' instead of 'device').");
#else
			SKIP("DPC++ does not allow for host accessors to be captured into kernels.");
#endif
		}

		SECTION("capturing device accessor into host task") {
			CHECK_THROWS_WITH(([&] {
				q.submit([&](handler& cgh) {
					accessor acc0{buf1, cgh, one_to_one{}, write_only_host_task};
					accessor acc1{buf0, cgh, one_to_one{}, write_only};
					cgh.host_task(on_master_node, [=]() {
						(void)acc0;
						(void)acc1;
					});
				});
			})(),
			    "Accessor 1 for buffer 0 has wrong target ('device' instead of 'host_task').");
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler throws when accessing host objects within device tasks", "[handler]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation

		distr_queue q;
#if !defined(__SYCL_COMPILER_VERSION) // TODO: This may break when using hipSYCL w/ DPC++ as compiler
		experimental::host_object<size_t> ho;

		CHECK_THROWS_WITH(([&] {
			q.submit([&](handler& cgh) {
				experimental::side_effect se{ho, cgh};
				cgh.parallel_for(range<1>(1), [=](item<1>) { (void)se; });
			});
		})(),
		    "Side effects can only be used in host tasks.");
#else
		SKIP("DPC++ does not allow for side effects to be captured into kernels.");
#endif
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler throws when not all accessors / side-effects are copied into the kernel", "[handler]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation

		distr_queue q;

		buffer<size_t, 1> buf0{1};
		buffer<size_t, 1> buf1{1};
		experimental::host_object<size_t> ho;

		CHECK_THROWS_WITH(([&] {
			q.submit([&](handler& cgh) {
				accessor acc0{buf0, cgh, one_to_one{}, write_only};
				accessor acc1{buf1, cgh, one_to_one{}, write_only};
				// DPC++ has its own compile-time check for this, so we can't actually capture anything by reference
#if !defined(__SYCL_COMPILER_VERSION) // TODO: This may break when using hipSYCL w/ DPC++ as compiler
				cgh.parallel_for(range<1>(1), [acc0, &acc1 /* oops */](item<1>) {
					(void)acc0;
					(void)acc1;
				});
#else
				cgh.parallel_for(range<1>(1), [acc0](item<1>) { (void)acc0; });
#endif
			});
		})(),
		    "Accessor 1 for buffer 1 is not being copied into the kernel. This indicates a bug. Make sure the accessor is captured by value and not by "
		    "reference, or remove it entirely.");

		CHECK_THROWS_WITH(([&] {
			q.submit([&](handler& cgh) {
				accessor acc0{buf0, cgh, one_to_one{}, write_only_host_task};
				accessor acc1{buf1, cgh, one_to_one{}, write_only_host_task};
				cgh.host_task(on_master_node, [&acc0 /* oops */, acc1]() {
					(void)acc0;
					(void)acc1;
				});
			});
		})(),
		    "Accessor 0 for buffer 0 is not being copied into the kernel. This indicates a bug. Make sure the accessor is captured by value and not by "
		    "reference, or remove it entirely.");

		CHECK_THROWS_WITH(([&] {
			q.submit([&](handler& cgh) {
				accessor acc0{buf0, cgh, one_to_one{}, write_only_host_task};
				experimental::side_effect se{ho, cgh};
				cgh.host_task(on_master_node, [acc0, &se /* oops */]() {
					(void)acc0;
					(void)se;
				});
			});
		})(),
		    "The number of side effects copied into the kernel is fewer (0) than expected (1). This may be indicative of a bug. Make sure all side effects are "
		    "captured by value and not by reference, and remove unused ones.");
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler does not throw when void side effects are not copied into a kernel", "[handler]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation

		distr_queue q;
		experimental::host_object<void> ho;

		CHECK_NOTHROW(([&] {
			q.submit([&](handler& cgh) {
				// This is just used to establish an order between tasks, so capturing it doesn't make sense.
				experimental::side_effect se{ho, cgh};
				cgh.host_task(on_master_node, []() {});
			});
		})());
	}

	// This test checks that the diagnostic is not simply implemented by counting the number captured of accessors;
	// instead it can distinguish between different accessors and copies of the same accessor.
	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler recognizes copies of same accessor being captured multiple times", "[handler]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // throwing in submit() will warn about unconsumed task_id reservation

		distr_queue q;
		buffer<size_t, 1> buf1{1};

		CHECK_THROWS_WITH(([&] {
			q.submit([&](handler& cgh) {
				accessor acc1{buf1, cgh, one_to_one{}, write_only};
				auto acc1a = acc1;
				accessor acc2{buf1, cgh, one_to_one{}, write_only};
				cgh.parallel_for(range<1>(1), [acc1, acc1a](item<1>) {
					(void)acc1;
					(void)acc1a;
				});
			});
		})(),
		    "Accessor 1 for buffer 0 is not being copied into the kernel. This indicates a bug. Make sure the accessor is captured by value and not by "
		    "reference, or remove it entirely.");
	}

	// Since the diagnostic for side effects is based on a simple count, we currently cannot tell whether
	// a single side effect is copied several times (which is fine), versus another not being copied.
	TEST_CASE_METHOD(test_utils::runtime_fixture, "handler recognizes copies of same side-effect being captured multiple times", "[handler][!shouldfail]") {
		distr_queue q;

		experimental::host_object<size_t> ho1;
		experimental::host_object<size_t> ho2;

		CHECK_THROWS_WITH(([&] {
			q.submit([&](handler& cgh) {
				experimental::side_effect se1{ho1, cgh};
				auto se2 = se1;
				experimental::side_effect se3{ho2, cgh};
				cgh.host_task(on_master_node, [se1, se2, &se3 /* oops! */]() {
					(void)se1;
					(void)se2;
					(void)se3;
				});
			});
		})(),
		    "(NYI)");
	}

	// This test case requires actual command execution, which is why it is not in graph_compaction_tests
	TEST_CASE_METHOD(test_utils::runtime_fixture, "tasks behind the deletion horizon are deleted", "[task_manager][task-graph][task-horizon]") {
		using namespace cl::sycl::access;

		constexpr int horizon_step_size = 2;

		distr_queue q;
		auto& tm = runtime::get_instance().get_task_manager();
		tm.set_horizon_step(horizon_step_size);

		const int init = 42;
		buffer<int, 0> buf_a(&init, {});

		SECTION("in a simple linear chain of tasks") {
			std::mutex m;
			int completed_step = -1;
			std::condition_variable cv;

			constexpr int chain_length = 1000;
			for(int i = 0; i < chain_length; ++i) {
				q.submit([&](handler& cgh) {
					accessor acc{buf_a, cgh, celerity::access::all{}, celerity::read_write_host_task};
					cgh.host_task(on_master_node, [&, acc, i] {
						(void)acc;
						std::lock_guard lock(m);
						completed_step = i;
						cv.notify_all();
					});
				});

				// We need to wait in each iteration, so that tasks are still generated after some have already been executed (and after they therefore
				// triggered their horizons). We can't use slow_full_sync for this as it will begin a new epoch and force-prune the task graph itself.
				std::unique_lock lock(m);
				cv.wait(lock, [&] { return completed_step == i; });
			}

			// There are 2 sets of `horizon_step_size` host tasks after the current effective epoch, 2 horizon tasks, plus up to `horizon_step_size` additional
			// host tasks that will be deleted on the next submission.
			constexpr int visible_horizons = 2;
			constexpr int max_visible_host_tasks = (visible_horizons + 1) * horizon_step_size;
			constexpr int task_limit = max_visible_host_tasks + visible_horizons;
			CHECK(tm.get_current_task_count() <= task_limit);
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

		q.submit([&](handler& cgh) {
			experimental::side_effect append_owned{owned_ho, cgh};
			experimental::side_effect append_ref{ref_ho, cgh};
			experimental::side_effect track_void{void_ho, cgh};
			cgh.host_task(on_master_node, [=] {
				(*append_owned).push_back(1);
				(*append_ref).push_back(1);
			});
		});

		q.submit([&](handler& cgh) {
			experimental::side_effect append_owned{owned_ho, cgh};
			experimental::side_effect append_ref{ref_ho, cgh};
			experimental::side_effect track_void{void_ho, cgh};
			cgh.host_task(on_master_node, [=] {
				append_owned->push_back(2);
				append_ref->push_back(2);
			});
		});

		q.submit([&](handler& cgh) {
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
		auto& exec = *utils::as<live_executor>(&runtime_testspy::get_exec(rt));

		const auto scheduler_thread_name = get_thread_name(scheduler_testspy::get_thread(schdlr).native_handle());
		CHECK(scheduler_thread_name == "cy-scheduler");

		const auto executor_thread_name = get_thread_name(executor_testspy::get_thread(exec).native_handle());
		CHECK(executor_thread_name == "cy-executor");

		q.submit([](handler& cgh) {
			cgh.host_task(experimental::collective, [&](experimental::collective_partition) {
				const auto base_name = std::string("cy-host-");
				const auto worker_thread_name = get_thread_name(get_current_thread_handle());
				CHECK_THAT(worker_thread_name, Catch::Matchers::StartsWith(base_name));
			});
		});
	}

#endif

	const std::string dryrun_envvar_name = "CELERITY_DRY_RUN_NODES";

	TEST_CASE_METHOD(test_utils::runtime_fixture, "dry run generates commands for an arbitrary number of simulated worker nodes", "[dryrun]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // dry run unconditionally warns when enabled

		const size_t num_nodes = GENERATE(values({4, 8, 16}));

		env::scoped_test_environment ste(std::unordered_map<std::string, std::string>{{dryrun_envvar_name, std::to_string(num_nodes)}});

		distr_queue q{std::vector{sycl::device{sycl::default_selector_v}}}; // Initialize runtime with a single device so we don't get multiple chunks

		auto& rt = runtime::get_instance();
		auto& tm = rt.get_task_manager();
		tm.set_horizon_step(2);

		REQUIRE(rt.is_dry_run());

		buffer<int, 1> buf{range<1>(10)};
		q.submit([&](handler& cgh) {
			accessor acc{buf, cgh, all{}, write_only_host_task, no_init};
			cgh.host_task(on_master_node, [=] { (void)acc; });
		});
		q.submit([&](handler& cgh) {
			accessor acc{buf, cgh, all{}, read_only_host_task};
			cgh.host_task(range<1>{num_nodes * 2}, [=](partition<1>) { (void)acc; });
		});

		// intial epoch + master-node task + 1 push per node + host task
		// (dry runs currently always simulate node 0, hence the master-node task)
		CHECK(scheduler_testspy::get_command_count(runtime_testspy::get_schdlr(rt)) == 3 + num_nodes);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "dry run proceeds on fences", "[dryrun]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // dry run unconditionally warns when enabled

		env::scoped_test_environment ste(std::unordered_map<std::string, std::string>{{dryrun_envvar_name, "1"}});

		distr_queue q;
		REQUIRE(runtime::get_instance().is_dry_run());

		SECTION("for buffers") {
			buffer<bool, 0> buf{false};
			q.submit([&](handler& cgh) {
				accessor acc{buf, cgh, all{}, write_only_host_task};
				cgh.host_task(on_master_node, [=] { acc = true; });
			});

			auto ret = q.fence(buf);
			REQUIRE(ret.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
			CHECK_FALSE(*ret.get()); // extra check that the task was not actually executed
		}

		SECTION("for host objects") {
			experimental::host_object<bool> obj(false);
			q.submit([&](handler& cgh) {
				experimental::side_effect eff(obj, cgh);
				cgh.host_task(on_master_node, [=] { *eff = true; });
			});

			auto ret = q.fence(obj);
			REQUIRE(ret.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
			CHECK_FALSE(ret.get()); // extra check that the task was not actually executed
		}

		CHECK(test_utils::log_contains_substring(log_level::warn, "Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set"));
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "dry run processes horizons", "[dryrun]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // dry run unconditionally warns when enabled

		env::scoped_test_environment ste(std::unordered_map<std::string, std::string>{{dryrun_envvar_name, "1"}});

		distr_queue q;

		auto& rt = runtime::get_instance();
		auto& tm = rt.get_task_manager();
		tm.set_horizon_step(1); // horizon step 1 to make testing easy and reproducable with config changes

		REQUIRE(rt.is_dry_run());

		auto latest_hor = task_manager_testspy::get_latest_horizon_reached(tm);
		CHECK_FALSE(latest_hor.has_value());

		q.submit([&](handler& cgh) { cgh.host_task(on_master_node, [=] {}); });

		// we can't slow_full_sync in this test, so we just try until the horizons have been processed
		// 100*10ms is one second in total; if the horizon hasn't happened at that point, it's not happening
		constexpr int max_num_tries = 100;
		for(int i = 0; i < max_num_tries; ++i) {
			latest_hor = task_manager_testspy::get_latest_horizon_reached(tm);
			if(latest_hor.has_value()) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		CHECK(latest_hor.has_value());
	}

	TEST_CASE_METHOD(test_utils::mpi_fixture, "Config reads environment variables correctly", "[env-vars][config]") {
		test_utils::allow_max_log_level(detail::log_level::warn); // setting CELERITY_DRY_RUN_NODES unconditionally warns

		const std::unordered_map<std::string, std::string> env_map{
		    {"CELERITY_LOG_LEVEL", "debug"},
		    {"CELERITY_PROFILE_KERNEL", "1"},
		    {"CELERITY_DRY_RUN_NODES", "4"},
		    {"CELERITY_PRINT_GRAPHS", "true"},
		};
		const auto test_env = env::scoped_test_environment(env_map);
		auto cfg = config(nullptr, nullptr);

		CHECK(cfg.get_log_level() == spdlog::level::debug);
		const auto has_prof = cfg.get_enable_device_profiling();
		REQUIRE(has_prof.has_value());
		CHECK((*has_prof) == true);
		CHECK(cfg.get_dry_run_nodes() == 4);
		CHECK(cfg.should_print_graphs() == true);
	}

	TEST_CASE_METHOD(test_utils::mpi_fixture, "config reports incorrect environment varibles", "[env-vars][config]") {
		test_utils::allow_max_log_level(detail::log_level::err);

		const std::string error_string{"Failed to parse/validate environment variables."};
		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_LOG_LEVEL", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}

		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_GRAPH_PRINT_MAX_VERTS", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}

		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_DEVICES", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}

		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_DRY_RUN_NODES", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}

		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_PROFILE_KERNEL", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}
		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_FORCE_WG", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}

		{
			std::unordered_map<std::string, std::string> invalid_test_env_var{{"CELERITY_PROFILE_OCL", "a"}};
			const auto test_env = env::scoped_test_environment(invalid_test_env_var);
			CHECK_THROWS_WITH((celerity::detail::config(nullptr, nullptr)), error_string);
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "fences extract data from host objects", "[runtime][fence]") {
		experimental::host_object<int> ho{1};
		distr_queue q;

		q.submit([&](handler& cgh) {
			experimental::side_effect e(ho, cgh);
			cgh.host_task(on_master_node, [=] { *e = 2; });
		});
		auto v2 = q.fence(ho);

		q.submit([&](handler& cgh) {
			experimental::side_effect e(ho, cgh);
			cgh.host_task(on_master_node, [=] { *e = 3; });
		});
		auto v3 = q.fence(ho);

		CHECK(v2.get() == 2);
		CHECK(v3.get() == 3);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "fences extract data from buffers", "[runtime][fence]") {
		buffer<int, 2> buf(range<2>(4, 4));
		distr_queue q;

		q.submit([&](handler& cgh) {
			accessor acc(buf, cgh, one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(init)>(buf.get_range(), [=](celerity::item<2> item) { acc[item] = static_cast<int>(item.get_linear_id()); });
		});

		const auto check_snapshot = [&](const subrange<2>& sr, const std::vector<int>& expected_data) {
			const auto snapshot = q.fence(buf, sr).get();
			CHECK(snapshot.get_subrange() == sr);
			CHECK(memcmp(snapshot.get_data(), expected_data.data(), expected_data.size() * sizeof(int)) == 0);
		};

		// Each of these should require its own device-host transfers. Do not use generators / sections, because we want them to happen sequentially.
		check_snapshot(subrange<2>({0, 3}, {1, 1}), std::vector{3});
		check_snapshot(subrange<2>({2, 2}, {2, 1}), std::vector{10, 14});
		check_snapshot(subrange<2>({}, buf.get_range()), [&] {
			std::vector<int> all_data(buf.get_range().size());
			std::iota(all_data.begin(), all_data.end(), 0);
			return all_data;
		}());
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "fences extract data from 0-dimensional buffers", "[runtime][fence]") {
		buffer<int, 0> buf;
		distr_queue q;

		q.submit([&](handler& cgh) {
			accessor acc(buf, cgh, write_only, no_init);
			cgh.parallel_for<class UKN(init)>(buf.get_range(), [=](celerity::item<0> item) { *acc = 42; });
		});

		const auto snapshot = q.fence(buf).get();
		CHECK(*snapshot == 42);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "0-dimensional kernels work as expected", "[buffer]") {
		constexpr float value_a = 13.37f;
		constexpr float value_b = 42.0f;

		buffer<float, 0> buf_a(value_a);
		buffer<float, 0> buf_b(value_a);
		buffer<float, 0> buf_c(value_a);

		distr_queue q;

		q.submit([&](handler& cgh) {
			accessor acc_a(buf_a, cgh, write_only, no_init);
			cgh.parallel_for<class UKN(device)>(range<0>(), [=](item<0> /* it */) { *acc_a = value_b; });
		});
		q.submit([&](handler& cgh) {
			accessor acc_b(buf_b, cgh, write_only, no_init);
			cgh.parallel_for<class UKN(device)>(nd_range<0>(), [=](nd_item<0> /* it */) { *acc_b = value_b; });
		});
		q.submit([&](handler& cgh) {
			accessor acc_c(buf_c, cgh, write_only_host_task, no_init);
			cgh.host_task(range<0>(), [=](partition<0> /* part */) { *acc_c = value_b; });
		});

		q.submit([&](handler& cgh) {
			accessor acc_a(buf_a, cgh, read_only_host_task);
			accessor acc_b(buf_b, cgh, read_only_host_task);
			accessor acc_c(buf_c, cgh, read_only_host_task);
			cgh.host_task(on_master_node, [=] {
				CHECK(*acc_a == value_b);
				CHECK(*acc_b == value_b);
				CHECK(*acc_c == value_b);
			});
		});
	}

	TEST_CASE_METHOD(
	    test_utils::runtime_fixture, "runtime warns on uninitialized reads iff access pattern diagnostics are enabled", "[runtime][diagnostics]") //
	{
		test_utils::allow_max_log_level(log_level::warn);

		distr_queue q;
		buffer<int, 1> buf(1);

		std::string expected_warning_message;

		SECTION("in device kernels") {
			q.submit([&](handler& cgh) {
				accessor acc(buf, cgh, celerity::access::all(), celerity::read_only);
				cgh.parallel_for(range(1), [=](item<1>) { (void)acc; });
			});
			expected_warning_message = "Device kernel T1 declares a reading access on uninitialized B0 {[0,0,0] - [1,1,1]}. Make sure to construct the "
			                           "accessor with no_init if possible.";
		}

		SECTION("in host tasks") {
			q.submit([&](handler& cgh) {
				accessor acc(buf, cgh, celerity::access::all(), celerity::read_only_host_task);
				cgh.host_task(on_master_node, [=] { (void)acc; });
			});
			expected_warning_message = "Master-node host task T1 declares a reading access on uninitialized B0 {[0,0,0] - [1,1,1]}. Make sure to construct the "
			                           "accessor with no_init if possible.";
		}

		q.slow_full_sync();

		CHECK(test_utils::log_contains_exact(log_level::warn, expected_warning_message) == CELERITY_ACCESS_PATTERN_DIAGNOSTICS);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime logs errors on overlapping writes between instructions iff access pattern diagnostics are enabled",
	    "[runtime]") //
	{
		test_utils::allow_max_log_level(log_level::err);

		distr_queue q;
		const auto num_devices = runtime_testspy::get_num_local_devices(runtime::get_instance());
		if(num_devices < 2) { SKIP("Test needs at least 2 devices"); }

		buffer<int, 1> buf(1);
		q.submit([&](handler& cgh) {
			accessor acc(buf, cgh, celerity::access::all(), write_only, no_init);
			cgh.parallel_for(range(num_devices), [=](item<1>) { (void)acc; });
		});
		q.slow_full_sync();

		const auto expected_error_message =
		    "Device kernel T1 has overlapping writes on N0 in B0 {[0,0,0] - [1,1,1]}. Choose a non-overlapping range mapper "
		    "for this write access or constrain the split via experimental::constrain_split to make the access non-overlapping.";
		CHECK(test_utils::log_contains_exact(log_level::err, expected_error_message) == CELERITY_ACCESS_PATTERN_DIAGNOSTICS);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime types throw when used from the wrong thread", "[runtime]") {
		distr_queue q;
		buffer<int, 1> buf(range<1>(1));
		experimental::host_object<int> ho(42);

		constexpr auto what = "Celerity runtime, distr_queue, handler, buffer and host_object types must only be constructed, used, and destroyed from the "
		                      "application thread. Make sure that you did not accidentally capture one of these types in a host_task.";
		std::thread([&] {
			CHECK_THROWS_WITH((distr_queue{}), what);
			CHECK_THROWS_WITH((buffer<int, 1>{range<1>{1}}), what);
			CHECK_THROWS_WITH((experimental::host_object<int>{}), what);

			CHECK_THROWS_WITH(q.submit([&](handler& cgh) { (void)cgh; }), what);
			CHECK_THROWS_WITH(q.slow_full_sync(), what);
			CHECK_THROWS_WITH(q.fence(buf), what);
			CHECK_THROWS_WITH(q.fence(ho), what);

			// We can't easily test whether `~distr_queue()` et al. throw, because that would require marking the entire stack of destructors noexcept(false)
			// including the ~shared_ptr we use internally for reference semantics. Instead we verify that the runtime operations their trackers call throw.
			CHECK_THROWS_WITH(detail::runtime::get_instance().destroy_queue(), what);
			CHECK_THROWS_WITH(detail::runtime::get_instance().destroy_buffer(get_buffer_id(buf)), what);
			CHECK_THROWS_WITH(detail::runtime::get_instance().destroy_host_object(get_host_object_id(ho)), what);
		}).join();
	}

} // namespace detail
} // namespace celerity
