#include "cgf.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <iterator>
#include <set>

#include "test_utils.h"


namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	TEST_CASE("task_manager calls into delegate on task creation", "[task_manager]") {
		struct counter_delegate final : public task_manager::delegate {
			size_t counter = 0;
			void task_created(const task* /* tsk */) override { counter++; }
		};

		counter_delegate delegate;
		task_graph tdag;
		task_manager tm{1, tdag, nullptr, &delegate};
		tm.generate_epoch_task(epoch_action::init);
		CHECK(delegate.counter == 1);
		const range<2> gs = {1, 1};
		const id<2> go = {};
		tm.generate_command_group_task(invoke_command_group_function([=](handler& cgh) { cgh.parallel_for<class kernel>(gs, go, [](auto) {}); }));
		CHECK(delegate.counter == 2);
		tm.generate_command_group_task(invoke_command_group_function([](handler& cgh) { cgh.host_task(on_master_node, [] {}); }));
		CHECK(delegate.counter == 3);
	}

	TEST_CASE("task_manager correctly records compute task information", "[task_manager][task][device_compute_task]") {
		test_utils::task_test_context tt;
		auto buf_a = tt.mbf.create_buffer(range<2>(64, 152), true /* host_initialized */);
		auto buf_b = tt.mbf.create_buffer(range<3>(7, 21, 99));
		const auto tid = test_utils::add_compute_task(
		    tt.tm,
		    [&](handler& cgh) {
			    buf_a.get_access<access_mode::read>(cgh, one_to_one{});
			    buf_b.get_access<access_mode::discard_read_write>(cgh, fixed{subrange<3>{{}, {5, 18, 74}}});
		    },
		    range<2>{32, 128}, id<2>{32, 24});

		const auto tsk = test_utils::get_task(tt.tdag, tid);
		CHECK(tsk->get_type() == task_type::device_compute);
		CHECK(tsk->get_dimensions() == 2);
		CHECK(tsk->get_global_size() == range<3>{32, 128, 1});
		CHECK(tsk->get_global_offset() == id<3>{32, 24, 0});

		auto& bam = tsk->get_buffer_access_map();
		const auto bufs = bam.get_accessed_buffers();
		CHECK(bufs.size() == 2);
		CHECK(std::find(bufs.cbegin(), bufs.cend(), buf_a.get_id()) != bufs.cend());
		CHECK(std::find(bufs.cbegin(), bufs.cend(), buf_b.get_id()) != bufs.cend());
		CHECK(bam.get_nth_access(0) == std::pair{buf_a.get_id(), access_mode::read});
		CHECK(bam.get_nth_access(1) == std::pair{buf_b.get_id(), access_mode::discard_read_write});
		const auto reqs_a = bam.compute_consumed_region(buf_a.get_id(), subrange{tsk->get_global_offset(), tsk->get_global_size()});
		CHECK(reqs_a == box(subrange<3>({32, 24, 0}, {32, 128, 1})));
		const auto reqs_b = bam.compute_produced_region(buf_b.get_id(), subrange{tsk->get_global_offset(), tsk->get_global_size()});
		CHECK(reqs_b == box(subrange<3>({}, {5, 18, 74})));
	}

	TEST_CASE("buffer_access_map merges multiple accesses with the same mode", "[task][device_compute_task]") {
		std::vector<buffer_access> accs;
		accs.push_back(buffer_access{0, access_mode::read, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{3, 0}, {10, 20}}, range<2>{30, 30})});
		accs.push_back(buffer_access{0, access_mode::read, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{10, 0}, {7, 20}}, range<2>{30, 30})});
		const buffer_access_map bam{std::move(accs), task_geometry{2, {100, 100, 1}, {}, {}}};
		const auto req = bam.compute_consumed_region(0, subrange<3>({0, 0, 0}, {100, 100, 1}));
		CHECK(req == box(subrange<3>({3, 0, 0}, {14, 20, 1})));
	}

	TEST_CASE("tasks gracefully handle get_requirements() calls for buffers they don't access", "[task]") {
		const buffer_access_map bam;
		const auto req = bam.compute_consumed_region(0, subrange<3>({0, 0, 0}, {100, 1, 1}));
		CHECK(req == box<3>());
	}

	TEST_CASE("task_manager does not create multiple dependencies between the same tasks", "[task_manager][task-graph]") {
		auto tt = test_utils::task_test_context{};
		auto buf_a = tt.mbf.create_buffer(range<1>(128));
		auto buf_b = tt.mbf.create_buffer(range<1>(128));

		SECTION("true dependencies") {
			const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<access_mode::read>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<access_mode::read>(cgh, fixed<1>({0, 128}));
			});
			CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a));

			const auto its = test_utils::get_task(tt.tdag, tid_a)->get_dependents();
			REQUIRE(std::distance(its.begin(), its.end()) == 1);
		}

		SECTION("anti-dependencies") {
			const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a, dependency_kind::anti_dep));

			const auto its = test_utils::get_task(tt.tdag, tid_a)->get_dependents();
			REQUIRE(std::distance(its.begin(), its.end()) == 1);
		}

		// Here we also check that true dependencies always take precedence
		SECTION("true and anti-dependencies combined") {
			SECTION("if true is declared first") {
				const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
				});
				const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<access_mode::read>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
				});
				CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a));
				CHECK_FALSE(test_utils::has_dependency(tt.tdag, tid_b, tid_a, dependency_kind::anti_dep));

				const auto its = test_utils::get_task(tt.tdag, tid_a)->get_dependents();
				REQUIRE(std::distance(its.begin(), its.end()) == 1);
			}

			SECTION("if anti is declared first") {
				const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
				});
				const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<access_mode::read>(cgh, fixed<1>({0, 128}));
				});
				CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a));
				CHECK_FALSE(test_utils::has_dependency(tt.tdag, tid_b, tid_a, dependency_kind::anti_dep));

				const auto its = test_utils::get_task(tt.tdag, tid_a)->get_dependents();
				REQUIRE(std::distance(its.begin(), its.end()) == 1);
			}
		}
	}

	TEST_CASE("task_manager respects range mapper results for finding dependencies", "[task_manager][task-graph]") {
		auto tt = test_utils::task_test_context{};
		auto buf = tt.mbf.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

		const auto tid_a =
		    test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{0, 64}}); });
		const auto tid_b =
		    test_utils::add_compute_task<class UKN(task_b)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, fixed<1>{{0, 128}}); });
		CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a));
		CHECK(test_utils::has_dependency(tt.tdag, tid_b, tt.initial_epoch_task)); // for read of the host-initialized part

		const auto tid_c =
		    test_utils::add_compute_task<class UKN(task_c)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, fixed<1>{{64, 128}}); });
		CHECK_FALSE(test_utils::has_dependency(tt.tdag, tid_c, tid_a));
		CHECK(test_utils::has_dependency(tt.tdag, tid_c, tt.initial_epoch_task)); // for read of the host-initialized part
	}

	TEST_CASE("task_manager correctly generates anti-dependencies", "[task_manager][task-graph]") {
		auto tt = test_utils::task_test_context{};
		auto buf = tt.mbf.create_buffer(range<1>(128));

		// Write to the full buffer
		const auto tid_a =
		    test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{0, 128}}); });
		// Read the first half of the buffer
		const auto tid_b =
		    test_utils::add_compute_task<class UKN(task_b)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, fixed<1>{{0, 64}}); });
		CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a));
		// Overwrite the second half - no anti-dependency onto task_b should exist (but onto task_a)
		const auto tid_c =
		    test_utils::add_compute_task<class UKN(task_c)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{64, 64}}); });
		REQUIRE(test_utils::has_dependency(tt.tdag, tid_c, tid_a, dependency_kind::anti_dep));
		REQUIRE_FALSE(test_utils::has_dependency(tt.tdag, tid_c, tid_b, dependency_kind::anti_dep));
		// Overwrite the first half - now only an anti-dependency onto task_b should exist
		const auto tid_d =
		    test_utils::add_compute_task<class UKN(task_d)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{0, 64}}); });
		REQUIRE_FALSE(test_utils::has_dependency(tt.tdag, tid_d, tid_a, dependency_kind::anti_dep));
		REQUIRE(test_utils::has_dependency(tt.tdag, tid_d, tid_b, dependency_kind::anti_dep));
	}

	TEST_CASE("task_manager correctly handles host-initialized buffers", "[task_manager][task-graph]") {
		// we explicitly test reading from non_host_init_buf
		task_manager::policy_set tm_policy;
		tm_policy.uninitialized_read_error = error_policy::ignore;

		auto tt = test_utils::task_test_context(tm_policy);
		auto host_init_buf = tt.mbf.create_buffer(range<1>(128), true /* mark_as_host_initialized */);
		auto non_host_init_buf = tt.mbf.create_buffer(range<1>(128), false /* mark_as_host_initialized */);
		auto artificial_dependency_buf = tt.mbf.create_buffer(range<1>(1), false /* mark_as_host_initialized */);

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) {
			host_init_buf.get_access<access_mode::read>(cgh, fixed<1>{{0, 128}});
			artificial_dependency_buf.get_access<access_mode::discard_write>(cgh, all{});
		});
		CHECK(test_utils::has_dependency(tt.tdag, tid_a, tt.initial_epoch_task));

		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tt.tm, [&](handler& cgh) {
			non_host_init_buf.get_access<access_mode::read>(cgh, fixed<1>{{0, 128}});
			// introduce an arbitrary true-dependency to avoid the fallback epoch dependency that is generated for tasks without other true-dependencies
			artificial_dependency_buf.get_access<access_mode::read>(cgh, all{});
		});
		CHECK_FALSE(test_utils::has_dependency(tt.tdag, tid_b, tt.initial_epoch_task));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(
		    tt.tm, [&](handler& cgh) { host_init_buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{0, 128}}); });
		CHECK(test_utils::has_dependency(tt.tdag, tid_c, tid_a, dependency_kind::anti_dep));
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(
		    tt.tm, [&](handler& cgh) { non_host_init_buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{0, 128}}); });
		// Since task b is essentially reading uninitialized garbage, it doesn't make a difference if we write into it concurrently
		CHECK_FALSE(test_utils::has_dependency(tt.tdag, tid_d, tid_b, dependency_kind::anti_dep));
	}

	template <int Dims, typename Handler, typename Functor>
	void dispatch_get_access(test_utils::mock_buffer<Dims>& mb, Handler& handler, access_mode mode, Functor rmfn) {
		switch(mode) {
		case access_mode::read: mb.template get_access<access_mode::read>(handler, rmfn); break;
		case access_mode::write: mb.template get_access<access_mode::write>(handler, rmfn); break;
		case access_mode::read_write: mb.template get_access<access_mode::read_write>(handler, rmfn); break;
		case access_mode::discard_write: mb.template get_access<access_mode::discard_write>(handler, rmfn); break;
		case access_mode::discard_read_write: mb.template get_access<access_mode::discard_read_write>(handler, rmfn); break;
		default: utils::unreachable(); // LCOV_EXCL_LINE
		}
	}

	TEST_CASE("task_manager correctly handles dependencies for R/W modes", "[task_manager][task-graph]") {
		// A read-write access can also be implicitly created using a separate write and read, which is why we operate on "mode sets" here.
		const std::vector<std::vector<access_mode>> rw_mode_sets = {
		    {access_mode::discard_read_write}, {access_mode::read_write}, {access_mode::discard_write, access_mode::read}};

		for(const auto& mode_set : rw_mode_sets) {
			auto tt = test_utils::task_test_context{};
			auto buf = tt.mbf.create_buffer(range<1>(128), true);

			const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) {
				for(const auto& m : mode_set) {
					dispatch_get_access(buf, cgh, m, fixed<1>{{0, 128}});
				}
			});
			const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(
			    tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<1>{{0, 128}}); });
			REQUIRE(test_utils::has_dependency(tt.tdag, tid_b, tid_a, dependency_kind::anti_dep));
		}
	}

	TEST_CASE("task_manager handles all producer/consumer combinations correctly", "[task_manager][task-graph]") {
		constexpr access_mode consumer_modes[] = {access_mode::read, access_mode::read_write, access_mode::write};
		constexpr access_mode producer_modes[] = {access_mode::discard_read_write, access_mode::discard_write, access_mode::read_write, access_mode::write};

		for(const auto& consumer_mode : consumer_modes) {
			for(const auto& producer_mode : producer_modes) {
				CAPTURE(consumer_mode);
				CAPTURE(producer_mode);

				auto tt = test_utils::task_test_context{};
				auto buf = tt.mbf.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

				const task_id tid_a =
				    test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) { dispatch_get_access(buf, cgh, producer_mode, all()); });

				const task_id tid_b =
				    test_utils::add_compute_task<class UKN(task_b)>(tt.tm, [&](handler& cgh) { dispatch_get_access(buf, cgh, consumer_mode, all()); });
				CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_a));

				const task_id tid_c =
				    test_utils::add_compute_task<class UKN(task_c)>(tt.tm, [&](handler& cgh) { dispatch_get_access(buf, cgh, producer_mode, all()); });
				const bool pure_consumer = consumer_mode == access_mode::read;
				const bool pure_producer = producer_mode == access_mode::discard_read_write || producer_mode == access_mode::discard_write;
				CHECK(
				    test_utils::has_dependency(tt.tdag, tid_c, tid_b, pure_consumer || pure_producer ? dependency_kind::anti_dep : dependency_kind::true_dep));
			}
		}
	}

	TEST_CASE("task_manager generates pseudo-dependencies for collective host tasks", "[task_manager][task-graph]") {
		auto tt = test_utils::task_test_context{};
		experimental::collective_group group;
		auto tid_master = test_utils::add_host_task(tt.tm, on_master_node, [](handler&) {});
		auto tid_collective_implicit_1 = test_utils::add_host_task(tt.tm, experimental::collective, [](handler&) {});
		auto tid_collective_implicit_2 = test_utils::add_host_task(tt.tm, experimental::collective, [](handler&) {});
		auto tid_collective_explicit_1 = test_utils::add_host_task(tt.tm, experimental::collective(group), [](handler&) {});
		auto tid_collective_explicit_2 = test_utils::add_host_task(tt.tm, experimental::collective(group), [](handler&) {});

		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_master, tid_collective_implicit_1));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_master, tid_collective_implicit_2));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_master, tid_collective_explicit_1));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_master, tid_collective_explicit_2));

		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_1, tid_master));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_1, tid_collective_implicit_2));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_1, tid_collective_explicit_1));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_1, tid_collective_explicit_2));

		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_2, tid_master));
		CHECK(test_utils::has_dependency(tt.tdag, tid_collective_implicit_2, tid_collective_implicit_1, dependency_kind::true_dep));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_2, tid_collective_explicit_1));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_implicit_2, tid_collective_explicit_2));

		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_1, tid_master));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_1, tid_collective_implicit_1));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_1, tid_collective_implicit_2));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_1, tid_collective_explicit_2));

		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_2, tid_master));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_2, tid_collective_implicit_1));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_collective_explicit_2, tid_collective_implicit_2));
		CHECK(test_utils::has_dependency(tt.tdag, tid_collective_explicit_2, tid_collective_explicit_1, dependency_kind::true_dep));
	}

	void check_path_length_and_front(const task_manager& tm, const task_graph& tdag, int path_length, const std::unordered_set<task_id>& exec_front) {
		{
			INFO("path length");
			CHECK(task_manager_testspy::get_max_pseudo_critical_path_length(tm) == path_length);
		}
		{
			INFO("execution front");
			std::unordered_set<task*> task_exec_front;
			std::transform(exec_front.cbegin(), exec_front.cend(), std::inserter(task_exec_front, task_exec_front.begin()),
			    [&](const task_id tid) { return const_cast<task*>(test_utils::get_task(tdag, tid)); });
			CHECK(task_manager_testspy::get_execution_front(tm) == task_exec_front);
		}
	}

	TEST_CASE("task_manager keeps track of max pseudo critical path length and task front", "[task_manager][task-graph][task-front]") {
		auto tt = test_utils::task_test_context{};
		auto buf_a = tt.mbf.create_buffer(range<1>(128));

		const auto tid_a =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128})); });
		check_path_length_and_front(tt.tm, tt.tdag, 1, {tid_a}); // 1: we always depend on the initial epoch task

		const auto tid_b =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });
		check_path_length_and_front(tt.tm, tt.tdag, 2, {tid_b});

		const auto tid_c =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read>(cgh, fixed<1>({0, 128})); });
		check_path_length_and_front(tt.tm, tt.tdag, 3, {tid_c});

		const auto tid_d = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {});
		check_path_length_and_front(tt.tm, tt.tdag, 3, {tid_c, tid_d});
	}

	TEST_CASE("task horizons are being generated with correct dependencies", "[task_manager][task-graph][task-horizon]") {
		auto tt = test_utils::task_test_context{};

		tt.tm.set_horizon_step(2);
		auto buf_a = tt.mbf.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

		test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128})); });

		auto current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
		CHECK(current_horizon == nullptr);

		const auto tid_c =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read>(cgh, fixed<1>({0, 128})); });

		current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_id() == tid_c + 1);
		CHECK(test_utils::get_num_live_horizons(tt.tdag) == 1);

		auto horizon_dependencies = current_horizon->get_dependencies();

		CHECK(std::distance(horizon_dependencies.begin(), horizon_dependencies.end()) == 1);
		CHECK(horizon_dependencies.begin()->node->get_id() == tid_c);

		std::set<task_id> expected_dependency_ids;

		// current horizon is always part of the active task front
		expected_dependency_ids.insert(current_horizon->get_id());
		expected_dependency_ids.insert(test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {}));
		expected_dependency_ids.insert(test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {}));
		expected_dependency_ids.insert(test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {}));
		CHECK(test_utils::get_num_live_horizons(tt.tdag) == 1);

		test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });
		const auto tid_d =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });
		expected_dependency_ids.insert(tid_d);

		current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_id() == tid_d + 1);
		CHECK(test_utils::get_num_live_horizons(tt.tdag) == 2);

		horizon_dependencies = current_horizon->get_dependencies();
		CHECK(std::distance(horizon_dependencies.begin(), horizon_dependencies.end()) == 5);

		std::set<task_id> actual_dependecy_ids;
		for(auto dep : horizon_dependencies) {
			actual_dependecy_ids.insert(dep.node->get_id());
		}
		CHECK(expected_dependency_ids == actual_dependecy_ids);
	}

	TEST_CASE("task horizons are being generated for the parallelism limit", "[task_manager][task-graph][task-horizon]") {
		auto tt = test_utils::task_test_context{};

		// we set a high step but low max parallelism to make sure that all horizons in this test are generated due to the parallelism limit,
		// regardless of what the defaults for these values are
		tt.tm.set_horizon_step(256);
		const auto max_para = 3;
		tt.tm.set_horizon_max_parallelism(max_para);

		const size_t buff_size = 128;
		const size_t num_tasks = 9;
		const size_t buff_elem_per_task = buff_size / num_tasks;
		auto buf_a = tt.mbf.create_buffer(range<1>(buff_size), true /* mark_as_host_initialized */);

		auto current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
		CHECK(current_horizon == nullptr);

		for(size_t i = 0; i < num_tasks; ++i) {
			const auto offset = buff_elem_per_task * i;
			test_utils::add_host_task(
			    tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({offset, buff_elem_per_task})); });
		}

		// divided by "max_para - 1" since there is also always the previous horizon in the set
		const auto expected_num_horizons = num_tasks / (max_para - 1);
		CHECK(test_utils::get_num_live_horizons(tt.tdag) == expected_num_horizons);

		// the most recent horizon should have 3 predecessors: 1 other horizon and 2 host tasks we generated
		current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_dependencies().size() == 3);
	}

	static inline region<3> make_region(size_t min, size_t max) { return box<3>({min, 0, 0}, {max, 1, 1}); }

	TEST_CASE("task horizons update previous writer data structure", "[task_manager][task-graph][task-horizon]") {
		auto tt = test_utils::task_test_context{};

		tt.tm.set_horizon_step(2);
		auto buf_a = tt.mbf.create_buffer(range<1>(128));
		auto buf_b = tt.mbf.create_buffer(range<1>(128));

		const task_id tid_1 = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 64}));
			buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
		});
		const task_id tid_2 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({64, 64})); });
		[[maybe_unused]] const task_id tid_3 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({32, 64})); });
		const task_id tid_4 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({32, 64})); });

		const auto horizon = task_manager_testspy::get_current_horizon(tt.tm);
		CHECK(test_utils::get_num_live_horizons(tt.tdag) == 1);
		CHECK(horizon != nullptr);

		[[maybe_unused]] const task_id tid_6 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_b.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });
		[[maybe_unused]] const task_id tid_7 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_b.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });

		{
			INFO("check that previous tasks are still last writers before the first horizon is applied");
			const auto& region_map_a = task_manager_testspy::get_last_writer(tt.tm, buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 32)).front().second == test_utils::get_task(tt.tdag, tid_1));
			CHECK(region_map_a.get_region_values(make_region(96, 128)).front().second == test_utils::get_task(tt.tdag, tid_2));
			CHECK(region_map_a.get_region_values(make_region(32, 96)).front().second == test_utils::get_task(tt.tdag, tid_4));
		}

		[[maybe_unused]] const task_id tid_8 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_b.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });

		CHECK(test_utils::get_num_live_horizons(tt.tdag) == 2);

		{
			INFO("check that only the previous horizon is the last writer of buff_a");
			const auto& region_map_a = task_manager_testspy::get_last_writer(tt.tm, buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 128)).front().second == horizon);
		}

		const task_id tid_9 =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({64, 64})); });

		{
			INFO("check that the previous horizon and task 11 are last writers of buff_a");
			const auto& region_map_a = task_manager_testspy::get_last_writer(tt.tm, buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 64)).front().second == horizon);
			CHECK(region_map_a.get_region_values(make_region(64, 128)).front().second == test_utils::get_task(tt.tdag, tid_9));
		}
	}

	TEST_CASE("previous task horizon is used as last writer for host-initialized buffers", "[task_manager][task-graph][task-horizon]") {
		auto tt = test_utils::task_test_context{};
		tt.tm.set_horizon_step(2);

		task_id initial_last_writer_id = -1;
		{
			auto buf = tt.mbf.create_buffer(range<1>(1), true);
			const auto tid = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::read_write>(cgh, all{}); });
			const auto& deps = test_utils::get_task(tt.tdag, tid)->get_dependencies();
			CHECK(std::distance(deps.begin(), deps.end()) == 1);
			initial_last_writer_id = deps.begin()->node->get_id();
		}
		CHECK(test_utils::has_task(tt.tdag, initial_last_writer_id));

		// Create a bunch of tasks to trigger horizon cleanup
		{
			auto buf = tt.mbf.create_buffer(range<1>(1));
			const task* last_executed_horizon = nullptr;
			// We need 7 tasks to generate a pseudo-critical path length of 6 (3x2 horizon step size),
			// and another one that triggers the actual deferred deletion.
			for(int i = 0; i < 8; ++i) {
				test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
				const auto current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
				if(last_executed_horizon != nullptr && current_horizon->get_id() > last_executed_horizon->get_id()) {
					tt.tdag.erase_before_epoch(last_executed_horizon->get_id());
				}
				if(current_horizon != nullptr) { last_executed_horizon = current_horizon; }
			}
		}

		INFO("initial last writer with id " << initial_last_writer_id << " has been deleted");
		CHECK_FALSE(test_utils::has_task(tt.tdag, initial_last_writer_id));

		auto buf = tt.mbf.create_buffer(range<1>(1), true);
		const auto tid = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::read_write>(cgh, all{}); });
		const auto& deps = test_utils::get_task(tt.tdag, tid)->get_dependencies();
		CHECK(std::distance(deps.begin(), deps.end()) == 1);
		const auto* new_last_writer = deps.begin()->node;
		CHECK(new_last_writer->get_type() == task_type::horizon);

		const auto current_horizon = task_manager_testspy::get_current_horizon(tt.tm);
		REQUIRE(current_horizon);
		INFO("previous horizon is being used");
		CHECK(new_last_writer->get_id() < current_horizon->get_id());
	}

	TEST_CASE("collective host tasks do not order-depend on their predecessor if it is shadowed by a horizon", "[task_manager][task-graph][task-horizon]") {
		// Regression test: the order-dependencies between host tasks in the same collective group are built by tracking the last task in each collective group.
		// Once a horizon is inserted, new collective host tasks must order-depend on that horizon instead.

		auto tt = test_utils::task_test_context{};
		tt.tm.set_horizon_step(2);
		auto buf = tt.mbf.create_buffer(range<1>(1));

		[[maybe_unused]] const auto first_collective = test_utils::add_host_task(tt.tm, experimental::collective, [&](handler& cgh) {});

		// generate exactly two horizons
		for(int i = 0; i < 4; ++i) {
			test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
		}

		// This must depend on the first horizon, not first_collective
		const auto second_collective =
		    test_utils::add_host_task(tt.tm, experimental::collective, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, all{}); });

		const auto second_collective_deps = test_utils::get_task(tt.tdag, second_collective)->get_dependencies();
		const auto master_node_dep = std::find_if(second_collective_deps.begin(), second_collective_deps.end(),
		    [](const task::dependency d) { return d.node->get_type() == task_type::master_node; });
		const auto horizon_dep = std::find_if(second_collective_deps.begin(), second_collective_deps.end(), //
		    [](const task::dependency d) { return d.node->get_type() == task_type::horizon; });

		CHECK(std::distance(second_collective_deps.begin(), second_collective_deps.end()) == 2);
		REQUIRE(master_node_dep != second_collective_deps.end());
		CHECK(master_node_dep->kind == dependency_kind::true_dep);
		REQUIRE(horizon_dep != second_collective_deps.end());
		CHECK(horizon_dep->kind == dependency_kind::true_dep);
	}

	TEST_CASE("buffer accesses with empty ranges do not generate data-flow dependencies", "[task_manager][task-graph]") {
		auto tt = test_utils::task_test_context{};
		auto buf = tt.mbf.create_buffer(range<2>(32, 32), true /* mark_as_host_initialized */);

		const auto write_sr = GENERATE(values({subrange<2>{{16, 16}, {0, 0}}, subrange<2>{{16, 16}, {8, 8}}}));
		const auto read_sr = GENERATE(values({subrange<2>{{1, 1}, {0, 0}}, subrange<2>{{8, 8}, {16, 16}}}));

		const auto read_empty = read_sr.range.size() == 0;
		const auto write_empty = write_sr.range.size() == 0;
		CAPTURE(read_empty);
		CAPTURE(write_empty);

		const auto write_tid =
		    test_utils::add_compute_task<class UKN(write)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<2>{write_sr}); });
		const auto read_tid =
		    test_utils::add_compute_task<class UKN(read)>(tt.tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, fixed<2>{read_sr}); });

		CHECK(test_utils::has_any_dependency(tt.tdag, read_tid, write_tid) == (!write_empty && !read_empty));
	}

	TEST_CASE("side effects generate appropriate task-dependencies", "[task_manager][task-graph][side-effect]") {
		using order = experimental::side_effect_order;
		static constexpr auto side_effect_orders = {order::sequential};

		// TODO placeholder: complete with dependency types for other side effect orders
		const auto expected_dependencies = std::unordered_map<std::pair<order, order>, std::optional<dependency_kind>, utils::pair_hash>{
		    {{order::sequential, order::sequential}, dependency_kind::true_dep}};

		const auto order_a = GENERATE(values(side_effect_orders));
		const auto order_b = GENERATE(values(side_effect_orders));

		CAPTURE(order_a);
		CAPTURE(order_b);

		auto tt = test_utils::task_test_context{};
		auto ho_common = tt.mhof.create_host_object(); // should generate dependencies
		auto ho_a = tt.mhof.create_host_object();      // should NOT generate dependencies
		auto ho_b = tt.mhof.create_host_object();      // -"-
		const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
			ho_common.add_side_effect(cgh, order_a);
			ho_a.add_side_effect(cgh, order_a);
		});
		const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) {
			ho_common.add_side_effect(cgh, order_b);
			ho_b.add_side_effect(cgh, order_b);
		});

		const auto deps_a = test_utils::get_task(tt.tdag, tid_a)->get_dependencies();
		REQUIRE(std::distance(deps_a.begin(), deps_a.end()) == 1);
		CHECK(deps_a.front().node->get_id() == tt.initial_epoch_task);

		const auto deps_b = test_utils::get_task(tt.tdag, tid_b)->get_dependencies();
		const auto expected_b = expected_dependencies.at({order_a, order_b});
		CHECK(std::distance(deps_b.begin(), deps_b.end()) == expected_b.has_value());
		if(expected_b) {
			CHECK(deps_b.front().node == test_utils::get_task(tt.tdag, tid_a));
			CHECK(deps_b.front().kind == *expected_b);
		}
	}

	TEST_CASE("side-effect dependencies are correctly subsumed by horizons", "[task_manager][task-graph][task-horizon]") {
		auto tt = test_utils::task_test_context{};
		tt.tm.set_horizon_step(2);
		auto ho = tt.mhof.create_host_object();

		[[maybe_unused]] const auto first_task =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); });

		// generate exactly two horizons
		auto buf = tt.mbf.create_buffer(range<1>(1));
		for(int i = 0; i < 5; ++i) {
			test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
		}

		// This must depend on the first horizon, not first_task
		const auto second_task =
		    test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); });

		const auto& second_deps = test_utils::get_task(tt.tdag, second_task)->get_dependencies();
		CHECK(std::distance(second_deps.begin(), second_deps.end()) == 1);
		for(const auto& dep : second_deps) {
			const auto type = dep.node->get_type();
			CHECK(type == task_type::horizon);
			CHECK(dep.kind == dependency_kind::true_dep);
		}
	}

	TEST_CASE("epochs create appropriate dependencies to predecessors and successors", "[task_manager][task-graph][epoch]") {
		auto tt = test_utils::task_test_context{};

		auto buf_a = tt.mbf.create_buffer(range<1>(1));
		const auto tid_a =
		    test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) { buf_a.get_access<access_mode::discard_write>(cgh, all{}); });

		auto buf_b = tt.mbf.create_buffer(range<1>(1));
		const auto tid_b =
		    test_utils::add_compute_task<class UKN(task_b)>(tt.tm, [&](handler& cgh) { buf_b.get_access<access_mode::discard_write>(cgh, all{}); });

		const auto tid_epoch = tt.tm.generate_epoch_task(epoch_action::none);

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tt.tm, [&](handler& cgh) { buf_a.get_access<access_mode::read>(cgh, all{}); });
		const auto tid_d =
		    test_utils::add_compute_task<class UKN(task_d)>(tt.tm, [&](handler& cgh) { buf_b.get_access<access_mode::discard_write>(cgh, all{}); });
		const auto tid_e = test_utils::add_compute_task<class UKN(task_e)>(tt.tm, [&](handler& cgh) {});
		const auto tid_f = test_utils::add_compute_task<class UKN(task_f)>(tt.tm, [&](handler& cgh) { buf_b.get_access<access_mode::read>(cgh, all{}); });
		const auto tid_g =
		    test_utils::add_compute_task<class UKN(task_g)>(tt.tm, [&](handler& cgh) { buf_b.get_access<access_mode::discard_write>(cgh, all{}); });

		CHECK(test_utils::has_dependency(tt.tdag, tid_epoch, tid_a));
		CHECK(test_utils::has_dependency(tt.tdag, tid_epoch, tid_b));
		CHECK(test_utils::has_dependency(tt.tdag, tid_c, tid_epoch));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_c, tid_a));
		CHECK(test_utils::has_dependency(tt.tdag, tid_d, tid_epoch)); // needs a true_dep on barrier since it only has anti_deps otherwise
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_d, tid_b));
		CHECK(test_utils::has_dependency(tt.tdag, tid_e, tid_epoch));
		CHECK(test_utils::has_dependency(tt.tdag, tid_f, tid_d));
		CHECK_FALSE(test_utils::has_any_dependency(tt.tdag, tid_f, tid_epoch));
		CHECK(test_utils::has_dependency(tt.tdag, tid_g, tid_f, dependency_kind::anti_dep));
		CHECK(test_utils::has_dependency(tt.tdag, tid_g, tid_epoch)); // needs a true_dep on barrier since it only has anti_deps otherwise
	}

	TEST_CASE("inserting epochs resets the need for horizons", "[task_manager][task-graph][task-horizon][epoch]") {
		auto tt = test_utils::task_test_context{};
		tt.tm.set_horizon_step(2);
		auto buf = tt.mbf.create_buffer(range<1>(1));

		for(int i = 0; i < 3; ++i) {
			test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
			tt.tm.generate_epoch_task(epoch_action::none);
		}

		CHECK(test_utils::get_num_live_horizons(tt.tdag) == 0);
	}

	TEST_CASE("a sequence of epochs without intermediate tasks has defined behavior", "[task_manager][task-graph][epoch]") {
		auto tt = test_utils::task_test_context{};

		auto tid_before = tt.initial_epoch_task;
		for(const auto action : {epoch_action::barrier, epoch_action::shutdown}) {
			const auto tid = tt.tm.generate_epoch_task(action);
			CAPTURE(tid_before, tid);
			const auto deps = test_utils::get_task(tt.tdag, tid)->get_dependencies();
			CHECK(std::distance(deps.begin(), deps.end()) == 1);
			for(const auto& d : deps) {
				CHECK(d.kind == dependency_kind::true_dep);
				CHECK(d.node->get_id() == tid_before);
			}
			tid_before = tid;
		}
	}

	TEST_CASE("fences introduce dependencies on host objects", "[task_manager][task-graph][fence]") {
		auto tt = test_utils::task_test_context{};
		auto ho = tt.mhof.create_host_object();

		const auto tid_a = test_utils::add_host_task(
		    tt.tm, celerity::experimental::collective, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); });
		const auto tid_fence = test_utils::add_fence_task(tt.tm, ho);
		const auto tid_b = test_utils::add_host_task(
		    tt.tm, celerity::experimental::collective, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); });

		CHECK(test_utils::has_dependency(tt.tdag, tid_fence, tid_a));
		CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_fence));
	}

	TEST_CASE("fences introduce data dependencies", "[task_manager][task-graph][fence]") {
		auto tt = test_utils::task_test_context{};
		auto buf = tt.mbf.create_buffer<1>({1});

		const auto tid_a = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
		const auto tid_fence = test_utils::add_fence_task(tt.tm, buf);
		const auto tid_b = test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });

		CHECK(test_utils::has_dependency(tt.tdag, tid_fence, tid_a));
		CHECK(test_utils::has_dependency(tt.tdag, tid_b, tid_fence, dependency_kind::anti_dep));
	}

	TEST_CASE("task_manager throws in tests if it detects an uninitialized read", "[task_manager]") {
		test_utils::task_test_context tt;

		SECTION("on a fully uninitialized buffer") {
			auto buf = tt.mbf.create_buffer<1>({1});

			CHECK_THROWS_WITH((test_utils::add_compute_task(
			                      tt.tm, [&](handler& cgh) { debug::set_task_name(cgh, "uninit_read"), buf.get_access<access_mode::read>(cgh, all{}); })),
			    "Device kernel T1 \"uninit_read\" declares a reading access on uninitialized B0 {[0,0,0] - [1,1,1]}.");
		}

		SECTION("on a partially initialized buffer") {
			auto buf = tt.mbf.create_buffer<2>({64, 64});
			test_utils::add_compute_task<class UKN(uninit_read)>(
			    tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<2>({{0, 0}, {32, 32}})); });

			CHECK_THROWS_WITH((test_utils::add_compute_task(
			                      tt.tm, [&](handler& cgh) { debug::set_task_name(cgh, "uninit_read"), buf.get_access<access_mode::write>(cgh, all{}); })),
			    "Device kernel T2 \"uninit_read\" declares a consuming access on uninitialized B0 {[0,32,0] - [32,64,1], [32,0,0] - [64,64,1]}. Make sure to "
			    "construct the accessor with no_init if this was unintentional.");
		}
	}

	TEST_CASE("task_manager warns when when long-running programs frequently epoch-synchronize", "[task_manager]") {
		test_utils::allow_max_log_level(log_level::warn);

		const auto action = GENERATE(values({epoch_action::none, epoch_action::barrier}));

		task_graph tdag;
		task_manager tm(1 /* num collective nodes */, tdag, nullptr /* recorder */, nullptr /* delegate */);
		tm.generate_epoch_task(epoch_action::init);
		for(int i = 0; i <= 25; ++i) {
			for(int j = 0; j < 5; ++j) {
				tm.generate_command_group_task(invoke_command_group_function([](handler& cgh) { cgh.host_task(celerity::once, [] {}); }));
			}
			tm.generate_epoch_task(action);
		}
		tm.generate_epoch_task(epoch_action::shutdown);

		CHECK(test_utils::log_contains_exact(log_level::warn,
		    "Your program appears to call queue::wait() excessively, which may lead to performance degradation. Consider using queue::fence() "
		    "for data-dependent branching and employ queue::wait() for timing only on a very coarse granularity."));
	}

} // namespace detail
} // namespace celerity
