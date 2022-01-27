#include "allscale/api/user/data/grid.h"
#include "task.h"
#include "task_manager.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <celerity.h>
#include <iterator>

#include "test_utils.h"


namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;

	TEST_CASE("task_manager does not create multiple dependencies between the same tasks", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(128));

		SECTION("true dependencies") {
			const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<mode::read>(cgh, fixed<1>({0, 128}));
			});
			CHECK(has_dependency(tm, tid_b, tid_a));

			const auto its = tm.get_task(tid_a)->get_dependents();
			REQUIRE(std::distance(its.begin(), its.end()) == 1);

			test_utils::maybe_print_graph(tm);
		}

		SECTION("anti-dependencies") {
			const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
				buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			CHECK(has_dependency(tm, tid_b, tid_a, dependency_kind::ANTI_DEP));

			const auto its = tm.get_task(tid_a)->get_dependents();
			REQUIRE(std::distance(its.begin(), its.end()) == 1);

			test_utils::maybe_print_graph(tm);
		}

		// Here we also check that true dependencies always take precedence
		SECTION("true and anti-dependencies combined") {
			SECTION("if true is declared first") {
				const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
				});
				const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<mode::read>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
				});
				CHECK(has_dependency(tm, tid_b, tid_a));
				CHECK_FALSE(has_dependency(tm, tid_b, tid_a, dependency_kind::ANTI_DEP));

				const auto its = tm.get_task(tid_a)->get_dependents();
				REQUIRE(std::distance(its.begin(), its.end()) == 1);

				test_utils::maybe_print_graph(tm);
			}

			SECTION("if anti is declared first") {
				const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
				});
				const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
					buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
					buf_b.get_access<mode::read>(cgh, fixed<1>({0, 128}));
				});
				CHECK(has_dependency(tm, tid_b, tid_a));
				CHECK_FALSE(has_dependency(tm, tid_b, tid_a, dependency_kind::ANTI_DEP));

				const auto its = tm.get_task(tid_a)->get_dependents();
				REQUIRE(std::distance(its.begin(), its.end()) == 1);

				test_utils::maybe_print_graph(tm);
			}
		}
	}

	TEST_CASE("task_manager respects range mapper results for finding dependencies", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 64}});
		});
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, fixed<1>{{0, 128}}); });
		REQUIRE(has_dependency(tm, tid_b, tid_a));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, fixed<1>{{64, 128}}); });
		REQUIRE_FALSE(has_dependency(tm, tid_c, tid_a));

		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly generates anti-dependencies", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		// Write to the full buffer
		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
		});
		// Read the first half of the buffer
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, fixed<1>{{0, 64}}); });
		CHECK(has_dependency(tm, tid_b, tid_a));
		// Overwrite the second half - no anti-dependency onto task_b should exist (but onto task_a)
		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>{{64, 64}});
		});
		REQUIRE(has_dependency(tm, tid_c, tid_a, dependency_kind::ANTI_DEP));
		REQUIRE_FALSE(has_dependency(tm, tid_c, tid_b, dependency_kind::ANTI_DEP));
		// Overwrite the first half - now only an anti-dependency onto task_b should exist
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 64}});
		});
		REQUIRE_FALSE(has_dependency(tm, tid_d, tid_a, dependency_kind::ANTI_DEP));
		REQUIRE(has_dependency(tm, tid_d, tid_b, dependency_kind::ANTI_DEP));

		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly handles host-initialized buffers", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), true);
		auto non_host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), false);
		auto artificial_dependency_buf = mbf.create_buffer(cl::sycl::range<1>(1), false);

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::read>(cgh, fixed<1>{{0, 128}});
			artificial_dependency_buf.get_access<mode::discard_write>(cgh, all{});
		});
		CHECK(has_dependency(tm, tid_a, 0)); // This task has a dependency on the initial epoch task (tid 0)

		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::read>(cgh, fixed<1>{{0, 128}});
			// introduce an arbitrary true-dependency to avoid the fallback epoch dependency that is generated for tasks without other true-dependencies
			artificial_dependency_buf.get_access<mode::read>(cgh, all{});
		});
		CHECK_FALSE(has_dependency(tm, tid_b, 0));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
		});
		CHECK(has_dependency(tm, tid_c, tid_a, dependency_kind::ANTI_DEP));
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
		});
		// Since task b is essentially reading uninitialized garbage, it doesn't make a difference if we write into it concurrently
		CHECK_FALSE(has_dependency(tm, tid_d, tid_b, dependency_kind::ANTI_DEP));

		test_utils::maybe_print_graph(tm);
	}

	template <int Dims, typename Handler, typename Functor>
	void dispatch_get_access(test_utils::mock_buffer<Dims>& mb, Handler& handler, cl::sycl::access::mode mode, Functor rmfn) {
		using namespace cl::sycl::access;
		switch(mode) {
		case mode::read: mb.template get_access<mode::read>(handler, rmfn); break;
		case mode::write: mb.template get_access<mode::write>(handler, rmfn); break;
		case mode::read_write: mb.template get_access<mode::read_write>(handler, rmfn); break;
		case mode::discard_write: mb.template get_access<mode::discard_write>(handler, rmfn); break;
		case mode::discard_read_write: mb.template get_access<mode::discard_read_write>(handler, rmfn); break;
		case mode::atomic: mb.template get_access<mode::atomic>(handler, rmfn); break;
		default: assert(false);
		}
	}

	TEST_CASE("task_manager correctly handles dependencies for R/W modes", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		// A read-write access can also be implicitly created using a separate write and read, which is why we operate on "mode sets" here.
		const std::vector<std::vector<mode>> rw_mode_sets = {{mode::discard_read_write}, {mode::read_write}, {mode::atomic}, {mode::discard_write, mode::read}};

		for(const auto& mode_set : rw_mode_sets) {
			task_manager tm{1, nullptr, nullptr};
			test_utils::mock_buffer_factory mbf(&tm);
			auto buf = mbf.create_buffer(cl::sycl::range<1>(128), true);

			const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
				for(const auto& m : mode_set) {
					dispatch_get_access(buf, cgh, m, fixed<1>{{0, 128}});
				}
			});
			const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
				buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
			});
			REQUIRE(has_dependency(tm, tid_b, tid_a, dependency_kind::ANTI_DEP));
		}
	}

	TEST_CASE("task_manager handles all producer/consumer combinations correctly", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		for(const auto& consumer_mode : detail::access::consumer_modes) {
			for(const auto& producer_mode : detail::access::producer_modes) {
				CAPTURE(consumer_mode);
				CAPTURE(producer_mode);
				task_manager tm{1, nullptr, nullptr};
				test_utils::mock_buffer_factory mbf(&tm);
				auto buf = mbf.create_buffer(cl::sycl::range<1>(128), false);

				const task_id tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, producer_mode, fixed<1>{{0, 128}});
				});

				const task_id tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, consumer_mode, fixed<1>{{0, 128}});
				});
				REQUIRE(has_dependency(tm, tid_b, tid_a));

				const task_id tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, producer_mode, fixed<1>{{0, 128}});
				});
				const bool pure_consumer = consumer_mode == mode::read;
				const bool pure_producer = producer_mode == mode::discard_read_write || producer_mode == mode::discard_write;
				REQUIRE(has_dependency(tm, tid_c, tid_b, pure_consumer || pure_producer ? dependency_kind::ANTI_DEP : dependency_kind::TRUE_DEP));
			}
		}
	}

	TEST_CASE("task_manager generates pseudo-dependencies for collective host tasks", "[task_manager][task-graph]") {
		task_manager tm{1, nullptr, nullptr};
		experimental::collective_group group;
		auto tid_master = test_utils::add_host_task(tm, on_master_node, [](handler&) {});
		auto tid_collective_implicit_1 = test_utils::add_host_task(tm, experimental::collective, [](handler&) {});
		auto tid_collective_implicit_2 = test_utils::add_host_task(tm, experimental::collective, [](handler&) {});
		auto tid_collective_explicit_1 = test_utils::add_host_task(tm, experimental::collective(group), [](handler&) {});
		auto tid_collective_explicit_2 = test_utils::add_host_task(tm, experimental::collective(group), [](handler&) {});

		CHECK_FALSE(has_any_dependency(tm, tid_master, tid_collective_implicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_master, tid_collective_implicit_2));
		CHECK_FALSE(has_any_dependency(tm, tid_master, tid_collective_explicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_master, tid_collective_explicit_2));

		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_1, tid_master));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_1, tid_collective_implicit_2));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_1, tid_collective_explicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_1, tid_collective_explicit_2));

		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_2, tid_master));
		CHECK(has_dependency(tm, tid_collective_implicit_2, tid_collective_implicit_1, dependency_kind::TRUE_DEP));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_2, tid_collective_explicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_2, tid_collective_explicit_2));

		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_master));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_collective_implicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_collective_implicit_2));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_collective_explicit_2));

		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_2, tid_master));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_2, tid_collective_implicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_2, tid_collective_implicit_2));
		CHECK(has_dependency(tm, tid_collective_explicit_2, tid_collective_explicit_1, dependency_kind::TRUE_DEP));
	}

	void check_path_length_and_front(task_manager& tm, int path_length, std::unordered_set<task_id> exec_front) {
		{
			INFO("path length");
			CHECK(task_manager_testspy::get_max_pseudo_critical_path_length(tm) == path_length);
		}
		{
			INFO("execution front");
			std::unordered_set<task*> task_exec_front;
			std::transform(exec_front.cbegin(), exec_front.cend(), std::inserter(task_exec_front, task_exec_front.begin()),
			    [&tm](task_id tid) { return const_cast<task*>(tm.get_task(tid)); });
			CHECK(task_manager_testspy::get_execution_front(tm) == task_exec_front);
		}
	}

	TEST_CASE("task_manager keeps track of max pseudo critical path length and task front", "[task_manager][task-graph][task-front]") {
		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));

		const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
		});
		check_path_length_and_front(tm, 1, {tid_a}); // 1: we always depend on the initial epoch task

		const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128}));
		});
		check_path_length_and_front(tm, 2, {tid_b});

		const auto tid_c = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read>(cgh, fixed<1>({0, 128})); });
		check_path_length_and_front(tm, 3, {tid_c});

		const auto tid_d = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {});
		check_path_length_and_front(tm, 3, {tid_c, tid_d});

		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE("task horizons are being generated with correct dependencies", "[task_manager][task-graph][task-horizon]") {
		task_manager tm{1, nullptr, nullptr};
		tm.set_horizon_step(2);

		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));

		test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128})); });

		auto* current_horizon = task_manager_testspy::get_current_horizon_task(tm);
		CHECK(current_horizon == nullptr);

		const auto tid_c = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read>(cgh, fixed<1>({0, 128})); });

		current_horizon = task_manager_testspy::get_current_horizon_task(tm);
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_id() == tid_c + 1);
		CHECK(task_manager_testspy::get_num_horizons(tm) == 1);

		auto horizon_dependencies = current_horizon->get_dependencies();

		CHECK(std::distance(horizon_dependencies.begin(), horizon_dependencies.end()) == 1);
		CHECK(horizon_dependencies.begin()->node->get_id() == tid_c);

		std::set<task_id> expected_dependency_ids;

		// current horizon is always part of the active task front
		expected_dependency_ids.insert(current_horizon->get_id());
		expected_dependency_ids.insert(test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {}));
		expected_dependency_ids.insert(test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {}));
		expected_dependency_ids.insert(test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {}));
		CHECK(task_manager_testspy::get_num_horizons(tm) == 1);

		test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128})); });
		const auto tid_d = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128}));
		});
		expected_dependency_ids.insert(tid_d);

		current_horizon = task_manager_testspy::get_current_horizon_task(tm);
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_id() == tid_d + 1);
		CHECK(task_manager_testspy::get_num_horizons(tm) == 2);

		horizon_dependencies = current_horizon->get_dependencies();
		CHECK(std::distance(horizon_dependencies.begin(), horizon_dependencies.end()) == 5);

		std::set<task_id> actual_dependecy_ids;
		for(auto dep : horizon_dependencies) {
			actual_dependecy_ids.insert(dep.node->get_id());
		}
		CHECK(expected_dependency_ids == actual_dependecy_ids);
	}

	static inline GridRegion<3> make_region(int min, int max) { return GridRegion<3>(GridPoint<3>(min, 0, 0), GridPoint<3>(max, 1, 1)); }

	TEST_CASE("task horizons update previous writer data structure", "[task_manager][task-graph][task-horizon]") {
		task_manager tm{1, nullptr, nullptr};
		tm.set_horizon_step(2);

		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(128));

		task_id tid_1 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 64}));
			buf_b.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 128}));
		});
		task_id tid_2 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::discard_write>(cgh, fixed<1>({64, 64}));
		});
		task_id tid_3 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({32, 64}));
		});
		task_id tid_4 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({32, 64}));
		});

		auto* horizon_tsk = task_manager_testspy::get_current_horizon_task(tm);
		CHECK(task_manager_testspy::get_num_horizons(tm) == 1);
		CHECK(horizon_tsk != nullptr);

		task_id tid_6 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_b.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128}));
		});
		task_id tid_7 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_b.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128}));
		});

		{
			INFO("check that previous tasks are still last writers before the first horizon is applied");
			auto region_map_a = task_manager_testspy::get_last_writer(tm, buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 32)).front().second.value() == tid_1);
			CHECK(region_map_a.get_region_values(make_region(96, 128)).front().second.value() == tid_2);
			CHECK(region_map_a.get_region_values(make_region(32, 96)).front().second.value() == tid_4);
		}

		task_id tid_8 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_b.get_access<access_mode::read_write>(cgh, fixed<1>({0, 128}));
		});

		CHECK(task_manager_testspy::get_num_horizons(tm) == 2);

		{
			INFO("check that only the previous horizon is the last writer of buff_a");
			auto region_map_a = task_manager_testspy::get_last_writer(tm, buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 128)).front().second.value() == horizon_tsk->get_id());
		}

		task_id tid_9 = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_a.get_access<access_mode::read_write>(cgh, fixed<1>({64, 64}));
		});

		{
			INFO("check that the previous horizon and task 11 are last writers of buff_a");
			auto region_map_a = task_manager_testspy::get_last_writer(tm, buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 64)).front().second.value() == horizon_tsk->get_id());
			CHECK(region_map_a.get_region_values(make_region(64, 128)).front().second.value() == tid_9);
		}

		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE("previous task horizon is used as last writer for host-initialized buffers", "[task_manager][task-graph][task-horizon]") {
		task_manager tm{1, nullptr, nullptr};
		tm.set_horizon_step(2);

		test_utils::mock_buffer_factory mbf(&tm);

		task_id initial_last_writer_id = -1;
		{
			auto buf = mbf.create_buffer(cl::sycl::range<1>(1), true);
			const auto tid = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::read_write>(cgh, all{}); });
			const auto& deps = tm.get_task(tid)->get_dependencies();
			CHECK(std::distance(deps.begin(), deps.end()) == 1);
			initial_last_writer_id = deps.begin()->node->get_id();
		}
		CHECK(tm.has_task(initial_last_writer_id));

		// Create a bunch of tasks to trigger horizon cleanup
		{
			auto buf = mbf.create_buffer(cl::sycl::range<1>(1));
			task_id last_executed_horizon = 0;
			// We need 7 tasks to generate a pseudo-critical path length of 6 (3x2 horizon step size),
			// and another one that triggers the actual deferred deletion.
			for(int i = 0; i < 8; ++i) {
				const auto tid = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
				const auto* current_horizon = task_manager_testspy::get_current_horizon_task(tm);
				if(current_horizon != nullptr && current_horizon->get_id() > last_executed_horizon) {
					last_executed_horizon = current_horizon->get_id();
					tm.notify_horizon_executed(last_executed_horizon);
				}
			}
		}

		INFO("initial last writer with id " << initial_last_writer_id << " has been deleted");
		CHECK_FALSE(tm.has_task(initial_last_writer_id));

		auto buf = mbf.create_buffer(cl::sycl::range<1>(1), true);
		const auto tid = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::read_write>(cgh, all{}); });
		const auto& deps = tm.get_task(tid)->get_dependencies();
		CHECK(std::distance(deps.begin(), deps.end()) == 1);
		const auto* new_last_writer = deps.begin()->node;
		CHECK(new_last_writer->get_type() == task_type::HORIZON);

		const auto* current_horizon = task_manager_testspy::get_current_horizon_task(tm);
		INFO("previous horizon is being used");
		CHECK(new_last_writer->get_id() < current_horizon->get_id());

		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE("collective host tasks do not order-depend on their predecessor if it is shadowed by a horizon", "[task_manager][task-graph][task-horizon]") {
		// Regression test: the order-dependencies between host tasks in the same collective group are built by tracking the last task in each collective group.
		// Once a horizon is inserted, new collective host tasks must order-depend on that horizon instead.

		task_manager tm{1, nullptr, nullptr};
		tm.set_horizon_step(2);

		const auto first_collective = test_utils::add_host_task(tm, experimental::collective, [&](handler& cgh) {});

		// generate exactly two horizons
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(range<1>(1));
		for(int i = 0; i < 4; ++i) {
			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
		}

		// This must depend on the first horizon, not first_collective
		const auto second_collective =
		    test_utils::add_host_task(tm, experimental::collective, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, all{}); });

		const auto second_collective_deps = tm.get_task(second_collective)->get_dependencies();
		const auto master_node_dep = std::find_if(second_collective_deps.begin(), second_collective_deps.end(),
		    [](const task::dependency d) { return d.node->get_type() == task_type::MASTER_NODE; });
		const auto horizon_dep = std::find_if(second_collective_deps.begin(), second_collective_deps.end(), //
		    [](const task::dependency d) { return d.node->get_type() == task_type::HORIZON; });

		CHECK(std::distance(second_collective_deps.begin(), second_collective_deps.end()) == 2);
		REQUIRE(master_node_dep != second_collective_deps.end());
		CHECK(master_node_dep->kind == dependency_kind::TRUE_DEP);
		REQUIRE(horizon_dep != second_collective_deps.end());
		CHECK(horizon_dep->kind == dependency_kind::TRUE_DEP);

		test_utils::maybe_print_graph(tm);
	}

	TEST_CASE("buffer accesses with empty ranges do not generate data-flow dependencies", "[task_manager][task-graph]") {
		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(range<2>(32, 32));

		const auto write_sr = GENERATE(values({subrange<2>{{16, 16}, {0, 0}}, subrange<2>{{16, 16}, {8, 8}}}));
		const auto read_sr = GENERATE(values({subrange<2>{{1, 1}, {0, 0}}, subrange<2>{{8, 8}, {16, 16}}}));

		const auto read_empty = read_sr.range.size() == 0;
		const auto write_empty = write_sr.range.size() == 0;
		CAPTURE(read_empty);
		CAPTURE(write_empty);

		const auto write_tid =
		    test_utils::add_compute_task<class UKN(write)>(tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, fixed<2>{write_sr}); });
		const auto read_tid =
		    test_utils::add_compute_task<class UKN(read)>(tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, fixed<2>{read_sr}); });

		CHECK(has_any_dependency(tm, read_tid, write_tid) == (!write_empty && !read_empty));
	}

	TEST_CASE("side effects generate appropriate task-dependencies", "[task_manager][task-graph][side-effect]") {
		using order = experimental::side_effect_order;
		static constexpr auto side_effect_orders = {order::sequential};

		// TODO placeholder: complete with dependency types for other side effect orders
		const auto expected_dependencies = std::unordered_map<std::pair<order, order>, std::optional<dependency_kind>, pair_hash>{
		    {{order::sequential, order::sequential}, dependency_kind::TRUE_DEP}};

		const auto order_a = GENERATE(values(side_effect_orders));
		const auto order_b = GENERATE(values(side_effect_orders));

		CAPTURE(order_a);
		CAPTURE(order_b);

		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_host_object_factory mhof;

		auto ho_common = mhof.create_host_object(); // should generate dependencies
		auto ho_a = mhof.create_host_object();      // should NOT generate dependencies
		auto ho_b = mhof.create_host_object();      // -"-
		const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			ho_common.add_side_effect(cgh, order_a);
			ho_a.add_side_effect(cgh, order_a);
		});
		const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			ho_common.add_side_effect(cgh, order_b);
			ho_b.add_side_effect(cgh, order_b);
		});

		const auto deps_a = tm.get_task(tid_a)->get_dependencies();
		REQUIRE(std::distance(deps_a.begin(), deps_a.end()) == 1);
		CHECK(deps_a.front().node->get_id() == task_manager::initial_epoch_task);

		const auto deps_b = tm.get_task(tid_b)->get_dependencies();
		const auto expected_b = expected_dependencies.at({order_a, order_b});
		CHECK(std::distance(deps_b.begin(), deps_b.end()) == expected_b.has_value());
		if(expected_b) {
			CHECK(deps_b.front().node == tm.get_task(tid_a));
			CHECK(deps_b.front().kind == *expected_b);
		}
	}

	TEST_CASE("side-effect dependencies are correctly subsumed by horizons", "[task_manager][task-graph][task-horizon]") {
		task_manager tm{1, nullptr, nullptr};
		tm.set_horizon_step(2);

		test_utils::mock_host_object_factory mhof;
		auto ho = mhof.create_host_object();
		const auto first_task =
		    test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); });

		// generate exactly two horizons
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(range<1>(1));
		for(int i = 0; i < 5; ++i) {
			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); });
		}

		// This must depend on the first horizon, not first_task
		const auto second_task =
		    test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); });

		const auto& second_deps = tm.get_task(second_task)->get_dependencies();
		CHECK(std::distance(second_deps.begin(), second_deps.end()) == 1);
		for(const auto& dep : second_deps) {
			const auto type = dep.node->get_type();
			CHECK(type == task_type::HORIZON);
			CHECK(dep.kind == dependency_kind::TRUE_DEP);
		}

		test_utils::maybe_print_graph(tm);
	}
} // namespace detail
} // namespace celerity