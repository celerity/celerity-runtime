#include "task.h"
#include "task_manager.h"
#include "unit_test_suite_celerity.h"

#include <catch2/catch.hpp>

#include <celerity.h>

#include "test_utils.h"


namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;


	bool has_dependency(const task_manager& tm, task_id dependent, task_id dependency, dependency_kind kind = dependency_kind::TRUE_DEP) {
		for(auto dep : tm.get_task(dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency && dep.kind == kind) return true;
		}
		return false;
	}

	bool has_any_dependency(const task_manager& tm, task_id dependent, task_id dependency) {
		for(auto dep : tm.get_task(dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency) return true;
		}
		return false;
	}

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

			maybe_print_graph(tm);
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

			maybe_print_graph(tm);
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

				maybe_print_graph(tm);
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

				maybe_print_graph(tm);
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

		maybe_print_graph(tm);
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

		maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly handles host-initialized buffers", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), true);
		auto non_host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), false);

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::read>(cgh, fixed<1>{{0, 128}});
		});
		REQUIRE(has_dependency(tm, tid_a, 0)); // This task has a dependency on the init task (tid 0)
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::read>(cgh, fixed<1>{{0, 128}});
		});
		REQUIRE_FALSE(has_dependency(tm, tid_b, 0));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
		});
		REQUIRE(has_dependency(tm, tid_c, tid_a, dependency_kind::ANTI_DEP));
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
		});
		// Since task b is essentially reading uninitialized garbage, it doesn't make a difference if we write into it concurrently
		REQUIRE_FALSE(has_dependency(tm, tid_d, tid_b, dependency_kind::ANTI_DEP));

		maybe_print_graph(tm);
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
		CHECK(has_dependency(tm, tid_collective_implicit_2, tid_collective_implicit_1, dependency_kind::ORDER_DEP));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_2, tid_collective_explicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_implicit_2, tid_collective_explicit_2));

		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_master));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_collective_implicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_collective_implicit_2));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_1, tid_collective_explicit_2));

		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_2, tid_master));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_2, tid_collective_implicit_1));
		CHECK_FALSE(has_any_dependency(tm, tid_collective_explicit_2, tid_collective_implicit_2));
		CHECK(has_dependency(tm, tid_collective_explicit_2, tid_collective_explicit_1, dependency_kind::ORDER_DEP));
	}

	void check_path_length_and_front(task_manager& tm, unsigned path_length, std::unordered_set<task_id> exec_front) {
		{
			INFO("path length");
			CHECK(tm.get_max_pseudo_critical_path_length() == path_length);
		}
		{
			INFO("execution front");
			std::unordered_set<task*> task_exec_front;
			std::transform(exec_front.cbegin(), exec_front.cend(), std::inserter(task_exec_front, task_exec_front.begin()),
			    [&tm](task_id tid) { return const_cast<task*>(tm.get_task(tid)); });
			CHECK(tm.get_execution_front() == task_exec_front);
		}
	}

	TEST_CASE("task_manager keeps track of max pseudo critical path length and task front", "[task_manager][task-graph][task-front]") {
		using namespace cl::sycl::access;
		task_manager tm{1, nullptr, nullptr};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));

		SECTION("with true dependencies") {
			const auto tid_a = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128}));
			});
			check_path_length_and_front(tm, 0, {tid_a});

			const auto tid_b = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read_write>(cgh, fixed<1>({0, 128}));
			});
			check_path_length_and_front(tm, 1, {tid_b});

			const auto tid_c = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, fixed<1>({0, 128})); });
			check_path_length_and_front(tm, 2, {tid_c});

			const auto tid_d = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {});
			check_path_length_and_front(tm, 2, {tid_c, tid_d});

			maybe_print_graph(tm);
		}
	}

	struct task_manager_testspy {
		static task* get_previous_horizon_task(task_manager& tm) { return tm.previous_horizon_task; }
		static int get_num_horizons(task_manager& tm) {
			int horizon_counter = 0;
			for(auto& [_, task_ptr] : tm.task_map) {
				if(task_ptr->get_type() == task_type::HORIZON) { horizon_counter++; }
			}
			return horizon_counter;
		}
	};

	TEST_CASE("task horizons are being generated", "[task_manager][task-graph][task-horizon]") {
		task_manager tm{1, nullptr, nullptr};
		tm.set_horizon_step(2);

		using namespace cl::sycl::access;
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));

		SECTION("with true dependencies") {
			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 128})); });

			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, fixed<1>({0, 128})); });

			auto* previous_horizon = task_manager_testspy::get_previous_horizon_task(tm);
			CHECK(previous_horizon == nullptr);

			const auto tid_c = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, fixed<1>({0, 128})); });

			previous_horizon = task_manager_testspy::get_previous_horizon_task(tm);
			REQUIRE(previous_horizon != nullptr);
			CHECK(previous_horizon->get_id() == tid_c + 1);
			CHECK(task_manager_testspy::get_num_horizons(tm) == 1);

			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {});
			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {});
			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {});
			CHECK(task_manager_testspy::get_num_horizons(tm) == 1);

			test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, fixed<1>({0, 128})); });
			const auto tid_d = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read_write>(cgh, fixed<1>({0, 128}));
			});

			previous_horizon = task_manager_testspy::get_previous_horizon_task(tm);
			REQUIRE(previous_horizon != nullptr);
			CHECK(previous_horizon->get_id() == tid_d + 1);
			CHECK(task_manager_testspy::get_num_horizons(tm) == 2);
		}
	}

} // namespace detail
} // namespace celerity