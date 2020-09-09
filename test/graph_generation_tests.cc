#include "unit_test_suite_celerity.h"

#include <optional>
#include <set>
#include <unordered_set>

#include <catch2/catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

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
		task_manager tm{1, nullptr, true};
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
		task_manager tm{1, nullptr, true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({0, 64}));
		});
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, fixed<2, 1>({0, 128})); });
		REQUIRE(has_dependency(tm, tid_b, tid_a));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, fixed<2, 1>({64, 128})); });
		REQUIRE_FALSE(has_dependency(tm, tid_c, tid_a));

		maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly generates anti-dependencies", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{1, nullptr, true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		// Write to the full buffer
		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({0, 128}));
		});
		// Read the first half of the buffer
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, fixed<2, 1>({0, 64})); });
		CHECK(has_dependency(tm, tid_b, tid_a));
		// Overwrite the second half - no anti-dependency onto task_b should exist (but onto task_a)
		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({64, 64}));
		});
		REQUIRE(has_dependency(tm, tid_c, tid_a, dependency_kind::ANTI_DEP));
		REQUIRE_FALSE(has_dependency(tm, tid_c, tid_b, dependency_kind::ANTI_DEP));
		// Overwrite the first half - now only an anti-dependency onto task_b should exist
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({0, 64}));
		});
		REQUIRE_FALSE(has_dependency(tm, tid_d, tid_a, dependency_kind::ANTI_DEP));
		REQUIRE(has_dependency(tm, tid_d, tid_b, dependency_kind::ANTI_DEP));

		maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly handles host-initialized buffers", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{1, nullptr, true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), true);
		auto non_host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), false);

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::read>(cgh, fixed<2, 1>({0, 128}));
		});
		REQUIRE(has_dependency(tm, tid_a, 0)); // This task has a dependency on the init task (tid 0)
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::read>(cgh, fixed<2, 1>({0, 128}));
		});
		REQUIRE_FALSE(has_dependency(tm, tid_b, 0));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({0, 128}));
		});
		REQUIRE(has_dependency(tm, tid_c, tid_a, dependency_kind::ANTI_DEP));
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({0, 128}));
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
			task_manager tm{1, nullptr, true};
			test_utils::mock_buffer_factory mbf(&tm);
			auto buf = mbf.create_buffer(cl::sycl::range<1>(128), true);

			const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
				for(const auto& m : mode_set) {
					dispatch_get_access(buf, cgh, m, fixed<2, 1>({0, 128}));
				}
			});
			const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
				buf.get_access<mode::discard_write>(cgh, fixed<2, 1>({0, 128}));
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
				task_manager tm{1, nullptr, true};
				test_utils::mock_buffer_factory mbf(&tm);
				auto buf = mbf.create_buffer(cl::sycl::range<1>(128), false);

				const task_id tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, producer_mode, fixed<2, 1>({0, 128}));
				});

				const task_id tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, consumer_mode, fixed<2, 1>({0, 128}));
				});
				REQUIRE(has_dependency(tm, tid_b, tid_a));

				const task_id tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, producer_mode, fixed<2, 1>({0, 128}));
				});
				const bool pure_consumer = consumer_mode == mode::read;
				const bool pure_producer = producer_mode == mode::discard_read_write || producer_mode == mode::discard_write;
				REQUIRE(has_dependency(tm, tid_c, tid_b, pure_consumer || pure_producer ? dependency_kind::ANTI_DEP : dependency_kind::TRUE_DEP));
			}
		}
	}

	TEST_CASE("task_manager generates pseudo-dependencies for collective host tasks", "[task_manager][task-graph]") {
		task_manager tm{1, nullptr, true};
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

	TEST_CASE("command_graph keeps track of created commands", "[command_graph][command-graph]") {
		command_graph cdag;
		auto cmd0 = cdag.create<task_command>(0, 0, subrange<3>{});
		auto cmd1 = cdag.create<task_command>(0, 1, subrange<3>{});
		REQUIRE(cmd0->get_cid() != cmd1->get_cid());
		REQUIRE(cdag.get(cmd0->get_cid()) == cmd0);
		REQUIRE(cdag.command_count() == 2);
		REQUIRE(cdag.task_command_count(0) == 1);
		REQUIRE(cdag.task_command_count(1) == 1);

		cdag.erase(cmd1);
		REQUIRE(cdag.command_count() == 1);
		REQUIRE(cdag.task_command_count(1) == 0);
	}

	TEST_CASE("command_graph allows to iterate over all raw command pointers", "[command_graph][command-graph]") {
		command_graph cdag;
		std::unordered_set<abstract_command*> cmds;
		cmds.insert(cdag.create<task_command>(0, 0, subrange<3>{}));
		cmds.insert(cdag.create<nop_command>(0));
		cmds.insert(cdag.create<push_command>(0, 0, 0, subrange<3>{}));
		for(auto cmd : cdag.all_commands()) {
			REQUIRE(cmds.find(cmd) != cmds.end());
			cmds.erase(cmd);
		}
		REQUIRE(cmds.empty());
	}

	TEST_CASE("command_graph keeps track of execution fronts", "[command_graph][command-graph]") {
		command_graph cdag;

		auto build_testing_graph_on_node = [&cdag](node_id node) {
			std::unordered_set<abstract_command*> expected_front;

			auto t0 = cdag.create<task_command>(node, 0, subrange<3>{});
			expected_front.insert(t0);
			REQUIRE(expected_front == cdag.get_execution_front(node));

			expected_front.insert(cdag.create<task_command>(node, 1, subrange<3>{}));
			REQUIRE(expected_front == cdag.get_execution_front(node));

			expected_front.erase(t0);
			auto t2 = cdag.create<task_command>(node, 2, subrange<3>{});
			expected_front.insert(t2);
			cdag.add_dependency(t2, t0);
			REQUIRE(expected_front == cdag.get_execution_front(node));
			return expected_front;
		};

		auto node_0_expected_front = build_testing_graph_on_node(0u);

		SECTION("for individual nodes") { build_testing_graph_on_node(1u); }

		REQUIRE(node_0_expected_front == cdag.get_execution_front(0));
	}

	TEST_CASE("isa<> RTTI helper correctly handles command hierarchies", "[rtti][command-graph]") {
		command_graph cdag;
		auto np = cdag.create<nop_command>(0);
		REQUIRE(isa<abstract_command>(np));
		auto hec = cdag.create<task_command>(0, 0, subrange<3>{});
		REQUIRE(isa<task_command>(hec));
		auto pc = cdag.create<push_command>(0, 0, 0, subrange<3>{});
		REQUIRE(isa<abstract_command>(pc));
		auto apc = cdag.create<await_push_command>(0, pc);
		REQUIRE(isa<abstract_command>(apc));
	}

	TEST_CASE("graph_generator generates required data transfer commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(4);
		auto& inspector = ctx.get_inspector();

		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(300));

		const auto tid_a = test_utils::build_and_flush(ctx, 4,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf.get_access<mode::discard_write>(cgh, [](chunk<1> chnk) {
				        switch(chnk.offset[0]) {
				        case 0: return subrange<1>(chnk);
				        case 75: return subrange<1>(150, 75);
				        case 150: return subrange<1>(75, 75);
				        case 225: return subrange<1>(chnk);
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
		        },
		        cl::sycl::range<1>{300}));

		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 4);
		CHECK(inspector.get_commands(tid_a, node_id(1), command_type::TASK).size() == 1);
		CHECK(inspector.get_commands(tid_a, node_id(2), command_type::TASK).size() == 1);
		CHECK(inspector.get_commands(tid_a, node_id(3), command_type::TASK).size() == 1);

		test_utils::build_and_flush(ctx, 4,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::read>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{300}));

		REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 2);
		REQUIRE(inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH).size() == 1);
		REQUIRE(inspector.get_commands(std::nullopt, node_id(2), command_type::PUSH).size() == 1);
		REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 2);
		REQUIRE(inspector.get_commands(std::nullopt, node_id(1), command_type::AWAIT_PUSH).size() == 1);
		REQUIRE(inspector.get_commands(std::nullopt, node_id(2), command_type::AWAIT_PUSH).size() == 1);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator doesn't generate data transfer commands for the same buffer and range more than once", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));

		SECTION("when used in the same task") {
			test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				// Both of theses are consumer modes, meaning that both have a requirement on the buffer range produced in task_a
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
				buf_a.get_access<mode::write>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::TASK).size() == 3);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);

			maybe_print_graphs(ctx);
		}

		SECTION("when used in the same task by different chunks on the same worker node") {
			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](auto& mah) {
				buf_a.get_access<mode::discard_write>(mah, fixed<1>({0, 100}));
			}));
			// Create 4 chunks, two of which will be assigned to the worker node
			const auto tid_b = test_utils::build_and_flush(ctx, 2, 4,
			    test_utils::add_compute_task<class task_b>(
			        ctx.get_task_manager(),
			        [&](auto& cgh) {
				        // All chunks read the same subrange (the full buffer)
				        buf_a.get_access<mode::read>(cgh, fixed<1, 1>(subrange<1>(0, 100)));
			        },
			        cl::sycl::range<1>(100)));

			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 4);
			const auto computes = inspector.get_commands(tid_b, node_id(1), command_type::TASK);
			CHECK(computes.size() == 2);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(0), command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(1), command_type::AWAIT_PUSH).size() == 1);

			maybe_print_graphs(ctx);
		}

		SECTION("when used in consecutive tasks") {
			auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

			test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
				buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
			}));

			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);

			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
				buf_b.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::TASK).size() == 4);
			// Assert that the number of PUSHes / AWAIT_PUSHes hasn't changed
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);

			maybe_print_graphs(ctx);
		}

		SECTION("when used in parallel tasks") {
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);

			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));

			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);

			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::TASK).size() == 4);
			// Assert that the number of PUSHes / AWAIT_PUSHes hasn't changed
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);

			maybe_print_graphs(ctx);
		}
	}

	TEST_CASE("graph_generator uses original producer as source for PUSH rather than building dependency chain", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		constexpr int NUM_NODES = 3;
		test_utils::cdag_test_context ctx(NUM_NODES);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto full_range = cl::sycl::range<1>(300);
		auto buf_a = mbf.create_buffer(full_range);

		test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(producer)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, full_range));

		SECTION("when distributing a single reading task across nodes") {
			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(producer)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, one_to_one<1>()); }, full_range));
		}

		SECTION("when distributing a single read-write task across nodes") {
			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(producer)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, one_to_one<1>()); }, full_range));
		}

		SECTION("when running multiple reading task on separate nodes") {
			auto full_range_for_single_node = [=](node_id node) {
				return [=](chunk<1> chnk) -> subrange<1> {
					if(chnk.range == full_range) return chnk;
					if(chnk.offset[0] == (full_range.size() / NUM_NODES) * node) { return {0, full_range}; }
					return {0, 0};
				};
			};

			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(producer)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, full_range_for_single_node(1)); }, full_range));

			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(producer)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, full_range_for_single_node(2)); }, full_range));
		}

		CHECK(inspector.get_commands(std::nullopt, node_id(0), command_type::PUSH).size() == 2);
		CHECK(inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH).size() == 0);
		CHECK(inspector.get_commands(std::nullopt, node_id(2), command_type::PUSH).size() == 0);
		CHECK(inspector.get_commands(std::nullopt, node_id(1), command_type::AWAIT_PUSH).size() == 1);
		CHECK(inspector.get_commands(std::nullopt, node_id(2), command_type::AWAIT_PUSH).size() == 1);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator consolidates PUSH commands for adjacent subranges", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();

		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		const auto tid_a = test_utils::build_and_flush(ctx, 2,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{64},
		        cl::sycl::id<1>{0}));
		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);

		const auto tid_b = test_utils::build_and_flush(ctx, 2,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        // Swap the two chunks so we write a contiguous range on the worker node across tasks a and b
			        buf.get_access<mode::discard_write>(cgh, [](chunk<1> chnk) {
				        switch(chnk.offset[0]) {
				        case 64: return subrange<1>(96, 32);
				        case 96: return subrange<1>(64, 32);
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
		        },
		        cl::sycl::range<1>{64}, cl::sycl::id<1>{64}));
		CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 2);

		test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
			buf.get_access<mode::read>(cgh, fixed<1>({0, 128}));
		}));

		auto push_commands = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
		REQUIRE(push_commands.size() == 1);
		REQUIRE(inspector.get_dependency_count(*push_commands.cbegin()) == 2);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator builds dependencies to all local commands if a given range is produced by multiple", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(1);
		auto& inspector = ctx.get_inspector();

		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{64},
		        cl::sycl::id<1>{0}));
		test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{32},
		        cl::sycl::id<1>{64}));
		test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_c)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{32},
		        cl::sycl::id<1>{96}));

		auto master_task = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
			buf.get_access<mode::read>(cgh, fixed<1>({0, 128}));
		}));

		auto master_cmds = inspector.get_commands(master_task, std::nullopt, std::nullopt);
		CHECK(master_cmds.size() == 1);

		auto master_cmd = *master_cmds.cbegin();
		CHECK(inspector.get_dependency_count(master_cmd) == 3);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator generates dependencies for PUSH commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();

		SECTION("if data is produced by an execution command") {
			test_utils::mock_buffer_factory mbf(ctx);
			auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);
			const auto computes = inspector.get_commands(tid_a, node_id(1), command_type::TASK);
			CHECK(computes.size() == 1);

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).empty());
			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			const auto pushes = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
			CHECK(pushes.size() == 1);

			REQUIRE(inspector.has_dependency(*pushes.cbegin(), *computes.cbegin()));

			maybe_print_graphs(ctx);
		}

		SECTION("if data is produced by an AWAIT_PUSH command") {
			// There currently is no good way of reliably testing this because the source node for a PUSH is currently
			// selected "randomly" (i.e. the first in an unordered_set is used, ordering depends on STL implementation)
			// TODO: Revisit in the future
		}
	}

	TEST_CASE("graph_generator generates anti-dependencies for AWAIT_PUSH commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100), true);

		SECTION("if writing to region used by execution command") {
			// The master node starts by reading from buf (which is host-initialized)
			const auto tid_a = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);
			const auto master_node_tasks_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
			CHECK(master_node_tasks_a.size() == 1);

			// Meanwhile, the worker node writes to buf
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 2);
			const auto computes_b_0 = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(computes_b_0.size() == 1);
			CHECK(inspector.has_dependency(*computes_b_0.cbegin(), *master_node_tasks_a.cbegin()));

			// Finally the master node reads again from buf, which is now the version written to by the worker node.
			const auto tid_c = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			CHECK(await_pushes.size() == 1);
			const auto master_node_tasks_c = inspector.get_commands(tid_c, node_id(0), command_type::TASK);
			CHECK(master_node_tasks_c.size() == 1);
			CHECK(inspector.has_dependency(*master_node_tasks_c.cbegin(), *await_pushes.cbegin()));

			// The AWAIT_PUSH command has to wait until the MASTER_NODE in task_a is complete.
			REQUIRE(inspector.has_dependency(*await_pushes.cbegin(), *master_node_tasks_a.cbegin()));

			maybe_print_graphs(ctx);
		}

		SECTION("if writing to region used by PUSH command") {
			// Worker node writes to buf
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);

			// Master node reads from buf, requiring a PUSH, while also writing to it
			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read_write>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			const auto pushes = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
			CHECK(pushes.size() == 1);

			// Finally, the worker node reads buf again, requiring an AWAIT_PUSH
			// Note that in this example the AWAIT_PUSH can never occur during the PUSH to master, as they are effectively
			// in a distributed dependency relationship, however more complex examples could give rise to situations where this can happen.
			const auto tid_c = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::read>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::TASK).size() == 2);
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 2);
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(1), command_type::AWAIT_PUSH);
			CHECK(await_pushes.size() == 1);

			REQUIRE(inspector.has_dependency(*await_pushes.cbegin(), *pushes.cbegin()));

			maybe_print_graphs(ctx);
		}

		SECTION("if writing to region used by another AWAIT_PUSH command") {
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);

			const auto tid_b = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			CHECK(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);
			const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(master_node_tasks_b.size() == 1);

			const auto tid_c = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::TASK).size() == 2);

			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			CHECK(await_pushes.size() == 2);

			// The anti-dependency is delegated to the reader (i.e. the master_node_task)
			REQUIRE_FALSE(inspector.has_dependency(*await_pushes.crbegin(), *await_pushes.cbegin()));
			REQUIRE(inspector.has_dependency(*await_pushes.crbegin(), *master_node_tasks_b.cbegin()));

			maybe_print_graphs(ctx);
		}
	}

	TEST_CASE("graph_generator generates anti-dependencies with subrange precision", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

		SECTION("for execution commands") {
			// task_a writes the first half
			const auto tid_a = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf.get_access<mode::discard_write>(cgh, fixed<1, 1>({0, 50}));
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
			CHECK(computes_a.size() == 1);

			// task_b reads the first half
			const auto tid_b = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf.get_access<mode::read>(cgh, fixed<1, 1>({0, 50}));
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(computes_b.size() == 1);
			CHECK(inspector.has_dependency(*computes_b.cbegin(), *computes_a.cbegin()));

			// task_c writes the second half
			const auto tid_c = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf.get_access<mode::discard_write>(cgh, fixed<1, 1>({50, 50}));
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_c = inspector.get_commands(tid_c, node_id(0), command_type::TASK);
			CHECK(computes_c.size() == 1);

			// task_c should not have an anti-dependency onto task_b (or task_a)
			REQUIRE_FALSE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));
			REQUIRE_FALSE(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));

			maybe_print_graphs(ctx);
		}

		SECTION("for AWAIT_PUSH commands") {
			// task_a writes the full buffer
			const auto tid_a = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
			}));
			const auto master_node_tasks_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
			CHECK(master_node_tasks_a.size() == 1);

			// task_b only reads the second half
			const auto tid_b = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({50, 50}));
			}));
			const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(master_node_tasks_b.size() == 1);

			// task_c writes to the first half
			const auto tid_c = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{50}));
			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::TASK).size() == 2);

			// task_d reads the first half
			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 50}));
			}));

			// This should generate an AWAIT_PUSH command that does NOT have an anti-dependency onto task_b, only task_a
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			REQUIRE(inspector.has_dependency(*await_pushes.cbegin(), *master_node_tasks_a.cbegin()));
			REQUIRE_FALSE(inspector.has_dependency(*await_pushes.cbegin(), *master_node_tasks_b.cbegin()));

			maybe_print_graphs(ctx);
		}
	}

	TEST_CASE("graph_generator generates dependencies for execution commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		SECTION("if data is produced remotely") {
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 2);
			const auto tid_c = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
				buf_b.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			REQUIRE(await_pushes.size() == 2);
			const auto master_node_tasks = inspector.get_commands(tid_c, node_id(0), command_type::TASK);
			CHECK(master_node_tasks.size() == 1);
			REQUIRE(inspector.has_dependency(*master_node_tasks.cbegin(), *await_pushes.cbegin()));
			REQUIRE(inspector.has_dependency(*master_node_tasks.cbegin(), *(await_pushes.cbegin()++)));

			maybe_print_graphs(ctx);
		}

		SECTION("if data is produced remotely but already available from an earlier task") {
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);
			test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			const auto await_pushes_b = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			REQUIRE(await_pushes_b.size() == 1);
			const auto tid_c = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			// Assert that the number of AWAIT_PUSHes hasn't changed (i.e., none were added)
			REQUIRE(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);
			const auto master_node_tasks = inspector.get_commands(tid_c, node_id(0), command_type::TASK);
			REQUIRE(master_node_tasks.size() == 1);
			REQUIRE(inspector.has_dependency(*master_node_tasks.cbegin(), *await_pushes_b.cbegin()));

			maybe_print_graphs(ctx);
		}

		SECTION("if data is produced locally") {
			const auto tid_a = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
			const auto tid_b = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			const auto tid_c = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf_a.get_access<mode::read>(cgh, one_to_one<1>());
				        buf_b.get_access<mode::read>(cgh, one_to_one<1>());
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_c = inspector.get_commands(tid_c, node_id(0), command_type::TASK);
			REQUIRE(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));
			REQUIRE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));

			maybe_print_graphs(ctx);
		}
	}

	// This test case currently fails and exists for documentation purposes:
	//	- Having fixed write access to a buffer results in unclear semantics when it comes to splitting the task into chunks.
	//  - We could check for write access when using the built-in fixed range mapper and warn / throw.
	//		- But of course this is the easy case; the user could just as well write the same by hand.
	//
	// Really the most sensible thing to do might be to check whether chunks write to overlapping regions and abort if so.
	TEST_CASE("graph_generator handles fixed write access", "[graph_generator][command-graph][!shouldfail]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(3);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100), true);

		const auto tid_a = test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf.get_access<mode::write>(cgh, fixed<1, 1>({0, 100}));
		        },
		        cl::sycl::range<1>{100}));

		// Another solution could be to not split the task at all
		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);

		test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf.get_access<mode::read>(cgh, fixed<1, 1>({0, 100}));
		        },
		        cl::sycl::range<1>{100}));

		// Right now this generates a push command from the second node to the first, which also doesn't make much sense
		CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).empty());

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator allows chunks to require empty buffer ranges", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(3);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
			buf_a.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
			buf_b.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
		}));
		const auto tid_b = test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        // NOTE: It's important to construct range-mappers in such a way that passing the
			        // global size (during tdag generation) still returns the correct result!
			        buf_a.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				        switch(chnk.offset[0]) {
				        case 0: return chnk;
				        case 33: return chnk;
				        case 66: return {0, 0}; // Node 2 does not read buffer a
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
			        buf_b.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				        switch(chnk.offset[0]) {
				        case 0: return chnk;
				        case 33: return {0, 0}; // Node 1 does not read buffer b
				        case 66: return chnk;
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
		        },
		        cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 3);
		const auto computes_node1 = inspector.get_commands(tid_b, node_id(1), command_type::TASK);
		CHECK(computes_node1.size() == 1);
		const auto computes_node2 = inspector.get_commands(tid_b, node_id(2), command_type::TASK);
		CHECK(computes_node2.size() == 1);
		const auto await_pushes_node1 = inspector.get_commands(std::nullopt, node_id(1), command_type::AWAIT_PUSH);
		REQUIRE(await_pushes_node1.size() == 1);
		CHECK(inspector.has_dependency(*computes_node1.cbegin(), *await_pushes_node1.cbegin()));
		const auto await_pushes_node2 = inspector.get_commands(std::nullopt, node_id(2), command_type::AWAIT_PUSH);
		REQUIRE(await_pushes_node2.size() == 1);
		CHECK(inspector.has_dependency(*computes_node2.cbegin(), *await_pushes_node2.cbegin()));

		maybe_print_graphs(ctx);
	}

	// This is a highly constructed and unrealistic example, but we'd still like the behavior to be clearly defined.
	TEST_CASE("graph_generator generates anti-dependencies for execution commands that have a task-level true dependency", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		test_utils::cdag_test_context ctx(3);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		// Initialize both buffers
		const auto tid_a = test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>());
			        buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>());
		        },
		        cl::sycl::range<1>{100}));
		const auto computes_a_node1 = inspector.get_commands(tid_a, node_id(1), command_type::TASK);
		CHECK(computes_a_node1.size() == 1);
		const auto computes_a_node2 = inspector.get_commands(tid_a, node_id(2), command_type::TASK);
		CHECK(computes_a_node2.size() == 1);

		// Read from buf_a but overwrite buf_b
		// Importantly, we only read on the first worker node node, making it so the second worker does not have a true dependency on the previous task.
		const auto tid_b = test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf_a.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				        if(chnk.range[0] == 100) return chnk; // Return full chunk during tdag generation
				        switch(chnk.offset[0]) {
				        case 0: return {0, 0};
				        case 33: return chnk;
				        case 66: return {0, 0};
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
			        buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>());
		        },
		        cl::sycl::range<1>{100}));
		const auto computes_b_node1 = inspector.get_commands(tid_b, node_id(1), command_type::TASK);
		CHECK(computes_b_node1.size() == 1);
		const auto computes_b_node2 = inspector.get_commands(tid_b, node_id(2), command_type::TASK);
		CHECK(computes_b_node2.size() == 1);

		CHECK(inspector.has_dependency(*computes_b_node1.cbegin(), *computes_a_node1.cbegin()));
		REQUIRE(inspector.has_dependency(*computes_b_node2.cbegin(), *computes_a_node2.cbegin()));

		maybe_print_graphs(ctx);
	}

	// This test covers implementation details rather than graph-level constructs, however it's important that we deal with this gracefully.
	TEST_CASE("graph_generator correctly handles anti-dependency edge cases", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		test_utils::cdag_test_context ctx(1);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		// task_a writes both buffers
		const auto tid_a = test_utils::build_and_flush(ctx, test_utils::add_compute_task<class UKN(task_a)>(
		                                                        ctx.get_task_manager(),
		                                                        [&](handler& cgh) {
			                                                        buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>());
			                                                        buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>());
		                                                        },
		                                                        cl::sycl::range<1>{100}));

		task_id tid_b, tid_c;

		SECTION("correctly handles false anti-dependencies that consume a different buffer from the last writer") {
			// task_b reads buf_a
			tid_b = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_b)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, one_to_one<1>()); }, cl::sycl::range<1>(100)));

			// task_c writes buf_b, initially making task_b a potential anti-dependency (as it is a dependent of task_a). However, since the
			// two tasks don't actually touch the same buffers at all, nothing needs to be done.
			tid_c = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_c)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::read_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>(100)));
		}

		SECTION("does not consider anti-dependants of last writer as potential anti-dependencies") {
			// task_b writes buf_a, making task_a an anti-dependency
			tid_b = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_b)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>(100)));

			// task_c writes buf_b. Since task_b is not a true dependent of task_a, we don't consider it as a potential anti-dependency.
			tid_c = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_c)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>(100)));
		}

		// Even though we're testing for different conditions, we can use the same assertions here.

		const auto computes = inspector.get_commands(std::nullopt, std::nullopt, command_type::TASK);
		CHECK(computes.size() == 3);

		const auto computes_a = inspector.get_commands(tid_a, std::nullopt, command_type::TASK);
		CHECK(computes_a.size() == 1);
		const auto computes_b = inspector.get_commands(tid_b, std::nullopt, command_type::TASK);
		CHECK(computes_b.size() == 1);
		CHECK(inspector.has_dependency(*computes_b.cbegin(), *computes_a.cbegin()));
		const auto computes_c = inspector.get_commands(tid_c, std::nullopt, command_type::TASK);
		CHECK(computes_c.size() == 1);
		CHECK(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));

		REQUIRE_FALSE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator generates anti-dependencies onto the original producer if no consumer exists in between", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		test_utils::cdag_test_context ctx(3);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

		const auto tid_a = test_utils::build_and_flush(ctx, 3, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
		}));
		const auto master_node_tasks_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
		const auto tid_b = test_utils::build_and_flush(ctx, 3, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
		}));
		const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
		CHECK(master_node_tasks_b.size() == 1);
		REQUIRE(inspector.has_dependency(*master_node_tasks_b.cbegin(), *master_node_tasks_a.cbegin()));

		maybe_print_graphs(ctx);
	}

	// TODO: This test is too white-boxy. Come up with a different solution (ideally by simplifying the approach inside graph_generator).
	TEST_CASE("graph_generator generates anti-dependencies for execution commands onto PUSHes within the same task", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

		// NOTE: These two sections are handled by different mechanisms inside the graph_generator:
		//	   - The first is done by generate_anti_dependencies during the initial sweep.
		// 	   - The second is done by the "intra-task" loop at the end.
		// TODO DRY this up

		SECTION("if the PUSH is generated before the execution command") {
			test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        // Both nodes read the full buffer
				        buf.get_access<mode::read>(cgh, fixed<1, 1>({0, 100}));

				        // Only the worker also writes to the buffer
				        buf.get_access<mode::read_write>(cgh, [](chunk<1> chnk) -> subrange<1> {
					        if(chnk.range[0] == 100) return chnk; // Return full chunk during tdag generation
					        switch(chnk.offset[0]) {
					        case 0: return {0, 0};
					        case 50: return chnk;
					        default: CATCH_ERROR("Unexpected offset");
					        }
				        });
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 2);
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 2);
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 2);

			const auto pushes_master = inspector.get_commands(std::nullopt, node_id(0), command_type::PUSH);
			CHECK(pushes_master.size() == 1);
			const auto computes_master = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(computes_master.size() == 1);
			// Since the master node does not write to the buffer, there is no anti-dependency...
			REQUIRE_FALSE(inspector.has_dependency(*computes_master.cbegin(), *pushes_master.cbegin()));

			const auto pushes_node1 = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
			CHECK(pushes_node1.size() == 1);
			const auto computes_node1 = inspector.get_commands(tid_b, node_id(1), command_type::TASK);
			CHECK(computes_node1.size() == 1);
			// ...however for the worker, there is.
			REQUIRE(inspector.has_dependency(*computes_node1.cbegin(), *pushes_node1.cbegin()));

			maybe_print_graphs(ctx);
		}

		SECTION("if the PUSH is generated after the execution command") {
			test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        // Both nodes read the full buffer
				        buf.get_access<mode::read>(cgh, fixed<1, 1>({0, 100}));

				        // Only the master also writes to the buffer
				        buf.get_access<mode::read_write>(cgh, [](chunk<1> chnk) -> subrange<1> {
					        if(chnk.range[0] == 100) return chnk; // Return full chunk during tdag generation
					        switch(chnk.offset[0]) {
					        case 0: return chnk;
					        case 50: return {0, 0};
					        default: CATCH_ERROR("Unexpected offset");
					        }
				        });
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 2);
			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 2);
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 2);

			const auto pushes_node1 = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
			CHECK(pushes_node1.size() == 1);
			const auto computes_node1 = inspector.get_commands(tid_b, node_id(1), command_type::TASK);
			CHECK(computes_node1.size() == 1);
			// Since the worker node does not write to the buffer, there is no anti-dependency...
			REQUIRE_FALSE(inspector.has_dependency(*computes_node1.cbegin(), *pushes_node1.cbegin()));

			const auto pushes_master = inspector.get_commands(std::nullopt, node_id(0), command_type::PUSH);
			CHECK(pushes_master.size() == 1);
			const auto computes_master = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(computes_master.size() == 1);
			// ...however for the master, there is.
			REQUIRE(inspector.has_dependency(*computes_master.cbegin(), *pushes_master.cbegin()));

			maybe_print_graphs(ctx);
		}
	}

	TEST_CASE("graph_generator generates anti-dependencies for commands accessing host-initialized buffers", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(1);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		// We have two host initialized buffers
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100), true);
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100), true);

		// task_a reads from host-initialized buffer a
		const auto tid_a = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);
		const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
		CHECK(computes_a.size() == 1);

		// task_b writes to the same buffer a
		const auto tid_b = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 1);
		const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
		CHECK(computes_b.size() == 1);
		// task_b should have an anti-dependency onto task_a
		REQUIRE(inspector.has_dependency(*computes_b.cbegin(), *computes_a.cbegin()));

		// task_c writes to a different buffer b
		const auto tid_c = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_c)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one<1>()); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::TASK).size() == 1);
		const auto computes_c = inspector.get_commands(tid_c, node_id(0), command_type::TASK);
		CHECK(computes_c.size() == 1);
		// task_c should not have any anti-dependencies at all
		REQUIRE(inspector.get_dependency_count(*computes_c.cbegin()) == 0);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator generates pseudo-dependencies for collective commands on the same collective group") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();

		auto all_command_dependencies = [&](task_id depender, task_id dependency, auto predicate) {
			auto& cdag = ctx.get_command_graph();
			auto depender_commands = inspector.get_commands(depender, std::nullopt, command_type::TASK);
			auto dependency_commands = inspector.get_commands(dependency, std::nullopt, command_type::TASK);
			for(auto depender_cid : depender_commands) {
				auto depender_cmd = cdag.get(depender_cid);
				for(auto dependency_cid : dependency_commands) {
					auto dependency_cmd = cdag.get(dependency_cid);
					if(!predicate(depender_cmd, dependency_cmd)) return false;
				}
			}
			return true;
		};

		auto has_dependencies_on_same_node = [&](task_id depender, task_id dependency) {
			return all_command_dependencies(depender, dependency, [](auto depender_cmd, auto dependency_cmd) {
				return depender_cmd->has_dependency(dependency_cmd, dependency_kind::ORDER_DEP) == (depender_cmd->get_nid() == dependency_cmd->get_nid());
			});
		};

		auto has_no_dependencies = [&](task_id depender, task_id dependency) {
			return all_command_dependencies(
			    depender, dependency, [](auto depender_cmd, auto dependency_cmd) { return !depender_cmd->has_dependency(dependency_cmd); });
		};

		experimental::collective_group group;
		auto tid_master = test_utils::build_and_flush(ctx, 2, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler&) {}));
		auto tid_collective_implicit_1 =
		    test_utils::build_and_flush(ctx, 2, test_utils::add_host_task(ctx.get_task_manager(), experimental::collective, [&](handler&) {}));
		auto tid_collective_implicit_2 =
		    test_utils::build_and_flush(ctx, 2, test_utils::add_host_task(ctx.get_task_manager(), experimental::collective, [&](handler&) {}));
		auto tid_collective_explicit_1 =
		    test_utils::build_and_flush(ctx, 2, test_utils::add_host_task(ctx.get_task_manager(), experimental::collective(group), [&](handler&) {}));
		auto tid_collective_explicit_2 =
		    test_utils::build_and_flush(ctx, 2, test_utils::add_host_task(ctx.get_task_manager(), experimental::collective(group), [&](handler&) {}));

		CHECK(has_no_dependencies(tid_master, tid_collective_implicit_1));
		CHECK(has_no_dependencies(tid_master, tid_collective_implicit_2));
		CHECK(has_no_dependencies(tid_master, tid_collective_explicit_1));
		CHECK(has_no_dependencies(tid_master, tid_collective_explicit_2));

		CHECK(has_no_dependencies(tid_collective_implicit_1, tid_master));
		CHECK(has_no_dependencies(tid_collective_implicit_1, tid_collective_implicit_2));
		CHECK(has_no_dependencies(tid_collective_implicit_1, tid_collective_explicit_1));
		CHECK(has_no_dependencies(tid_collective_implicit_1, tid_collective_explicit_2));

		CHECK(has_no_dependencies(tid_collective_implicit_2, tid_master));
		CHECK(has_dependencies_on_same_node(tid_collective_implicit_2, tid_collective_implicit_1));
		CHECK(has_no_dependencies(tid_collective_implicit_2, tid_collective_explicit_1));
		CHECK(has_no_dependencies(tid_collective_implicit_2, tid_collective_explicit_2));

		CHECK(has_no_dependencies(tid_collective_explicit_1, tid_master));
		CHECK(has_no_dependencies(tid_collective_explicit_1, tid_collective_implicit_1));
		CHECK(has_no_dependencies(tid_collective_explicit_1, tid_collective_implicit_2));
		CHECK(has_no_dependencies(tid_collective_explicit_1, tid_collective_explicit_2));

		CHECK(has_no_dependencies(tid_collective_explicit_2, tid_master));
		CHECK(has_no_dependencies(tid_collective_explicit_2, tid_collective_implicit_1));
		CHECK(has_no_dependencies(tid_collective_explicit_2, tid_collective_implicit_2));
		CHECK(has_dependencies_on_same_node(tid_collective_explicit_2, tid_collective_explicit_1));

		maybe_print_graphs(ctx);
	}

} // namespace detail
} // namespace celerity
