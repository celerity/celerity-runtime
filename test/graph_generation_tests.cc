#include "unit_test_suite_celerity.h"

#include <optional>
#include <set>
#include <unordered_set>

#include <catch2/catch.hpp>

#include <celerity.h>

#include "access_modes.h"

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
		cmds.insert(cdag.create<push_command>(0, 0, 0, 0, subrange<3>{}));
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
		auto pc = cdag.create<push_command>(0, 0, 0, 0, subrange<3>{});
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
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::read>(cgh, one_to_one{}); }, cl::sycl::range<1>{300}));

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
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
				        buf_a.get_access<mode::read>(cgh, fixed<1>{{0, 100}});
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, full_range));

		SECTION("when distributing a single reading task across nodes") {
			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(producer)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, one_to_one{}); }, full_range));
		}

		SECTION("when distributing a single read-write task across nodes") {
			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(producer)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, one_to_one{}); }, full_range));
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
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{64},
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
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{64},
		        cl::sycl::id<1>{0}));
		test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{32},
		        cl::sycl::id<1>{64}));
		test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_c)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{32},
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::read>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);

			const auto tid_b = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			CHECK(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);
			const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			CHECK(master_node_tasks_b.size() == 1);

			const auto tid_c = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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
				        buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 50}});
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
				        buf.get_access<mode::read>(cgh, fixed<1>{{0, 50}});
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
				        buf.get_access<mode::discard_write>(cgh, fixed<1>{{50, 50}});
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{50}));
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 2);
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
			const auto tid_b = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 1);
			const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
			const auto tid_c = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf_a.get_access<mode::read>(cgh, one_to_one{});
				        buf_b.get_access<mode::read>(cgh, one_to_one{});
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
			        buf.get_access<mode::write>(cgh, fixed<1>{{0, 100}});
		        },
		        cl::sycl::range<1>{100}));

		// Another solution could be to not split the task at all
		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);

		test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf.get_access<mode::read>(cgh, fixed<1>{{0, 100}});
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
				        case 34: return chnk;
				        case 67: return {0, 0}; // Node 2 does not read buffer a
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
			        buf_b.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				        switch(chnk.offset[0]) {
				        case 0: return chnk;
				        case 34: return {0, 0}; // Node 1 does not read buffer b
				        case 67: return chnk;
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
			        buf_a.get_access<mode::discard_write>(cgh, one_to_one{});
			        buf_b.get_access<mode::discard_write>(cgh, one_to_one{});
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
				        case 34: return chnk;
				        case 67: return {0, 0};
				        default: CATCH_ERROR("Unexpected offset");
				        }
			        });
			        buf_b.get_access<mode::discard_write>(cgh, one_to_one{});
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
			                                                        buf_a.get_access<mode::discard_write>(cgh, one_to_one{});
			                                                        buf_b.get_access<mode::discard_write>(cgh, one_to_one{});
		                                                        },
		                                                        cl::sycl::range<1>{100}));

		task_id tid_b, tid_c;

		SECTION("correctly handles false anti-dependencies that consume a different buffer from the last writer") {
			// task_b reads buf_a
			tid_b = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_b)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, one_to_one{}); }, cl::sycl::range<1>(100)));

			// task_c writes buf_b, initially making task_b a potential anti-dependency (as it is a dependent of task_a). However, since the
			// two tasks don't actually touch the same buffers at all, nothing needs to be done.
			tid_c = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_c)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::read_write>(cgh, one_to_one{}); }, cl::sycl::range<1>(100)));
		}

		SECTION("does not consider anti-dependants of last writer as potential anti-dependencies") {
			// task_b writes buf_a, making task_a an anti-dependency
			tid_b = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_b)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>(100)));

			// task_c writes buf_b. Since task_b is not a true dependent of task_a, we don't consider it as a potential anti-dependency.
			tid_c = test_utils::build_and_flush(
			    ctx, test_utils::add_compute_task<class UKN(task_c)>(
			             ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>(100)));
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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        // Both nodes read the full buffer
				        buf.get_access<mode::read>(cgh, fixed<1>{{0, 100}});

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
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        // Both nodes read the full buffer
				        buf.get_access<mode::read>(cgh, fixed<1>{{0, 100}});

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
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::TASK).size() == 1);
		const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::TASK);
		CHECK(computes_a.size() == 1);

		// task_b writes to the same buffer a
		const auto tid_b = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::TASK).size() == 1);
		const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::TASK);
		CHECK(computes_b.size() == 1);
		// task_b should have an anti-dependency onto task_a
		REQUIRE(inspector.has_dependency(*computes_b.cbegin(), *computes_a.cbegin()));

		// task_c writes to a different buffer b
		const auto tid_c = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_c)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

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

	TEST_CASE("graph_generator generates reduction command trees", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto range = cl::sycl::range<1>(64);
		auto buf_0 = mbf.create_buffer(range);
		auto buf_1 = mbf.create_buffer(cl::sycl::range<1>(1));

		const auto tid_initialize = test_utils::add_compute_task<class UKN(task_initialize)>(
		    tm, [&](handler& cgh) { buf_1.get_access<mode::discard_write>(cgh, one_to_one{}); }, range);
		test_utils::build_and_flush(ctx, num_nodes, tid_initialize);

		const auto tid_produce = test_utils::add_compute_task<class UKN(task_produce)>(
		    tm, [&](handler& cgh) { buf_0.get_access<mode::discard_write>(cgh, one_to_one{}); }, range);
		test_utils::build_and_flush(ctx, num_nodes, tid_produce);

		const auto tid_reduce = test_utils::add_compute_task<class UKN(task_reduce)>(
		    tm,
		    [&](handler& cgh) {
			    buf_0.get_access<mode::read>(cgh, one_to_one{});
			    test_utils::add_reduction(cgh, rm, buf_1, true /* include_current_buffer_value */);
		    },
		    range);
		test_utils::build_and_flush(ctx, num_nodes, tid_reduce);

		const auto tid_consume = test_utils::add_compute_task<class UKN(task_consume)>(
		    tm,
		    [&](handler& cgh) {
			    buf_1.get_access<mode::read>(cgh, fixed<1>({0, 1}));
		    },
		    range);
		test_utils::build_and_flush(ctx, num_nodes, tid_consume);

		CHECK(has_dependency(tm, tid_reduce, tid_initialize));
		CHECK(has_dependency(tm, tid_reduce, tid_produce));
		CHECK(has_dependency(tm, tid_consume, tid_reduce));

		auto consume_cmds = ctx.get_inspector().get_commands(tid_consume, std::nullopt, std::nullopt);
		CHECK(consume_cmds.size() == num_nodes);

		auto reduce_task_cmds = ctx.get_inspector().get_commands(tid_reduce, std::nullopt, std::nullopt);
		CHECK(reduce_task_cmds.size() == num_nodes);

		reduction_id rid = 0;
		for(auto cid : consume_cmds) {
			// Each consume command has a reduction as its direct predecessor
			auto deps = ctx.get_inspector().get_dependencies(cid);
			REQUIRE(deps.size() == 1);
			auto* rcmd = dynamic_cast<reduction_command*>(ctx.get_command_graph().get(deps[0]));
			REQUIRE(rcmd);
			if(rid) {
				CHECK(rcmd->get_rid() == rid);
			} else {
				rid = rcmd->get_rid();
			}

			// Reduction commands have exactly one dependency to the local parent task_command and one dependency to await_push_commands from all other nodes
			auto rdeps = ctx.get_inspector().get_dependencies(deps[0]);
			CHECK(rdeps.size() == num_nodes);
			bool have_local_dep = false;
			std::unordered_set<node_id> await_push_sources;
			for(auto rdcid : rdeps) {
				auto* rdcmd = ctx.get_command_graph().get(rdcid);
				if(auto* tdcmd = dynamic_cast<task_command*>(rdcmd)) {
					CHECK(!have_local_dep);
					have_local_dep = true;
				} else {
					auto* apdcmd = dynamic_cast<await_push_command*>(rdcmd);
					REQUIRE(apdcmd);
					CHECK(apdcmd->get_source()->get_rid() == rid);
					auto source_nid = apdcmd->get_source()->get_nid();
					CHECK(source_nid != rcmd->get_nid());
					CHECK(!await_push_sources.count(source_nid));
					await_push_sources.emplace(source_nid);
				}
			}
			CHECK(have_local_dep);
		}

		maybe_print_graphs(ctx);
	}

	TEST_CASE("single-node configurations do not generate reduction commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		size_t num_nodes = 1;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto range = cl::sycl::range<1>(64);
		auto buf_0 = mbf.create_buffer(range);

		const auto tid_reduce = test_utils::add_compute_task<class UKN(task_reduce)>(
		    tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, false /* include_current_buffer_value */); }, range);
		test_utils::build_and_flush(ctx, num_nodes, tid_reduce);

		const auto tid_consume = test_utils::add_compute_task<class UKN(task_consume)>(tm, [&](handler& cgh) {
			buf_0.get_access<mode::read>(cgh, fixed<1>({0, 1}));
		});
		test_utils::build_and_flush(ctx, num_nodes, tid_consume);

		CHECK(has_dependency(tm, tid_consume, tid_reduce));
		CHECK(ctx.get_inspector().get_commands(std::nullopt, std::nullopt, command_type::REDUCTION).empty());

		maybe_print_graphs(ctx);
	}

	TEST_CASE("discarding the reduction result from a task_command will not generate a reduction command", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_reduction)>(tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, false); }));

		test_utils::build_and_flush(ctx, num_nodes, test_utils::add_compute_task<class UKN(task_discard)>(tm, [&](handler& cgh) {
			buf_0.get_access<mode::discard_write>(cgh, fixed<1>({0, 1}));
		}));

		CHECK(ctx.get_inspector().get_commands(std::nullopt, std::nullopt, command_type::REDUCTION).empty());

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator does not generate multiple reduction commands for redundant requirements", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 4;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_reduction)>(
		        tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, false); }, {num_nodes, 1}));

		test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_0.get_access<mode::read>(cgh, fixed<1>({0, 1}));
			buf_0.get_access<mode::read_write>(cgh, fixed<1>({0, 1}));
			buf_0.get_access<mode::write>(cgh, fixed<1>({0, 1}));
		}));

		CHECK(ctx.get_inspector().get_commands(std::nullopt, std::nullopt, command_type::REDUCTION).size() == 1);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator does not generate unnecessary anti-dependencies around reduction commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		auto compute_tid = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_reduction)>(tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, false); }));

		auto host_tid = test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_0.get_access<mode::read_write>(cgh, fixed<1>({0, 1}));
		}));

		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();
		for(auto host_cid : inspector.get_commands(host_tid, std::nullopt, std::nullopt)) {
			for(auto compute_cid : inspector.get_commands(compute_tid, std::nullopt, std::nullopt)) {
				CHECK(!cdag.get(host_cid)->has_dependency(cdag.get(compute_cid), dependency_kind::ANTI_DEP));
			}
		}

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator designates a reduction initializer command that does not require transfers", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		test_utils::build_and_flush(
		    ctx, num_nodes, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_0.get_access<mode::discard_write>(cgh, all{}); }));

		auto compute_tid = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_reduction)>(tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, true); }));

		test_utils::build_and_flush(
		    ctx, num_nodes, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf_0.get_access<mode::read>(cgh, all{}); }));

		// Although there are two writing tasks
		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();
		size_t have_initializers = 0;
		for(auto cid : inspector.get_commands(compute_tid, std::nullopt, std::nullopt)) {
			auto* tcmd = dynamic_cast<task_command*>(cdag.get(cid));
			CHECK((tcmd->get_nid() == 0) == tcmd->is_reduction_initializer());
			have_initializers += tcmd->is_reduction_initializer();
		}
		CHECK(have_initializers == 1);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator respects task granularity when splitting", "[graph_generator]") {
		using namespace cl::sycl::access;

		const size_t num_nodes = 4;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();

		auto simple_1d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(simple_1d)>(
		        tm, [&](handler& cgh) {}, cl::sycl::range<1>{255}));

		auto simple_2d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(simple_2d)>(
		        tm, [&](handler& cgh) {}, cl::sycl::range<2>{255, 19}));

		auto simple_3d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(simple_3d)>(
		        tm, [&](handler& cgh) {}, cl::sycl::range<3>{255, 19, 31}));

		auto perfect_1d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_nd_range_compute_task<class UKN(perfect_1d)>(
		        tm, [&](handler& cgh) {}, celerity::nd_range<1>{{256}, {32}}));

		auto perfect_2d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_nd_range_compute_task<class UKN(perfect_2d)>(
		        tm, [&](handler& cgh) {}, celerity::nd_range<2>{{256, 19}, {32, 19}}));

		auto perfect_3d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_nd_range_compute_task<class UKN(perfect_3d)>(
		        tm, [&](handler& cgh) {}, celerity::nd_range<3>{{256, 19, 31}, {32, 19, 31}}));

		auto rebalance_1d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_nd_range_compute_task<class UKN(rebalance_1d)>(
		        tm, [&](handler& cgh) {}, celerity::nd_range<1>{{320}, {32}}));

		auto rebalance_2d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_nd_range_compute_task<class UKN(rebalance_2d)>(
		        tm, [&](handler& cgh) {}, celerity::nd_range<2>{{320, 19}, {32, 19}}));

		auto rebalance_3d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_nd_range_compute_task<class UKN(rebalance_3d)>(
		        tm, [&](handler& cgh) {}, celerity::nd_range<3>{{320, 19, 31}, {32, 19, 31}}));

		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();

		for(auto tid : {simple_1d, simple_2d, simple_3d}) {
			size_t total_range_dim0 = 0;
			for(auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				auto* tcmd = dynamic_cast<task_command*>(cdag.get(cid));
				auto range_dim0 = tcmd->get_execution_range().range[0];
				// Don't waste compute resources by creating over- or undersized chunks
				CHECK((range_dim0 == 63 || range_dim0 == 64));
				total_range_dim0 += range_dim0;
			}
			CHECK(total_range_dim0 == 255);
		}

		for(auto tid : {perfect_1d, perfect_2d, perfect_3d}) {
			for(auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				auto* tcmd = dynamic_cast<task_command*>(cdag.get(cid));
				CHECK(tcmd->get_execution_range().range[0] == 64); // Can be split perfectly
			}
		}

		for(auto tid : {rebalance_1d, rebalance_2d, rebalance_3d}) {
			size_t total_range_dim0 = 0;
			for(auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				auto* tcmd = dynamic_cast<task_command*>(cdag.get(cid));
				auto range_dim0 = tcmd->get_execution_range().range[0];
				// Don't waste compute resources by creating over- or undersized chunks
				CHECK((range_dim0 == 64 || range_dim0 == 96));
				total_range_dim0 += range_dim0;
			}
			CHECK(total_range_dim0 == 320);
		}

		maybe_print_graphs(ctx);
	}

	template <int Dims>
	class simple_task;

	template <int Dims>
	class nd_range_task;

	TEMPLATE_TEST_CASE_SIG("graph_generator does not create empty chunks", "[graph_generator]", ((int Dims), Dims), 1, 2, 3) {
		const size_t num_nodes = 3;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();

		range<Dims> task_range = zero_range;
		task_id tid = -1;

		SECTION("for simple tasks") {
			task_range = range_cast<Dims>(range<3>(2, 2, 2));
			tid = test_utils::build_and_flush(ctx, num_nodes,
			    test_utils::add_compute_task<simple_task<Dims>>(
			        tm, [&](handler& cgh) {}, task_range));
		}

		SECTION("for nd-range tasks") {
			task_range = range_cast<Dims>(range<3>(16, 2, 2));
			const auto local_range = range_cast<Dims>(range<3>(8, 1, 1));
			tid = test_utils::build_and_flush(ctx, num_nodes,
			    test_utils::add_nd_range_compute_task<nd_range_task<Dims>>(
			        tm, [&](handler& cgh) {}, nd_range<Dims>(task_range, local_range)));
		}

		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();

		const auto cmds = inspector.get_commands(tid, std::nullopt, std::nullopt);
		CHECK(cmds.size() == 2);
		for(auto tid : cmds) {
			auto* tcmd = dynamic_cast<task_command*>(cdag.get(tid));
			auto split_range = range_cast<3>(task_range);
			split_range[0] /= 2;
			CHECK(tcmd->get_execution_range().range == split_range);
		}
	}

	TEST_CASE("buffer accesses with empty ranges do not generate pushes or data-flow dependencies", "[graph_generator][command-graph]") {
		constexpr size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();

		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		const range<1> buf_range{16};
		auto buf = mbf.create_buffer(buf_range);

		const auto write_rm = [&](chunk<1> chnk) {
			const range<1> rg{chnk.offset[0] < 8 ? 8 - chnk.offset[0] : 0};
			return subrange<1>{chnk.offset, rg};
		};
		const auto write_tid = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(write)>(
		        tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, write_rm); }, buf_range));

		const auto read_rm = [&](chunk<1> chnk) {
			subrange<1> sr;
			sr.offset[0] = 4;
			sr.range[0] = chnk.offset[0] + chnk.range[0] > 4 ? chnk.offset[0] + chnk.range[0] - 4 : 0;
			return sr;
		};
		const auto read_tid = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(read)>(
		        tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, read_rm); }, buf_range / 2));

		maybe_print_graphs(ctx);

		CHECK(has_dependency(tm, read_tid, write_tid));

		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();

		const abstract_command* write_cmds[num_nodes];
		{
			const auto write_cids = inspector.get_commands(write_tid, std::nullopt, std::nullopt);
			REQUIRE(write_cids.size() == num_nodes); // naive split
			for(const auto cid : write_cids) {
				const auto cmd = cdag.get(cid);
				write_cmds[cmd->get_nid()] = cmd;
			}
		}

		const abstract_command* read_cmds[num_nodes];
		{
			const auto read_cids = inspector.get_commands(read_tid, std::nullopt, std::nullopt);
			REQUIRE(read_cids.size() == num_nodes); // naive split
			for(const auto cid : read_cids) {
				const auto cmd = cdag.get(cid);
				read_cmds[cmd->get_nid()] = cmd;
			}
		}

		CHECK(!inspector.has_dependency(read_cmds[0]->get_cid(), write_cmds[0]->get_cid()));
		CHECK(!inspector.has_dependency(read_cmds[1]->get_cid(), write_cmds[1]->get_cid()));

		const abstract_command* push_cmd;
		{
			const auto pushes = inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH);
			REQUIRE(pushes.size() == 1);
			push_cmd = cdag.get(*pushes.begin());
			CHECK(push_cmd->get_nid() == 0);
			const auto push_dependencies = push_cmd->get_dependencies();
			REQUIRE(std::distance(push_dependencies.begin(), push_dependencies.end()) == 1);
			CHECK(push_dependencies.begin()->node == write_cmds[0]);
		}

		const abstract_command* await_push_cmd;
		{
			const auto await_pushes = inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH);
			REQUIRE(await_pushes.size() == 1);
			await_push_cmd = cdag.get(*await_pushes.begin());
			CHECK(await_push_cmd->get_nid() == 1);
			const auto await_push_dependents = await_push_cmd->get_dependents();
			REQUIRE(std::distance(await_push_dependents.begin(), await_push_dependents.end()) == 1);
			CHECK(await_push_dependents.begin()->node == read_cmds[1]);
		}
	}

} // namespace detail
} // namespace celerity
