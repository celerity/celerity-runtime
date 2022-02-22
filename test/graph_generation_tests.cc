#include <optional>
#include <unordered_set>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/internal/catch_enforce.hpp> // for CATCH_ERROR

#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	TEST_CASE("command_graph keeps track of created commands", "[command_graph][command-graph]") {
		command_graph cdag;
		auto cmd0 = cdag.create<execution_command>(0, 0, subrange<3>{});
		auto cmd1 = cdag.create<execution_command>(0, 1, subrange<3>{});
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
		cmds.insert(cdag.create<execution_command>(0, 0, subrange<3>{}));
		cmds.insert(cdag.create<epoch_command>(0, task_manager::initial_epoch_task, epoch_action::none));
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

			auto t0 = cdag.create<execution_command>(node, 0, subrange<3>{});
			expected_front.insert(t0);
			REQUIRE(expected_front == cdag.get_execution_front(node));

			expected_front.insert(cdag.create<execution_command>(node, 1, subrange<3>{}));
			REQUIRE(expected_front == cdag.get_execution_front(node));

			expected_front.erase(t0);
			auto t2 = cdag.create<execution_command>(node, 2, subrange<3>{});
			expected_front.insert(t2);
			cdag.add_dependency(t2, t0, dependency_kind::TRUE_DEP, dependency_origin::dataflow);
			REQUIRE(expected_front == cdag.get_execution_front(node));
			return expected_front;
		};

		auto node_0_expected_front = build_testing_graph_on_node(0u);

		SECTION("for individual nodes") { build_testing_graph_on_node(1u); }

		REQUIRE(node_0_expected_front == cdag.get_execution_front(0));
	}

	TEST_CASE("isa<> RTTI helper correctly handles command hierarchies", "[rtti][command-graph]") {
		command_graph cdag;
		auto np = cdag.create<epoch_command>(0, task_manager::initial_epoch_task, epoch_action::none);
		REQUIRE(isa<abstract_command>(np));
		auto hec = cdag.create<execution_command>(0, 0, subrange<3>{});
		REQUIRE(isa<execution_command>(hec));
		auto pc = cdag.create<push_command>(0, 0, 0, 0, subrange<3>{});
		REQUIRE(isa<abstract_command>(pc));
		auto apc = cdag.create<await_push_command>(0, pc);
		REQUIRE(isa<abstract_command>(apc));
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
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 2);
			const auto tid_c = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, fixed<1>({0, 100}));
				buf_b.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			REQUIRE(await_pushes.size() == 2);
			const auto master_node_tasks = inspector.get_commands(tid_c, node_id(0), command_type::EXECUTION);
			CHECK(master_node_tasks.size() == 1);
			REQUIRE(inspector.has_dependency(*master_node_tasks.cbegin(), *await_pushes.cbegin()));
			REQUIRE(inspector.has_dependency(*master_node_tasks.cbegin(), *(await_pushes.cbegin()++)));

			test_utils::maybe_print_graphs(ctx);
		}

		SECTION("if data is produced remotely but already available from an earlier task") {
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);
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
			const auto master_node_tasks = inspector.get_commands(tid_c, node_id(0), command_type::EXECUTION);
			REQUIRE(master_node_tasks.size() == 1);
			REQUIRE(inspector.has_dependency(*master_node_tasks.cbegin(), *await_pushes_b.cbegin()));

			test_utils::maybe_print_graphs(ctx);
		}

		SECTION("if data is produced locally") {
			const auto tid_a = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::EXECUTION);
			const auto tid_b = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
			const auto tid_c = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf_a.get_access<mode::read>(cgh, one_to_one{});
				        buf_b.get_access<mode::read>(cgh, one_to_one{});
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto computes_c = inspector.get_commands(tid_c, node_id(0), command_type::EXECUTION);
			REQUIRE(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));
			REQUIRE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));

			test_utils::maybe_print_graphs(ctx);
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
		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 1);

		test_utils::build_and_flush(ctx, 3,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf.get_access<mode::read>(cgh, fixed<1>{{0, 100}});
		        },
		        cl::sycl::range<1>{100}));

		// Right now this generates a push command from the second node to the first, which also doesn't make much sense
		CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).empty());

		test_utils::maybe_print_graphs(ctx);
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
		const auto computes_a_node1 = inspector.get_commands(tid_a, node_id(1), command_type::EXECUTION);
		CHECK(computes_a_node1.size() == 1);
		const auto computes_a_node2 = inspector.get_commands(tid_a, node_id(2), command_type::EXECUTION);
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
		const auto computes_b_node1 = inspector.get_commands(tid_b, node_id(1), command_type::EXECUTION);
		CHECK(computes_b_node1.size() == 1);
		const auto computes_b_node2 = inspector.get_commands(tid_b, node_id(2), command_type::EXECUTION);
		CHECK(computes_b_node2.size() == 1);

		CHECK(inspector.has_dependency(*computes_b_node1.cbegin(), *computes_a_node1.cbegin()));
		REQUIRE(inspector.has_dependency(*computes_b_node2.cbegin(), *computes_a_node2.cbegin()));

		test_utils::maybe_print_graphs(ctx);
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

		const auto computes = inspector.get_commands(std::nullopt, std::nullopt, command_type::EXECUTION);
		CHECK(computes.size() == 3);

		const auto computes_a = inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION);
		CHECK(computes_a.size() == 1);
		const auto computes_b = inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION);
		CHECK(computes_b.size() == 1);
		CHECK(inspector.has_dependency(*computes_b.cbegin(), *computes_a.cbegin()));
		const auto computes_c = inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION);
		CHECK(computes_c.size() == 1);
		CHECK(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));

		REQUIRE_FALSE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));

		test_utils::maybe_print_graphs(ctx);
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
		const auto master_node_tasks_a = inspector.get_commands(tid_a, node_id(0), command_type::EXECUTION);
		const auto tid_b = test_utils::build_and_flush(ctx, 3, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 100}));
		}));
		const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
		CHECK(master_node_tasks_b.size() == 1);
		REQUIRE(inspector.has_dependency(*master_node_tasks_b.cbegin(), *master_node_tasks_a.cbegin()));

		test_utils::maybe_print_graphs(ctx);
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
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 2);

			const auto pushes_master = inspector.get_commands(std::nullopt, node_id(0), command_type::PUSH);
			CHECK(pushes_master.size() == 1);
			const auto computes_master = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
			CHECK(computes_master.size() == 1);
			// Since the master node does not write to the buffer, there is no anti-dependency...
			REQUIRE_FALSE(inspector.has_dependency(*computes_master.cbegin(), *pushes_master.cbegin()));

			const auto pushes_node1 = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
			CHECK(pushes_node1.size() == 1);
			const auto computes_node1 = inspector.get_commands(tid_b, node_id(1), command_type::EXECUTION);
			CHECK(computes_node1.size() == 1);
			// ...however for the worker, there is.
			REQUIRE(inspector.has_dependency(*computes_node1.cbegin(), *pushes_node1.cbegin()));

			test_utils::maybe_print_graphs(ctx);
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
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 2);

			const auto pushes_node1 = inspector.get_commands(std::nullopt, node_id(1), command_type::PUSH);
			CHECK(pushes_node1.size() == 1);
			const auto computes_node1 = inspector.get_commands(tid_b, node_id(1), command_type::EXECUTION);
			CHECK(computes_node1.size() == 1);
			// Since the worker node does not write to the buffer, there is no anti-dependency...
			REQUIRE_FALSE(inspector.has_dependency(*computes_node1.cbegin(), *pushes_node1.cbegin()));

			const auto pushes_master = inspector.get_commands(std::nullopt, node_id(0), command_type::PUSH);
			CHECK(pushes_master.size() == 1);
			const auto computes_master = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
			CHECK(computes_master.size() == 1);
			// ...however for the master, there is.
			REQUIRE(inspector.has_dependency(*computes_master.cbegin(), *pushes_master.cbegin()));

			test_utils::maybe_print_graphs(ctx);
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

		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 1);
		const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::EXECUTION);
		CHECK(computes_a.size() == 1);

		// task_b writes to the same buffer a
		const auto tid_b = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 1);
		const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
		CHECK(computes_b.size() == 1);
		// task_b should have an anti-dependency onto task_a
		REQUIRE(inspector.has_dependency(*computes_b.cbegin(), *computes_a.cbegin()));

		// task_c writes to a different buffer b
		const auto tid_c = test_utils::build_and_flush(ctx, 1,
		    test_utils::add_compute_task<class UKN(task_c)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION).size() == 1);
		const auto computes_c = inspector.get_commands(tid_c, node_id(0), command_type::EXECUTION);
		CHECK(computes_c.size() == 1);
		// task_c should not have any anti-dependencies at all
		REQUIRE(inspector.get_dependency_count(*computes_c.cbegin()) == 0);

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator generates pseudo-dependencies for collective commands on the same collective group", "[graph_generator][collectives]") {
		using namespace cl::sycl::access;

		test_utils::cdag_test_context ctx(2);
		auto& inspector = ctx.get_inspector();

		auto all_command_dependencies = [&](task_id depender, task_id dependency, auto predicate) {
			auto& cdag = ctx.get_command_graph();
			auto depender_commands = inspector.get_commands(depender, std::nullopt, command_type::EXECUTION);
			auto dependency_commands = inspector.get_commands(dependency, std::nullopt, command_type::EXECUTION);
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
				return depender_cmd->has_dependency(dependency_cmd, dependency_kind::TRUE_DEP) == (depender_cmd->get_nid() == dependency_cmd->get_nid());
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

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE("side effects generate appropriate command-dependencies", "[graph_generator][command-graph][side-effect]") {
		using order = experimental::side_effect_order;

		// Must be static for Catch2 GENERATE, which implicitly generates sections for each value and therefore cannot depend on runtime values
		static constexpr auto side_effect_orders = {order::sequential};

		constexpr size_t num_nodes = 2;
		const range<1> node_range{num_nodes};

		// TODO placeholder: complete with dependency types for other side effect orders
		const auto expected_dependencies = std::unordered_map<std::pair<order, order>, std::optional<dependency_kind>, pair_hash>{
		    {{order::sequential, order::sequential}, dependency_kind::TRUE_DEP},
		};

		const auto order_a = GENERATE(values(side_effect_orders));
		const auto order_b = GENERATE(values(side_effect_orders));
		CAPTURE(order_a);
		CAPTURE(order_b);

		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		test_utils::mock_host_object_factory mhof;

		auto ho_common = mhof.create_host_object(); // should generate dependencies
		auto ho_a = mhof.create_host_object();      // should NOT generate dependencies
		auto ho_b = mhof.create_host_object();      // -"-
		const auto tid_0 = test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, node_range, [&](handler& cgh) { //
			ho_a.add_side_effect(cgh, order_a);
		}));
		const auto tid_1 = test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, node_range, [&](handler& cgh) { //
			ho_common.add_side_effect(cgh, order_a);
			ho_b.add_side_effect(cgh, order_b);
		}));
		const auto tid_2 = test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, node_range, [&](handler& cgh) { //
			ho_common.add_side_effect(cgh, order_b);
		}));

		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();

		for(auto tid : {tid_0, tid_1}) {
			for(auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				const auto deps = cdag.get(cid)->get_dependencies();
				REQUIRE(std::distance(deps.begin(), deps.end()) == 1);
				CHECK(isa<epoch_command>(deps.front().node));
			}
		}

		const auto expected_2 = expected_dependencies.at({order_a, order_b});
		for(auto cid_2 : inspector.get_commands(tid_2, std::nullopt, std::nullopt)) {
			const auto deps_2 = cdag.get(cid_2)->get_dependencies();
			// This assumes no oversubscription in the split, adjust if necessary:
			CHECK(std::distance(deps_2.begin(), deps_2.end()) == expected_2.has_value());
			if(expected_2) {
				const auto& dep_tcmd = dynamic_cast<const task_command&>(*deps_2.front().node);
				CHECK(dep_tcmd.get_tid() == tid_1);
			}
		}
	}

	TEST_CASE("epochs serialize commands on every node", "[graph_generator][command-graph][epoch]") {
		using namespace cl::sycl::access;

		const size_t num_nodes = 2;
		const range<2> node_range{num_nodes, 1};
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();

		test_utils::mock_buffer_factory mbf(&tm, &ggen);

		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();

		const auto get_single_command = [&](const char* const task, const task_id tid, const node_id nid) {
			INFO(task);
			const auto set = inspector.get_commands(tid, nid, {});
			REQUIRE(set.size() == 1);
			return cdag.get(*set.begin());
		};

		const auto tid_a = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_a)>(
		        tm, [&](handler& cgh) {}, node_range));
		const auto tid_b = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_b)>(
		        tm, [&](handler& cgh) {}, node_range));

		const auto tid_epoch = test_utils::build_and_flush(ctx, num_nodes, tm.finish_epoch(epoch_action::none));

		for(node_id nid = 0; nid < num_nodes; ++nid) {
			CAPTURE(nid);

			const auto cmd_a = get_single_command("a", tid_a, nid);
			const auto a_deps = cmd_a->get_dependencies();
			REQUIRE(std::distance(a_deps.begin(), a_deps.end()) == 1);
			CHECK(a_deps.front().origin == dependency_origin::current_epoch);

			const auto cmd_b = get_single_command("b", tid_b, nid);
			const auto b_deps = cmd_b->get_dependencies();
			REQUIRE(std::distance(b_deps.begin(), b_deps.end()) == 1);
			CHECK(b_deps.front().origin == dependency_origin::current_epoch);

			const auto cmd_epoch = get_single_command("epoch", tid_epoch, nid);
			const auto epoch_deps = cmd_epoch->get_dependencies();
			CHECK(std::distance(epoch_deps.begin(), epoch_deps.end()) == 2);
			CHECK(inspector.has_dependency(cmd_epoch->get_cid(), cmd_a->get_cid()));
			CHECK(inspector.has_dependency(cmd_epoch->get_cid(), cmd_b->get_cid()));
		}

		auto buf = mbf.create_buffer(range<1>{1}, true /* host_initialized */);
		const auto tid_c = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_c)>(
		        tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); }, node_range));
		const auto tid_d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_d)>(
		        tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); }, node_range));

		for(node_id nid = 0; nid < num_nodes; ++nid) {
			CAPTURE(nid);

			const auto cmd_epoch = get_single_command("epoch", tid_epoch, nid);

			const auto cmd_c = get_single_command("c", tid_c, nid);
			const auto c_deps = cmd_c->get_dependencies();
			CHECK(std::distance(c_deps.begin(), c_deps.end()) == 1);
			CHECK(inspector.has_dependency(cmd_c->get_cid(), cmd_epoch->get_cid()));

			const auto cmd_d = get_single_command("d", tid_d, nid);
			const auto d_deps = cmd_d->get_dependencies();
			CHECK(std::distance(d_deps.begin(), d_deps.end()) == 2);
			CHECK(inspector.has_dependency(cmd_d->get_cid(), cmd_epoch->get_cid()));
			CHECK(inspector.has_dependency(cmd_d->get_cid(), cmd_c->get_cid()));
		}

		maybe_print_graphs(ctx);
	}

	TEST_CASE("a sequence of epochs without intermediate commands has defined behavior", "[graph_generator][command-graph][epoch]") {
		const size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();

		auto tid_before = task_manager::initial_epoch_task;
		for(const auto action : {epoch_action::barrier, epoch_action::shutdown}) {
			const auto tid = test_utils::build_and_flush(ctx, num_nodes, tm.finish_epoch(action));
			CAPTURE(tid_before, tid);
			for(const auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				CAPTURE(cid);
				const auto deps = cdag.get(cid)->get_dependencies();
				CHECK(std::distance(deps.begin(), deps.end()) == 1);
				for(const auto& d : deps) {
					CHECK(d.kind == dependency_kind::TRUE_DEP);
					CHECKED_IF(isa<task_command>(d.node)) { CHECK(static_cast<task_command*>(d.node)->get_tid() == tid_before); }
				}
			}
			tid_before = tid;
		}

		maybe_print_graphs(ctx);
	}

} // namespace detail
} // namespace celerity
