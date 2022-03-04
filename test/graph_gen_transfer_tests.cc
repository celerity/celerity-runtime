#include "unit_test_suite_celerity.h"

#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_enforce.hpp> // for CATCH_ERROR

#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::fixed;
	using celerity::access::one_to_one;

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

		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 4);
		CHECK(inspector.get_commands(tid_a, node_id(1), command_type::EXECUTION).size() == 1);
		CHECK(inspector.get_commands(tid_a, node_id(2), command_type::EXECUTION).size() == 1);
		CHECK(inspector.get_commands(tid_a, node_id(3), command_type::EXECUTION).size() == 1);

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

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::EXECUTION).size() == 3);
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

			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 4);
			const auto computes = inspector.get_commands(tid_b, node_id(1), command_type::EXECUTION);
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

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::EXECUTION).size() == 4);
			// Assert that the number of PUSHes / AWAIT_PUSHes hasn't changed
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);

			maybe_print_graphs(ctx);
		}

		SECTION("when used in parallel tasks") {
			const auto tid_a = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);

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

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::EXECUTION).size() == 4);
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
		CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);

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
		CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 2);

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
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);
			const auto computes = inspector.get_commands(tid_a, node_id(1), command_type::EXECUTION);
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

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto master_node_tasks_a = inspector.get_commands(tid_a, node_id(0), command_type::EXECUTION);
			CHECK(master_node_tasks_a.size() == 1);

			// Meanwhile, the worker node writes to buf
			const auto tid_b = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 2);
			const auto computes_b_0 = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
			CHECK(computes_b_0.size() == 1);
			CHECK(inspector.has_dependency(*computes_b_0.cbegin(), *master_node_tasks_a.cbegin()));

			// Finally the master node reads again from buf, which is now the version written to by the worker node.
			const auto tid_c = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));

			CHECK(inspector.get_commands(std::nullopt, std::nullopt, command_type::AWAIT_PUSH).size() == 1);
			const auto await_pushes = inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH);
			CHECK(await_pushes.size() == 1);
			const auto master_node_tasks_c = inspector.get_commands(tid_c, node_id(0), command_type::EXECUTION);
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

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);

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

			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION).size() == 2);
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

			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 2);

			const auto tid_b = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({0, 100}));
			}));
			CHECK(inspector.get_commands(std::nullopt, node_id(0), command_type::AWAIT_PUSH).size() == 1);
			const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
			CHECK(master_node_tasks_b.size() == 1);

			const auto tid_c = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION).size() == 2);

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
			CHECK(inspector.get_commands(tid_a, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto computes_a = inspector.get_commands(tid_a, node_id(0), command_type::EXECUTION);
			CHECK(computes_a.size() == 1);

			// task_b reads the first half
			const auto tid_b = test_utils::build_and_flush(ctx, 1,
			    test_utils::add_compute_task<class UKN(task_b)>(
			        ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf.get_access<mode::read>(cgh, fixed<1>{{0, 50}});
			        },
			        cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_b, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto computes_b = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
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
			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION).size() == 1);
			const auto computes_c = inspector.get_commands(tid_c, node_id(0), command_type::EXECUTION);
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
			const auto master_node_tasks_a = inspector.get_commands(tid_a, node_id(0), command_type::EXECUTION);
			CHECK(master_node_tasks_a.size() == 1);

			// task_b only reads the second half
			const auto tid_b = test_utils::build_and_flush(ctx, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) {
				buf.get_access<mode::read>(cgh, fixed<1>({50, 50}));
			}));
			const auto master_node_tasks_b = inspector.get_commands(tid_b, node_id(0), command_type::EXECUTION);
			CHECK(master_node_tasks_b.size() == 1);

			// task_c writes to the first half
			const auto tid_c = test_utils::build_and_flush(ctx, 2,
			    test_utils::add_compute_task<class UKN(task_c)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, cl::sycl::range<1>{50}));
			CHECK(inspector.get_commands(tid_c, std::nullopt, command_type::EXECUTION).size() == 2);

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

} // namespace detail
} // namespace celerity
