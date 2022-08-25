#include <unordered_set>

#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	TEST_CASE("graph_generator generates reduction command trees", "[graph_generator][command-graph][reductions]") {
		using namespace cl::sycl::access;
		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(ctx);

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

			// Reduction commands have exactly one dependency to the local parent execution_command and one dependency to await_push_commands from all other
			// nodes
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

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE("single-node configurations do not generate reduction commands", "[graph_generator][command-graph][reductions]") {
		using namespace cl::sycl::access;
		size_t num_nodes = 1;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(ctx);

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
		CHECK(ctx.get_inspector().get_commands(std::nullopt, std::nullopt, command_type::reduction).empty());

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE(
	    "discarding the reduction result from a execution_command will not generate a reduction command", "[graph_generator][command-graph][reductions]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(ctx);

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_reduction)>(tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, false); }));

		test_utils::build_and_flush(ctx, num_nodes, test_utils::add_compute_task<class UKN(task_discard)>(tm, [&](handler& cgh) {
			buf_0.get_access<mode::discard_write>(cgh, fixed<1>({0, 1}));
		}));

		CHECK(ctx.get_inspector().get_commands(std::nullopt, std::nullopt, command_type::reduction).empty());

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator does not generate multiple reduction commands for redundant requirements", "[graph_generator][command-graph][reductions]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 4;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(ctx);

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(task_reduction)>(
		        tm, [&](handler& cgh) { test_utils::add_reduction(cgh, rm, buf_0, false); }, {num_nodes, 1}));

		test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
			buf_0.get_access<mode::read>(cgh, fixed<1>({0, 1}));
			buf_0.get_access<mode::read_write>(cgh, fixed<1>({0, 1}));
			buf_0.get_access<mode::write>(cgh, fixed<1>({0, 1}));
		}));

		CHECK(ctx.get_inspector().get_commands(std::nullopt, std::nullopt, command_type::reduction).size() == 1);

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator does not generate unnecessary anti-dependencies around reduction commands", "[graph_generator][command-graph][reductions]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(ctx);

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
				CHECK(!cdag.get(host_cid)->has_dependency(cdag.get(compute_cid), dependency_kind::anti_dep));
			}
		}

		test_utils::maybe_print_graphs(ctx);
	}

	TEST_CASE("graph_generator designates a reduction initializer command that does not require transfers", "[graph_generator][command-graph][reductions]") {
		using namespace cl::sycl::access;

		size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();
		auto& rm = ctx.get_reduction_manager();
		test_utils::mock_buffer_factory mbf(ctx);

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
			auto* ecmd = dynamic_cast<execution_command*>(cdag.get(cid));
			CHECK((ecmd->get_nid() == 0) == ecmd->is_reduction_initializer());
			have_initializers += ecmd->is_reduction_initializer();
		}
		CHECK(have_initializers == 1);

		test_utils::maybe_print_graphs(ctx);
	}

} // namespace detail
} // namespace celerity
