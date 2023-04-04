#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::one_to_one;

	TEST_CASE("graph_generator respects task granularity when splitting", "[graph_generator]") {
		using namespace cl::sycl::access;

		const size_t num_nodes = 4;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();

		auto simple_1d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(simple_1d)>(
		        tm, [&](handler& cgh) {}, range<1>{255}));

		auto simple_2d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(simple_2d)>(
		        tm, [&](handler& cgh) {}, range<2>{255, 19}));

		auto simple_3d = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(simple_3d)>(
		        tm, [&](handler& cgh) {}, range<3>{255, 19, 31}));

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
				auto* ecmd = dynamic_cast<execution_command*>(cdag.get(cid));
				auto range_dim0 = ecmd->get_execution_range().range[0];
				// Don't waste compute resources by creating over- or undersized chunks
				CHECK((range_dim0 == 63 || range_dim0 == 64));
				total_range_dim0 += range_dim0;
			}
			CHECK(total_range_dim0 == 255);
		}

		for(auto tid : {perfect_1d, perfect_2d, perfect_3d}) {
			for(auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				auto* ecmd = dynamic_cast<execution_command*>(cdag.get(cid));
				CHECK(ecmd->get_execution_range().range[0] == 64); // Can be split perfectly
			}
		}

		for(auto tid : {rebalance_1d, rebalance_2d, rebalance_3d}) {
			size_t total_range_dim0 = 0;
			for(auto cid : inspector.get_commands(tid, std::nullopt, std::nullopt)) {
				auto* ecmd = dynamic_cast<execution_command*>(cdag.get(cid));
				auto range_dim0 = ecmd->get_execution_range().range[0];
				// Don't waste compute resources by creating over- or undersized chunks
				CHECK((range_dim0 == 64 || range_dim0 == 96));
				total_range_dim0 += range_dim0;
			}
			CHECK(total_range_dim0 == 320);
		}

		test_utils::maybe_print_graphs(ctx);
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
		for(const auto cid : cmds) {
			const auto ecmd = cdag.get<execution_command>(cid);
			auto split_range = range_cast<3>(task_range);
			split_range[0] /= 2;
			CHECK(ecmd->get_execution_range().range == split_range);
		}
	}

	TEST_CASE("buffer accesses with empty ranges do not generate pushes or data-flow dependencies", "[graph_generator][command-graph]") {
		constexpr size_t num_nodes = 2;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		auto& ggen = ctx.get_graph_generator();

		test_utils::mock_buffer_factory mbf(ctx);
		const range<1> buf_range{16};
		auto buf = mbf.create_buffer(buf_range);

		const auto write_tid = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(write)>(
		        tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, one_to_one{}); }, buf_range));

		const auto read_rm = [&](chunk<1> chnk) {
			const auto chunk_end = chnk.offset[0] + chnk.range[0];
			const auto window_start = 4;
			const auto window_length = chunk_end > window_start ? chunk_end - window_start : 0;
			return subrange<1>{window_start, window_length};
		};
		const auto read_tid = test_utils::build_and_flush(ctx, num_nodes,
		    test_utils::add_compute_task<class UKN(read)>(
		        tm, [&](handler& cgh) { buf.get_access<access_mode::read>(cgh, read_rm); }, buf_range / 2));

		test_utils::maybe_print_graphs(ctx);

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

		{
			const auto pushes = inspector.get_commands(std::nullopt, std::nullopt, command_type::push);
			REQUIRE(pushes.size() == 1);
			const auto push_cmd = cdag.get<push_command>(*pushes.begin());
			CHECK(push_cmd->get_nid() == 0);
			CHECK(push_cmd->get_target() == 1);
		}

		{
			const auto await_pushes = inspector.get_commands(std::nullopt, std::nullopt, command_type::await_push);
			REQUIRE(await_pushes.size() == 1);
			const auto await_push_cmd = cdag.get<await_push_command>(*await_pushes.begin());
			CHECK(await_push_cmd->get_nid() == 1);
		}
	}

} // namespace detail
} // namespace celerity
