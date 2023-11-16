#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "distributed_graph_generator_test_utils.h"

#include "distributed_graph_generator.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("distributed_graph_generator respects task granularity when splitting", "[distributed_graph_generator]") {
	const size_t num_nodes = 4;
	dist_cdag_test_context dctx(num_nodes);

	const auto simple_1d = dctx.device_compute<class UKN(simple_1d)>(range<1>{255}).submit();
	const auto simple_2d = dctx.device_compute<class UKN(simple_2d)>(range<2>{255, 19}).submit();
	const auto simple_3d = dctx.device_compute<class UKN(simple_3d)>(range<3>{255, 19, 31}).submit();
	const auto perfect_1d = dctx.device_compute<class UKN(perfect_1d)>(celerity::nd_range<1>{{256}, {32}}).submit();
	const auto perfect_2d = dctx.device_compute<class UKN(perfect_2d)>(celerity::nd_range<2>{{256, 19}, {32, 19}}).submit();
	const auto perfect_3d = dctx.device_compute<class UKN(perfect_3d)>(celerity::nd_range<3>{{256, 19, 31}, {32, 19, 31}}).submit();
	const auto rebalance_1d = dctx.device_compute<class UKN(rebalance_1d)>(celerity::nd_range<1>{{320}, {32}}).submit();
	const auto rebalance_2d = dctx.device_compute<class UKN(rebalance_2d)>(celerity::nd_range<2>{{320, 19}, {32, 19}}).submit();
	const auto rebalance_3d = dctx.device_compute<class UKN(rebalance_3d)>(celerity::nd_range<3>{{320, 19, 31}, {32, 19, 31}}).submit();

	for(auto tid : {simple_1d, simple_2d, simple_3d}) {
		size_t total_range_dim0 = 0;
		for(const auto* cmd : dctx.query(tid).get_raw()) {
			const auto* ecmd = dynamic_cast<const execution_command*>(cmd);
			auto range_dim0 = ecmd->get_execution_range().range[0];
			// Don't waste compute resources by creating over- or undersized chunks
			REQUIRE_LOOP((range_dim0 == 63 || range_dim0 == 64));
			total_range_dim0 += range_dim0;
		}
		CHECK(total_range_dim0 == 255);
	}

	for(auto tid : {perfect_1d, perfect_2d, perfect_3d}) {
		for(const auto* cmd : dctx.query(tid).get_raw()) {
			const auto* ecmd = dynamic_cast<const execution_command*>(cmd);
			REQUIRE_LOOP(ecmd->get_execution_range().range[0] == 64); // Can be split perfectly
		}
	}

	for(auto tid : {rebalance_1d, rebalance_2d, rebalance_3d}) {
		size_t total_range_dim0 = 0;
		for(const auto* cmd : dctx.query(tid).get_raw()) {
			const auto* ecmd = dynamic_cast<const execution_command*>(cmd);
			const auto range_dim0 = ecmd->get_execution_range().range[0];
			// Don't waste compute resources by creating over- or undersized chunks
			REQUIRE_LOOP((range_dim0 == 64 || range_dim0 == 96));
			total_range_dim0 += range_dim0;
		}
		CHECK(total_range_dim0 == 320);
	}
}

TEST_CASE("distributed_graph_generator respects split constraints", "[distributed_graph_generator]") {
	const size_t num_nodes = 4;
	dist_cdag_test_context dctx(num_nodes);

	// Split constraints use the same underlying mechanisms as task granularity (tested above), so we'll keep this brief
	const auto tid_a = dctx.device_compute<class UKN(task)>(range<1>{128}).constrain_split(range<1>{64}).submit();
	REQUIRE(dctx.query(tid_a).count() == 2);
	CHECK(dynamic_cast<const execution_command*>(dctx.query(tid_a).get_raw(0)[0])->get_execution_range().range == range<3>{64, 1, 1});
	CHECK(dynamic_cast<const execution_command*>(dctx.query(tid_a).get_raw(1)[0])->get_execution_range().range == range<3>{64, 1, 1});

	// The more interesting aspect is that a constrained nd-range kernel uses the least common multiple of the two constraints
	const auto tid_b = dctx.device_compute<class UKN(task)>(nd_range<1>{{192}, {32}}).constrain_split(range<1>{3}).submit();
	REQUIRE(dctx.query(tid_b).count() == 2);
	CHECK(dynamic_cast<const execution_command*>(dctx.query(tid_b).get_raw(0)[0])->get_execution_range().range == range<3>{96, 1, 1});
	CHECK(dynamic_cast<const execution_command*>(dctx.query(tid_b).get_raw(1)[0])->get_execution_range().range == range<3>{96, 1, 1});
}

TEST_CASE("distributed_graph_generator creates 2-dimensional chunks when providing the split_2d hint", "[distributed_graph_generator][split][task-hints]") {
	const size_t num_nodes = 4;
	dist_cdag_test_context dctx(num_nodes);
	const auto tid_a = dctx.device_compute<class UKN(task)>(range<2>{128, 128}).hint(experimental::hints::split_2d{}).submit();
	REQUIRE(dctx.query(tid_a).count() == 4);
	for(node_id nid = 0; nid < 4; ++nid) {
		CHECK(dynamic_cast<const execution_command*>(dctx.query(tid_a).get_raw(nid)[0])->get_execution_range().range == range<3>{64, 64, 1});
	}
}

template <int Dims>
class simple_task;

template <int Dims>
class nd_range_task;

TEMPLATE_TEST_CASE_SIG("distributed_graph_generator does not create empty chunks", "[distributed_graph_generator]", ((int Dims), Dims), 1, 2, 3) {
	const size_t num_nodes = 3;
	dist_cdag_test_context dctx(num_nodes);

	range<Dims> task_range = zeros;
	task_id tid = -1;

	SECTION("for simple tasks") {
		task_range = truncate_range<Dims>({2, 2, 2});
		tid = dctx.device_compute<simple_task<Dims>>(task_range).submit();
	}

	SECTION("for nd-range tasks") {
		task_range = truncate_range<Dims>({16, 2, 2});
		const auto local_range = truncate_range<Dims>({8, 1, 1});
		tid = dctx.device_compute<nd_range_task<Dims>>(nd_range<Dims>(task_range, local_range)).submit();
	}

	const auto cmds = dctx.query(tid).get_raw();
	CHECK(cmds.size() == 2);
	for(const auto* cmd : cmds) {
		const auto* ecmd = dynamic_cast<const execution_command*>(cmd);
		auto split_range = range_cast<3>(task_range);
		split_range[0] /= 2; // We're assuming a 1D split here
		CHECK(ecmd->get_execution_range().range == split_range);
	}
}

TEST_CASE("buffer accesses with empty ranges do not generate pushes or data-flow dependencies", "[distributed_graph_generator][command-graph]") {
	constexpr size_t num_nodes = 2;
	dist_cdag_test_context dctx(num_nodes);

	const range<1> buf_range{16};
	auto buf = dctx.create_buffer(buf_range);

	const auto write_tid = dctx.device_compute<class UKN(write)>(buf_range).discard_write(buf, acc::one_to_one{}).submit();
	const auto read_rm = [&](chunk<1> chnk) {
		const auto chunk_end = chnk.offset[0] + chnk.range[0];
		const auto window_start = 4;
		const auto window_length = chunk_end > window_start ? chunk_end - window_start : 0;
		return subrange<1>{window_start, window_length};
	};
	const auto read_tid = dctx.device_compute<class UKN(read)>(buf_range / 2).read(buf, read_rm).submit();

	CHECK(has_dependency(dctx.get_task_manager(), read_tid, write_tid));

	CHECK_FALSE(dctx.query(node_id(0), write_tid).have_successors(dctx.query(node_id(0), read_tid)));
	CHECK_FALSE(dctx.query(node_id(1), write_tid).have_successors(dctx.query(node_id(1), read_tid)));

	CHECK(dctx.query(command_type::push).count() == 1);
	CHECK(dctx.query(command_type::push, node_id(0)).count() == 1);

	CHECK(dctx.query(command_type::await_push).count() == 1);
	CHECK(dctx.query(command_type::await_push, node_id(1)).count() == 1);
}
