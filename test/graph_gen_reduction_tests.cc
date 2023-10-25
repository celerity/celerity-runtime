#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "distributed_graph_generator_test_utils.h"

#include "distributed_graph_generator.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("distributed_graph_generator generates reduction command trees", "[distributed_distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(range<1>{1});

	const auto tid_initialize = dctx.device_compute<class UKN(initialize_1)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
	const auto tid_produce = dctx.device_compute<class UKN(produce_0)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_reduce =
	    dctx.device_compute<class UKN(reduce)>(test_range).read(buf0, acc::one_to_one{}).reduce(buf1, true /* include_current_buffer_value */).submit();
	const auto tid_consume = dctx.device_compute<class UKN(consume_1)>(test_range).read(buf1, acc::all{}).submit();

	CHECK(has_dependency(dctx.get_task_manager(), tid_reduce, tid_initialize));
	CHECK(has_dependency(dctx.get_task_manager(), tid_reduce, tid_produce));
	CHECK(has_dependency(dctx.get_task_manager(), tid_consume, tid_reduce));

	CHECK(dctx.query(node_id(0), tid_initialize).have_successors(dctx.query(node_id(0), tid_reduce)));
	CHECK(dctx.query(tid_produce).have_successors(dctx.query(tid_reduce)));
	// Reduction commands have exactly one dependency to the local parent execution_command and one dependency to an await_push command
	CHECK(dctx.query(tid_reduce).have_successors(dctx.query(command_type::reduction)));
	CHECK(dctx.query(command_type::await_push).have_successors(dctx.query(command_type::reduction)));
	// Each consume command has a reduction as its direct predecessor
	CHECK(dctx.query(command_type::reduction).have_successors(dctx.query(tid_consume), dependency_kind::true_dep));
}

TEST_CASE("single-node configurations do not generate reduction commands", "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(1);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(range<1>(1));

	dctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	dctx.device_compute<class UKN(consume)>(test_range).read(buf0, acc::all{}).submit();
	CHECK(dctx.query(command_type::reduction).empty());
}

TEST_CASE("discarding the reduction result from a execution_command will not generate a reduction command",
    "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(range<1>(1));

	const auto tid_reduce = dctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	const auto tid_discard = dctx.device_compute<class UKN(discard)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	// Now consume the result to check that the buffer was no longer in a pending reduction state (=> regression test)
	dctx.device_compute<class UKN(consume)>(test_range).read(buf0, acc::one_to_one{}).submit();
	CHECK(dctx.query(command_type::reduction).empty());
	// On node 0 (where buf0 is actually being overwritten) there should be an anti-dependency between the two
	CHECK(dctx.query(node_id(0), tid_reduce).have_successors(dctx.query(node_id(0), tid_discard), dependency_kind::anti_dep));
}

TEST_CASE("distributed_graph_generator does not generate multiple reduction commands for redundant requirements",
    "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(4);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(range<1>(1));

	dctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();

	SECTION("in a single task") {
		dctx.master_node_host_task().read(buf0, acc::all{}).read_write(buf0, acc::all{}).write(buf0, acc::all{}).submit();
		CHECK(dctx.query(command_type::reduction).count() == 1);
	}

	SECTION("across multiple tasks") {
		dctx.master_node_host_task().read(buf0, acc::all{}).submit();
		dctx.master_node_host_task().read_write(buf0, acc::all{}).submit();
		dctx.master_node_host_task().write(buf0, acc::all{}).submit();
		CHECK(dctx.query(command_type::reduction).count() == 1);
	}
}

TEST_CASE("distributed_graph_generator does not generate unnecessary anti-dependencies around reduction commands",
    "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(1);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(range<1>(1));

	const auto tid_reduce = dctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	const auto tid_host = dctx.master_node_host_task().read_write(buf0, acc::all{}).submit();
	CHECK_FALSE(dctx.query(tid_reduce).have_successors(dctx.query(tid_host), dependency_kind::anti_dep));
}

TEST_CASE(
    "commands overwriting a buffer generate anti-dependencies on preceding reduction pushes", "[distributed_graph_generator][command-graph][reductions]") {
	// regression test - this reproduces the essence of distr_tests "multiple chained reductions produce correct results"
	const size_t num_nodes = 2;
	dist_cdag_test_context dctx(num_nodes);

	auto buf0 = dctx.create_buffer(range<1>(1));

	dctx.device_compute<class UKN(reduce_a)>(range<1>(num_nodes)).reduce(buf0, false /* include_current_buffer_value */).submit();
	const auto tid_b = dctx.device_compute<class UKN(reduce_b)>(range<1>(num_nodes)).reduce(buf0, true /* include_current_buffer_value */).submit();

	const auto n1_pushes = dctx.query(node_id(1), command_type::push);
	const auto n1_tb_execs = dctx.query(node_id(1), tid_b);
	CHECK(n1_pushes.have_successors(n1_tb_execs, dependency_kind::anti_dep));
}

TEST_CASE("distributed_graph_generator forwards final reduction result if required by another node in a later task",
    "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(4);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(range<1>(1));

	dctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	dctx.master_node_host_task().read(buf0, acc::all{}).submit();
	dctx.collective_host_task().read(buf0, acc::all{}).submit();

	// There should only be a single reduction on node 0
	CHECK(dctx.query(command_type::reduction).count() == 1);
	// ...and the result is subsequently pushed to all other nodes
	CHECK(dctx.query(command_type::reduction).find_successors(command_type::push).count() == 3);
}

TEST_CASE("multiple chained reductions produce appropriate data transfers", "[distributed_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 2;
	dist_cdag_test_context dctx(num_nodes);

	auto buf0 = dctx.create_buffer(range<1>(1));

	dctx.device_compute<class UKN(reduce_a)>(range<1>(num_nodes)).reduce(buf0, false /* include_current_buffer_value */).submit();
	dctx.device_compute<class UKN(reduce_b)>(range<1>(num_nodes)).reduce(buf0, true /* include_current_buffer_value */).submit();
	const auto reduction1 = dctx.query(command_type::reduction);
	CHECK(reduction1.count() == 1);
	dctx.master_node_host_task().read(buf0, acc::all{}).submit();
	const auto reduction2 = dctx.query(command_type::reduction) - reduction1;
	CHECK(reduction2.count() == 1);

	// Both reductions are preceeded by await_pushes
	CHECK(reduction1.find_predecessors(command_type::await_push, dependency_kind::true_dep).count() == 1);
	CHECK(reduction2.find_predecessors(command_type::await_push, dependency_kind::true_dep).count() == 1);

	CHECK(dctx.query(node_id(0), command_type::push).empty());
	CHECK(dctx.query(node_id(1), command_type::push).count() == 2);
}

TEST_CASE("reductions that overwrite the previous buffer contents do not generate data transfers", "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {64};
	auto buf0 = dctx.create_buffer(range<1>(1));

	const auto only1 = [&](chunk<1> chnk) -> subrange<1> {
		if(chnk.range == chnk.global_size) return {0, 1};
		switch(chnk.offset[0]) {
		case 0: return {0, 0};
		case 32: return {0, 1};
		default: FAIL("Unexpected offset");
		}
		return {};
	};
	// Node 1 initializes the buffer, then both nodes reduce into it without keeping the data from task_a.
	dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, only1).submit();
	dctx.device_compute<class UKN(task_b)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	// This should not generate any data transfers.
	CHECK(dctx.query(command_type::push).empty());
	CHECK(dctx.query(command_type::await_push).empty());
}

TEST_CASE("nodes that do not own pending reduction don't include it in final reduction result", "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(3);
	auto buf0 = dctx.create_buffer(range<1>(1));

	dctx.device_compute<class UKN(reduce)>(nd_range<1>(64, 32)).reduce(buf0, false /* include_current_buffer_value */).submit();
	CHECK(dctx.query(command_type::execution).count() == 2);

	SECTION("remote reductions generate empty notification-only pushes") {
		dctx.master_node_host_task().read(buf0, acc::all{}).submit();
		const auto pushes = dctx.query(command_type::push);
		CHECK(pushes.count() == 2);
		auto push2 = pushes.get_raw(2);
		REQUIRE(push2.size() == 1);
		CHECK(dynamic_cast<const push_command*>(push2[0])->get_range().range.size() == 0);
		// The push only has a dependency on the initial epoch
		CHECK(dctx.query()
		          .find_all(node_id(2), command_type::epoch)
		          .have_successors(pushes.find_all(node_id(2)), dependency_kind::true_dep, dependency_origin::last_epoch));
	}

	SECTION("local reductions don't have a dependency on the last writer") {
		// The last writer in this case is the initial epoch
		dctx.device_compute<class UKN(consume)>(range<1>(96)).read(buf0, acc::all{}).submit();
		CHECK(dctx.query(node_id(2), command_type::reduction).find_predecessors(dependency_kind::true_dep).assert_count(1).have_type(command_type::await_push));
	}
}

TEST_CASE("reductions that do not include the current value generate anti-dependencies onto previous writer",
    "[distributed_graph_generator][command-graph][reductions]") {
	dist_cdag_test_context dctx(1);
	auto buf0 = dctx.create_buffer(range<1>(1));

	const auto tid_write = dctx.master_node_host_task().discard_write(buf0, acc::all{}).submit();
	const auto tid_reduce = dctx.device_compute<class UKN(reduce)>(range<1>(1)).reduce(buf0, false /* include_current_buffer_value */).submit();
	CHECK(dctx.query(tid_write).have_successors(dctx.query(tid_reduce), dependency_kind::anti_dep));
}

TEST_CASE("reduction commands anti-depend on their partial-result push commands", "[distributed_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 2;
	dist_cdag_test_context dctx(2);
	auto buf = dctx.create_buffer(range<1>(1));

	const auto tid_producer = dctx.device_compute(range<1>(num_nodes)).reduce(buf, false /* include_current_buffer_value */).submit();
	/* const auto tid_consumer = */ dctx.device_compute(range<1>(num_nodes)).read(buf, acc::all{}).submit();

	CHECK(dctx.query(tid_producer)
	          .assert_count_per_node(1)
	          .find_successors(command_type::push, dependency_kind::true_dep)
	          .assert_count_per_node(1)
	          .have_successors(dctx.query(command_type::reduction).assert_count_per_node(1), dependency_kind::anti_dep));
}

TEST_CASE("reduction in a single-node task does not generate a reduction command, but the result is await-pushed on other nodes",
    "[distributed_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 3;
	dist_cdag_test_context dctx(num_nodes);
	auto buf = dctx.create_buffer(range<1>(1));

	const auto tid_producer = dctx.device_compute(range<1>(1)).reduce(buf, false /* include_current_buffer_value */).submit();
	const auto tid_consumer = dctx.device_compute(range<1>(num_nodes)).read(buf, acc::all()).submit();

	CHECK(dctx.query(command_type::reduction).count() == 0);
	CHECK(dctx.query(tid_producer).assert_count(1).have_successors(dctx.query(node_id(0), command_type::push).assert_count(2)));
	for(node_id nid_await : {node_id(1), node_id(2)}) {
		CHECK(dctx.query(nid_await, command_type::await_push).assert_count(1).have_successors(dctx.query(nid_await, tid_consumer)));
	}
}
