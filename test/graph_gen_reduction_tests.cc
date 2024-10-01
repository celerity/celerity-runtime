#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "command_graph_generator_test_utils.h"

#include "command_graph_generator.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("command_graph_generator generates reduction command trees", "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(test_range);
	auto buf1 = cctx.create_buffer(range<1>{1});

	const auto tid_initialize = cctx.device_compute<class UKN(initialize_1)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
	const auto tid_produce = cctx.device_compute<class UKN(produce_0)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_reduce =
	    cctx.device_compute<class UKN(reduce)>(test_range).read(buf0, acc::one_to_one{}).reduce(buf1, true /* include_current_buffer_value */).submit();
	const auto tid_consume = cctx.device_compute<class UKN(consume_1)>(test_range).read(buf1, acc::all{}).submit();

	CHECK(has_dependency(cctx.get_task_manager(), tid_reduce, tid_initialize));
	CHECK(has_dependency(cctx.get_task_manager(), tid_reduce, tid_produce));
	CHECK(has_dependency(cctx.get_task_manager(), tid_consume, tid_reduce));

	CHECK(cctx.query(node_id(0), tid_initialize).have_successors(cctx.query(node_id(0), tid_reduce)));
	CHECK(cctx.query(tid_produce).have_successors(cctx.query(tid_reduce)));
	// Reduction commands have exactly one dependency to the local parent execution_command and one dependency to an await_push command
	CHECK(cctx.query(tid_reduce).have_successors(cctx.query(command_type::reduction)));
	CHECK(cctx.query(command_type::await_push).have_successors(cctx.query(command_type::reduction)));
	// Each consume command has a reduction as its direct predecessor
	CHECK(cctx.query(command_type::reduction).have_successors(cctx.query(tid_consume), dependency_kind::true_dep));
}

TEST_CASE("single-node configurations do not generate reduction commands", "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(1);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(range<1>(1));

	cctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	cctx.device_compute<class UKN(consume)>(test_range).read(buf0, acc::all{}).submit();
	CHECK(cctx.query(command_type::reduction).empty());
}

TEST_CASE(
    "discarding the reduction result from a execution_command will not generate a reduction command", "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(range<1>(1));

	const auto tid_reduce = cctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	const auto tid_discard = cctx.device_compute<class UKN(discard)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	// Now consume the result to check that the buffer was no longer in a pending reduction state (=> regression test)
	cctx.device_compute<class UKN(consume)>(test_range).read(buf0, acc::one_to_one{}).submit();
	CHECK(cctx.query(command_type::reduction).empty());
	// On node 0 (where buf0 is actually being overwritten) there should be an anti-dependency between the two
	CHECK(cctx.query(node_id(0), tid_reduce).have_successors(cctx.query(node_id(0), tid_discard), dependency_kind::anti_dep));
}

TEST_CASE("command_graph_generator does not generate multiple reduction commands for redundant requirements",
    "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(4);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(range<1>(1));

	cctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();

	SECTION("in a single task") {
		cctx.master_node_host_task().read(buf0, acc::all{}).read_write(buf0, acc::all{}).write(buf0, acc::all{}).submit();
		CHECK(cctx.query(command_type::reduction).count() == 1);
	}

	SECTION("across multiple tasks") {
		cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		cctx.master_node_host_task().read_write(buf0, acc::all{}).submit();
		cctx.master_node_host_task().write(buf0, acc::all{}).submit();
		CHECK(cctx.query(command_type::reduction).count() == 1);
	}
}

TEST_CASE("command_graph_generator does not generate unnecessary anti-dependencies around reduction commands",
    "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(1);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(range<1>(1));

	const auto tid_reduce = cctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	const auto tid_host = cctx.master_node_host_task().read_write(buf0, acc::all{}).submit();
	CHECK_FALSE(cctx.query(tid_reduce).have_successors(cctx.query(tid_host), dependency_kind::anti_dep));
}

TEST_CASE("commands overwriting a buffer generate anti-dependencies on preceding reduction pushes", "[command_graph_generator][command-graph][reductions]") {
	// regression test - this reproduces the essence of distr_tests "multiple chained reductions produce correct results"
	const size_t num_nodes = 2;
	cdag_test_context cctx(num_nodes);

	auto buf0 = cctx.create_buffer(range<1>(1));

	cctx.device_compute<class UKN(reduce_a)>(range<1>(num_nodes)).reduce(buf0, false /* include_current_buffer_value */).submit();
	const auto tid_b = cctx.device_compute<class UKN(reduce_b)>(range<1>(num_nodes)).reduce(buf0, true /* include_current_buffer_value */).submit();

	const auto n1_pushes = cctx.query(node_id(1), command_type::push);
	const auto n1_tb_execs = cctx.query(node_id(1), tid_b);
	CHECK(n1_pushes.have_successors(n1_tb_execs, dependency_kind::anti_dep));
}

TEST_CASE("command_graph_generator forwards final reduction result if required by another node in a later task",
    "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(4);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(range<1>(1));

	cctx.device_compute<class UKN(reduce)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	cctx.master_node_host_task().read(buf0, acc::all{}).submit();
	cctx.collective_host_task().read(buf0, acc::all{}).submit();

	// There should only be a single reduction on node 0
	CHECK(cctx.query(command_type::reduction).count() == 1);
	// ...and the result is subsequently pushed to all other nodes
	CHECK(cctx.query(command_type::reduction).find_successors(command_type::push).count() == 3);
}

TEST_CASE("multiple chained reductions produce appropriate data transfers", "[command_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 2;
	cdag_test_context cctx(num_nodes);

	auto buf0 = cctx.create_buffer(range<1>(1));

	cctx.device_compute<class UKN(reduce_a)>(range<1>(num_nodes)).reduce(buf0, false /* include_current_buffer_value */).submit();
	cctx.device_compute<class UKN(reduce_b)>(range<1>(num_nodes)).reduce(buf0, true /* include_current_buffer_value */).submit();
	const auto reduction1 = cctx.query(command_type::reduction);
	CHECK(reduction1.count() == 1);
	cctx.master_node_host_task().read(buf0, acc::all{}).submit();
	const auto reduction2 = cctx.query(command_type::reduction) - reduction1;
	CHECK(reduction2.count() == 1);

	// Both reductions are preceeded by await_pushes
	CHECK(reduction1.find_predecessors(command_type::await_push, dependency_kind::true_dep).count() == 1);
	CHECK(reduction2.find_predecessors(command_type::await_push, dependency_kind::true_dep).count() == 1);

	CHECK(cctx.query(node_id(0), command_type::push).empty());
	CHECK(cctx.query(node_id(1), command_type::push).count() == 2);
}

TEST_CASE("reductions that overwrite the previous buffer contents do not generate data transfers", "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {64};
	auto buf0 = cctx.create_buffer(range<1>(1));

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
	cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, only1).submit();
	cctx.device_compute<class UKN(task_b)>(test_range).reduce(buf0, false /* include_current_buffer_value */).submit();
	// This should not generate any data transfers.
	CHECK(cctx.query(command_type::push).empty());
	CHECK(cctx.query(command_type::await_push).empty());
}

TEST_CASE("nodes that do not own pending reduction don't include it in final reduction result", "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(3);
	auto buf0 = cctx.create_buffer(range<1>(1));

	cctx.device_compute<class UKN(reduce)>(nd_range<1>(64, 32)).reduce(buf0, false /* include_current_buffer_value */).submit();
	CHECK(cctx.query(command_type::execution).count() == 2);

	SECTION("remote reductions generate empty notification-only pushes") {
		cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		const auto pushes = cctx.query(command_type::push);
		CHECK(pushes.count() == 2);
		auto push2 = pushes.get_raw(2);
		REQUIRE(push2.size() == 1);
		CHECK(dynamic_cast<const push_command*>(push2[0])->get_range().range.size() == 0);
		// The push only has a dependency on the initial epoch
		CHECK(cctx.query()
		          .find_all(node_id(2), command_type::epoch)
		          .have_successors(pushes.find_all(node_id(2)), dependency_kind::true_dep, dependency_origin::last_epoch));
	}

	SECTION("local reductions don't have a dependency on the last writer") {
		// The last writer in this case is the initial epoch
		cctx.device_compute<class UKN(consume)>(range<1>(96)).read(buf0, acc::all{}).submit();
		CHECK(cctx.query(node_id(2), command_type::reduction).find_predecessors(dependency_kind::true_dep).assert_count(1).have_type(command_type::await_push));
	}
}

TEST_CASE("reductions that do not include the current value generate anti-dependencies onto previous writer",
    "[command_graph_generator][command-graph][reductions]") {
	cdag_test_context cctx(1);
	auto buf0 = cctx.create_buffer(range<1>(1));

	const auto tid_write = cctx.master_node_host_task().discard_write(buf0, acc::all{}).submit();
	const auto tid_reduce = cctx.device_compute<class UKN(reduce)>(range<1>(1)).reduce(buf0, false /* include_current_buffer_value */).submit();
	CHECK(cctx.query(tid_write).have_successors(cctx.query(tid_reduce), dependency_kind::anti_dep));
}

TEST_CASE("reduction commands anti-depend on their partial-result push commands", "[command_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 2;
	cdag_test_context cctx(2);
	auto buf = cctx.create_buffer(range<1>(1));

	const auto tid_producer = cctx.device_compute(range<1>(num_nodes)).reduce(buf, false /* include_current_buffer_value */).submit();
	/* const auto tid_consumer = */ cctx.device_compute(range<1>(num_nodes)).read(buf, acc::all{}).submit();

	CHECK(cctx.query(tid_producer)
	          .assert_count_per_node(1)
	          .find_successors(command_type::push, dependency_kind::true_dep)
	          .assert_count_per_node(1)
	          .have_successors(cctx.query(command_type::reduction).assert_count_per_node(1), dependency_kind::anti_dep));
}

TEST_CASE("reduction in a single-node task does not generate a reduction command, but the result is await-pushed on other nodes",
    "[command_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 3;
	cdag_test_context cctx(num_nodes);
	auto buf = cctx.create_buffer(range<1>(1));

	const auto tid_producer = cctx.device_compute(range<1>(1)).reduce(buf, false /* include_current_buffer_value */).submit();
	const auto tid_consumer = cctx.device_compute(range<1>(num_nodes)).read(buf, acc::all()).submit();

	CHECK(cctx.query(command_type::reduction).count() == 0);
	CHECK(cctx.query(tid_producer).assert_count(1).have_successors(cctx.query(node_id(0), command_type::push).assert_count(2)));
	for(node_id nid_await : {node_id(1), node_id(2)}) {
		CHECK(cctx.query(nid_await, command_type::await_push).assert_count(1).have_successors(cctx.query(nid_await, tid_consumer)));
	}
}

TEST_CASE("nodes that do not participate in reduction generate await-pushes when reading the result afterwards",
    "[command_graph_generator][command-graph][reductions]") {
	const size_t num_nodes = 4;
	cdag_test_context cctx(num_nodes);
	auto buf = cctx.create_buffer(range<1>(1));

	const auto tid_producer = cctx.device_compute(range<1>(num_nodes)).reduce(buf, false /* include_current_buffer_value */).submit();

	SECTION("when reducing on a single node") {
		const auto tid_reducer = cctx.device_compute(range<1>(1)).read(buf, acc::all()).submit();
		// Theres a push on nodes 1-3
		CHECK((cctx.query(command_type::push) - cctx.query(node_id(0))).count() == 3);

		const auto tid_consumer = cctx.device_compute(range<1>(num_nodes)).read(buf, acc::all()).submit();

		CHECK(cctx.query(command_type::reduction).count() == 1);
		CHECK(cctx.query(command_type::reduction).have_successors(cctx.query(tid_reducer)));
		// Node 0 pushes the result to all other nodes
		CHECK(cctx.query(command_type::reduction).have_successors(cctx.query(node_id(0), command_type::push).assert_count(3), dependency_kind::true_dep));
		// There's an await push on nodes 1-3 before the consumer task
		CHECK((cctx.query(command_type::await_push) - cctx.query(node_id(0)))
		          .assert_count(3)
		          .have_successors(cctx.query(tid_consumer) - cctx.query(node_id(0)), dependency_kind::true_dep));
		// No new pushes have been added on nodes 1-3
		CHECK((cctx.query(command_type::push) - cctx.query(node_id(0))).count() == 3);
	}

	// This is currently unsupported
	SECTION("when reducing on a subset of nodes") {
		CHECK_THROWS_WITH((cctx.device_compute(range<1>(2)).name("mytask").read(buf, acc::all()).submit()),
		    "Device kernel T2 \"mytask\" requires a reduction on B0 that is not performed on all nodes. This is currently not supported. Either "
		    "ensure that all nodes receive a chunk that reads from the buffer, or reduce the data on a single node.");
	}
}
