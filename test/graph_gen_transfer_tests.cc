#include <catch2/catch_test_macros.hpp>

#include "distributed_graph_generator_test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("distributed_graph_generator generates required data transfer commands", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(4);

	const range<1> test_range = {256};
	auto buf = dctx.create_buffer(test_range);

	const auto rm = [&](const chunk<1>& chnk) { return subrange(id(test_range[0] - chnk.offset[0] - chnk.range[0]), chnk.range); };
	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, rm).submit();
	CHECK(dctx.query(tid_a, command_type::execution).count_per_node() == 1);

	dctx.device_compute<class UKN(task_b)>(test_range).read(buf, acc::one_to_one{}).submit();
	CHECK(dctx.query(command_type::push).count() == 4);
	CHECK(dctx.query(command_type::push).count_per_node() == 1);
	CHECK(dctx.query(command_type::await_push).count() == 4);
	CHECK(dctx.query(command_type::await_push).count_per_node() == 1);
}

TEST_CASE("distributed_graph_generator doesn't generate data transfer commands for the same buffer and range more than once",
    "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);

	SECTION("when used in the same task") {
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		// Both of theses are consumer modes, meaning that both have a requirement on the buffer range produced in task_a
		dctx.master_node_host_task().read(buf0, acc::all{}).write(buf0, acc::all{}).submit();
		CHECK(dctx.query(command_type::push, node_id(1)).count() == 1);
		CHECK(dctx.query(command_type::await_push, node_id(0)).count() == 1);
	}

	SECTION("when used in the same task by different chunks on the same worker node") {
		// FIXME: Bring this back once we support oversubscription
		SKIP("Oversubscription NYI");
	}

	SECTION("when used in consecutive tasks") {
		auto buf1 = dctx.create_buffer(test_range);
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		dctx.master_node_host_task().read(buf0, acc::all{}).discard_write(buf1, acc::all{}).submit();
		CHECK(dctx.query(command_type::push, node_id(1)).count() == 1);
		CHECK(dctx.query(command_type::await_push, node_id(0)).count() == 1);
		dctx.master_node_host_task().read(buf0, acc::all{}).read(buf1, acc::all{}).submit();
		// Assert that the number of pushes / await_pushes hasn't changed
		CHECK(dctx.query(command_type::push, node_id(1)).count() == 1);
		CHECK(dctx.query(command_type::await_push, node_id(0)).count() == 1);
	}

	SECTION("when used in parallel tasks") {
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		dctx.master_node_host_task().read(buf0, acc::all{}).submit();
		CHECK(dctx.query(command_type::push, node_id(1)).count() == 1);
		CHECK(dctx.query(command_type::await_push, node_id(0)).count() == 1);
		dctx.master_node_host_task().read(buf0, acc::all{}).submit();
		// Assert that the number of pushes / await_pushes hasn't changed
		CHECK(dctx.query(command_type::push, node_id(1)).count() == 1);
		CHECK(dctx.query(command_type::await_push, node_id(0)).count() == 1);
	}
}

TEST_CASE("distributed_graph_generator uses original producer as source for push rather than building dependency chain",
    "[distributed_graph_generator][command-graph]") {
	const size_t num_nodes = 3;
	dist_cdag_test_context dctx(num_nodes);

	const range<1> test_range = {300};
	auto buf = dctx.create_buffer(test_range);

	dctx.master_node_host_task().discard_write(buf, acc::all{}).submit();

	SECTION("when distributing a single reading task across nodes") {
		dctx.device_compute<class UKN(task_b)>(test_range).read(buf, acc::one_to_one{}).submit();
	}

	SECTION("when distributing a single read-write task across nodes") {
		dctx.device_compute<class UKN(task_c)>(test_range).read_write(buf, acc::one_to_one{}).submit();
	}

	SECTION("when running multiple reading tasks on separate nodes") {
		auto full_range_for_single_node = [=](node_id node) {
			return [=](chunk<1> chnk) -> subrange<1> {
				if(chnk.range == chnk.global_size) return chnk;
				if(chnk.offset[0] == (test_range.size() / num_nodes) * node) { return {0, test_range}; }
				return {0, 0};
			};
		};
		dctx.device_compute<class UKN(task_d)>(test_range).read(buf, full_range_for_single_node(1)).submit();
		dctx.device_compute<class UKN(task_e)>(test_range).read(buf, full_range_for_single_node(2)).submit();
	}

	CHECK(dctx.query(node_id(0), command_type::push).count() == 2);
	CHECK(dctx.query(command_type::push).count() == 2);
	CHECK(dctx.query(node_id(1), command_type::await_push).count() == 1);
	CHECK(dctx.query(node_id(2), command_type::await_push).count() == 1);
}

// NOTE: This behavior changed between master/worker and distributed scheduling; we no longer consolidate pushes.
//       In part this is because of the way data is being tracked (on a per-command last writer basis),
//       however importantly it also enables better communication/computation overlapping in some cases.
//       This behavior may change again in the future!
TEST_CASE("distributed_graph_generator consolidates push commands for adjacent subranges", "[distributed_graph_generator][command-graph][!shouldfail]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(range<1>{test_range[0] / 2}).discard_write(buf, acc::one_to_one{}).submit();
	// Swap the two chunks so we write a contiguous range on node 1 across tasks a and b
	const auto swap_rm = [](chunk<1> chnk) -> subrange<1> {
		if(chnk.range == chnk.global_size) return chnk;
		switch(chnk.offset[0]) {
		case 64: return {96, 32};
		case 96: return {64, 32};
		default: FAIL("Unexpected offset");
		}
		return {};
	};
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(range<1>{test_range[0] / 2}, id<1>{test_range[0] / 2}).discard_write(buf, swap_rm).submit();
	dctx.master_node_host_task().read(buf, acc::all{}).submit();

	CHECK(dctx.query(command_type::push).count() == 1);
	CHECK(dctx.query(tid_a).have_successors(dctx.query(command_type::push)));
	CHECK(dctx.query(tid_b).have_successors(dctx.query(command_type::push)));
}

// Regression test: While we generate separate pushes for each last writer (see above), unless a last writer box gets fragmented
// further by subsequent writes, we should only ever generate a single push command. This was not the case, because we additionally have
// to check whether the data in question has already been (partially) replicated to the target node. Without a subsequent merging step,
// the number of pushes was effectively being dictated by the replication map, NOT the last writer.
TEST_CASE(
    "distributed_graph_generator does not unnecessarily divide push commands due to partial replication", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(3);

	const range<1> test_range = {96};
	auto buf = dctx.create_buffer(test_range);
	dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
	// Assuming standard 1D split
	CHECK(subrange_cast<1>(dynamic_cast<const execution_command&>(*dctx.query().get_raw(0)[0]).get_execution_range()) == subrange<1>{0, 32});
	CHECK(subrange_cast<1>(dynamic_cast<const execution_command&>(*dctx.query().get_raw(1)[0]).get_execution_range()) == subrange<1>{32, 32});
	CHECK(subrange_cast<1>(dynamic_cast<const execution_command&>(*dctx.query().get_raw(2)[0]).get_execution_range()) == subrange<1>{64, 32});
	// Require partial data from nodes 1 and 2
	dctx.master_node_host_task().read(buf, acc::fixed{subrange<1>{48, 32}}).submit();
	const auto pushes1 = dctx.query(command_type::push);
	CHECK(pushes1.count() == 2);
	// Now exchange data between nodes 1 and 2. Node 0 doesn't read anything.
	auto rm = [](const chunk<1>& chnk) {
		if(chnk.offset[0] + chnk.range[0] >= 64) return subrange<1>{32, 64};
		return subrange<1>{0, 0};
	};
	dctx.device_compute<class UKN(task_c)>(test_range).read(buf, rm).submit();
	const auto pushes2 = dctx.query(command_type::push) - pushes1;
	CHECK(pushes2.count() == 2);
}

TEST_CASE("distributed_graph_generator generates dependencies for push commands", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
	dctx.master_node_host_task().read(buf, acc::all{}).submit();
	CHECK(dctx.query(tid_a).have_successors(dctx.query(command_type::push)));
}

TEST_CASE("distributed_graph_generator generates anti-dependencies for await_push commands", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf = dctx.create_buffer(test_range, true);

	SECTION("if writing to region used by execution command") {
		// Node 0 starts by reading from buf (which is host-initialized)
		const auto tid_a = dctx.master_node_host_task().read(buf, acc::all{}).submit();
		// Then both nodes write to it
		dctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads it again, generating a transfer
		dctx.master_node_host_task().read(buf, acc::all{}).submit();
		// The await_push command has to wait until task_a is complete
		CHECK(dctx.query(node_id(0), tid_a).have_successors(dctx.query(node_id(0), command_type::await_push), dependency_kind::anti_dep));
	}

	SECTION("if writing to region used by push command") {
		// Both nodes write to buffer
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads and writes the buffer, generating a push
		dctx.master_node_host_task().read_write(buf, acc::all{}).submit();
		// Finally, both nodes read the buffer again, requiring an await_push on node 1
		// Note that in this example the await_push is never at risk of actually running concurrently with the first push to node 0, as they are effectively
		// in a distributed dependency relationship, however more complex examples could give rise to situations where this can happen.
		dctx.device_compute<class UKN(task_c)>(test_range).read(buf, acc::one_to_one{}).submit();
		CHECK(
		    dctx.query().find_all(node_id(1), command_type::push).have_successors(dctx.query(node_id(1), command_type::await_push), dependency_kind::anti_dep));
	}

	SECTION("if writing to region written by another await_push command") {
		// Both nodes write to buffer
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads the whole buffer
		const auto tid_b = dctx.master_node_host_task().read(buf, acc::all{}).submit();
		const auto first_await_push = dctx.query(command_type::await_push);
		CHECK(first_await_push.count() == 1);
		// Both nodes write it again
		dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads it again
		dctx.master_node_host_task().read(buf, acc::all{}).submit();
		const auto second_await_push = dctx.query(command_type::await_push) - first_await_push;
		// The first await push last wrote the data, but the anti-dependency is delegated to the reading successor task
		CHECK(dctx.query(tid_b).have_successors(second_await_push));
	}
}

TEST_CASE("distributed_graph_generator generates anti-dependencies with subrange precision", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf = dctx.create_buffer(test_range, true);

	SECTION("for execution commands") {
		// task_a writes the first half
		const auto tid_a = dctx.device_compute<class UKN(task_a)>(range<1>(test_range[0] / 2)).discard_write(buf, acc::one_to_one{}).submit();
		// task_b reads the first half
		const auto tid_b = dctx.device_compute<class UKN(task_b)>(range<1>(test_range[0] / 2)).read(buf, acc::one_to_one{}).submit();
		// task_c writes the second half
		const auto tid_c =
		    dctx.device_compute<class UKN(task_c)>(range<1>(test_range[0] / 2), id<1>(test_range[0] / 2)).discard_write(buf, acc::one_to_one{}).submit();
		// task_c should not have an anti-dependency onto task_b (or task_a)
		CHECK_FALSE(dctx.query(tid_a).have_successors(dctx.query(tid_c), dependency_kind::anti_dep));
		CHECK_FALSE(dctx.query(tid_b).have_successors(dctx.query(tid_c), dependency_kind::anti_dep));
	}

	SECTION("for await_push commands") {
		// task_a writes the full buffer
		const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// task_b reads the second half
		const auto tid_b = dctx.master_node_host_task().read(buf, acc::fixed<1>{{test_range[0] / 2, test_range[0] / 2}}).submit();
		// task_c writes to the first half
		dctx.device_compute<class UKN(task_c)>(range<1>(test_range[0] / 2)).discard_write(buf, acc::one_to_one{}).submit();
		// task_d reads the first half
		const auto tid_d = dctx.master_node_host_task().read(buf, acc::fixed<1>{{0, test_range[0] / 2}}).submit();
		// This should generate an await_push command that does NOT have an anti-dependency onto task_b, only task_a
		const auto await_push = dctx.query(tid_d).find_predecessors(command_type::await_push);
		CHECK(await_push.count() == 1);
		CHECK(dctx.query(node_id(0), tid_a).have_successors(await_push, dependency_kind::anti_dep));
		CHECK_FALSE(dctx.query(node_id(0), tid_b).have_successors(await_push, dependency_kind::anti_dep));
	}
}
