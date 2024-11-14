#include <catch2/catch_test_macros.hpp>

#include "command_graph_generator_test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("command_graph_generator generates required data transfer commands", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(4);

	SECTION("when using 1D buffers") {
		const range<1> test_range = {256};
		auto buf = cctx.create_buffer(test_range);

		const auto rm = [&](const chunk<1>& chnk) { return subrange(id(test_range[0] - chnk.offset[0] - chnk.range[0]), chnk.range); };
		cctx.device_compute(test_range).name("init").discard_write(buf, rm).submit();
		CHECK(cctx.query<execution_command_record>("init").count_per_node() == 1);

		cctx.device_compute(test_range).read(buf, acc::one_to_one{}).submit();
		CHECK(cctx.query<push_command_record>().total_count() == 4);
		CHECK(cctx.query<await_push_command_record>().total_count() == 4);

		CHECK(cctx.query<push_command_record>().on(0)->target_regions == push_regions<1>({{3, box<1>{192, 256}}}));
		CHECK(cctx.query<await_push_command_record>().on(0)->await_region == box_cast<3>(box<1>{0, 64}));
		CHECK(cctx.query<push_command_record>().on(1)->target_regions == push_regions<1>({{2, box<1>{128, 192}}}));
		CHECK(cctx.query<await_push_command_record>().on(1)->await_region == box_cast<3>(box<1>{64, 128}));
		CHECK(cctx.query<push_command_record>().on(2)->target_regions == push_regions<1>({{1, box<1>{64, 128}}}));
		CHECK(cctx.query<await_push_command_record>().on(2)->await_region == box_cast<3>(box<1>{128, 192}));
		CHECK(cctx.query<push_command_record>().on(3)->target_regions == push_regions<1>({{0, box<1>{0, 64}}}));
		CHECK(cctx.query<await_push_command_record>().on(3)->await_region == box_cast<3>(box<1>{192, 256}));
	}

	const auto run_2d_3d_test = [&](auto dims) {
		const auto test_range = truncate_range<dims.value>(range<3>{256, 256, 256});
		auto buf = cctx.create_buffer(test_range);

		// TODO: Revisit once we have split_3d{}
		cctx.device_compute(test_range).name("init").hint(experimental::hints::split_2d{}).discard_write(buf, acc::one_to_one{}).submit();
		CHECK(cctx.query<execution_command_record>("init").count_per_node() == 1);

		cctx.device_compute(test_range).read(buf, acc::all{}).submit();
		CHECK(cctx.query<push_command_record>().total_count() == 4);
		CHECK(cctx.query<await_push_command_record>().total_count() == 4);

		const auto push0 = truncate_box<dims.value>(box<3>{{0, 0, 0}, {128, 128, 256}});
		const auto await0 = region_difference(truncate_box<dims.value>(box<3>{{0, 0, 0}, {256, 256, 256}}), push0);
		CHECK(cctx.query<push_command_record>().on(0)->target_regions == push_regions<dims.value>({{1, push0}, {2, push0}, {3, push0}}));
		CHECK(cctx.query<await_push_command_record>().on(0)->await_region == region_cast<3>(await0));

		const auto push1 = truncate_box<dims.value>(box<3>{{0, 128, 0}, {128, 256, 256}});
		const auto await1 = region_difference(truncate_box<dims.value>(box<3>{{0, 0, 0}, {256, 256, 256}}), push1);
		CHECK(cctx.query<push_command_record>().on(1)->target_regions == push_regions<dims.value>({{0, push1}, {2, push1}, {3, push1}}));
		CHECK(cctx.query<await_push_command_record>().on(1)->await_region == region_cast<3>(await1));

		const auto push2 = truncate_box<dims.value>(box<3>{{128, 0, 0}, {256, 128, 256}});
		const auto await2 = region_difference(truncate_box<dims.value>(box<3>{{0, 0, 0}, {256, 256, 256}}), push2);
		CHECK(cctx.query<push_command_record>().on(2)->target_regions == push_regions<dims.value>({{0, push2}, {1, push2}, {3, push2}}));
		CHECK(cctx.query<await_push_command_record>().on(2)->await_region == region_cast<3>(await2));

		const auto push3 = truncate_box<dims.value>(box<3>{{128, 128, 0}, {256, 256, 256}});
		const auto await3 = region_difference(truncate_box<dims.value>(box<3>{{0, 0, 0}, {256, 256, 256}}), push3);
		CHECK(cctx.query<push_command_record>().on(3)->target_regions == push_regions<dims.value>({{0, push3}, {1, push3}, {2, push3}}));
		CHECK(cctx.query<await_push_command_record>().on(3)->await_region == region_cast<3>(await3));
	};

	SECTION("when using 2D buffers") { run_2d_3d_test(std::integral_constant<int, 2>{}); }
	SECTION("when using 3D buffers") { run_2d_3d_test(std::integral_constant<int, 3>{}); }
}

TEST_CASE("command_graph_generator generates a single push command for multiple recipients", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(4);

	const range<1> test_range = {256};
	auto buf = cctx.create_buffer(test_range);

	// Initialize buffer on node 0
	cctx.master_node_host_task().discard_write(buf, acc::all{}).submit();
	// Read buffer on all nodes
	cctx.device_compute(test_range).read(buf, acc::all{}).submit();

	CHECK(cctx.query<push_command_record>().total_count() == 1);
	CHECK(cctx.query<push_command_record>().on(0)->target_regions == push_regions<1>({{1, box<1>{0, 256}}, {2, box<1>{0, 256}}, {3, box<1>{0, 256}}}));
}

TEST_CASE("command_graph_generator generates a single push command per buffer and task", "[command_graph_generator][command-graph]") { //
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);
	auto buf1 = cctx.create_buffer(test_range);

	// Initialize buffers in two separate tasks (= multiple last writers)
	cctx.device_compute(test_range / 2).discard_write(buf0, acc::one_to_one{}).discard_write(buf1, acc::one_to_one{}).submit();
	cctx.device_compute(test_range / 2, id_cast<1>(test_range / 2)).discard_write(buf0, acc::one_to_one{}).discard_write(buf1, acc::one_to_one{}).submit();

	// Read entire buffer 0 and first half of buffer 1 on both nodes
	cctx.device_compute(test_range).read(buf0, acc::all{}).read(buf1, acc::fixed<1>{{0, 64}}).submit();
	// Read remainder of buffer 1 on both nodes, but split task into 4 chunks each
	cctx.set_test_chunk_multiplier(4);
	cctx.device_compute(test_range).read(buf1, acc::fixed<1>{{64, 128}}).submit();

	CHECK(cctx.query<push_command_record>().total_count() == 6);

	// Pushes for first task
	CHECK(cctx.query<push_command_record>(buf0.get_id()).on(0)[0]->target_regions == push_regions<1>({{1, region<1>{{box<1>{0, 32}, box<1>{64, 96}}}}}));
	CHECK(cctx.query<push_command_record>(buf1.get_id()).on(0)[0]->target_regions == push_regions<1>({{1, region<1>{{box<1>{0, 32}}}}}));
	CHECK(cctx.query<push_command_record>(buf0.get_id()).on(1)[0]->target_regions == push_regions<1>({{0, region<1>{{box<1>{32, 64}, box<1>{96, 128}}}}}));
	CHECK(cctx.query<push_command_record>(buf1.get_id()).on(1)[0]->target_regions == push_regions<1>({{0, region<1>{{box<1>{32, 64}}}}}));

	// Pushes for second task
	CHECK(cctx.query<push_command_record>(buf1.get_id()).on(0)[1]->target_regions == push_regions<1>({{1, region<1>{{box<1>{64, 96}}}}}));
	CHECK(cctx.query<push_command_record>(buf1.get_id()).on(1)[1]->target_regions == push_regions<1>({{0, region<1>{{box<1>{96, 128}}}}}));
}

TEST_CASE("command_graph_generator doesn't generate data transfer commands for the same buffer and range more than once",
    "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);

	SECTION("when used in the same task") {
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		// Both of theses are consumer modes, meaning that both have a requirement on the buffer range produced in task_a
		cctx.master_node_host_task().read(buf0, acc::all{}).write(buf0, acc::all{}).submit();
		CHECK(cctx.query<push_command_record>().on(1).count() == 1);
		CHECK(cctx.query<push_command_record>().on(1)->target_regions == push_regions<1>({{0, box<1>{64, 128}}}));
		CHECK(cctx.query<await_push_command_record>().on(0).count() == 1);
	}

	SECTION("when used in the same task by different chunks on the same worker node") {
		cctx.master_node_host_task().discard_write(buf0, acc::all{}).submit();
		cctx.set_test_chunk_multiplier(2);
		cctx.device_compute(test_range).read(buf0, acc::all{}).submit();
		CHECK(cctx.query<push_command_record>().total_count() == 1);
		CHECK(cctx.query<push_command_record>().on(0)->target_regions == push_regions<1>({{1, box<1>{0, 128}}}));
	}

	SECTION("when used in consecutive tasks") {
		auto buf1 = cctx.create_buffer(test_range);
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		cctx.master_node_host_task().read(buf0, acc::all{}).discard_write(buf1, acc::all{}).submit();
		CHECK(cctx.query<push_command_record>().on(1).count() == 1);
		CHECK(cctx.query<await_push_command_record>().on(0).count() == 1);
		cctx.master_node_host_task().read(buf0, acc::all{}).read(buf1, acc::all{}).submit();
		// Assert that the number of pushes / await_pushes hasn't changed
		CHECK(cctx.query<push_command_record>().on(1).count() == 1);
		CHECK(cctx.query<await_push_command_record>().on(0).count() == 1);
	}

	SECTION("when used in parallel tasks") {
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		CHECK(cctx.query<push_command_record>().on(1).count() == 1);
		CHECK(cctx.query<await_push_command_record>().on(0).count() == 1);
		cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		// Assert that the number of pushes / await_pushes hasn't changed
		CHECK(cctx.query<push_command_record>().on(1).count() == 1);
		CHECK(cctx.query<await_push_command_record>().on(0).count() == 1);
	}
}

TEST_CASE("only the original producer (owner) of data generates pushes", "[command_graph_generator][command-graph]") {
	const size_t num_nodes = 3;
	cdag_test_context cctx(num_nodes);

	const range<1> test_range = {300};
	auto buf = cctx.create_buffer(test_range);

	// Buffer is initialized on node 0
	cctx.master_node_host_task().discard_write(buf, acc::all{}).submit();

	SECTION("when distributing a single reading task across nodes") {
		cctx.device_compute(test_range).read(buf, acc::one_to_one{}).submit();
		CHECK(cctx.query<push_command_record>().total_count() == 1);
		CHECK(cctx.query<push_command_record>().on(0).count() == 1);
	}

	SECTION("when distributing a single read-write task across nodes") {
		cctx.device_compute(test_range).read_write(buf, acc::one_to_one{}).submit();
		CHECK(cctx.query<push_command_record>().total_count() == 1);
		CHECK(cctx.query<push_command_record>().on(0).count() == 1);
	}

	auto full_range_for_single_node = [=](node_id node) {
		return [=](chunk<1> chnk) -> subrange<1> {
			if(chnk.range == chnk.global_size) return chnk;
			if(chnk.offset[0] == (test_range.size() / num_nodes) * node) { return {0, test_range}; }
			return {0, 0};
		};
	};

	SECTION("when running multiple reading tasks on separate nodes") {
		SECTION("one direction") {
			cctx.device_compute(test_range).read(buf, full_range_for_single_node(1)).submit();
			cctx.device_compute(test_range).read(buf, full_range_for_single_node(2)).submit();
		}
		SECTION("other direction") {
			cctx.device_compute(test_range).read(buf, full_range_for_single_node(2)).submit();
			cctx.device_compute(test_range).read(buf, full_range_for_single_node(1)).submit();
		}
		CHECK(cctx.query<push_command_record>().total_count() == 2);
		CHECK(cctx.query<push_command_record>().on(0).count() == 2);
	}

	CHECK(cctx.query<await_push_command_record>().on(1).count() == 1);
	CHECK(cctx.query<await_push_command_record>().on(2).count() == 1);
}

// UPDATE: Disregard the comment below, as we now generate a single fat-push command per buffer and task. Keeping for additional coverage.
// Regression test: While we generate separate pushes for each last writer (see above), unless a last writer box gets fragmented
// further by subsequent writes, we should only ever generate a single push command. This was not the case, because we additionally have
// to check whether the data in question has already been (partially) replicated to the target node. Without a subsequent merging step,
// the number of pushes was effectively being dictated by the replication map, NOT the last writer.
TEST_CASE("command_graph_generator does not unnecessarily divide push commands due to partial replication", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(3);

	const range<1> test_range = {96};
	auto buf = cctx.create_buffer(test_range);
	cctx.device_compute(test_range).name("task a").discard_write(buf, acc::one_to_one{}).submit();
	// Assuming standard 1D split
	CHECK(subrange_cast<1>(cctx.query<execution_command_record>("task a").on(0)->execution_range) == subrange<1>{0, 32});
	CHECK(subrange_cast<1>(cctx.query<execution_command_record>("task a").on(1)->execution_range) == subrange<1>{32, 32});
	CHECK(subrange_cast<1>(cctx.query<execution_command_record>("task a").on(2)->execution_range) == subrange<1>{64, 32});
	// Require partial data from nodes 1 and 2
	cctx.master_node_host_task().read(buf, acc::fixed{subrange<1>{48, 32}}).submit();
	const auto pushes1 = cctx.query<push_command_record>();
	CHECK(pushes1.total_count() == 2);
	// Now exchange data between nodes 1 and 2. Node 0 doesn't read anything.
	auto rm = [](const chunk<1>& chnk) {
		if(chnk.offset[0] + chnk.range[0] >= 64) return subrange<1>{32, 64};
		return subrange<1>{0, 0};
	};
	cctx.device_compute<class UKN(task_c)>(test_range).read(buf, rm).submit();
	const auto pushes2 = difference_of(cctx.query<push_command_record>(), pushes1);
	CHECK(pushes2.total_count() == 2);
}

TEST_CASE("command_graph_generator generates dependencies for push commands", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf = cctx.create_buffer(test_range);

	cctx.device_compute(test_range / 2).name("init first half").discard_write(buf, acc::one_to_one{}).submit();
	cctx.device_compute(test_range / 2, id_cast<1>(test_range / 2)).name("init second half").discard_write(buf, acc::one_to_one{}).submit();
	cctx.master_node_host_task().read(buf, acc::all{}).submit();
	CHECK(cctx.query<push_command_record>().on(1).count() == 1);
	CHECK(cctx.query("init first half").successors().contains(cctx.query<push_command_record>()));
	CHECK(cctx.query("init second half").successors().contains(cctx.query<push_command_record>()));
}

TEST_CASE("command_graph_generator generates anti-dependencies for await_push commands", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf = cctx.create_buffer(test_range, true);

	SECTION("if writing to region used by execution command") {
		// Node 0 starts by reading from buf (which is host-initialized)
		const auto tid_a = cctx.master_node_host_task().read(buf, acc::all{}).submit();
		// Then both nodes write to it
		cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads it again, generating a transfer
		cctx.master_node_host_task().read(buf, acc::all{}).submit();
		// The await_push command has to wait until task_a is complete
		CHECK(cctx.query(tid_a).on(0).successors().contains(cctx.query<await_push_command_record>().on(0)));
	}

	SECTION("if writing to region used by push command") {
		// Both nodes write to buffer
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads and writes the buffer, generating a push
		cctx.master_node_host_task().read_write(buf, acc::all{}).submit();
		// Finally, both nodes read the buffer again, requiring an await_push on node 1
		// Note that in this example the await_push is never at risk of actually running concurrently with the first push to node 0, as they are effectively
		// in a distributed dependency relationship, however more complex examples could give rise to situations where this can happen.
		cctx.device_compute<class UKN(task_c)>(test_range).read(buf, acc::one_to_one{}).submit();
		CHECK(cctx.query<push_command_record>().on(1).successors().contains(cctx.query<await_push_command_record>().on(1)));
	}

	SECTION("if writing to region written by another await_push command") {
		// Both nodes write to buffer
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads the whole buffer
		const auto tid_b = cctx.master_node_host_task().read(buf, acc::all{}).submit();
		const auto first_await_push = cctx.query<await_push_command_record>();
		CHECK(first_await_push.total_count() == 1);
		// Both nodes write it again
		cctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// Node 0 reads it again
		cctx.master_node_host_task().read(buf, acc::all{}).submit();
		const auto second_await_push = difference_of(cctx.query<await_push_command_record>(), first_await_push);
		// The first await push last wrote the data, but the anti-dependency is delegated to the reading successor task
		CHECK(cctx.query(tid_b).successors().contains(second_await_push));
	}
}

TEST_CASE("command_graph_generator generates anti-dependencies with subrange precision", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf = cctx.create_buffer(test_range, true);

	SECTION("for execution commands") {
		// task_a writes the first half
		const auto tid_a = cctx.device_compute<class UKN(task_a)>(range<1>(test_range[0] / 2)).discard_write(buf, acc::one_to_one{}).submit();
		// task_b reads the first half
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(range<1>(test_range[0] / 2)).read(buf, acc::one_to_one{}).submit();
		// task_c writes the second half
		const auto tid_c =
		    cctx.device_compute<class UKN(task_c)>(range<1>(test_range[0] / 2), id<1>(test_range[0] / 2)).discard_write(buf, acc::one_to_one{}).submit();
		// task_c should not have an anti-dependency onto task_b (or task_a)
		CHECK(cctx.query(tid_a).is_concurrent_with(cctx.query(tid_c)));
		CHECK(cctx.query(tid_b).is_concurrent_with(cctx.query(tid_c)));
	}

	SECTION("for await_push commands") {
		// task_a writes the full buffer
		const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf, acc::one_to_one{}).submit();
		// task_b reads the second half
		const auto tid_b = cctx.master_node_host_task().read(buf, acc::fixed<1>{{test_range[0] / 2, test_range[0] / 2}}).submit();
		// task_c writes to the first half
		cctx.device_compute<class UKN(task_c)>(range<1>(test_range[0] / 2)).discard_write(buf, acc::one_to_one{}).submit();
		// task_d reads the first half
		const auto tid_d = cctx.master_node_host_task().read(buf, acc::fixed<1>{{0, test_range[0] / 2}}).submit();
		// This should generate an await_push command that does NOT have an anti-dependency onto task_b, only task_a
		const auto await_push = cctx.query(tid_d).predecessors().select_all<await_push_command_record>();
		CHECK(await_push.total_count() == 1);
		CHECK(cctx.query(tid_a).on(0).successors().contains(await_push.on(0)));
		CHECK(cctx.query(tid_b).on(0).is_concurrent_with(await_push.on(0)));
	}
}

TEST_CASE("5-point stencil program with 2D split does not exchange boundaries with diagonal neighbor nodes", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(4 /* num nodes */);

	constexpr range<2> buffer_size(256, 256);
	auto buf_a = cctx.create_buffer<2>(buffer_size);

	cctx.device_compute(range(buffer_size)).name("init").hint(experimental::hints::split_2d()).discard_write(buf_a, acc::one_to_one()).submit();
	cctx.device_compute(range(buffer_size))
	    .name("read")
	    .hint(experimental::hints::split_2d())
	    .read(buf_a, acc::neighborhood({1, 1}, neighborhood_shape::along_axes))
	    .submit();

	const std::vector<std::pair<node_id, node_id>> diagonals{{0, 3}, {1, 2}, {2, 1}, {3, 0}};
	for(const auto& [nid, diag] : diagonals) {
		CHECK(std::ranges::none_of(cctx.query<push_command_record>().on(nid)->target_regions, [diag = diag](auto& tr) { return tr.first == diag; }));
	}
}
