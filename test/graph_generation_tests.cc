#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "distributed_graph_generator_test_utils.h"

#include "distributed_graph_generator.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("command_graph keeps track of created commands", "[command_graph][command-graph]") {
	command_graph cdag;
	auto* const cmd0 = cdag.create<execution_command>(0, subrange<3>{});
	auto* const cmd1 = cdag.create<execution_command>(1, subrange<3>{});
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
	cmds.insert(cdag.create<execution_command>(0, subrange<3>{}));
	cmds.insert(cdag.create<epoch_command>(task_manager::initial_epoch_task, epoch_action::none));
	cmds.insert(cdag.create<push_command>(0, 0, 0, 0, subrange<3>{}));
	for(auto* cmd : cdag.all_commands()) {
		REQUIRE(cmds.find(cmd) != cmds.end());
		cmds.erase(cmd);
	}
	REQUIRE(cmds.empty());
}

TEST_CASE("command_graph keeps track of execution front", "[command_graph][command-graph]") {
	command_graph cdag;

	std::unordered_set<abstract_command*> expected_front;

	auto* const t0 = cdag.create<execution_command>(0, subrange<3>{});
	expected_front.insert(t0);
	REQUIRE(expected_front == cdag.get_execution_front());

	expected_front.insert(cdag.create<execution_command>(1, subrange<3>{}));
	REQUIRE(expected_front == cdag.get_execution_front());

	expected_front.erase(t0);
	auto* const t2 = cdag.create<execution_command>(2, subrange<3>{});
	expected_front.insert(t2);
	cdag.add_dependency(t2, t0, dependency_kind::true_dep, dependency_origin::dataflow);
	REQUIRE(expected_front == cdag.get_execution_front());
}

TEST_CASE("isa<> RTTI helper correctly handles command hierarchies", "[rtti][command-graph]") {
	command_graph cdag;
	auto* const np = cdag.create<epoch_command>(task_manager::initial_epoch_task, epoch_action::none);
	REQUIRE(utils::isa<abstract_command>(np));
	auto* const hec = cdag.create<execution_command>(0, subrange<3>{});
	REQUIRE(utils::isa<execution_command>(hec));
	auto* const pc = cdag.create<push_command>(0, 0, 0, 0, subrange<3>{});
	REQUIRE(utils::isa<abstract_command>(pc));
	auto* const apc = cdag.create<await_push_command>(0, 0, 0, region<3>{});
	REQUIRE(utils::isa<abstract_command>(apc));
}

TEST_CASE("distributed_graph_generator generates dependencies for execution commands", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	SECTION("if data is produced remotely") {
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		dctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		const auto tid_c = dctx.master_node_host_task().read(buf0, acc::all{}).read(buf1, acc::all{}).submit();
		CHECK(dctx.query(master_node_id, command_type::await_push).assert_count(2).have_successors(dctx.query(tid_c), dependency_kind::true_dep));
	}

	SECTION("if data is produced remotely but already available from an earlier task") {
		dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		dctx.master_node_host_task().read(buf0, acc::all{}).submit();
		const auto await_pushes = dctx.query(master_node_id, command_type::await_push).assert_count(1);

		const auto tid_c = dctx.master_node_host_task().read(buf0, acc::all{}).submit();
		// Assert that the number of await_pushes hasn't changed (i.e., none were added)
		CHECK(dctx.query(master_node_id, command_type::await_push).count() == await_pushes.count());
		// ...and the command for task c depends on the earlier await_push
		CHECK(await_pushes.have_successors(dctx.query(tid_c), dependency_kind::true_dep));
	}

	SECTION("if data is produced locally") {
		const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		const auto tid_c = dctx.device_compute<class UKN(task_c)>(test_range).read(buf0, acc::one_to_one{}).read(buf1, acc::one_to_one{}).submit();
		CHECK(dctx.query(tid_a).have_successors(dctx.query(tid_c)));
		CHECK(dctx.query(tid_b).have_successors(dctx.query(tid_c)));
	}
}

TEST_CASE("distributed_graph_generator builds dependencies to all local commands if a given range is produced by multiple",
    "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(1);

	const range<1> test_range = {96};
	const range<1> one_third = {test_range / 3};
	auto buf = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(one_third, id<1>{0 * one_third}).discard_write(buf, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(one_third, id<1>{1 * one_third}).discard_write(buf, acc::one_to_one{}).submit();
	const auto tid_c = dctx.device_compute<class UKN(task_c)>(one_third, id<1>{2 * one_third}).discard_write(buf, acc::one_to_one{}).submit();

	const auto tid_d = dctx.device_compute<class UKN(task_d)>(test_range).read(buf, acc::one_to_one{}).submit();
	CHECK(dctx.query(tid_a).have_successors(dctx.query(tid_d)));
	CHECK(dctx.query(tid_b).have_successors(dctx.query(tid_d)));
	CHECK(dctx.query(tid_c).have_successors(dctx.query(tid_d)));
}

// This test case currently fails and exists for documentation purposes:
//	- Having fixed write access to a buffer results in unclear semantics when it comes to splitting the task into chunks.
//  - We could check for write access when using the built-in fixed range mapper and warn / throw.
//		- But of course this is the easy case; the user could just as well write the same by hand.
//
// Really the most sensible thing to do might be to check whether chunks write to overlapping regions and abort if so.
TEST_CASE("distributed_graph_generator handles fixed write access", "[distributed_graph_generator][command-graph][!shouldfail]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::all{}).submit();
	// Another solution could be to not split the task at all
	CHECK(dctx.query(tid_a).count() == 1);

	dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::all{}).submit();
	// Right now this generates push commands, which also doesn't make much sense
	CHECK(dctx.query(command_type::push).empty());
}

// This is a highly constructed and unrealistic example, but we'd still like the behavior to be clearly defined.
TEST_CASE("distributed_graph_generator generates anti-dependencies for execution commands that have a task-level true dependency",
    "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	// Initialize both buffers
	const auto tid_a =
	    dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).discard_write(buf1, acc::one_to_one{}).submit();

	// Read from buf0 but overwrite buf1
	// Importantly, we only read on the first node, making it so the second node does not have a true dependency on the previous execution command.
	const auto node_1_writes = [=](chunk<1> chnk) -> subrange<1> {
		if(chnk.range[0] == test_range) return subrange<1>{64, 64};
		switch(chnk.offset[0]) {
		case 0: return {0, 0};
		case 64: return chnk;
		default: FAIL("Unexpected offset");
		}
		return {};
	};
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, node_1_writes).discard_write(buf1, acc::one_to_one{}).submit();

	CHECK(dctx.query(tid_a, node_id(0)).have_successors(dctx.query(tid_b, node_id(0)), dependency_kind::anti_dep));
	CHECK(dctx.query(tid_a, node_id(1)).have_successors(dctx.query(tid_b, node_id(1)), dependency_kind::true_dep));
}

TEST_CASE("distributed_graph_generator correctly handles anti-dependency edge cases", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(1);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	// task_a writes both buffers
	dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).discard_write(buf1, acc::one_to_one{}).submit();

	SECTION("correctly handles false anti-dependencies that consume a different buffer from the last writer") {
		// task_b reads buf0
		const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::one_to_one{}).submit();
		// task_c writes buf1, initially making task_b a potential anti-dependency (as it is a successor of task_a).
		const auto tid_c = dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		// However, since the two tasks don't actually touch the same buffers at all, nothing needs to be done.
		CHECK_FALSE(dctx.query(tid_b).have_successors(dctx.query(tid_c), dependency_kind::anti_dep));
	}

	SECTION("does not consider anti-successors of last writer as potential anti-dependencies") {
		// task_b writes buf0, making task_a an anti-dependency
		const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		// task_c writes buf1. Since task_b is not a true successor of task_a, we don't consider it as a potential anti-dependency.
		const auto tid_c = dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		CHECK_FALSE(dctx.query(tid_b).have_successors(dctx.query(tid_c), dependency_kind::anti_dep));
	}
}

TEST_CASE("distributed_graph_generator generates anti-dependencies onto the original producer if no consumer exists in between",
    "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	CHECK(dctx.query(tid_a).have_successors(dctx.query(tid_b), dependency_kind::anti_dep));
}

TEST_CASE("distributed_graph_generator generates anti-dependencies for execution commands onto pushes within the same task",
    "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	auto buf0 = dctx.create_buffer(test_range);

	const auto run_test = [&](const node_id writing_node, const node_id other_node) {
		const auto only_one_writes = [=](chunk<1> chnk) -> subrange<1> {
			if(chnk.range[0] == test_range) return subrange<1>{writing_node == 0 ? 0 : 64, 64};
			switch(chnk.offset[0]) {
			case 0: return writing_node == 0 ? chnk : subrange<1>{0, 0};
			case 64: return writing_node == 1 ? chnk : subrange<1>{0, 0};
			default: FAIL("Unexpected offset");
			}
			return {};
		};

		// Both nodes write parts of the buffer.
		const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();

		// Both nodes read the full buffer, but writing_node also writes to it.
		const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::all{}).discard_write(buf0, only_one_writes).submit();

		// Each node pushes data to the other.
		const auto push_w = dctx.query(writing_node, command_type::push);
		CHECK(push_w.count() == 1);
		const auto push_o = dctx.query(other_node, command_type::push);
		CHECK(push_o.count() == 1);

		// Since other_node does not write to the buffer, there is no anti-dependency...
		CHECK_FALSE(push_o.have_successors(dctx.query(other_node, tid_b), dependency_kind::anti_dep));
		// ...however for node 1, there is.
		CHECK(push_w.have_successors(dctx.query(writing_node, tid_b), dependency_kind::anti_dep));
	};

	// NOTE: These two sections are handled by different mechanisms inside the distributed_graph_generator:
	//	   - The first is done by generate_anti_dependencies during the initial sweep.
	// 	   - The second is done by the "intra-task" loop at the end.
	// NOTE: This ("before" / "after") assumes that chunks are processed in node id order
	SECTION("if the push is generated before the execution command") { run_test(1, 0); }
	SECTION("if the push is generated after the execution command") { run_test(0, 1); }
}

TEST_CASE(
    "distributed_graph_generator generates anti-dependencies for commands accessing host-initialized buffers", "[distributed_graph_generator][command-graph]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};
	// We have two host initialized buffers
	auto buf0 = dctx.create_buffer(test_range, true);
	auto buf1 = dctx.create_buffer(test_range, true);

	// task_a reads from host-initialized buffer 0
	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).read(buf0, acc::one_to_one{}).submit();

	// task_b writes to the same buffer 0
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	// task_b should have an anti-dependency onto task_a
	CHECK(dctx.query(tid_a).have_successors(dctx.query(tid_b)));

	// task_c writes to a different buffer 1
	const auto tid_c = dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
	// task_c should not have any anti-dependencies at all
	CHECK(dctx.query(tid_c).find_predecessors(dependency_kind::anti_dep).empty());
}

TEST_CASE("distributed_graph_generator generates pseudo-dependencies for collective commands on the same collective group",
    "[distributed_graph_generator][collectives]") {
	dist_cdag_test_context dctx(2);

	experimental::collective_group group;
	const auto tid_a = dctx.master_node_host_task().submit();
	const auto tid_collective_implicit_1 = dctx.collective_host_task().submit();
	const auto tid_collective_implicit_2 = dctx.collective_host_task().submit();
	const auto tid_collective_explicit_1 = dctx.collective_host_task(group).submit();
	const auto tid_collective_explicit_2 = dctx.collective_host_task(group).submit();

	CHECK_FALSE(dctx.query(tid_collective_implicit_1).have_successors(dctx.query(tid_a)));
	CHECK_FALSE(dctx.query(tid_collective_implicit_2).have_successors(dctx.query(tid_a)));
	CHECK_FALSE(dctx.query(tid_collective_explicit_1).have_successors(dctx.query(tid_a)));
	CHECK_FALSE(dctx.query(tid_collective_explicit_2).have_successors(dctx.query(tid_a)));

	CHECK_FALSE(dctx.query(tid_a).assert_count(1).have_successors(dctx.query(master_node_id, tid_collective_implicit_1)));
	CHECK_FALSE(dctx.query(tid_collective_implicit_2).have_successors(dctx.query(tid_collective_implicit_1)));
	CHECK_FALSE(dctx.query(tid_collective_explicit_1).have_successors(dctx.query(tid_collective_implicit_1)));
	CHECK_FALSE(dctx.query(tid_collective_explicit_2).have_successors(dctx.query(tid_collective_implicit_1)));

	CHECK_FALSE(dctx.query(tid_a).assert_count(1).have_successors(dctx.query(master_node_id, tid_collective_implicit_2)));
	CHECK(dctx.query()
	          .find_all(tid_collective_implicit_1)
	          .have_successors(dctx.query(tid_collective_implicit_2), dependency_kind::true_dep, dependency_origin::collective_group_serialization));
	CHECK_FALSE(dctx.query(tid_collective_explicit_1).have_successors(dctx.query(tid_collective_implicit_2)));
	CHECK_FALSE(dctx.query(tid_collective_explicit_2).have_successors(dctx.query(tid_collective_implicit_2)));

	CHECK_FALSE(dctx.query(tid_a).assert_count(1).have_successors(dctx.query(master_node_id, tid_collective_explicit_1)));
	CHECK_FALSE(dctx.query(tid_collective_implicit_1).have_successors(dctx.query(tid_collective_explicit_1)));
	CHECK_FALSE(dctx.query(tid_collective_implicit_2).have_successors(dctx.query(tid_collective_explicit_1)));
	CHECK_FALSE(dctx.query(tid_collective_explicit_1).have_successors(dctx.query(tid_collective_explicit_1)));

	CHECK_FALSE(dctx.query(tid_a).assert_count(1).have_successors(dctx.query(master_node_id, tid_collective_explicit_2)));
	CHECK_FALSE(dctx.query(tid_collective_implicit_1).have_successors(dctx.query(tid_collective_explicit_2)));
	CHECK_FALSE(dctx.query(tid_collective_implicit_2).have_successors(dctx.query(tid_collective_explicit_2)));
	CHECK(dctx.query()
	          .find_all(tid_collective_explicit_1)
	          .have_successors(dctx.query(tid_collective_explicit_2), dependency_kind::true_dep, dependency_origin::collective_group_serialization));
}

TEST_CASE("side effects generate appropriate command-dependencies", "[distributed_graph_generator][command-graph][side-effect]") {
	using order = experimental::side_effect_order;

	// Must be static for Catch2 GENERATE, which implicitly generates sections for each value and therefore cannot depend on runtime values
	static constexpr auto side_effect_orders = {order::sequential};

	constexpr size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	// TODO placeholder: complete with dependency types for other side effect orders
	const auto expected_dependencies = std::unordered_map<std::pair<order, order>, std::optional<dependency_kind>, utils::pair_hash>{
	    {{order::sequential, order::sequential}, dependency_kind::true_dep},
	};

	const auto order_a = GENERATE(values(side_effect_orders));
	const auto order_b = GENERATE(values(side_effect_orders));
	CAPTURE(order_a);
	CAPTURE(order_b);

	dist_cdag_test_context dctx(2);

	auto ho_common = dctx.create_host_object(); // should generate dependencies
	auto ho_a = dctx.create_host_object();      // should NOT generate dependencies
	auto ho_b = dctx.create_host_object();      // -"-

	const auto tid_0 = dctx.host_task(node_range).affect(ho_a, order_a).submit();
	const auto tid_1 = dctx.host_task(node_range).affect(ho_common, order_a).affect(ho_b, order_b).submit();
	const auto tid_2 = dctx.host_task(node_range).affect(ho_common, order_b).submit();

	CHECK(dctx.query(tid_0).find_predecessors(command_type::epoch).count() == 2);
	CHECK(dctx.query(tid_1).find_predecessors(command_type::epoch).count() == 2);

	const auto expected_2 = expected_dependencies.at({order_a, order_b});
	CHECK(dctx.query(tid_2).find_predecessors().count_per_node() == expected_2.has_value());
	if(expected_2) { CHECK(dctx.query(tid_2).find_predecessors(tid_1).count_per_node() == 1); }
}

TEST_CASE("epochs serialize task commands on every node", "[distributed_graph_generator][command-graph][epoch]") {
	using namespace cl::sycl::access;

	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	dist_cdag_test_context dctx(num_nodes);

	const auto tid_init = task_manager::initial_epoch_task;
	const auto tid_a = dctx.device_compute(node_range).submit();
	const auto tid_b = dctx.device_compute(node_range).submit();
	const auto tid_epoch_1 = dctx.epoch(epoch_action::none);

	CHECK(dctx.query(tid_init).count_per_node() == 1);
	CHECK(dctx.query(tid_a).count_per_node() == 1);
	CHECK(dctx.query(tid_b).count_per_node() == 1);
	CHECK(dctx.query(tid_epoch_1).count_per_node() == 1);
	CHECK(dctx.query(tid_init).have_successors(dctx.query(tid_a), dependency_kind::true_dep, dependency_origin::last_epoch));
	CHECK(dctx.query(tid_init).have_successors(dctx.query(tid_b), dependency_kind::true_dep, dependency_origin::last_epoch));
	CHECK(dctx.query().find_all(tid_a).have_successors(dctx.query(tid_epoch_1), dependency_kind::true_dep, dependency_origin::execution_front));
	CHECK(dctx.query().find_all(tid_b).have_successors(dctx.query(tid_epoch_1), dependency_kind::true_dep, dependency_origin::execution_front));

	// commands always depend on the *last* epoch

	auto buf = dctx.create_buffer<1>(node_range, true /* host initialized */);

	const auto tid_c = dctx.device_compute(node_range).read_write(buf, celerity::access::one_to_one()).submit();
	const auto tid_d = dctx.device_compute(node_range).discard_write(buf, celerity::access::one_to_one()).submit();

	CHECK(dctx.query(tid_c).count_per_node() == 1);
	CHECK(dctx.query(tid_d).count_per_node() == 1);
	CHECK(dctx.query(tid_epoch_1).have_successors(dctx.query(tid_c), dependency_kind::true_dep, dependency_origin::dataflow));
	CHECK(dctx.query().find_all(tid_epoch_1).have_successors(dctx.query(tid_d), dependency_kind::true_dep, dependency_origin::last_epoch));
	CHECK(dctx.query(tid_c).have_successors(dctx.query(tid_d), dependency_kind::anti_dep));

	// TODO we should test that dependencies never pass over epochs.
	// Doing this properly needs dist_cdag_test_context to keep commands alive even when the dggen deletes them after inserting the new epoch
}

TEST_CASE("a sequence of epochs without intermediate commands has defined behavior", "[distributed_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	dist_cdag_test_context dctx(num_nodes);

	auto tid_before = task_manager::initial_epoch_task;
	for(const auto action : {epoch_action::barrier, epoch_action::shutdown}) {
		const auto tid = dctx.epoch(action);
		CAPTURE(tid_before, tid);
		CHECK(dctx.query(tid).find_predecessors().count_per_node() == 1);
		CHECK(dctx.query(tid_before).find_successors().count_per_node() == 1);
		CHECK(dctx.query(tid_before).have_successors(dctx.query(tid), dependency_kind::true_dep));
		tid_before = tid;
	}
}

TEST_CASE("all commands have a transitive true-dependency on the preceding epoch", "[distributed_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	dist_cdag_test_context dctx(num_nodes);
	dctx.set_horizon_step(99); // no horizon interference

	auto buf_1 = dctx.create_buffer<1>({num_nodes}, true /* host_initialized */);
	auto buf_2 = dctx.create_buffer<1>({1});
	auto buf_3 = dctx.create_buffer<1>({1});

	dctx.device_compute(range<1>{num_nodes}).discard_write(buf_2, celerity::access::one_to_one()).reduce(buf_3, false).submit();
	const auto epoch = dctx.epoch(epoch_action::none);
	dctx.device_compute(node_range).discard_write(buf_1, celerity::access::one_to_one()).read(buf_2, celerity::access::all()).submit();
	dctx.device_compute(node_range).read(buf_3, celerity::access::all()).submit();

	for(node_id nid = 0; nid < num_nodes; ++nid) {
		CAPTURE(nid);

		const auto all_commands = dctx.query(nid).get_raw(nid);
		const auto epoch_only = dctx.query(epoch).get_raw(nid);

		// Iteratively build the set of commands transitively true-depending on an epoch
		std::set<const abstract_command*> commands_after_epoch(all_commands.begin(), all_commands.end());
		std::set<const abstract_command*> transitive_dependents;
		std::set<const abstract_command*> dependent_front(epoch_only.begin(), epoch_only.end());
		while(!dependent_front.empty()) {
			transitive_dependents.insert(dependent_front.begin(), dependent_front.end());
			std::set<const abstract_command*> new_dependent_front;
			for(const auto* const cmd : dependent_front) {
				for(const auto& dep : cmd->get_dependents()) {
					if(dep.kind == dependency_kind::true_dep && transitive_dependents.count(dep.node) == 0) { new_dependent_front.insert(dep.node); }
				}
			}
			dependent_front = std::move(new_dependent_front);
		}

		CHECK(commands_after_epoch == transitive_dependents);
	}
}

TEST_CASE("fences introduce dependencies on host objects", "[distributed_graph_generator][command-graph][fence]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	dist_cdag_test_context dctx(num_nodes);

	auto ho = dctx.create_host_object();

	const auto tid_a = dctx.collective_host_task().affect(ho).submit();
	const auto tid_fence = dctx.fence(ho);
	const auto tid_b = dctx.collective_host_task().affect(ho).submit();

	CHECK(dctx.query(tid_a).count_per_node() == 1);
	CHECK(dctx.query(tid_a).have_successors(dctx.query(tid_fence)));
	CHECK(dctx.query(tid_fence).have_successors(dctx.query(tid_b)));
}

TEST_CASE("fences introduce dependencies on buffers", "[distributed_graph_generator][command-graph][fence]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	dist_cdag_test_context dctx(num_nodes);

	auto buf = dctx.create_buffer<1>({num_nodes});

	const auto tid_a = dctx.master_node_host_task().discard_write(buf, celerity::access::all()).submit();
	const auto tid_fence = dctx.fence(buf);
	const auto tid_b = dctx.collective_host_task().discard_write(buf, celerity::access::one_to_one()).submit();

	for(node_id nid = 0; nid < num_nodes; ++nid) {
		CAPTURE(nid);

		CHECK(dctx.query(tid_a, nid).count() == (nid == 0));
		CHECK(dctx.query(command_type::push, nid).count() == (nid == 0));
		CHECK(dctx.query(command_type::await_push, nid).count() == (nid == 1));
		if(nid == 0) {
			CHECK(dctx.query(nid, tid_a).have_successors(dctx.query(nid, command_type::push)));
			CHECK(dctx.query(nid, tid_a).have_successors(dctx.query(nid, tid_fence)));
		} else {
			CHECK(dctx.query(nid, command_type::await_push).have_successors(dctx.query(nid, tid_fence)));
		}
		CHECK(dctx.query(tid_b, nid).count() == 1);
		CHECK(dctx.query(tid_fence, nid).have_successors(dctx.query(tid_b, nid)));
	}
}
