#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "command_graph_generator_test_utils.h"

#include "command_graph_generator.h"


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
	command_set cmds;
	cmds.insert(cdag.create<execution_command>(0, subrange<3>{}));
	cmds.insert(cdag.create<epoch_command>(task_manager::initial_epoch_task, epoch_action::none, std::vector<reduction_id>{}));
	cmds.insert(cdag.create<push_command>(0, transfer_id(0, 0, 0), subrange<3>{}));
	for(auto* cmd : cdag.all_commands()) {
		REQUIRE(cmds.find(cmd) != cmds.end());
		cmds.erase(cmd);
	}
	REQUIRE(cmds.empty());
}

TEST_CASE("command_graph keeps track of execution front", "[command_graph][command-graph]") {
	command_graph cdag;

	command_set expected_front;

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
	auto* const np = cdag.create<epoch_command>(task_manager::initial_epoch_task, epoch_action::none, std::vector<reduction_id>{});
	REQUIRE(utils::isa<abstract_command>(np));
	auto* const hec = cdag.create<execution_command>(0, subrange<3>{});
	REQUIRE(utils::isa<execution_command>(hec));
	auto* const pc = cdag.create<push_command>(0, transfer_id(0, 0, 0), subrange<3>{});
	REQUIRE(utils::isa<abstract_command>(pc));
	auto* const apc = cdag.create<await_push_command>(transfer_id(0, 0, 0), region<3>{});
	REQUIRE(utils::isa<abstract_command>(apc));
}

TEST_CASE("command_graph_generator generates dependencies for execution commands", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);
	auto buf1 = cctx.create_buffer(test_range);

	SECTION("if data is produced remotely") {
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		const auto tid_c = cctx.master_node_host_task().read(buf0, acc::all{}).read(buf1, acc::all{}).submit();
		CHECK(cctx.query(master_node_id, command_type::await_push).assert_count(2).have_successors(cctx.query(tid_c), dependency_kind::true_dep));
	}

	SECTION("if data is produced remotely but already available from an earlier task") {
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		const auto await_pushes = cctx.query(master_node_id, command_type::await_push).assert_count(1);

		const auto tid_c = cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		// Assert that the number of await_pushes hasn't changed (i.e., none were added)
		CHECK(cctx.query(master_node_id, command_type::await_push).count() == await_pushes.count());
		// ...and the command for task c depends on the earlier await_push
		CHECK(await_pushes.have_successors(cctx.query(tid_c), dependency_kind::true_dep));
	}

	SECTION("if data is produced locally") {
		const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).read(buf0, acc::one_to_one{}).read(buf1, acc::one_to_one{}).submit();
		CHECK(cctx.query(tid_a).have_successors(cctx.query(tid_c)));
		CHECK(cctx.query(tid_b).have_successors(cctx.query(tid_c)));
	}
}

TEST_CASE(
    "command_graph_generator builds dependencies to all local commands if a given range is produced by multiple", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(1);

	const range<1> test_range = {96};
	const range<1> one_third = {test_range / 3};
	auto buf = cctx.create_buffer(test_range);

	const auto tid_a = cctx.device_compute<class UKN(task_a)>(one_third, id<1>{0 * one_third}).discard_write(buf, acc::one_to_one{}).submit();
	const auto tid_b = cctx.device_compute<class UKN(task_b)>(one_third, id<1>{1 * one_third}).discard_write(buf, acc::one_to_one{}).submit();
	const auto tid_c = cctx.device_compute<class UKN(task_c)>(one_third, id<1>{2 * one_third}).discard_write(buf, acc::one_to_one{}).submit();

	const auto tid_d = cctx.device_compute<class UKN(task_d)>(test_range).read(buf, acc::one_to_one{}).submit();
	CHECK(cctx.query(tid_a).have_successors(cctx.query(tid_d)));
	CHECK(cctx.query(tid_b).have_successors(cctx.query(tid_d)));
	CHECK(cctx.query(tid_c).have_successors(cctx.query(tid_d)));
}

// This is a highly constructed and unrealistic example, but we'd still like the behavior to be clearly defined.
TEST_CASE("command_graph_generator generates anti-dependencies for execution commands that have a task-level true dependency",
    "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);
	auto buf1 = cctx.create_buffer(test_range);

	// Initialize both buffers
	const auto tid_a =
	    cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).discard_write(buf1, acc::one_to_one{}).submit();

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
	const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).read(buf0, node_1_writes).discard_write(buf1, acc::one_to_one{}).submit();

	CHECK(cctx.query(tid_a, node_id(0)).have_successors(cctx.query(tid_b, node_id(0)), dependency_kind::anti_dep));
	CHECK(cctx.query(tid_a, node_id(1)).have_successors(cctx.query(tid_b, node_id(1)), dependency_kind::true_dep));
}

TEST_CASE("command_graph_generator correctly handles anti-dependency edge cases", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(1);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);
	auto buf1 = cctx.create_buffer(test_range);

	// task_a writes both buffers
	cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).discard_write(buf1, acc::one_to_one{}).submit();

	SECTION("correctly handles false anti-dependencies that consume a different buffer from the last writer") {
		// task_b reads buf0
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::one_to_one{}).submit();
		// task_c writes buf1, initially making task_b a potential anti-dependency (as it is a successor of task_a).
		const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		// However, since the two tasks don't actually touch the same buffers at all, nothing needs to be done.
		CHECK_FALSE(cctx.query(tid_b).have_successors(cctx.query(tid_c), dependency_kind::anti_dep));
	}

	SECTION("does not consider anti-successors of last writer as potential anti-dependencies") {
		// task_b writes buf0, making task_a an anti-dependency
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		// task_c writes buf1. Since task_b is not a true successor of task_a, we don't consider it as a potential anti-dependency.
		const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		CHECK_FALSE(cctx.query(tid_b).have_successors(cctx.query(tid_c), dependency_kind::anti_dep));
	}
}

TEST_CASE("command_graph_generator generates anti-dependencies onto the original producer if no consumer exists in between",
    "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);

	const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	CHECK(cctx.query(tid_a).have_successors(cctx.query(tid_b), dependency_kind::anti_dep));
}

TEST_CASE(
    "command_graph_generator generates anti-dependencies for execution commands onto pushes within the same task", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);

	const auto run_test = [&](const node_id writing_node, const node_id other_node) {
		const auto only_one_writes = [=](chunk<1> chnk) -> subrange<1> {
			if(chnk.range[0] == test_range) return subrange<1>{writing_node == 0 ? 0u : 64u, 64};
			switch(chnk.offset[0]) {
			case 0: return writing_node == 0 ? chnk : subrange<1>{0, 0};
			case 64: return writing_node == 1 ? chnk : subrange<1>{0, 0};
			default: FAIL("Unexpected offset");
			}
			return {};
		};

		// Both nodes write parts of the buffer.
		[[maybe_unused]] const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();

		// Both nodes read the full buffer, but writing_node also writes to it.
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::all{}).discard_write(buf0, only_one_writes).submit();

		// Each node pushes data to the other.
		const auto push_w = cctx.query(writing_node, command_type::push);
		CHECK(push_w.count() == 1);
		const auto push_o = cctx.query(other_node, command_type::push);
		CHECK(push_o.count() == 1);

		// Since other_node does not write to the buffer, there is no anti-dependency...
		CHECK_FALSE(push_o.have_successors(cctx.query(other_node, tid_b), dependency_kind::anti_dep));
		// ...however for node 1, there is.
		CHECK(push_w.have_successors(cctx.query(writing_node, tid_b), dependency_kind::anti_dep));
	};

	// NOTE: These two sections are handled by different mechanisms inside the command_graph_generator:
	//	   - The first is done by generate_anti_dependencies during the initial sweep.
	// 	   - The second is done by the "intra-task" loop at the end.
	// NOTE: This ("before" / "after") assumes that chunks are processed in node id order
	SECTION("if the push is generated before the execution command") { run_test(1, 0); }
	SECTION("if the push is generated after the execution command") { run_test(0, 1); }
}

TEST_CASE("command_graph_generator generates anti-dependencies for commands accessing host-initialized buffers", "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	// We have two host initialized buffers
	auto buf0 = cctx.create_buffer(test_range, true);
	auto buf1 = cctx.create_buffer(test_range, true);

	// task_a reads from host-initialized buffer 0
	const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).read(buf0, acc::one_to_one{}).submit();

	// task_b writes to the same buffer 0
	const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	// task_b should have an anti-dependency onto task_a
	CHECK(cctx.query(tid_a).have_successors(cctx.query(tid_b)));

	// task_c writes to a different buffer 1
	const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
	// task_c should not have any anti-dependencies at all
	CHECK(cctx.query(tid_c).find_predecessors(dependency_kind::anti_dep).empty());
}

TEST_CASE(
    "command_graph_generator generates pseudo-dependencies for collective commands on the same collective group", "[command_graph_generator][collectives]") {
	cdag_test_context cctx(2);

	experimental::collective_group group;
	const auto tid_a = cctx.master_node_host_task().submit();
	const auto tid_collective_implicit_1 = cctx.collective_host_task().submit();
	const auto tid_collective_implicit_2 = cctx.collective_host_task().submit();
	const auto tid_collective_explicit_1 = cctx.collective_host_task(group).submit();
	const auto tid_collective_explicit_2 = cctx.collective_host_task(group).submit();

	CHECK_FALSE(cctx.query(tid_collective_implicit_1).have_successors(cctx.query(tid_a)));
	CHECK_FALSE(cctx.query(tid_collective_implicit_2).have_successors(cctx.query(tid_a)));
	CHECK_FALSE(cctx.query(tid_collective_explicit_1).have_successors(cctx.query(tid_a)));
	CHECK_FALSE(cctx.query(tid_collective_explicit_2).have_successors(cctx.query(tid_a)));

	CHECK_FALSE(cctx.query(tid_a).assert_count(1).have_successors(cctx.query(master_node_id, tid_collective_implicit_1)));
	CHECK_FALSE(cctx.query(tid_collective_implicit_2).have_successors(cctx.query(tid_collective_implicit_1)));
	CHECK_FALSE(cctx.query(tid_collective_explicit_1).have_successors(cctx.query(tid_collective_implicit_1)));
	CHECK_FALSE(cctx.query(tid_collective_explicit_2).have_successors(cctx.query(tid_collective_implicit_1)));

	CHECK_FALSE(cctx.query(tid_a).assert_count(1).have_successors(cctx.query(master_node_id, tid_collective_implicit_2)));
	CHECK(cctx.query()
	          .find_all(tid_collective_implicit_1)
	          .have_successors(cctx.query(tid_collective_implicit_2), dependency_kind::true_dep, dependency_origin::collective_group_serialization));
	CHECK_FALSE(cctx.query(tid_collective_explicit_1).have_successors(cctx.query(tid_collective_implicit_2)));
	CHECK_FALSE(cctx.query(tid_collective_explicit_2).have_successors(cctx.query(tid_collective_implicit_2)));

	CHECK_FALSE(cctx.query(tid_a).assert_count(1).have_successors(cctx.query(master_node_id, tid_collective_explicit_1)));
	CHECK_FALSE(cctx.query(tid_collective_implicit_1).have_successors(cctx.query(tid_collective_explicit_1)));
	CHECK_FALSE(cctx.query(tid_collective_implicit_2).have_successors(cctx.query(tid_collective_explicit_1)));
	CHECK_FALSE(cctx.query(tid_collective_explicit_1).have_successors(cctx.query(tid_collective_explicit_1)));

	CHECK_FALSE(cctx.query(tid_a).assert_count(1).have_successors(cctx.query(master_node_id, tid_collective_explicit_2)));
	CHECK_FALSE(cctx.query(tid_collective_implicit_1).have_successors(cctx.query(tid_collective_explicit_2)));
	CHECK_FALSE(cctx.query(tid_collective_implicit_2).have_successors(cctx.query(tid_collective_explicit_2)));
	CHECK(cctx.query()
	          .find_all(tid_collective_explicit_1)
	          .have_successors(cctx.query(tid_collective_explicit_2), dependency_kind::true_dep, dependency_origin::collective_group_serialization));
}

TEST_CASE("side effects generate appropriate command-dependencies", "[command_graph_generator][command-graph][side-effect]") {
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

	cdag_test_context cctx(2);

	auto ho_common = cctx.create_host_object(); // should generate dependencies
	auto ho_a = cctx.create_host_object();      // should NOT generate dependencies
	auto ho_b = cctx.create_host_object();      // -"-

	const auto tid_0 = cctx.host_task(node_range).affect(ho_a, order_a).submit();
	const auto tid_1 = cctx.host_task(node_range).affect(ho_common, order_a).affect(ho_b, order_b).submit();
	const auto tid_2 = cctx.host_task(node_range).affect(ho_common, order_b).submit();

	CHECK(cctx.query(tid_0).find_predecessors(command_type::epoch).count() == 2);
	CHECK(cctx.query(tid_1).find_predecessors(command_type::epoch).count() == 2);

	const auto expected_2 = expected_dependencies.at({order_a, order_b});
	CHECK(cctx.query(tid_2).find_predecessors().count_per_node() == expected_2.has_value());
	if(expected_2) { CHECK(cctx.query(tid_2).find_predecessors(tid_1).count_per_node() == 1); }
}

TEST_CASE("epochs serialize task commands on every node", "[command_graph_generator][command-graph][epoch]") {
	using namespace sycl::access;

	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context cctx(num_nodes);

	const auto tid_init = task_manager::initial_epoch_task;
	const auto tid_a = cctx.device_compute(node_range).submit();
	const auto tid_b = cctx.device_compute(node_range).submit();
	const auto tid_epoch_1 = cctx.epoch(epoch_action::none);

	CHECK(cctx.query(tid_init).count_per_node() == 1);
	CHECK(cctx.query(tid_a).count_per_node() == 1);
	CHECK(cctx.query(tid_b).count_per_node() == 1);
	CHECK(cctx.query(tid_epoch_1).count_per_node() == 1);
	CHECK(cctx.query(tid_init).have_successors(cctx.query(tid_a), dependency_kind::true_dep, dependency_origin::last_epoch));
	CHECK(cctx.query(tid_init).have_successors(cctx.query(tid_b), dependency_kind::true_dep, dependency_origin::last_epoch));
	CHECK(cctx.query().find_all(tid_a).have_successors(cctx.query(tid_epoch_1), dependency_kind::true_dep, dependency_origin::execution_front));
	CHECK(cctx.query().find_all(tid_b).have_successors(cctx.query(tid_epoch_1), dependency_kind::true_dep, dependency_origin::execution_front));

	// commands always depend on the *last* epoch

	auto buf = cctx.create_buffer<1>(node_range, true /* host initialized */);

	const auto tid_c = cctx.device_compute(node_range).read_write(buf, celerity::access::one_to_one()).submit();
	const auto tid_d = cctx.device_compute(node_range).discard_write(buf, celerity::access::one_to_one()).submit();

	CHECK(cctx.query(tid_c).count_per_node() == 1);
	CHECK(cctx.query(tid_d).count_per_node() == 1);
	CHECK(cctx.query(tid_epoch_1).have_successors(cctx.query(tid_c), dependency_kind::true_dep, dependency_origin::dataflow));
	CHECK(cctx.query().find_all(tid_epoch_1).have_successors(cctx.query(tid_d), dependency_kind::true_dep, dependency_origin::last_epoch));
	CHECK(cctx.query(tid_c).have_successors(cctx.query(tid_d), dependency_kind::anti_dep));

	// TODO we should test that dependencies never pass over epochs.
	// Doing this properly needs cdag_test_context to keep commands alive even when the cggen deletes them after inserting the new epoch
}

TEST_CASE("a sequence of epochs without intermediate commands has defined behavior", "[command_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	cdag_test_context cctx(num_nodes);

	auto tid_before = task_manager::initial_epoch_task;
	for(const auto action : {epoch_action::barrier, epoch_action::shutdown}) {
		const auto tid = cctx.epoch(action);
		CAPTURE(tid_before, tid);
		CHECK(cctx.query(tid).find_predecessors().count_per_node() == 1);
		CHECK(cctx.query(tid_before).find_successors().count_per_node() == 1);
		CHECK(cctx.query(tid_before).have_successors(cctx.query(tid), dependency_kind::true_dep));
		tid_before = tid;
	}
}

TEST_CASE("all commands have a transitive true-dependency on the preceding epoch", "[command_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context cctx(num_nodes);
	cctx.set_horizon_step(99); // no horizon interference

	auto buf_1 = cctx.create_buffer<1>({num_nodes}, true /* host_initialized */);
	auto buf_2 = cctx.create_buffer<1>({1});
	auto buf_3 = cctx.create_buffer<1>({1});

	cctx.device_compute(range<1>{num_nodes}).discard_write(buf_2, celerity::access::one_to_one()).reduce(buf_3, false).submit();
	const auto epoch = cctx.epoch(epoch_action::none);
	cctx.device_compute(node_range).discard_write(buf_1, celerity::access::one_to_one()).read(buf_2, celerity::access::all()).submit();
	cctx.device_compute(node_range).read(buf_3, celerity::access::all()).submit();

	for(node_id nid = 0; nid < num_nodes; ++nid) {
		CAPTURE(nid);

		const auto all_commands = cctx.query(nid).get_raw(nid);
		const auto epoch_only = cctx.query(epoch).get_raw(nid);

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

TEST_CASE("fences introduce dependencies on host objects", "[command_graph_generator][command-graph][fence]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context cctx(num_nodes);

	auto ho = cctx.create_host_object();

	const auto tid_a = cctx.collective_host_task().affect(ho).submit();
	const auto tid_fence = cctx.fence(ho);
	const auto tid_b = cctx.collective_host_task().affect(ho).submit();

	CHECK(cctx.query(tid_a).count_per_node() == 1);
	CHECK(cctx.query(tid_a).have_successors(cctx.query(tid_fence)));
	CHECK(cctx.query(tid_fence).have_successors(cctx.query(tid_b)));
}

TEST_CASE("fences introduce dependencies on buffers", "[command_graph_generator][command-graph][fence]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context cctx(num_nodes);

	auto buf = cctx.create_buffer<1>({num_nodes});

	const auto tid_a = cctx.master_node_host_task().discard_write(buf, celerity::access::all()).submit();
	const auto tid_fence = cctx.fence(buf);
	const auto tid_b = cctx.collective_host_task().discard_write(buf, celerity::access::one_to_one()).submit();

	for(node_id nid = 0; nid < num_nodes; ++nid) {
		CAPTURE(nid);

		CHECK(cctx.query(tid_a, nid).count() == (nid == 0));
		CHECK(cctx.query(command_type::push, nid).count() == (nid == 0));
		CHECK(cctx.query(command_type::await_push, nid).count() == (nid == 1));
		if(nid == 0) {
			CHECK(cctx.query(nid, tid_a).have_successors(cctx.query(nid, command_type::push)));
			CHECK(cctx.query(nid, tid_a).have_successors(cctx.query(nid, tid_fence)));
		} else {
			CHECK(cctx.query(nid, command_type::await_push).have_successors(cctx.query(nid, tid_fence)));
		}
		CHECK(cctx.query(tid_b, nid).count() == 1);
		CHECK(cctx.query(tid_fence, nid).have_successors(cctx.query(tid_b, nid)));
	}
}

TEST_CASE("command_graph_generator throws in tests if it detects an uninitialized read", "[command_graph_generator]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context::policy_set policy;
	policy.tm.uninitialized_read_error = error_policy::ignore; // otherwise we get task-level errors first

	cdag_test_context cctx(num_nodes, policy);

	SECTION("on a fully uninitialized buffer") {
		auto buf = cctx.create_buffer<1>({1});
		CHECK_THROWS_WITH((cctx.device_compute(node_range).name("uninitialized").read(buf, acc::all()).submit()),
		    "Command C1 on N0, which executes [0,0,0] - [1,1,1] of device kernel T1 \"uninitialized\", reads B0 {[0,0,0] - [1,1,1]}, which has not been "
		    "written by any node.");
	}

	SECTION("on a partially, locally initialized buffer") {
		auto buf = cctx.create_buffer<1>(node_range);
		cctx.device_compute(range(1)).discard_write(buf, acc::one_to_one()).submit();
		CHECK_THROWS_WITH((cctx.device_compute(node_range).read(buf, acc::all()).submit()),
		    "Command C2 on N0, which executes [0,0,0] - [1,1,1] of device kernel T2, reads B0 {[1,0,0] - [2,1,1]}, which has not been written by any node.");
	}

	SECTION("on a partially, remotely initialized buffer") {
		auto buf = cctx.create_buffer<1>(node_range);
		cctx.device_compute(range(1)).discard_write(buf, acc::one_to_one()).submit();
		CHECK_THROWS_WITH((cctx.device_compute(node_range).read(buf, acc::one_to_one()).submit()),
		    "Command C1 on N1, which executes [1,0,0] - [2,1,1] of device kernel T2, reads B0 {[1,0,0] - [2,1,1]}, which has not been written by any node.");
	}
}

TEST_CASE("command_graph_generator throws in tests if it detects overlapping writes", "[command_graph_generator]") {
	cdag_test_context cctx(2);
	auto buf = cctx.create_buffer<2>({20, 20});

	SECTION("on all-write") {
		CHECK_THROWS_WITH((cctx.device_compute(buf.get_range()).discard_write(buf, acc::all()).submit()),
		    "Device kernel T1 has overlapping writes between multiple nodes in B0 {[0,0,0] - [20,20,1]}. Choose a non-overlapping "
		    "range mapper for this write access or constrain the split via experimental::constrain_split to make the access non-overlapping.");
	}

	SECTION("on neighborhood-write") {
		CHECK_THROWS_WITH((cctx.host_task(buf.get_range()).name("host neighborhood").discard_write(buf, acc::neighborhood(1, 1)).submit()),
		    "Host-compute task T1 \"host neighborhood\" has overlapping writes between multiple nodes in B0 {[9,0,0] - [11,20,1]}. Choose a non-overlapping "
		    "range mapper for this write access or constrain the split via experimental::constrain_split to make the access non-overlapping.");
	}
}
