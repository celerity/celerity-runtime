#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "command_graph_generator_test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("isa<> RTTI helper correctly handles command hierarchies", "[rtti][command-graph]") {
	auto tsk0 = task::make_epoch(0, epoch_action::none, nullptr);
	const auto np = std::make_unique<epoch_command>(command_id(), tsk0.get(), epoch_action::none, std::vector<reduction_id>{});
	REQUIRE(utils::isa<command>(np.get()));
	auto tsk1 = task::make_master_node(1, {}, {}, {});
	const auto hec = std::make_unique<execution_command>(command_id(), tsk1.get(), subrange<3>{}, false);
	REQUIRE(utils::isa<execution_command>(hec.get()));
	const auto pc = std::make_unique<push_command>(command_id(), transfer_id(0, 0, 0), std::vector<std::pair<node_id, region<3>>>{});
	REQUIRE(utils::isa<command>(pc.get()));
	const auto apc = std::make_unique<await_push_command>(command_id(), transfer_id(0, 0, 0), region<3>{});
	REQUIRE(utils::isa<command>(apc.get()));
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
		CHECK(cctx.query<await_push_command_record>().on(master_node_id).assert_count(2).successors().contains(cctx.query(tid_c).on(master_node_id)));
	}

	SECTION("if data is produced remotely but already available from an earlier task") {
		cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		const auto await_pushes = cctx.query<await_push_command_record>().on(master_node_id).assert_count(1);

		const auto tid_c = cctx.master_node_host_task().read(buf0, acc::all{}).submit();
		// Assert that the number of await_pushes hasn't changed (i.e., none were added)
		CHECK(cctx.query<await_push_command_record>().on(master_node_id).count() == await_pushes.count());
		// ...and the command for task c depends on the earlier await_push
		CHECK(await_pushes.successors().contains(cctx.query(tid_c).on(master_node_id)));
	}

	SECTION("if data is produced locally") {
		const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).read(buf0, acc::one_to_one{}).read(buf1, acc::one_to_one{}).submit();
		CHECK(cctx.query(tid_a).successors().contains(cctx.query(tid_c)));
		CHECK(cctx.query(tid_b).successors().contains(cctx.query(tid_c)));
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
	CHECK(cctx.query(tid_a).successors().contains(cctx.query(tid_d)));
	CHECK(cctx.query(tid_b).successors().contains(cctx.query(tid_d)));
	CHECK(cctx.query(tid_c).successors().contains(cctx.query(tid_d)));
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
	// Importantly, we only read on node 1, making it so that node 0 does not have a true dependency on the previous execution command.
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

	CHECK(cctx.query(tid_a).on(0).successors().contains(cctx.query(tid_b).on(0)));
	CHECK(cctx.query(tid_a).on(1).successors().contains(cctx.query(tid_b).on(1)));
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
		CHECK(cctx.query(tid_b).is_concurrent_with(cctx.query(tid_c)));
	}

	SECTION("does not consider anti-successors of last writer as potential anti-dependencies") {
		// task_b writes buf0, making task_a an anti-dependency
		const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
		// task_c writes buf1. Since task_b is not a true successor of task_a, we don't consider it as a potential anti-dependency.
		const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
		CHECK(cctx.query(tid_b).is_concurrent_with(cctx.query(tid_c)));
	}
}

TEST_CASE("command_graph_generator generates anti-dependencies onto the original producer if no consumer exists in between",
    "[command_graph_generator][command-graph]") {
	cdag_test_context cctx(2);

	const range<1> test_range = {128};
	auto buf0 = cctx.create_buffer(test_range);

	const auto tid_a = cctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = cctx.device_compute<class UKN(task_b)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	CHECK(cctx.query(tid_a).successors().contains(cctx.query(tid_b)));
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
		const auto push_w = cctx.query<push_command_record>().on(writing_node);
		CHECK(push_w.count() == 1);
		const auto push_o = cctx.query<push_command_record>().on(other_node);
		CHECK(push_o.count() == 1);

		// Since other_node does not write to the buffer, there is no anti-dependency...
		CHECK_FALSE(push_o.successors().contains(cctx.query(tid_b).on(other_node)));
		// ...however for node 1, there is.
		CHECK(push_w.successors().contains(cctx.query(tid_b).on(writing_node)));
	};

	// UPDATE: Which node writes and which reads is no longer relevant, as pushes are now always generated before execution commands.
	//         Nevertheless, keeping both cases doesn't hurt either.
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
	CHECK(cctx.query(tid_a).successors().contains(cctx.query(tid_b)));

	// task_c writes to a different buffer 1
	const auto tid_c = cctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf1, acc::one_to_one{}).submit();
	// task_c should not have any anti-dependencies at all
	CHECK(cctx.query(tid_c).is_concurrent_with(union_of(cctx.query(tid_a), cctx.query(tid_b))));
}

TEST_CASE(
    "command_graph_generator generates pseudo-dependencies for collective commands in the same collective group", "[command_graph_generator][collectives]") {
	cdag_test_context cctx(2);

	experimental::collective_group group;
	cctx.master_node_host_task().name("task a").submit();
	cctx.collective_host_task().name("implicit 1").submit();
	cctx.collective_host_task().name("implicit 2").submit();
	cctx.collective_host_task(group).name("explicit 1").submit();
	cctx.collective_host_task(group).name("explicit 2").submit();

	CHECK(cctx.query("implicit 1").on(master_node_id).is_concurrent_with(cctx.query("task a").on(master_node_id)));
	CHECK(cctx.query("implicit 2").on(master_node_id).is_concurrent_with(cctx.query("task a").on(master_node_id)));
	CHECK(cctx.query("explicit 1").on(master_node_id).is_concurrent_with(cctx.query("task a").on(master_node_id)));
	CHECK(cctx.query("explicit 2").on(master_node_id).is_concurrent_with(cctx.query("task a").on(master_node_id)));

	CHECK(cctx.query("implicit 1").is_concurrent_with(cctx.query("explicit 1")));
	CHECK(cctx.query("implicit 1").is_concurrent_with(cctx.query("explicit 2")));
	CHECK(cctx.query("implicit 2").is_concurrent_with(cctx.query("explicit 1")));
	CHECK(cctx.query("implicit 2").is_concurrent_with(cctx.query("explicit 2")));

	CHECK(cctx.query("implicit 1").successors().contains(cctx.query("implicit 2")));
	CHECK(cctx.query("explicit 1").successors().contains(cctx.query("explicit 2")));
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

	CHECK(cctx.query(tid_0).predecessors().select_all<epoch_command_record>().total_count() == 2);
	CHECK(cctx.query(tid_1).predecessors().select_all<epoch_command_record>().total_count() == 2);

	const auto expected_2 = expected_dependencies.at({order_a, order_b});
	CHECK(cctx.query(tid_2).predecessors().count_per_node() == expected_2.has_value());
	if(expected_2) { CHECK(cctx.query(tid_2).predecessors().select_all(tid_1).count_per_node() == 1); }
}

TEST_CASE("epochs serialize task commands on every node", "[command_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context cctx(num_nodes);

	const task_id tid_init = cctx.get_initial_epoch_task();
	cctx.device_compute(node_range).name("task a").submit();
	cctx.device_compute(node_range).name("task b").submit();
	const auto before_epoch = cctx.query();
	const auto epoch = cctx.epoch(epoch_action::none);

	CHECK(cctx.query(tid_init).count_per_node() == 1);
	CHECK(cctx.query("task a").count_per_node() == 1);
	CHECK(cctx.query("task b").count_per_node() == 1);
	CHECK(cctx.query(epoch).count_per_node() == 1);
	CHECK(cctx.query(tid_init).successors().contains(cctx.query("task a")));
	CHECK(cctx.query(tid_init).successors().contains(cctx.query("task b")));
	CHECK(cctx.query().select_all("task a").successors().contains(cctx.query(epoch)));
	CHECK(cctx.query().select_all("task b").successors().contains(cctx.query(epoch)));

	// commands always depend on the *last* epoch

	auto buf = cctx.create_buffer<1>(node_range, true /* host initialized */);

	cctx.device_compute(node_range).name("task c").read_write(buf, celerity::access::one_to_one()).submit();
	cctx.device_compute(node_range).name("task d").discard_write(buf, celerity::access::one_to_one()).submit();

	CHECK(cctx.query("task c").count_per_node() == 1);
	CHECK(cctx.query("task d").count_per_node() == 1);
	CHECK(cctx.query(epoch).successors().contains(cctx.query("task c")));
	CHECK(cctx.query(epoch).successors().contains(cctx.query("task d")));
	CHECK(cctx.query("task c").successors().contains(cctx.query("task d")));

	// Dependencies don't pass the epoch
	CHECK_FALSE(union_of(cctx.query("task a"), cctx.query("task b")).predecessors().contains(before_epoch));
}

TEST_CASE("a sequence of epochs without intermediate commands has defined behavior", "[command_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	cdag_test_context cctx(num_nodes);

	task_id tid_before = cctx.get_initial_epoch_task();
	for(const auto action : {epoch_action::barrier, epoch_action::shutdown}) {
		const auto tid = cctx.epoch(action);
		CAPTURE(tid_before, tid);
		CHECK(cctx.query(tid).predecessors().count_per_node() == 1);
		CHECK(cctx.query(tid_before).successors().count_per_node() == 1);
		CHECK(cctx.query(tid_before).successors().contains(cctx.query(tid)));
		tid_before = tid;
	}
}

TEST_CASE("all commands have a transitive true-dependency on the preceding epoch", "[command_graph_generator][command-graph][epoch]") {
	const size_t num_nodes = 2;
	const range<1> node_range{num_nodes};

	cdag_test_context cctx(num_nodes);

	auto buf_1 = cctx.create_buffer<1>({num_nodes}, true /* host_initialized */);
	auto buf_2 = cctx.create_buffer<1>({1});
	auto buf_3 = cctx.create_buffer<1>({1});

	cctx.device_compute(range<1>{num_nodes}).discard_write(buf_2, celerity::access::one_to_one()).reduce(buf_3, false).submit();
	const auto epoch = cctx.epoch(epoch_action::none);
	const auto after_epoch = cctx.query();
	cctx.device_compute(node_range).name("task a").discard_write(buf_1, celerity::access::one_to_one()).read(buf_2, celerity::access::all()).submit();
	cctx.device_compute(node_range).name("task b").read(buf_3, celerity::access::all()).submit();

	CHECK(cctx.query(epoch).transitive_successors() == difference_of(cctx.query(), after_epoch));
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
	CHECK(cctx.query(tid_a).successors().contains(cctx.query(tid_fence)));
	CHECK(cctx.query(tid_fence).successors().contains(cctx.query(tid_b)));
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

		CHECK(cctx.query(tid_a).on(nid).count() == (nid == 0));
		CHECK(cctx.query<push_command_record>().on(nid).count() == (nid == 0));
		CHECK(cctx.query<await_push_command_record>().on(nid).count() == (nid == 1));
		if(nid == 0) {
			CHECK(cctx.query(tid_a).on(nid).successors().contains(cctx.query<push_command_record>().on(nid)));
			CHECK(cctx.query(tid_a).on(nid).successors().contains(cctx.query(tid_fence).on(nid)));
		} else {
			CHECK(cctx.query<await_push_command_record>().on(nid).successors().contains(cctx.query(tid_fence).on(nid)));
		}
		CHECK(cctx.query(tid_b).on(nid).count() == 1);
		CHECK(cctx.query(tid_fence).on(nid).successors().contains(cctx.query(tid_b).on(nid)));
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
		    "Command C3 on N0, which executes [0,0,0] - [1,1,1] of device kernel T2, reads B0 {[1,0,0] - [2,1,1]}, which has not been written by any node.");
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
		CHECK_THROWS_WITH((cctx.host_task(buf.get_range()).name("host neighborhood").discard_write(buf, acc::neighborhood({1, 1})).submit()),
		    "Host-compute task T1 \"host neighborhood\" has overlapping writes between multiple nodes in B0 {[9,0,0] - [11,20,1]}. Choose a non-overlapping "
		    "range mapper for this write access or constrain the split via experimental::constrain_split to make the access non-overlapping.");
	}
}

TEST_CASE("results form generator kernels are never communicated between nodes", "[command_graph_generator][owner-computes]") {
	const bool split_2d = GENERATE(values({0, 1}));
	CAPTURE(split_2d);

	const size_t num_nodes = 4;
	cdag_test_context cctx(num_nodes);            // 4 nodes, so we can get a true 2D work assignment for the timestep kernel
	auto buf = cctx.create_buffer<2>({256, 256}); // a 256x256 buffer

	const auto tid_init = cctx.device_compute(buf.get_range()) //
	                          .discard_write(buf, celerity::access::one_to_one())
	                          .name("init")
	                          .submit();
	const auto tid_ts0 = cctx.device_compute(buf.get_range()) //
	                         .hint_if(split_2d, experimental::hints::split_2d())
	                         .read_write(buf, celerity::access::one_to_one())
	                         .name("timestep 0")
	                         .submit();
	const auto tid_ts1 = cctx.device_compute(buf.get_range()) //
	                         .hint_if(split_2d, experimental::hints::split_2d())
	                         .read_write(buf, celerity::access::one_to_one())
	                         .name("timestep 1")
	                         .submit();

	CHECK(cctx.query<execution_command_record>().count_per_node() == 3); // one for each task above
	CHECK(cctx.query<push_command_record>().total_count() == 0);
	CHECK(cctx.query<await_push_command_record>().total_count() == 0);

	const auto inits = cctx.query<execution_command_record>(tid_init);
	const auto ts0s = cctx.query<execution_command_record>(tid_ts0);
	const auto ts1s = cctx.query<execution_command_record>(tid_ts1);
	CHECK(inits.count_per_node() == 1);
	CHECK(ts0s.count_per_node() == 1);
	CHECK(ts1s.count_per_node() == 1);

	for(node_id nid = 0; nid < num_nodes; ++nid) {
		const auto n_init = inits.on(nid);
		REQUIRE(n_init->accesses.size() == 1);

		const auto generate = n_init->accesses.front();
		CHECK(generate.bid == buf.get_id());
		CHECK(generate.mode == access_mode::discard_write);

		const auto n_ts0 = ts0s.on(nid);
		CHECK(n_ts0.predecessors().contains(n_init));
		REQUIRE(n_ts0->accesses.size() == 1);

		const auto consume = n_ts0->accesses.front();
		CHECK(consume.bid == buf.get_id());
		CHECK(consume.mode == access_mode::read_write);

		// generator kernel "init" has generated exactly the buffer subrange that is consumed by "timestep 0"
		CHECK(consume.req == generate.req);

		const auto n_ts1 = ts1s.on(nid);
		CHECK(n_ts1.predecessors().contains(n_ts0));
		CHECK_FALSE(n_ts1.predecessors().contains(n_init));
	}
}
