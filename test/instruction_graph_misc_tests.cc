#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "instruction_graph_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::experimental;

namespace acc = celerity::access;


TEST_CASE("a command group without data access compiles to a trivial graph", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	const range<1> test_range = {256};
	ictx.device_compute(test_range).name("kernel").submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count() == 3);
	CHECK(all_instrs.count<epoch_instruction_record>() == 2);
	CHECK(all_instrs.count<device_kernel_instruction_record>() == 1);

	const auto kernel = all_instrs.select_unique<device_kernel_instruction_record>("kernel");
	CHECK(kernel->access_map.empty());
	CHECK(kernel->execution_range == box(subrange<3>(zeros, range_cast<3>(test_range))));
	CHECK(kernel->device_id == device_id(0));
	CHECK(kernel.predecessors().is_unique<epoch_instruction_record>());
	CHECK(kernel.successors().is_unique<epoch_instruction_record>());
}

TEST_CASE("side-effects introduce dependencies between host-task instructions", "[instruction_graph_generator][instruction-graph]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(999);
	auto ho1 = ictx.create_host_object();
	auto ho2 = ictx.create_host_object(false /* owns instance */);
	ictx.master_node_host_task().name("affect ho1 (a)").affect(ho1).submit();
	ictx.master_node_host_task().name("affect ho2 (a)").affect(ho2).submit();
	ictx.master_node_host_task().name("affect ho1 (b)").affect(ho1).submit();
	ictx.master_node_host_task().name("affect ho2 (b)").affect(ho2).submit();
	ictx.master_node_host_task().name("affect ho1 + ho2").affect(ho1).affect(ho2).submit();
	ictx.master_node_host_task().name("affect ho1 (c)").affect(ho1).submit();
	ictx.master_node_host_task().name("affect ho2 (c)").affect(ho2).submit();
	ictx.finish(); // destroys all live host objects

	const auto all_instrs = ictx.query_instructions();
	const auto affect_ho1_a = all_instrs.select_unique<host_task_instruction_record>("affect ho1 (a)");
	const auto affect_ho2_a = all_instrs.select_unique<host_task_instruction_record>("affect ho2 (a)");
	const auto affect_ho1_b = all_instrs.select_unique<host_task_instruction_record>("affect ho1 (b)");
	const auto affect_ho2_b = all_instrs.select_unique<host_task_instruction_record>("affect ho2 (b)");
	const auto affect_both = all_instrs.select_unique<host_task_instruction_record>("affect ho1 + ho2");
	const auto affect_ho1_c = all_instrs.select_unique<host_task_instruction_record>("affect ho1 (c)");
	const auto affect_ho2_c = all_instrs.select_unique<host_task_instruction_record>("affect ho2 (c)");

	// only ho1 owns its instance, so only one destroy_host_object_instruction is emitted
	const auto destroy_ho1 = all_instrs.select_unique<destroy_host_object_instruction_record>();
	CHECK(destroy_ho1->host_object_id == ho1.get_id());

	CHECK(affect_ho1_a.predecessors().is_unique<epoch_instruction_record>());
	CHECK(affect_ho2_a.predecessors().is_unique<epoch_instruction_record>());
	CHECK(affect_ho1_a.successors() == affect_ho1_b);
	CHECK(affect_ho2_a.successors() == affect_ho2_b);
	CHECK(affect_ho1_b.successors() == affect_both);
	CHECK(affect_ho2_b.successors() == affect_both);
	CHECK(affect_both.successors() == union_of(affect_ho1_c, affect_ho2_c));
	CHECK(affect_ho1_c.successors() == destroy_ho1);
	CHECK(destroy_ho1.successors().is_unique<epoch_instruction_record>());
	CHECK(affect_ho2_c.successors().is_unique<epoch_instruction_record>());
}

TEST_CASE("collective-group instructions follow a single global total order", "[instruction_graph_generator][instruction-graph]") {
	const auto local_nid = GENERATE(values<node_id>({0, 1}));

	test_utils::idag_test_context ictx(2 /* nodes */, local_nid, 1 /* devices */);
	ictx.set_horizon_step(999);

	// collective-groups are not explicitly registered with graph generators, so both IDAG tests and the runtime use the same mechanism to declare them
	experimental::collective_group custom_cg_1;
	experimental::collective_group custom_cg_2;

	ictx.collective_host_task().name("default-group (a)").submit();
	ictx.collective_host_task().name("default-group (b)").submit();
	ictx.collective_host_task(custom_cg_1).name("custom-group 1 (a)").submit();
	ictx.collective_host_task(custom_cg_2).name("custom-group 2").submit();
	ictx.collective_host_task(custom_cg_1).name("custom-group 1 (b)").submit();
	ictx.collective_host_task().name("default-group (c)").submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto init_epoch = all_instrs.select_unique(task_manager::initial_epoch_task);
	// the default collective group does not use the default communicator (aka MPI_COMM_WORLD) because host tasks are executed on a different thread
	const auto clone_for_default_group = init_epoch.successors().assert_unique<clone_collective_group_instruction_record>();
	CHECK(clone_for_default_group->original_collective_group_id == root_collective_group_id);
	CHECK(clone_for_default_group->new_collective_group_id != clone_for_default_group->original_collective_group_id);
	const auto default_cgid = clone_for_default_group->new_collective_group_id;

	const auto default_group_a = all_instrs.select_unique<host_task_instruction_record>("default-group (a)");
	CHECK(default_group_a->collective_group_id == default_cgid);
	CHECK(default_group_a.predecessors() == clone_for_default_group);

	const auto default_group_b = all_instrs.select_unique<host_task_instruction_record>("default-group (b)");
	CHECK(default_group_b->collective_group_id == default_cgid);
	CHECK(default_group_b.predecessors() == default_group_a); // collective-group ordering

	// even though "default-group (c)" is submitted last, it only depends on its predecessor in the same group
	const auto default_group_c = all_instrs.select_unique<host_task_instruction_record>("default-group (c)");
	CHECK(default_group_c->collective_group_id == default_cgid);
	CHECK(default_group_c.predecessors() == default_group_b); // collective-group ordering
	CHECK(default_group_c.successors().is_unique<epoch_instruction_record>());

	// clone-collective-group instructions are ordered, because cloning an MPI communicator is a collective operation as well
	const auto clone_for_custom_group_1 = clone_for_default_group.successors().select_unique<clone_collective_group_instruction_record>();
	CHECK(clone_for_custom_group_1->original_collective_group_id == root_collective_group_id);
	CHECK(clone_for_custom_group_1->new_collective_group_id != clone_for_custom_group_1->original_collective_group_id);
	const auto custom_cgid_1 = clone_for_custom_group_1->new_collective_group_id;
	CHECK(custom_cgid_1 != default_cgid);

	const auto custom_group_1_a = all_instrs.select_unique<host_task_instruction_record>("custom-group 1 (a)");
	CHECK(custom_group_1_a->collective_group_id == custom_cgid_1);
	CHECK(custom_group_1_a.predecessors() == clone_for_custom_group_1);

	const auto custom_group_1_b = all_instrs.select_unique<host_task_instruction_record>("custom-group 1 (b)");
	CHECK(custom_group_1_b->collective_group_id == custom_cgid_1);
	CHECK(custom_group_1_b.predecessors() == custom_group_1_a); // collective-group ordering
	CHECK(custom_group_1_b.successors().is_unique<epoch_instruction_record>());

	// clone-collective-group instructions are ordered, because cloning an MPI communicator is a collective operation as well
	const auto clone_for_custom_group_2 = clone_for_custom_group_1.successors().select_unique<clone_collective_group_instruction_record>();
	CHECK(clone_for_custom_group_2->original_collective_group_id == root_collective_group_id);
	CHECK(clone_for_custom_group_2->new_collective_group_id != clone_for_custom_group_2->original_collective_group_id);
	const auto custom_cgid_2 = clone_for_custom_group_2->new_collective_group_id;
	CHECK(custom_cgid_2 != default_cgid);
	CHECK(custom_cgid_2 != custom_cgid_1);

	const auto custom_group_2 = all_instrs.select_unique<host_task_instruction_record>("custom-group 2");
	CHECK(custom_group_2->collective_group_id == custom_cgid_2);
	CHECK(custom_group_2.predecessors() == clone_for_custom_group_2);
	CHECK(custom_group_2.successors().is_unique<epoch_instruction_record>());
}

TEMPLATE_TEST_CASE_SIG("buffer fences export data to user memory", "[instruction_graph_generator][instruction-graph][fence]", ((int Dims), Dims), 0, 1, 2, 3) {
	constexpr static auto full_range = test_utils::truncate_range<Dims>({256, 256, 256});
	const auto export_subrange = GENERATE(values({subrange<Dims>({}, full_range), test_utils::truncate_subrange<Dims>({{48, 64, 72}, {128, 128, 128}})}));

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto buf = ictx.create_buffer<int>(full_range);
	ictx.device_compute(full_range).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.fence(buf, export_subrange);
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto fence = all_instrs.select_unique<fence_instruction_record>();
	const auto& fence_buffer_info = std::get<fence_instruction_record::buffer_variant>(fence->variant);
	CHECK(fence_buffer_info.bid == buf.get_id());
	CHECK(fence_buffer_info.box.get_offset() == id_cast<3>(export_subrange.offset));
	CHECK(fence_buffer_info.box.get_range() == range_cast<3>(export_subrange.range));

	// copy to user memory
	const auto fence_copy = fence.predecessors().assert_unique<copy_instruction_record>();
	CHECK(fence_copy->buffer_id == buf.get_id());
	CHECK(fence_copy->source_allocation.id.get_memory_id() == host_memory_id);
	CHECK(fence_copy->dest_allocation.id.get_memory_id() == user_memory_id);
	CHECK(fence_copy->dest_allocation.offset_bytes == 0);
	CHECK(fence_copy->dest_box == box_cast<3>(box(export_subrange)));
	CHECK(fence_copy->copy_region == region_cast<3>(region(box(export_subrange))));
	CHECK(fence_copy->element_size == sizeof(int));

	// exports are done from host memory, so data needs to be staged from device memory
	const auto staging_copy = fence_copy.predecessors().assert_unique<copy_instruction_record>();
	CHECK(staging_copy->dest_allocation == fence_copy->source_allocation);
	CHECK(staging_copy->copy_region == fence_copy->copy_region);
	CHECK(staging_copy->buffer_id == buf.get_id());
}

TEST_CASE("instruction_graph_generator gracefully handles empty-range buffer fences", "[instruction_graph_generator][instruction-graph][fence]") {
	const auto local_nid = GENERATE(values<node_id>({0, 1}));

	test_utils::idag_test_context ictx(2 /* nodes */, local_nid, 1 /* devices */);
	auto buf = ictx.create_buffer<int>(range(256));
	ictx.device_compute(buf.get_range()).discard_write(buf, acc::one_to_one()).submit();
	ictx.fence(buf, subrange<1>());
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto init_epoch = all_instrs.select_unique(task_manager::initial_epoch_task);
	const auto fence = all_instrs.select_unique<fence_instruction_record>();
	const auto shutdown_epoch = all_instrs.select_unique<epoch_instruction_record>(
	    [](const epoch_instruction_record& einstr) { return einstr.epoch_action == epoch_action::shutdown; });
	CHECK(fence.predecessors() == init_epoch);
	CHECK(fence.successors() == shutdown_epoch);

	const auto& garbage = shutdown_epoch->garbage;
	CHECK(garbage.user_allocations.size() == 1);
	CHECK(garbage.user_allocations.at(0) != null_allocation_id);
	CHECK(garbage.user_allocations.at(0).get_memory_id() == user_memory_id);

	CHECK(all_instrs.count<send_instruction_record>() == 0);
	CHECK(all_instrs.count<receive_instruction_record>() == 0);
	CHECK(all_instrs.count<split_receive_instruction_record>() == 0);
	CHECK(all_instrs.count<await_receive_instruction_record>() == 0);
	CHECK(all_instrs.count<copy_instruction_record>() == 0);
}

TEST_CASE("horizons and epochs notify the executor of unreferenced user allocations after buffer fences",
    "[instruction_graph_generator][instruction-graph][fence]") //
{
	const auto trigger = GENERATE(values<std::string>({"horizon", "epoch"}));
	CAPTURE(trigger);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(trigger == "horizon" ? 2 : 999);
	auto buf = ictx.create_buffer<int>(range(256));
	ictx.host_task(buf.get_range()).name("writer").discard_write(buf, acc::one_to_one()).submit();
	ictx.fence(buf, subrange({}, buf.get_range()));
	if(trigger == "horizon") { ictx.host_task(buf.get_range()).name("horizon trigger").read_write(buf, acc::one_to_one()).submit(); }
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	instruction_garbage garbage;
	if(trigger == "horizon") {
		const auto horizon = all_instrs.select_unique<horizon_instruction_record>();
		garbage = horizon->garbage;
	} else {
		const auto epoch = all_instrs.select_unique<epoch_instruction_record>(
		    [](const epoch_instruction_record& einstr) { return einstr.epoch_action == epoch_action::shutdown; });
		garbage = epoch->garbage;
	}
	CHECK(garbage.user_allocations.size() == 1);
	CHECK(garbage.user_allocations.at(0) != null_allocation_id);
	CHECK(garbage.user_allocations.at(0).get_memory_id() == user_memory_id);
}

TEST_CASE("epochs notify the executor of unreferenced user allocations after a buffer is destroyed", "[instruction_graph_generator][instruction-graph]") {
	const bool host_initialized = GENERATE(values<int>({false, true}));
	CAPTURE(host_initialized);

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	(void)ictx.create_buffer<int>(range(256), host_initialized);
	ictx.finish(); // destroys all live buffers

	const auto all_instrs = ictx.query_instructions();
	const auto shutdown_epoch = all_instrs.select_unique<epoch_instruction_record>(
	    [](const epoch_instruction_record& einstr) { return einstr.epoch_action == epoch_action::shutdown; });

	const auto garbage = shutdown_epoch->garbage;
	if(host_initialized) {
		CHECK(garbage.user_allocations.size() == 1);
		CHECK(garbage.user_allocations.at(0) != null_allocation_id);
		CHECK(garbage.user_allocations.at(0).get_memory_id() == user_memory_id);
	} else {
		CHECK(garbage.user_allocations.empty());
	}
}

TEST_CASE("host-object fences introduce the appropriate dependencies", "[instruction_graph_generator][instruction-graph][fence]") {
	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	auto ho = ictx.create_host_object();
	ictx.master_node_host_task().name("task 1").affect(ho).submit();
	ictx.fence(ho);
	ictx.master_node_host_task().name("task 2").affect(ho).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto task_1 = all_instrs.select_unique<host_task_instruction_record>("task 1");
	const auto fence = all_instrs.select_unique<fence_instruction_record>();
	const auto task_2 = all_instrs.select_unique<host_task_instruction_record>("task 2");

	CHECK(task_1.successors() == fence);
	CHECK(fence.successors() == task_2);

	const auto& ho_fence_info = std::get<fence_instruction_record::host_object_variant>(fence->variant);
	CHECK(ho_fence_info.hoid == ho.get_id());
}

TEST_CASE("epochs serialize execution and compact dependency tracking", "[instruction_graph_generator][instruction-graph][compaction]") {
	enum test_mode_enum { baseline_without_barrier_epoch, test_with_barrier_epoch };
	const auto test_mode = GENERATE(values({baseline_without_barrier_epoch, test_with_barrier_epoch}));
	INFO((test_mode == baseline_without_barrier_epoch ? "baseline: no epoch is inserted" : "test: barrier epoch is inserted"));

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(999);

	auto buf = ictx.create_buffer(range(256));
	auto ho = ictx.create_host_object();
	// we initialize the buffer on the device to get a single source allocation for the d2h copy after the barrier
	ictx.device_compute(range(1)).name("producer").discard_write(buf, acc::all()).submit();
	// there are two concurrent writers to `buf` on D0, which would generate two concurrent d2h copies if there were no epoch before the read
	ictx.device_compute(range(1)).name("producer").discard_write(buf, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute(range(1)).name("producer").discard_write(buf, acc::fixed<1>({128, 256})).submit();
	ictx.master_node_host_task().name("producer").affect(ho).submit();
	std::optional<task_id> barrier_epoch_tid;
	if(test_mode == test_with_barrier_epoch) { barrier_epoch_tid = ictx.epoch(epoch_action::barrier); }
	ictx.master_node_host_task().name("consumer").read(buf, acc::all()).affect(ho).submit();
	ictx.finish();

	// this test is very explicit about each instruction in the graph - things might easily break when touching IDAG generation.
	const auto all_instrs = ictx.query_instructions();

	// we expect an init epoch + optional barrier epoch + shutdown epoch
	CHECK(all_instrs.count<epoch_instruction_record>() == (test_mode == baseline_without_barrier_epoch ? 2 : 3));

	if(test_mode == baseline_without_barrier_epoch) {
		// Rudimentary check that without an epoch, the IDAG splits the copy to enable concurrency.
		// For a more thorough test of this, see "local copies are split on writers to facilitate compute-copy overlap" in instruction_graph_memory_tests.
		CHECK(all_instrs.count<copy_instruction_record>() == 2);
		return;
	}

	const auto init_epoch = all_instrs.select_unique<epoch_instruction_record>(task_manager::initial_epoch_task);
	CHECK(init_epoch.predecessors().count() == 0);

	const auto all_device_allocs = all_instrs.select_all<alloc_instruction_record>(
	    [](const alloc_instruction_record& ainstr) { return ainstr.allocation_id.get_memory_id() != host_memory_id; });
	CHECK(all_device_allocs.predecessors() == init_epoch);

	const auto all_producers = all_instrs.select_all("producer");

	// the barrier epoch (aka slow_full_sync) will transitively depend on all previous instructions
	const auto barrier_epoch = all_instrs.select_unique<epoch_instruction_record>(barrier_epoch_tid.value());
	CHECK(barrier_epoch->epoch_action == epoch_action::barrier);
	CHECK(barrier_epoch.transitive_predecessors() == union_of(init_epoch, all_device_allocs, all_producers));

	// There will only be a single d2h copy, since inserting the epoch will compact the last-writer tracking structures, replacing the two concurrent device
	// kernels with the single epoch.
	const auto host_alloc = all_instrs.select_unique<alloc_instruction_record>(
	    [](const alloc_instruction_record& ainstr) { return ainstr.allocation_id.get_memory_id() == host_memory_id; });
	CHECK(host_alloc.predecessors() == barrier_epoch);
	const auto d2h_copy = host_alloc.successors().select_unique<copy_instruction_record>();

	const auto consumer = all_instrs.select_unique<host_task_instruction_record>("consumer");
	const auto all_frees = all_instrs.select_all<free_instruction_record>();
	const auto destroy_ho = all_instrs.select_unique<destroy_host_object_instruction_record>();
	const auto shutdown_epoch = consumer.transitive_successors().select_unique<epoch_instruction_record>();
	CHECK(shutdown_epoch->epoch_action == epoch_action::shutdown);
	CHECK(shutdown_epoch.successors().count() == 0);

	// all instructions generated after the barrier epoch must transitively depend on it
	CHECK(barrier_epoch.transitive_successors() == union_of(host_alloc, d2h_copy, consumer, all_frees, destroy_ho, shutdown_epoch));

	// there can be no dependencies from instructions after the epoch
	CHECK(union_of(all_device_allocs, all_producers, barrier_epoch) == union_of(init_epoch, all_device_allocs, all_producers).successors());

	// there can be no dependencies to instructions before the epoch
	CHECK(union_of(barrier_epoch, host_alloc, d2h_copy, consumer, all_frees, destroy_ho)
	          .contains(union_of(host_alloc, d2h_copy, consumer, all_frees, destroy_ho, shutdown_epoch).predecessors()));
}

TEST_CASE("horizon application serializes execution and compacts dependency tracking", "[instruction_graph_generator][instruction-graph][compaction]") {
	enum test_mode_enum { baseline_without_horizons = 0, baseline_with_unapplied_horizon = 1, test_with_applied_horizon = 2 };
	const auto test_mode = GENERATE(values({baseline_without_horizons, baseline_with_unapplied_horizon, test_with_applied_horizon}));
	INFO((test_mode == baseline_without_horizons         ? "baseline: no horizons are inserted"
	      : test_mode == baseline_with_unapplied_horizon ? "baseline: horizon is inserted, but not applied"
	                                                     : "test: applying horizon"));
	const auto expected_num_horizons = static_cast<int>(test_mode); // we have defined test_mode_enum accordingly
	CAPTURE(expected_num_horizons);

	const int horizon_step = 3; // so no producer below triggers a horizon

	test_utils::idag_test_context ictx(1 /* nodes */, 0 /* my nid */, 1 /* devices */);
	ictx.set_horizon_step(test_mode == baseline_without_horizons ? 999 : horizon_step);

	auto buf = ictx.create_buffer(range(256));
	auto test_ho = ictx.create_host_object();
	auto age_ho = ictx.create_host_object(); // we repeatedly affect this host object to trigger horizon generation

	ictx.master_node_host_task().name("producer").affect(test_ho).submit();
	ictx.device_compute(range(1)).name("producer").discard_write(buf, acc::all()).submit();
	ictx.device_compute(range(1)).name("producer").discard_write(buf, acc::fixed<1>({0, 128})).submit();
	ictx.device_compute(range(1)).name("producer").discard_write(buf, acc::fixed<1>({128, 128})).submit();

	for(int i = 0; i < expected_num_horizons * horizon_step; ++i) {
		ictx.master_node_host_task().name("age").affect(age_ho).submit();
	}

	ictx.master_node_host_task().name("consumer").read(buf, acc::all()).affect(test_ho).affect(age_ho).submit();
	ictx.finish();

	// this test is very explicit about each instruction in the graph - things might easily break when touching IDAG generation.
	const auto all_instrs = ictx.query_instructions();

	const auto all_horizons = all_instrs.select_all<horizon_instruction_record>();
	REQUIRE(all_horizons.count() == static_cast<size_t>(expected_num_horizons));

	if(test_mode != test_with_applied_horizon) {
		// Rudimentary check that without applying a horizon, the IDAG splits the copy to enable concurrency.
		// For a more thorough test of this, see "local copies are split on writers to facilitate compute-copy overlap" in instruction_graph_memory_tests.
		CHECK(all_instrs.count<copy_instruction_record>() == 2);
		return;
	}

	// instruction records are in the order they were generated
	const auto applied_horizon = all_horizons[0];
	const auto current_horizon = all_horizons[1];

	const auto all_device_allocs = all_instrs.select_all<alloc_instruction_record>(
	    [](const alloc_instruction_record& ainstr) { return ainstr.allocation_id.get_memory_id() != host_memory_id; });
	const auto all_producers = all_instrs.select_all("producer");
	CHECK(applied_horizon.transitive_predecessors().contains(union_of(all_device_allocs, all_producers)));

	// There will only be a single d2h copy, since inserting the epoch will compact the last-writer tracking structures, replacing the two concurrent device
	// kernels with the single epoch.
	const auto host_alloc = all_instrs.select_unique<alloc_instruction_record>(
	    [](const alloc_instruction_record& ainstr) { return ainstr.allocation_id.get_memory_id() == host_memory_id; });
	const auto d2h_copy = host_alloc.successors().select_unique<copy_instruction_record>();
	const auto consumer = all_instrs.select_unique<host_task_instruction_record>("consumer");
	const auto all_frees = all_instrs.select_all<free_instruction_record>();
	const auto all_destroy_hos = all_instrs.select_all<destroy_host_object_instruction_record>();
	CHECK(applied_horizon.transitive_successors().contains(union_of(host_alloc, d2h_copy, consumer, all_frees, all_destroy_hos, current_horizon)));

	// The current horizon has been generated through the dependency chain on `age_ho`, and before submission of the consumer.
	CHECK_FALSE(union_of(host_alloc, d2h_copy, consumer, all_frees, all_destroy_hos, current_horizon).transitive_successors().contains(current_horizon));

	// The current horizon has not been applied, so no instructions except the shutdown epoch depend on it.
	CHECK(current_horizon.successors().is_unique<epoch_instruction_record>());
}

TEST_CASE("instruction_graph_generator throws in tests if it detects an uninitialized read", "[instruction_graph_generator]") {
	const size_t num_devices = 2;
	const range<1> device_range{num_devices};

	test_utils::idag_test_context::policy_set policy;
	policy.tm.uninitialized_read_error = error_policy::ignore;    // otherwise we get task-level errors first
	policy.dggen.uninitialized_read_error = error_policy::ignore; // otherwise we get command-level errors first

	test_utils::idag_test_context ictx(1, 0, num_devices, true /* supports d2d copies */, policy);

	SECTION("from a read-accessor on a fully uninitialized buffer") {
		auto buf = ictx.create_buffer<1>({1});
		CHECK_THROWS_WITH((ictx.device_compute(device_range).read(buf, acc::all()).submit()),
		    "Instructions for device kernel T1 are trying to read B0 {[0,0,0] - [1,1,1]}, which is neither found locally nor has been await-pushed before.");
	}

	SECTION("from a read-accessor on a partially, locally initialized buffer") {
		auto buf = ictx.create_buffer<1>(device_range);
		ictx.device_compute(range(1)).discard_write(buf, acc::one_to_one()).submit();
		CHECK_THROWS_WITH((ictx.device_compute(device_range).read(buf, acc::all()).submit()),
		    "Instructions for device kernel T2 are trying to read B0 {[1,0,0] - [2,1,1]}, which is neither found locally nor has been await-pushed before.");
	}

	SECTION("from a read-accessor on a partially, remotely initialized buffer") {
		auto buf = ictx.create_buffer<1>(device_range);
		ictx.device_compute(range(1)).discard_write(buf, acc::one_to_one()).submit();
		CHECK_THROWS_WITH((ictx.device_compute(device_range).read(buf, acc::one_to_one()).submit()),
		    "Instructions for device kernel T2 are trying to read B0 {[1,0,0] - [2,1,1]}, which is neither found locally nor has been await-pushed before.");
	}

	SECTION("from a reduction including the current value of an uninitialized buffer") {
		auto buf = ictx.create_buffer<1>({1});
		CHECK_THROWS_WITH((ictx.device_compute(device_range).reduce(buf, true /* include current buffer value */).submit()),
		    "Instructions for device kernel T1 are trying to read B0 {[0,0,0] - [1,1,1]}, which is neither found locally nor has been await-pushed before.");
	}
}

TEST_CASE("instruction_graph_generator throws in tests if it detects overlapping writes", "[instruction_graph_generator]") {
	const size_t num_devices = 2;
	test_utils::idag_test_context ictx(1, 0, num_devices);
	auto buf = ictx.create_buffer<2>({20, 20});

	SECTION("on all-write") {
		CHECK_THROWS_WITH((ictx.device_compute(buf.get_range()).discard_write(buf, acc::all()).submit()),
		    "Device kernel T1 has overlapping writes on N0 in B0 {[0,0,0] - [20,20,1]}. Choose a non-overlapping range mapper for this write access or "
		    "constrain the split via experimental::constrain_split to make the access non-overlapping.");
	}

	SECTION("on neighborhood-write") {
		CHECK_THROWS_WITH((ictx.device_compute(buf.get_range()).discard_write(buf, acc::neighborhood(1, 1)).submit()),
		    "Device kernel T1 has overlapping writes on N0 in B0 {[9,0,0] - [11,20,1]}. Choose a non-overlapping range mapper for this write access or "
		    "constrain the split via experimental::constrain_split to make the access non-overlapping.");
	}
}

// If there is no local last writer nor a pending await push for a buffer region being read, instruction_graph_generator should treat the memory being read from
// as coherent and the alloc_instruction (or the horizon / epoch that subsumes it) as the last writer for establishing dependencies.
TEST_CASE("instruction_graph_generator gracefully handles uninitialized reads when check is disabled", "[instruction_graph_generator]") {
	test_utils::idag_test_context::policy_set policy;
	policy.tm.uninitialized_read_error = error_policy::ignore;
	policy.dggen.uninitialized_read_error = error_policy::ignore;
	policy.iggen.uninitialized_read_error = error_policy::ignore;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* local nid */, 1 /* num devices */, true /* supports d2d copies */, policy);
	auto buf = ictx.create_buffer<1>({1});

	SECTION("from a device-kernel read-accessor") { //
		ictx.device_compute(range(1)).read(buf, acc::all()).submit();
		ictx.finish();

		const auto all_instrs = ictx.query_instructions();
		const auto kernel = all_instrs.select_unique<device_kernel_instruction_record>();
		const auto alloc = all_instrs.select_unique<alloc_instruction_record>();
		const auto free = all_instrs.select_unique<free_instruction_record>();
		CHECK(kernel.predecessors().contains(alloc));
		CHECK(kernel.successors().contains(free));
	}

	SECTION("from a reduction including the current buffer value") {
		ictx.device_compute(range(1)).reduce(buf, true /* include current buffer value */).submit();
		ictx.finish();

		const auto all_instrs = ictx.query_instructions();
		const auto kernel = all_instrs.select_unique<device_kernel_instruction_record>();
		const auto local_reduce = all_instrs.select_unique<reduce_instruction_record>();
		const auto alloc_device = all_instrs.select_unique<alloc_instruction_record>([](const alloc_instruction_record& ainstr) {
			return ainstr.origin == alloc_instruction_record::alloc_origin::buffer && ainstr.allocation_id.get_memory_id() != host_memory_id;
		});
		const auto alloc_host = all_instrs.select_unique<alloc_instruction_record>([](const alloc_instruction_record& ainstr) {
			return ainstr.origin == alloc_instruction_record::alloc_origin::buffer && ainstr.allocation_id.get_memory_id() == host_memory_id;
		});
		const auto free_device = all_instrs.select_unique<free_instruction_record>(
		    [&](const free_instruction_record& finstr) { return finstr.allocation_id == alloc_device->allocation_id; });
		const auto free_host = all_instrs.select_unique<free_instruction_record>(
		    [&](const free_instruction_record& finstr) { return finstr.allocation_id == alloc_host->allocation_id; });

		CHECK(kernel.transitive_predecessors().contains(alloc_device));
		CHECK(kernel.transitive_successors_across<copy_instruction_record>().contains(free_device));
		CHECK(local_reduce.transitive_predecessors().contains(alloc_host));
		CHECK(local_reduce.successors().contains(free_host));
	}

	SECTION("from a host-task read-accessor") { //
		ictx.host_task(range(1)).read(buf, acc::all()).submit();
		ictx.finish();

		const auto all_instrs = ictx.query_instructions();
		const auto host_task = all_instrs.select_unique<host_task_instruction_record>();
		const auto alloc = all_instrs.select_unique<alloc_instruction_record>();
		const auto free = all_instrs.select_unique<free_instruction_record>();
		CHECK(host_task.predecessors().contains(alloc));
		CHECK(host_task.successors().contains(free));
	}
}

TEST_CASE("instruction_graph_generator gracefully handles overlapping writes when check is disabled", "[instruction_graph_generator]") {
	size_t num_devices = 1;
	size_t oversubscription = 1;
	SECTION("between chunks of one kernel split among multiple devices") {
		num_devices = 2;
		oversubscription = 1;
	}
	SECTION("between an oversubscribed kernel on a single device") {
		num_devices = 1;
		oversubscription = 2;
	}

	test_utils::idag_test_context::policy_set policy;
	policy.dggen.overlapping_write_error = error_policy::ignore;
	policy.iggen.overlapping_write_error = error_policy::ignore;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* local nid */, num_devices, true /* supports d2d copies */, policy);
	auto buf = ictx.create_buffer<1>({1});
	ictx.device_compute(range(2)).name("writer").hint(hints::oversubscribe(oversubscription)).discard_write(buf, acc::all()).submit();
	ictx.device_compute(range(1)).name("reader").read(buf, acc::all()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	const auto all_allocs = all_instrs.select_all<alloc_instruction_record>();
	const auto all_writers = all_instrs.select_all<device_kernel_instruction_record>("writer");
	const auto reader = all_instrs.select_unique<device_kernel_instruction_record>("reader");
	const auto all_frees = all_instrs.select_all<free_instruction_record>();
	const auto free_after_reader =
	    all_frees.select_unique([&](const free_instruction_record& finstr) { return finstr.allocation_id == reader->access_map.at(0).allocation_id; });

	CHECK(all_allocs.count() == num_devices);
	CHECK(all_writers.count() == num_devices * oversubscription);
	CHECK(all_allocs.successors().contains(all_writers));
	CHECK(reader.transitive_predecessors().contains(all_writers));
	CHECK(reader.successors() == free_after_reader);
	CHECK(all_frees.count() == all_allocs.count());

	for(const auto& writer : all_writers.iterate()) {
		const auto free_after_writer =
		    all_frees.select_unique([&](const free_instruction_record& finstr) { return finstr.allocation_id == writer->access_map.at(0).allocation_id; });
		CHECK(writer.transitive_successors().contains(free_after_writer));
	}
}

TEMPLATE_TEST_CASE_SIG("hints::split_2d results in a 2d split", "[instruction_graph_generator][instruction-graph]", ((int Dims), Dims), 2, 3) {
	const size_t num_nodes = 1;
	const node_id local_nid = 0;
	const size_t num_devices = 4;

	test_utils::idag_test_context ictx(num_nodes, local_nid, num_devices);
	const auto range = test_utils::truncate_range<Dims>({256, 256, 256});
	auto buf = ictx.create_buffer(range);
	ictx.device_compute(range).hint(hints::split_2d()).discard_write(buf, acc::one_to_one()).submit();
	ictx.finish();

	const auto all_instrs = ictx.query_instructions();

	const auto all_kernels = all_instrs.select_all<device_kernel_instruction_record>();
	CHECK(all_kernels.count() == num_devices);

	box_vector<3> kernel_boxes;
	for(const auto& kernel : all_kernels.iterate()) {
		// 2D split means that no tile spans either dimension 0 or 1 entirely
		CHECK(kernel->execution_range.get_range()[0] < range[0]);
		CHECK(kernel->execution_range.get_range()[1] < range[1]);
		REQUIRE(kernel->access_map.size() == 1);
		const auto& write = kernel->access_map.front();
		CHECK(write.accessed_box_in_buffer == box(kernel->execution_range));
		kernel_boxes.push_back(box(kernel->execution_range));
	}
	CHECK(region(std::move(kernel_boxes)) == box(subrange(id<3>(), range_cast<3>(range))));
}

TEMPLATE_TEST_CASE_SIG("oversubscription splits local chunks recursively", "[instruction_graph_generator][instruction-graph]", ((int Dims), Dims), 1, 2, 3) {
	const size_t num_nodes = 1;
	const node_id local_nid = 0;
	const size_t num_devices = 2;
	const size_t oversub_factor = 4;

	test_utils::idag_test_context ictx(num_nodes, local_nid, num_devices);
	auto buf = ictx.create_buffer(test_utils::truncate_range<Dims>({256, 256, 256}));

	// This code is duck-typing identical for {device_kernel and host_task}_instruction_record, so we use a generic lambda to DRY it
	const auto check_is_box_tiling = [&](const auto& all_device_kernels_or_host_tasks) {
		box_vector<3> kernel_boxes;
		for(const auto& kernel : all_device_kernels_or_host_tasks.iterate()) {
			const auto kernel_box = detail::box(kernel->execution_range);
			REQUIRE(kernel->access_map.size() == 1);
			const auto& write = kernel->access_map.front();
			CHECK(write.accessed_box_in_buffer == kernel_box);
			kernel_boxes.push_back(kernel_box);
		}
		CHECK(region(std::move(kernel_boxes)) == box(subrange(id<3>(), range_cast<3>(buf.get_range()))));
	};

	SECTION("for device kernels") {
		SECTION("with a 1d split") {
			ictx.device_compute(buf.get_range())
			    .hint(hints::split_1d())
			    .hint(hints::oversubscribe(oversub_factor))
			    .discard_write(buf, acc::one_to_one())
			    .submit();
		}
		SECTION("with a 2d split") {
			ictx.device_compute(buf.get_range())
			    .hint(hints::split_2d())
			    .hint(hints::oversubscribe(oversub_factor))
			    .discard_write(buf, acc::one_to_one())
			    .submit();
		}

		const auto all_instrs = ictx.query_instructions();
		const auto all_kernels = all_instrs.select_all<device_kernel_instruction_record>();
		CHECK(all_kernels.count() == num_devices * oversub_factor);
		CHECK(all_kernels.count(device_id(0)) == oversub_factor);
		CHECK(all_kernels.count(device_id(1)) == oversub_factor);
		check_is_box_tiling(all_kernels);
	}

	SECTION("for host tasks kernels") {
		SECTION("with a 1d split") {
			ictx.host_task(buf.get_range()).hint(hints::split_1d()).hint(hints::oversubscribe(oversub_factor)).discard_write(buf, acc::one_to_one()).submit();
		}
		SECTION("with a 2d split") {
			ictx.host_task(buf.get_range()).hint(hints::split_2d()).hint(hints::oversubscribe(oversub_factor)).discard_write(buf, acc::one_to_one()).submit();
		}

		const auto all_instrs = ictx.query_instructions();
		const auto all_host_tasks = all_instrs.select_all<host_task_instruction_record>();
		CHECK(all_host_tasks.count() == oversub_factor);
		check_is_box_tiling(all_host_tasks);
	}
}

TEST_CASE("instruction_graph_generator throws in tests when detecting unsafe oversubscription", "[instruction_graph_generator]") {
	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* local nid */, 1 /* num devices */);
	auto buf = ictx.create_buffer(range(1));
	auto ho = ictx.create_host_object();

	SECTION("in device-kernels with reductions") {
		CHECK_THROWS_WITH(ictx.device_compute(range(256)).hint(hints::oversubscribe(2)).reduce(buf, false /* include_current_buffer_value */).submit(),
		    "Refusing to oversubscribe device kernel T1 because it performs a reduction.");
	}

	SECTION("in master-node host tasks") {
		CHECK_THROWS_WITH(ictx.master_node_host_task().hint(hints::oversubscribe(2)).submit(),
		    "Refusing to oversubscribe master-node host task T1 because its iteration space cannot be split.");
	}

	SECTION("in collective tasks") {
		CHECK_THROWS_WITH(ictx.collective_host_task().hint(hints::oversubscribe(2)).submit(),
		    "Refusing to oversubscribe collective host task T1 because it participates in a collective group.");
	}

	SECTION("in host tasks with side effects") {
		CHECK_THROWS_WITH(ictx.host_task(range(256)).hint(hints::oversubscribe(2)).affect(ho).submit(),
		    "Refusing to oversubscribe host-compute task T1 because it has side effects.");
	}
}

TEST_CASE("instruction_graph_generator gracefully handles unsafe oversubscription when check is disabled", "[instruction_graph_generator]") {
	test_utils::idag_test_context::policy_set policy;
	policy.iggen.unsafe_oversubscription_error = error_policy::ignore;

	test_utils::idag_test_context ictx(1 /* num nodes */, 0 /* local nid */, 1 /* num devices */, true /* supports d2d copies */, policy);
	auto buf = ictx.create_buffer(range(1));
	auto ho = ictx.create_host_object();

	SECTION("in device-kernels with reductions") {
		ictx.device_compute(range(256)).hint(hints::oversubscribe(2)).reduce(buf, false /* include_current_buffer_value */).submit();
	}

	SECTION("on master-node host tasks") { //
		ictx.master_node_host_task().hint(hints::oversubscribe(2)).submit();
	}

	SECTION("on collective tasks") { //
		ictx.collective_host_task().hint(hints::oversubscribe(2)).submit();
	}

	SECTION("on host tasks with side effects") { //
		ictx.host_task(range(256)).hint(hints::oversubscribe(2)).affect(ho).submit();
	}

	ictx.finish();

	const auto all_instrs = ictx.query_instructions();
	CHECK(all_instrs.count<device_kernel_instruction_record>() + all_instrs.count<host_task_instruction_record>() == 1);
}
