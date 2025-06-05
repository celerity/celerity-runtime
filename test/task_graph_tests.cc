#include "cgf.h"
#include "task.h"
#include "task_graph_test_utils.h"
#include "task_manager.h"
#include "test_utils.h"
#include "types.h"

#include <iterator>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>


namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	/// Returns true if all recorded dependencies between the given tasks match the given kind and origin.
	bool all_dependencies_match(const task_recorder& recorder, const task_id predecessor, const task_id successor, const dependency_kind kind,
	    const dependency_origin origin = dependency_origin::dataflow) {
		const auto& deps = recorder.get_dependencies();
		return std::all_of(deps.begin(), deps.end(),
		    [&](const auto& dep) { return (dep.predecessor != predecessor || dep.successor != successor) || (dep.kind == kind && dep.origin == origin); });
	}

	/// Returns true if at least one recorded dependency between the given tasks matches the given kind and origin.
	bool some_dependencies_match(const task_recorder& recorder, const task_id predecessor, const task_id successor, const dependency_kind kind,
	    const dependency_origin origin = dependency_origin::dataflow) {
		const auto& deps = recorder.get_dependencies();
		return std::any_of(deps.begin(), deps.end(),
		    [&](const auto& dep) { return dep.predecessor == predecessor && dep.successor == successor && dep.kind == kind && dep.origin == origin; });
	}

	TEST_CASE("task_manager calls into delegate on task creation", "[task_manager]") {
		struct counter_delegate final : public task_manager::delegate {
			size_t counter = 0;
			void task_created(const task* /* tsk */) override { counter++; }
		};

		counter_delegate delegate;
		task_graph tdag;
		task_manager tm{1, tdag, nullptr, &delegate};
		tm.generate_epoch_task(epoch_action::init);
		CHECK(delegate.counter == 1);
		const range<2> gs = {1, 1};
		const id<2> go = {};
		tm.generate_command_group_task(invoke_command_group_function([=](handler& cgh) { cgh.parallel_for<class kernel>(gs, go, [](auto) {}); }));
		CHECK(delegate.counter == 2);
		tm.generate_command_group_task(invoke_command_group_function([](handler& cgh) { cgh.host_task(on_master_node, [] {}); }));
		CHECK(delegate.counter == 3);
	}

	TEST_CASE("task_manager correctly records compute task information", "[task_manager][task][device_compute_task]") {
		test_utils::tdag_test_context tctx(1 /* num collective nodes */);

		auto buf_a = tctx.create_buffer(range<2>(64, 152), true /* host_initialized */);
		auto buf_b = tctx.create_buffer(range<3>(7, 21, 99));
		const auto tid =
		    tctx.device_compute(range<2>(32, 128), id<2>(32, 24)).read(buf_a, one_to_one{}).discard_write(buf_b, fixed{subrange<3>{{}, {5, 18, 74}}}).submit();

		const auto tsk = test_utils::get_task(tctx.get_task_graph(), tid);
		CHECK(tsk->get_type() == task_type::device_compute);
		CHECK(tsk->get_dimensions() == 2);
		CHECK(tsk->get_global_size() == range<3>{32, 128, 1});
		CHECK(tsk->get_global_offset() == id<3>{32, 24, 0});

		auto& bam = tsk->get_buffer_access_map();
		const auto bufs = bam.get_accessed_buffers();
		CHECK(bufs.size() == 2);
		CHECK(bufs.contains(buf_a.get_id()));
		CHECK(bufs.contains(buf_b.get_id()));
		CHECK(bam.get_nth_access(0) == std::pair{buf_a.get_id(), access_mode::read});
		CHECK(bam.get_nth_access(1) == std::pair{buf_b.get_id(), access_mode::discard_write});
		const auto reqs_a = bam.compute_consumed_region(buf_a.get_id(), subrange{tsk->get_global_offset(), tsk->get_global_size()});
		CHECK(reqs_a == box(subrange<3>({32, 24, 0}, {32, 128, 1})));
		const auto reqs_b = bam.compute_produced_region(buf_b.get_id(), subrange{tsk->get_global_offset(), tsk->get_global_size()});
		CHECK(reqs_b == box(subrange<3>({}, {5, 18, 74})));
	}

	TEST_CASE("buffer_access_map merges multiple accesses with the same mode", "[task][device_compute_task]") {
		std::vector<buffer_access> accs;
		accs.push_back(buffer_access{0, access_mode::read, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{3, 0}, {10, 20}}, range<2>{30, 30})});
		accs.push_back(buffer_access{0, access_mode::read, std::make_unique<range_mapper<2, fixed<2>>>(subrange<2>{{10, 0}, {7, 20}}, range<2>{30, 30})});
		const buffer_access_map bam{std::move(accs), task_geometry{2, {100, 100, 1}, {}, {}}};
		const auto req = bam.compute_consumed_region(0, subrange<3>({0, 0, 0}, {100, 100, 1}));
		CHECK(req == box(subrange<3>({3, 0, 0}, {14, 20, 1})));
	}

	TEST_CASE("tasks gracefully handle get_requirements() calls for buffers they don't access", "[task]") {
		const buffer_access_map bam;
		const auto req = bam.compute_consumed_region(0, subrange<3>({0, 0, 0}, {100, 1, 1}));
		CHECK(req == box<3>());
	}

	TEST_CASE("task_manager respects range mapper results for finding dependencies", "[task_manager][task-graph]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto buf = tctx.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

		const auto tid_a = tctx.device_compute(range<1>(ones)).discard_write(buf, fixed<1>{{0, 64}}).submit();
		const auto tid_b = tctx.device_compute(range<1>(ones)).read(buf, fixed<1>{{0, 128}}).submit();
		CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_b)));
		CHECK(tctx.query_tasks(tctx.get_initial_epoch_task()).successors().contains(tctx.query_tasks(tid_b))); // for read of the host-initialized part

		const auto tid_c = tctx.device_compute(range<1>(ones)).read(buf, fixed<1>{{64, 128}}).submit();
		CHECK(tctx.query_tasks(tid_a).is_concurrent_with(tctx.query_tasks(tid_c)));
		CHECK(tctx.query_tasks(tctx.get_initial_epoch_task()).successors().contains(tctx.query_tasks(tid_c))); // for read of the host-initialized part
	}

	TEST_CASE("task_manager correctly generates anti-dependencies", "[task_manager][task-graph]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto buf = tctx.create_buffer(range<1>(128));

		// Write to the full buffer
		const auto tid_a = tctx.device_compute(range<1>(ones)).discard_write(buf, all{}).submit();
		// Read the first half of the buffer
		const auto tid_b = tctx.device_compute(range<1>(ones)).read(buf, fixed<1>{{0, 64}}).submit();
		CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_b)));
		// Overwrite the second half - no anti-dependency onto task_b should exist (but onto task_a)
		const auto tid_c = tctx.device_compute(range<1>(ones)).discard_write(buf, fixed<1>{{64, 64}}).submit();
		CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_c)));
		CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_a, tid_c, dependency_kind::anti_dep));
		CHECK(tctx.query_tasks(tid_b).is_concurrent_with(tctx.query_tasks(tid_c)));
		// Overwrite the first half - now only an anti-dependency onto task_b should exist
		const auto tid_d = tctx.device_compute(range<1>(ones)).discard_write(buf, fixed<1>{{0, 64}}).submit();
		CHECK(tctx.query_tasks(tid_b).successors().contains(tctx.query_tasks(tid_d)));
		CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_b, tid_d, dependency_kind::anti_dep));
		CHECK(tctx.query_tasks(tid_c).is_concurrent_with(tctx.query_tasks(tid_d)));
	}

	TEST_CASE("task_manager correctly handles host-initialized buffers", "[task_manager][task-graph]") {
		// we explicitly test reading from non_host_init_buf
		task_manager::policy_set tm_policy;
		tm_policy.uninitialized_read_error = error_policy::ignore;

		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */, {tm_policy});

		auto host_init_buf = tctx.create_buffer(range<1>(128), true /* mark_as_host_initialized */);
		auto non_host_init_buf = tctx.create_buffer(range<1>(128), false /* mark_as_host_initialized */);
		auto artificial_dependency_buf = tctx.create_buffer(range<1>(1), false /* mark_as_host_initialized */);

		const auto tid_a = tctx.device_compute(range<1>(ones)).read(host_init_buf, all{}).discard_write(artificial_dependency_buf, all{}).submit();
		CHECK(tctx.query_tasks(tctx.get_initial_epoch_task()).successors().contains(tctx.query_tasks(tid_a)));

		// introduce an arbitrary true-dependency to avoid the fallback epoch dependency that is generated for tasks without other true-dependencies
		const auto tid_b = tctx.device_compute(range<1>(ones)).read(non_host_init_buf, all{}).read(artificial_dependency_buf, all{}).submit();
		CHECK_FALSE(tctx.query_tasks(tctx.get_initial_epoch_task()).successors().contains(tctx.query_tasks(tid_b)));

		const auto tid_c = tctx.device_compute(range<1>(ones)).discard_write(host_init_buf, all{}).submit();
		CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_c)));
		CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_a, tid_c, dependency_kind::anti_dep));
		const auto tid_d = tctx.device_compute(range<1>(ones)).discard_write(non_host_init_buf, all{}).submit();
		// Since task b is essentially reading uninitialized garbage, it doesn't make a difference if we write into it concurrently
		CHECK(tctx.query_tasks(tid_b).is_concurrent_with(tctx.query_tasks(tid_d)));
	}

	template <typename Builder, int Dims, typename Functor>
	auto dispatch_get_access(Builder&& builder, test_utils::mock_buffer<Dims>& mb, access_mode mode, Functor rmfn) {
		switch(mode) {
		case access_mode::read: return std::forward<Builder>(builder.read(mb, rmfn)); break;
		case access_mode::write: return std::forward<Builder>(builder.write(mb, rmfn)); break;
		case access_mode::read_write: return std::forward<Builder>(builder.read_write(mb, rmfn)); break;
		case access_mode::discard_write: return std::forward<Builder>(builder.discard_write(mb, rmfn)); break;
		case access_mode::discard_read_write: return std::forward<Builder>(builder.discard_read_write(mb, rmfn)); break;
		default: utils::unreachable(); // LCOV_EXCL_LINE
		}
	}

	TEST_CASE("task_manager correctly handles dependencies for R/W modes", "[task_manager][task-graph]") {
		// A read-write access can also be implicitly created using a separate write and read, which is why we operate on "mode sets" here.
		const std::vector<std::vector<access_mode>> rw_mode_sets = {
		    {access_mode::discard_read_write}, {access_mode::read_write}, {access_mode::discard_write, access_mode::read}};

		for(const auto& mode_set : rw_mode_sets) {
			test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
			auto buf = tctx.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

			auto builder = tctx.device_compute(range<1>(ones));
			for(const auto& m : mode_set) {
				builder = dispatch_get_access(std::move(builder), buf, m, all{});
			}
			const auto tid_a = builder.submit();

			const auto tid_b = tctx.device_compute(range<1>(ones)).discard_write(buf, all{}).submit();
			CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_b)));
			CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_a, tid_b, dependency_kind::anti_dep));
		}
	}

	TEST_CASE("task_manager handles all producer/consumer combinations correctly", "[task_manager][task-graph]") {
		constexpr access_mode consumer_modes[] = {access_mode::read, access_mode::read_write, access_mode::write};
		constexpr access_mode producer_modes[] = {access_mode::discard_read_write, access_mode::discard_write, access_mode::read_write, access_mode::write};

		for(const auto& consumer_mode : consumer_modes) {
			for(const auto& producer_mode : producer_modes) {
				CAPTURE(consumer_mode);
				CAPTURE(producer_mode);

				test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
				auto buf = tctx.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

				const auto tid_a = dispatch_get_access(tctx.device_compute(range<1>(ones)), buf, producer_mode, all{}).submit();
				const auto tid_b = dispatch_get_access(tctx.device_compute(range<1>(ones)), buf, consumer_mode, all{}).submit();
				CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_b)));

				const auto tid_c = dispatch_get_access(tctx.device_compute(range<1>(ones)), buf, producer_mode, all{}).submit();
				const bool pure_consumer = consumer_mode == access_mode::read;
				const bool pure_producer = producer_mode == access_mode::discard_read_write || producer_mode == access_mode::discard_write;
				if(pure_consumer || pure_producer) {
					CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_b, tid_c, dependency_kind::anti_dep));
				} else {
					CHECK(some_dependencies_match(tctx.get_task_recorder(), tid_b, tid_c, dependency_kind::true_dep));
				}
			}
		}
	}

	TEST_CASE("task_manager generates pseudo-dependencies for collective host tasks", "[task_manager][task-graph]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		experimental::collective_group group;
		const auto tid_master = tctx.master_node_host_task().name("master").submit();
		const auto tid_collective_implicit_1 = tctx.collective_host_task().name("collective implicit 1").submit();
		const auto tid_collective_implicit_2 = tctx.collective_host_task().name("collective implicit 2").submit();
		const auto tid_collective_explicit_1 = tctx.collective_host_task(group).name("collective explicit 1").submit();
		const auto tid_collective_explicit_2 = tctx.collective_host_task(group).name("collective explicit 2").submit();

		CHECK(tctx.query_tasks(tid_master).is_concurrent_with(tctx.query_tasks(tid_collective_implicit_1)));
		CHECK(tctx.query_tasks(tid_master).is_concurrent_with(tctx.query_tasks(tid_collective_implicit_2)));
		CHECK(tctx.query_tasks(tid_master).is_concurrent_with(tctx.query_tasks(tid_collective_explicit_1)));
		CHECK(tctx.query_tasks(tid_master).is_concurrent_with(tctx.query_tasks(tid_collective_explicit_2)));

		CHECK(tctx.query_tasks(tid_collective_implicit_1).successors().contains(tctx.query_tasks(tid_collective_implicit_2)));
		CHECK(tctx.query_tasks(tid_collective_implicit_1).is_concurrent_with(tctx.query_tasks(tid_collective_explicit_1)));
		CHECK(tctx.query_tasks(tid_collective_implicit_1).is_concurrent_with(tctx.query_tasks(tid_collective_explicit_2)));

		CHECK(tctx.query_tasks(tid_collective_implicit_2).is_concurrent_with(tctx.query_tasks(tid_collective_explicit_1)));
		CHECK(tctx.query_tasks(tid_collective_implicit_2).is_concurrent_with(tctx.query_tasks(tid_collective_explicit_2)));

		CHECK(tctx.query_tasks(tid_collective_explicit_1).successors().contains(tctx.query_tasks(tid_collective_explicit_2)));
	}

	void check_path_length_and_front(const task_manager& tm, const task_graph& tdag, int path_length, const std::unordered_set<task_id>& exec_front) {
		{
			INFO("path length");
			CHECK(task_manager_testspy::get_max_pseudo_critical_path_length(tm) == path_length);
		}
		{
			INFO("execution front");
			std::unordered_set<task*> task_exec_front;
			std::transform(exec_front.cbegin(), exec_front.cend(), std::inserter(task_exec_front, task_exec_front.begin()),
			    [&](const task_id tid) { return const_cast<task*>(test_utils::get_task(tdag, tid)); });
			CHECK(task_manager_testspy::get_execution_front(tm) == task_exec_front);
		}
	}

	TEST_CASE("task_manager keeps track of max pseudo critical path length and task front", "[task_manager][task-graph][task-front]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto buf_a = tctx.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

		const auto tid_a = tctx.master_node_host_task().discard_write(buf_a, all{}).submit();
		check_path_length_and_front(tctx.get_task_manager(), tctx.get_task_graph(), 1, {tid_a}); // 1: we always depend on the initial epoch task

		const auto tid_b = tctx.master_node_host_task().read_write(buf_a, all{}).submit();
		check_path_length_and_front(tctx.get_task_manager(), tctx.get_task_graph(), 2, {tid_b});

		const auto tid_c = tctx.master_node_host_task().read(buf_a, all{}).submit();
		check_path_length_and_front(tctx.get_task_manager(), tctx.get_task_graph(), 3, {tid_c});

		const auto tid_d = tctx.master_node_host_task().submit();
		check_path_length_and_front(tctx.get_task_manager(), tctx.get_task_graph(), 3, {tid_c, tid_d});
	}

	TEST_CASE("task horizons are being generated with correct dependencies", "[task_manager][task-graph][task-horizon]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

		tctx.set_horizon_step(2);
		auto buf_a = tctx.create_buffer(range<1>(128), true /* mark_as_host_initialized */);

		tctx.master_node_host_task().discard_write(buf_a, all{}).submit();

		auto current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		CHECK(current_horizon == nullptr);

		const auto tid_c = tctx.master_node_host_task().read(buf_a, all{}).submit();

		current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_id() == tid_c + 1);
		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == 1);

		auto horizon_dependencies = current_horizon->get_dependencies();

		CHECK(std::distance(horizon_dependencies.begin(), horizon_dependencies.end()) == 1);
		CHECK(horizon_dependencies.begin()->node->get_id() == tid_c);

		std::set<task_id> expected_dependency_ids;

		// current horizon is always part of the active task front
		expected_dependency_ids.insert(current_horizon->get_id());
		expected_dependency_ids.insert(tctx.master_node_host_task().submit());
		expected_dependency_ids.insert(tctx.master_node_host_task().submit());
		expected_dependency_ids.insert(tctx.master_node_host_task().submit());
		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == 1);

		tctx.master_node_host_task().read_write(buf_a, all{}).submit();
		const auto tid_d = tctx.master_node_host_task().read_write(buf_a, all{}).submit();
		expected_dependency_ids.insert(tid_d);

		current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_id() == tid_d + 1);
		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == 2);

		horizon_dependencies = current_horizon->get_dependencies();
		CHECK(std::distance(horizon_dependencies.begin(), horizon_dependencies.end()) == 5);

		std::set<task_id> actual_dependecy_ids;
		for(auto dep : horizon_dependencies) {
			actual_dependecy_ids.insert(dep.node->get_id());
		}
		CHECK(expected_dependency_ids == actual_dependecy_ids);
	}

	TEST_CASE("task horizons are being generated for the parallelism limit", "[task_manager][task-graph][task-horizon]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

		// we set a high step but low max parallelism to make sure that all horizons in this test are generated due to the parallelism limit,
		// regardless of what the defaults for these values are
		tctx.set_horizon_step(256);
		const auto max_para = 3;
		tctx.get_task_manager().set_horizon_max_parallelism(max_para);

		const size_t buff_size = 128;
		const size_t num_tasks = 9;
		const size_t buff_elem_per_task = buff_size / num_tasks;
		auto buf_a = tctx.create_buffer(range<1>(buff_size), true /* mark_as_host_initialized */);

		auto current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		CHECK(current_horizon == nullptr);

		for(size_t i = 0; i < num_tasks; ++i) {
			const auto offset = buff_elem_per_task * i;
			tctx.master_node_host_task().read_write(buf_a, fixed<1>({offset, buff_elem_per_task})).submit();
		}

		// divided by "max_para - 1" since there is also always the previous horizon in the set
		const auto expected_num_horizons = num_tasks / (max_para - 1);
		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == expected_num_horizons);

		// the most recent horizon should have 3 predecessors: 1 other horizon and 2 host tasks we generated
		current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		REQUIRE(current_horizon != nullptr);
		CHECK(current_horizon->get_dependencies().size() == 3);
	}

	static inline region<3> make_region(size_t min, size_t max) { return box<3>({min, 0, 0}, {max, 1, 1}); }

	TEST_CASE("task horizons update previous writer data structure", "[task_manager][task-graph][task-horizon]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

		tctx.set_horizon_step(2);
		auto buf_a = tctx.create_buffer(range<1>(128));
		auto buf_b = tctx.create_buffer(range<1>(128));

		const auto tid_1 = tctx.master_node_host_task().discard_write(buf_a, fixed<1>({0, 64})).discard_write(buf_b, fixed<1>({0, 128})).submit();

		const auto tid_2 = tctx.master_node_host_task().discard_write(buf_a, fixed<1>({64, 64})).submit();
		tctx.master_node_host_task().read_write(buf_a, fixed<1>({32, 64})).submit();
		const auto tid_4 = tctx.master_node_host_task().read_write(buf_a, fixed<1>({32, 64})).submit();

		const auto horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == 1);
		CHECK(horizon != nullptr);

		tctx.master_node_host_task().discard_write(buf_b, fixed<1>({0, 128})).submit();
		tctx.master_node_host_task().discard_write(buf_b, fixed<1>({0, 128})).submit();

		{
			INFO("check that previous tasks are still last writers before the first horizon is applied");
			const auto& region_map_a = task_manager_testspy::get_last_writer(tctx.get_task_manager(), buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 32)).front().second == test_utils::get_task(tctx.get_task_graph(), tid_1));
			CHECK(region_map_a.get_region_values(make_region(96, 128)).front().second == test_utils::get_task(tctx.get_task_graph(), tid_2));
			CHECK(region_map_a.get_region_values(make_region(32, 96)).front().second == test_utils::get_task(tctx.get_task_graph(), tid_4));
		}

		[[maybe_unused]] const auto tid_8 = tctx.master_node_host_task().read_write(buf_b, fixed<1>({0, 128})).submit();

		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == 2);

		{
			INFO("check that only the previous horizon is the last writer of buff_a");
			const auto& region_map_a = task_manager_testspy::get_last_writer(tctx.get_task_manager(), buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 128)).front().second == horizon);
		}

		const auto tid_9 = tctx.master_node_host_task().read_write(buf_a, fixed<1>({64, 64})).submit();

		{
			INFO("check that the previous horizon and task 11 are last writers of buff_a");
			const auto& region_map_a = task_manager_testspy::get_last_writer(tctx.get_task_manager(), buf_a.get_id());
			CHECK(region_map_a.get_region_values(make_region(0, 64)).front().second == horizon);
			CHECK(region_map_a.get_region_values(make_region(64, 128)).front().second == test_utils::get_task(tctx.get_task_graph(), tid_9));
		}
	}

	TEST_CASE("previous task horizon is used as last writer for host-initialized buffers", "[task_manager][task-graph][task-horizon]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		tctx.set_horizon_step(2);

		task_id initial_last_writer_id = -1;
		{
			auto buf = tctx.create_buffer(range<1>(1), true /* mark_as_host_initialized */);
			const auto tid = tctx.master_node_host_task().read_write(buf, all{}).submit();
			const auto& deps = test_utils::get_task(tctx.get_task_graph(), tid)->get_dependencies();
			CHECK(std::distance(deps.begin(), deps.end()) == 1);
			initial_last_writer_id = deps.begin()->node->get_id();
		}
		CHECK(test_utils::has_task(tctx.get_task_graph(), initial_last_writer_id));

		// Create a bunch of tasks to trigger horizon cleanup
		{
			auto buf = tctx.create_buffer(range<1>(1));
			const task* last_executed_horizon = nullptr;
			// We need 7 tasks to generate a pseudo-critical path length of 6 (3x2 horizon step size),
			// and another one that triggers the actual deferred deletion.
			for(int i = 0; i < 8; ++i) {
				tctx.master_node_host_task().discard_write(buf, all{}).submit();
				const auto current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
				if(last_executed_horizon != nullptr && current_horizon->get_id() > last_executed_horizon->get_id()) {
					tctx.get_task_graph().erase_before_epoch(last_executed_horizon->get_id());
				}
				if(current_horizon != nullptr) { last_executed_horizon = current_horizon; }
			}
		}

		INFO("initial last writer with id " << initial_last_writer_id << " has been deleted");
		CHECK_FALSE(test_utils::has_task(tctx.get_task_graph(), initial_last_writer_id));

		auto buf = tctx.create_buffer(range<1>(1), true);
		const auto tid = tctx.master_node_host_task().read_write(buf, all{}).submit();
		const auto& deps = test_utils::get_task(tctx.get_task_graph(), tid)->get_dependencies();
		CHECK(std::distance(deps.begin(), deps.end()) == 1);
		const auto* new_last_writer = deps.begin()->node;
		CHECK(new_last_writer->get_type() == task_type::horizon);

		const auto current_horizon = task_manager_testspy::get_current_horizon(tctx.get_task_manager());
		REQUIRE(current_horizon);
		INFO("previous horizon is being used");
		CHECK(new_last_writer->get_id() < current_horizon->get_id());
	}

	TEST_CASE("collective host tasks do not order-depend on their predecessor if it is shadowed by a horizon", "[task_manager][task-graph][task-horizon]") {
		// Regression test: the order-dependencies between host tasks in the same collective group are built by tracking the last task in each collective group.
		// Once a horizon is inserted, new collective host tasks must order-depend on that horizon instead.

		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		tctx.set_horizon_step(2);
		auto buf = tctx.create_buffer(range<1>(1));

		tctx.collective_host_task().name("first_collective").submit();

		// generate exactly two horizons
		for(int i = 0; i < 4; ++i) {
			tctx.master_node_host_task().discard_write(buf, all{}).submit();
		}

		// This must depend on the first horizon, not first_collective
		const auto second_collective = tctx.collective_host_task().name("second_collective").read(buf, all{}).submit();

		const auto second_collective_deps = test_utils::get_task(tctx.get_task_graph(), second_collective)->get_dependencies();
		const auto master_node_dep = std::find_if(second_collective_deps.begin(), second_collective_deps.end(),
		    [](const task::dependency d) { return d.node->get_type() == task_type::master_node; });
		const auto horizon_dep = std::find_if(second_collective_deps.begin(), second_collective_deps.end(), //
		    [](const task::dependency d) { return d.node->get_type() == task_type::horizon; });

		CHECK(std::distance(second_collective_deps.begin(), second_collective_deps.end()) == 2);
		REQUIRE(master_node_dep != second_collective_deps.end());
		CHECK(master_node_dep->kind == dependency_kind::true_dep);
		REQUIRE(horizon_dep != second_collective_deps.end());
		CHECK(horizon_dep->kind == dependency_kind::true_dep);
	}

	TEST_CASE("buffer accesses with empty ranges do not generate data-flow dependencies", "[task_manager][task-graph]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto buf = tctx.create_buffer(range<2>(32, 32), true /* mark_as_host_initialized */);

		const auto write_sr = GENERATE(values({subrange<2>{{16, 16}, {0, 0}}, subrange<2>{{16, 16}, {8, 8}}}));
		const auto read_sr = GENERATE(values({subrange<2>{{1, 1}, {0, 0}}, subrange<2>{{8, 8}, {16, 16}}}));

		const auto read_empty = read_sr.range.size() == 0;
		const auto write_empty = write_sr.range.size() == 0;
		CAPTURE(read_empty);
		CAPTURE(write_empty);

		const auto write_tid = tctx.device_compute(range<2>(ones)).discard_write(buf, fixed<2>{write_sr}).submit();
		const auto read_tid = tctx.device_compute(range<2>(ones)).read(buf, fixed<2>{read_sr}).submit();

		if(read_empty || write_empty) {
			CHECK(tctx.query_tasks(write_tid).is_concurrent_with(tctx.query_tasks(read_tid)));
		} else {
			CHECK(tctx.query_tasks(write_tid).successors().contains(tctx.query_tasks(read_tid)));
		}
	}

	TEST_CASE("side effects generate appropriate task-dependencies", "[task_manager][task-graph][side-effect]") {
		using order = experimental::side_effect_order;
		static constexpr auto side_effect_orders = {order::sequential};

		// TODO placeholder: complete with dependency types for other side effect orders
		const auto expected_dependencies = std::unordered_map<std::pair<order, order>, std::optional<dependency_kind>, utils::pair_hash>{
		    {{order::sequential, order::sequential}, dependency_kind::true_dep}};

		const auto order_a = GENERATE(values(side_effect_orders));
		const auto order_b = GENERATE(values(side_effect_orders));

		CAPTURE(order_a);
		CAPTURE(order_b);

		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto ho_common = tctx.create_host_object(); // should generate dependencies
		auto ho_a = tctx.create_host_object();      // should NOT generate dependencies
		auto ho_b = tctx.create_host_object();      // -"-
		const auto tid_a = tctx.master_node_host_task().affect(ho_common, order_a).affect(ho_a, order_a).submit();
		const auto tid_b = tctx.master_node_host_task().affect(ho_common, order_b).affect(ho_b, order_b).submit();

		CHECK(tctx.query_tasks(tid_a).predecessors().assert_count(1).contains(tctx.query_tasks(tctx.get_initial_epoch_task())));

		const auto deps_b = test_utils::get_task(tctx.get_task_graph(), tid_b)->get_dependencies();
		const auto expected_b = expected_dependencies.at({order_a, order_b});
		CHECK(std::distance(deps_b.begin(), deps_b.end()) == expected_b.has_value());
		if(expected_b) {
			CHECK(deps_b.front().node == test_utils::get_task(tctx.get_task_graph(), tid_a));
			CHECK(deps_b.front().kind == *expected_b);
		}
	}

	TEST_CASE("side-effect dependencies are correctly subsumed by horizons", "[task_manager][task-graph][task-horizon]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		tctx.set_horizon_step(2);
		auto ho = tctx.create_host_object();

		tctx.master_node_host_task().name("first_task").affect(ho, experimental::side_effect_order::sequential).submit();

		// generate exactly two horizons
		auto buf = tctx.create_buffer(range<1>(1));
		for(int i = 0; i < 5; ++i) {
			tctx.master_node_host_task().discard_write(buf, all{}).submit();
		}
		const auto horizon_tid = task_manager_testspy::get_epoch_for_new_tasks(tctx.get_task_manager())->get_id();

		// This must depend on the first horizon, not first_task
		const auto second_task = tctx.master_node_host_task().name("second_task").affect(ho, experimental::side_effect_order::sequential).submit();
		CHECK(tctx.query_tasks(second_task).predecessors().assert_count(1).contains(tctx.query_tasks(horizon_tid)));
	}

	TEST_CASE("epochs create appropriate dependencies to predecessors and successors", "[task_manager][task-graph][epoch]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

		auto buf_a = tctx.create_buffer(range<1>(1));
		const auto tid_a = tctx.device_compute(range<1>(ones)).name("a").discard_write(buf_a, all{}).submit();

		auto buf_b = tctx.create_buffer(range<1>(1));
		const auto tid_b = tctx.device_compute(range<1>(ones)).name("b").discard_write(buf_b, all{}).submit();

		const auto tid_epoch = tctx.epoch(epoch_action::none);

		const auto tid_c = tctx.device_compute(range<1>(ones)).name("c").read(buf_a, all{}).submit();
		const auto tid_d = tctx.device_compute(range<1>(ones)).name("d").discard_write(buf_b, all{}).submit();
		const auto tid_e = tctx.device_compute(range<1>(ones)).name("e").discard_write(buf_a, all{}).submit();
		const auto tid_f = tctx.device_compute(range<1>(ones)).name("f").read(buf_b, all{}).submit();
		const auto tid_g = tctx.device_compute(range<1>(ones)).name("g").discard_write(buf_b, all{}).submit();

		CHECK(tctx.query_tasks(tid_a).successors().contains(tctx.query_tasks(tid_epoch)));
		CHECK(tctx.query_tasks(tid_b).successors().contains(tctx.query_tasks(tid_epoch)));
		CHECK(tctx.query_tasks(tid_epoch).successors().contains(tctx.query_tasks(tid_c)));
		CHECK_FALSE(tctx.query_tasks(tid_c).predecessors().contains(tctx.query_tasks(tid_a)));
		CHECK(tctx.query_tasks(tid_epoch).successors().contains(tctx.query_tasks(tid_d))); // needs a true_dep on barrier since it only has anti_deps otherwise
		CHECK_FALSE(tctx.query_tasks(tid_d).predecessors().contains(tctx.query_tasks(tid_b)));
		CHECK(tctx.query_tasks(tid_epoch).successors().contains(tctx.query_tasks(tid_e)));
		CHECK(tctx.query_tasks(tid_f).predecessors().contains(tctx.query_tasks(tid_d)));
		CHECK_FALSE(tctx.query_tasks(tid_f).predecessors().contains(tctx.query_tasks(tid_epoch)));
		CHECK(tctx.query_tasks(tid_g).predecessors().contains(tctx.query_tasks(tid_f)));
		CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_f, tid_g, dependency_kind::anti_dep));
		CHECK(
		    tctx.query_tasks(tid_g).predecessors().contains(tctx.query_tasks(tid_epoch))); // needs a true_dep on barrier since it only has anti_deps otherwise
	}

	TEST_CASE("inserting epochs resets the need for horizons", "[task_manager][task-graph][task-horizon][epoch]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		tctx.set_horizon_step(2);
		auto buf = tctx.create_buffer(range<1>(1));

		for(int i = 0; i < 3; ++i) {
			tctx.master_node_host_task().discard_write(buf, all{}).submit();
			tctx.epoch(epoch_action::none);
		}

		CHECK(test_utils::get_num_live_horizons(tctx.get_task_graph()) == 0);
	}

	TEST_CASE("a sequence of epochs without intermediate tasks has defined behavior", "[task_manager][task-graph][epoch]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

		auto tid_before = tctx.get_initial_epoch_task();
		for(const auto action : {epoch_action::barrier, epoch_action::shutdown}) {
			const auto tid = tctx.epoch(action);
			CAPTURE(tid_before, tid);
			CHECK(tctx.query_tasks(tid).predecessors().contains(tctx.query_tasks(tid_before)));
			tid_before = tid;
		}
	}

	TEST_CASE("fences introduce dependencies on host objects", "[task_manager][task-graph][fence]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto ho = tctx.create_host_object();

		const auto tid_a = tctx.collective_host_task().affect(ho, experimental::side_effect_order::sequential).submit();
		const auto tid_fence = tctx.fence(ho);
		const auto tid_b = tctx.collective_host_task().affect(ho, experimental::side_effect_order::sequential).submit();

		CHECK(tctx.query_tasks(tid_fence).predecessors().contains(tctx.query_tasks(tid_a)));
		CHECK(tctx.query_tasks(tid_b).predecessors().contains(tctx.query_tasks(tid_fence)));
	}

	TEST_CASE("fences introduce data dependencies", "[task_manager][task-graph][fence]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		auto buf = tctx.create_buffer(range<1>(1));

		const auto tid_a = tctx.master_node_host_task().discard_write(buf, all{}).submit();
		const auto tid_fence = tctx.fence(buf);
		const auto tid_b = tctx.master_node_host_task().discard_write(buf, all{}).submit();

		CHECK(tctx.query_tasks(tid_fence).predecessors().contains(tctx.query_tasks(tid_a)));
		CHECK(tctx.query_tasks(tid_b).predecessors().contains(tctx.query_tasks(tid_fence)));
		CHECK(all_dependencies_match(tctx.get_task_recorder(), tid_fence, tid_b, dependency_kind::anti_dep));
	}

	TEST_CASE("task_manager throws in tests if it detects an uninitialized read", "[task_manager]") {
		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);

		SECTION("on a fully uninitialized buffer") {
			auto buf = tctx.create_buffer<1>({1});

			CHECK_THROWS_WITH((tctx.device_compute(range<1>(ones)).name("uninit_read").read(buf, all{}).submit()),
			    "Device kernel T1 \"uninit_read\" declares a reading access on uninitialized B0 {[0,0,0] - [1,1,1]}.");
		}

		SECTION("on a partially initialized buffer") {
			auto buf = tctx.create_buffer<2>({64, 64});
			tctx.device_compute(range<2>(ones)).discard_write(buf, fixed<2>({{0, 0}, {32, 32}})).submit();

			CHECK_THROWS_WITH((tctx.device_compute(range<1>(ones)).name("uninit_read").read(buf, all{}).submit()),
			    "Device kernel T2 \"uninit_read\" declares a reading access on uninitialized B0 {[0,32,0] - [32,64,1], [32,0,0] - [64,64,1]}.");
		}
	}

	TEST_CASE("task_manager warns when when long-running programs frequently epoch-synchronize", "[task_manager]") {
		test_utils::allow_max_log_level(log_level::warn);

		const auto action = GENERATE(values({epoch_action::none, epoch_action::barrier}));

		test_utils::tdag_test_context tctx(1 /* num_collective_nodes */);
		for(int i = 0; i <= 25; ++i) {
			for(int j = 0; j < 5; ++j) {
				tctx.master_node_host_task().submit();
			}
			tctx.epoch(action);
		}
		tctx.epoch(epoch_action::shutdown);

		CHECK(test_utils::log_contains_exact(log_level::warn,
		    "Your program appears to call queue::wait() excessively, which may lead to performance degradation. Consider using queue::fence() "
		    "for data-dependent branching and employ queue::wait() for timing only on a very coarse granularity."));
	}

} // namespace detail
} // namespace celerity
