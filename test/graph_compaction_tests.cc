#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "command_graph_generator_test_utils.h"

#include "command_graph_generator.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

namespace celerity::detail {

struct region_map_testspy {
	template <typename T>
	static size_t get_num_unique_values(const region_map<T>& map) {
		std::unordered_set<T> values;
		const auto cb = [&values](const auto& /* box */, const T& value) { values.insert(value); };
		switch(map.m_dims) {
		case 0: return 1; break;
		case 1: map.template get_map<1>().for_each(cb); break;
		case 2: map.template get_map<2>().for_each(cb); break;
		case 3: map.template get_map<3>().for_each(cb); break;
		}
		return values.size();
	}
};

struct command_graph_generator_testspy {
	static size_t get_last_writer_num_regions(const command_graph_generator& cggen, const buffer_id bid) {
		return region_map_testspy::get_num_unique_values(cggen.m_buffers.at(bid).local_last_writer);
	}

	static size_t get_command_buffer_reads_size(const command_graph_generator& cggen) { return cggen.m_command_buffer_reads.size(); }
};

} // namespace celerity::detail

TEST_CASE("horizons prevent tracking data structures from growing indefinitely", "[horizon][command-graph]") {
	constexpr int num_timesteps = 100;

	cdag_test_context cctx(1);
	const size_t buffer_width = 300;
	auto buf_a = cctx.create_buffer(range<2>(num_timesteps, buffer_width));

	const int horizon_step_size = GENERATE(values({1, 2, 3}));
	CAPTURE(horizon_step_size);

	cctx.set_horizon_step(horizon_step_size);

	for(int t = 0; t < num_timesteps; ++t) {
		CAPTURE(t);
		const auto read_accessor = [=](celerity::chunk<1> chnk) {
			celerity::subrange<2> ret;
			ret.range = range<2>(t, buffer_width);
			ret.offset = id<2>(0, 0);
			return ret;
		};
		const auto write_accessor = [=](celerity::chunk<1> chnk) {
			celerity::subrange<2> ret;
			ret.range = range<2>(1, buffer_width);
			ret.offset = id<2>(t, 0);
			return ret;
		};
		cctx.device_compute<class UKN(timestep)>(range<1>(buffer_width)).read(buf_a, read_accessor).discard_write(buf_a, write_accessor).submit();

		auto& ggen = cctx.get_graph_generator(0);

		// Assert once we've reached steady state as to not overcomplicate things
		if(t > 2 * horizon_step_size) {
			const auto num_regions = command_graph_generator_testspy::get_last_writer_num_regions(ggen, buf_a.get_id());
			const size_t cmds_before_applied_horizon = 1;
			const size_t cmds_after_applied_horizon = horizon_step_size + ((t + 1) % horizon_step_size);
			REQUIRE_LOOP(num_regions == cmds_before_applied_horizon + cmds_after_applied_horizon);

			// Pruning happens with a one step delay after a horizon has been applied
			const size_t expected_reads = horizon_step_size + (t % horizon_step_size) + 1;
			REQUIRE_LOOP(command_graph_generator_testspy::get_command_buffer_reads_size(ggen) == expected_reads);
		}

		size_t horizon_count = 0;
		for(const auto* cmd : cctx.get_graph_generator(0).get_command_graph().all_commands()) {
			if(cmd->get_type() == command_type::horizon) { ++horizon_count; }
		}
		REQUIRE_LOOP(horizon_count <= 3);
	}
}

TEST_CASE("horizons correctly deal with antidependencies", "[horizon][command-graph]") {
	constexpr int num_nodes = 1;
	cdag_test_context cctx(num_nodes);

	// For this test, we need to generate 2 horizons but still have the first one be relevant
	// after the second is generated -> use 2 buffers A and B, with a longer task chain on A, and write to B later
	// step size is set to ensure expected horizons
	cctx.set_horizon_step(2);

	const auto full_range = range<1>(100);
	auto buf_a = cctx.create_buffer<1>(full_range);
	auto buf_b = cctx.create_buffer<1>(full_range);

	// write to buf_a and buf_b
	cctx.device_compute<class UKN(init_a_b)>(full_range).discard_write(buf_a, acc::one_to_one{}).discard_write(buf_b, acc::one_to_one{}).submit();

	// then read from buf_b to later induce anti-dependence
	cctx.device_compute<class UKN(read_b_before_first_horizon)>(full_range).read(buf_b, acc::one_to_one{}).submit();

	// here, the first horizon should have been generated
	const auto first_horizon = cctx.query<horizon_command_record>();
	CHECK(first_horizon.total_count() == 1);

	// do 3 more read/writes on buf_a to generate another horizon and apply the first one
	task_id buf_a_rw = -1;
	for(int i = 0; i < 3; ++i) {
		buf_a_rw = cctx.device_compute<class UKN(read_b_before_first_horizon)>(full_range).read_write(buf_a, acc::one_to_one{}).submit();
	}

	// now, do a write on buf_b which should generate an anti-dependency on the first horizon
	auto write_b_after_first_horizon = cctx.device_compute<class UKN(write_b_after_first_horizon)>(full_range)
	                                       // introduce an artificial true dependency to avoid the fallback epoch dependency generated for ordering
	                                       .read(buf_a, acc::one_to_one{})
	                                       .discard_write(buf_b, acc::one_to_one{})
	                                       .submit();

	CHECK(cctx.query(buf_a_rw).successors().contains(cctx.query(write_b_after_first_horizon)));
	CHECK(first_horizon.successors().contains(cctx.query(write_b_after_first_horizon)));
}

TEST_CASE("previous horizons are used as last writers for host-initialized buffers", "[command_graph_generator][horizon][command-graph]") {
	constexpr int num_nodes = 2;
	cdag_test_context cctx(num_nodes);

	cctx.set_horizon_step(2);

	const auto buf_range = range<1>(100);

	std::array<command_id, 2> initial_last_writer_ids = {-1, -1};
	{
		auto buf = cctx.create_buffer(buf_range, true /* mark_as_host_initialized */);

		cctx.device_compute(buf_range).name("access_host_init_buf").read_write(buf, acc::one_to_one{}).submit();
		const auto cmds = cctx.query<execution_command_record>("access_host_init_buf");
		REQUIRE(cmds.count_per_node() == 1);
		initial_last_writer_ids = {cmds.on(0)->id, cmds.on(1)->id};
	}

	// Create bunch of tasks to trigger horizon cleanup
	{
		auto buf = cctx.create_buffer(buf_range);
		task_id last_horizon_reached = task_manager::initial_epoch_task;
		// We need 7 tasks to generate a pseudo-critical path length of 6 (3x2 horizon step size),
		// and another one that triggers the actual deferred deletion.
		for(int i = 0; i < 8; ++i) {
			cctx.device_compute<class UKN(generate_horizon)>(buf_range).discard_write(buf, acc::one_to_one{}).submit();
			const auto current_horizon = task_manager_testspy::get_current_horizon(cctx.get_task_manager());
			if(current_horizon && *current_horizon > last_horizon_reached) {
				last_horizon_reached = *current_horizon;
				cctx.get_task_manager().notify_horizon_reached(last_horizon_reached);
			}
		}
	}

	// Check that initial last writers have been deleted
	CHECK_FALSE(cctx.get_graph_generator(0).get_command_graph().has(initial_last_writer_ids[0]));
	CHECK_FALSE(cctx.get_graph_generator(1).get_command_graph().has(initial_last_writer_ids[1]));

	auto buf = cctx.create_buffer(buf_range, true /* mark_as_host_initialized */);
	cctx.device_compute(buf_range).name("access_host_init_buf").read_write(buf, acc::one_to_one{}).submit();

	const auto new_last_writers = cctx.query("access_host_init_buf").predecessors();
	CHECK(difference_of(new_last_writers, cctx.query<epoch_command_record>()).assert_all<horizon_command_record>().total_count() == 2);
}

TEST_CASE("commands for collective host tasks do not order-depend on their predecessor if it is shadowed by a horizon",
    "[command_graph_generator][command-graph][horizon]") {
	// Regression test: the order-dependencies between host tasks in the same collective group are built by tracking the last task command in each
	// collective group. Once a horizon is inserted, commands for new collective host tasks must order-depend on that horizon command instead.
	const size_t num_nodes = 1;
	cdag_test_context cctx(num_nodes);
	cctx.set_horizon_step(2);

	cctx.collective_host_task().submit();

	// generate exactly two horizons
	auto buf = cctx.create_buffer(range<1>(1));
	for(int i = 0; i < 4; ++i) {
		cctx.master_node_host_task().discard_write(buf, acc::all{}).submit();
	}

	// This must depend on the first horizon, not first_collective
	const auto second_collective = cctx.collective_host_task().read(buf, acc::all{}).submit();
	const auto predecessors = cctx.query(second_collective).predecessors();
	CHECK(predecessors.total_count() == 2);
	const auto execution_cmd = predecessors.select_all<execution_command_record>();
	const auto horizon_cmd = predecessors.select_all<horizon_command_record>();
	CHECK(execution_cmd.successors().contains(cctx.query(second_collective)));
	CHECK(horizon_cmd.successors().contains(cctx.query(second_collective)));
}

TEST_CASE("side-effect dependencies are correctly subsumed by horizons", "[command_graph_generator][command-graph][horizon]") {
	const size_t num_nodes = 1;
	cdag_test_context cctx(num_nodes);
	cctx.set_horizon_step(2);

	auto ho = cctx.create_host_object();
	cctx.master_node_host_task().affect(ho, experimental::side_effect_order::sequential).submit();

	// generate exactly two horizons
	auto buf = cctx.create_buffer(range<1>(1));
	for(int i = 0; i < 4; ++i) {
		cctx.master_node_host_task().discard_write(buf, acc::all{}).submit();
	}

	// This must depend on the first horizon, not first_task
	const auto second_task = cctx.master_node_host_task().affect(ho, experimental::side_effect_order::sequential).submit();
	CHECK(cctx.query(second_task).predecessors().assert_all<horizon_command_record>().total_count() == 1);
}

TEST_CASE("reaching an epoch will prune all nodes of the preceding task graph", "[task_manager][task-graph][epoch]") {
	constexpr int num_nodes = 2;

	auto tt = test_utils::task_test_context{};

	const auto check_task_has_exact_dependencies = [&](const char* info, const task_id dependent,
	                                                   const std::initializer_list<std::tuple<task_id, dependency_kind, dependency_origin>> dependencies) {
		INFO(info);
		CAPTURE(dependent);
		const auto actual = tt.tm.get_task(dependent)->get_dependencies();
		CHECK(static_cast<size_t>(std::distance(actual.begin(), actual.end())) == dependencies.size());
		for(const auto& [tid, kind, origin] : dependencies) {
			CAPTURE(tid);
			size_t actual_count = 0;
			for(const auto& actual_dep : actual) {
				if(actual_dep.node->get_id() == tid) {
					CHECK(actual_dep.kind == kind);
					CHECK(actual_dep.origin == origin);
					actual_count += 1;
				}
			}
			CHECK(actual_count == 1);
		}
	};

	const auto node_range = range<1>{num_nodes};
	const auto init_tid = task_manager::initial_epoch_task;

	auto early_host_initialized_buf = tt.mbf.create_buffer(node_range, true);
	auto buf_written_from_kernel = tt.mbf.create_buffer(node_range, false);

	const auto writer_tid = test_utils::add_compute_task<class UKN(writer)>(
	    tt.tm, [&](handler& cgh) { buf_written_from_kernel.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, node_range);

	const auto epoch_tid = tt.tm.generate_epoch_task(epoch_action::none);

	const auto reader_writer_tid = test_utils::add_compute_task<class UKN(reader_writer)>(
	    tt.tm, [&](handler& cgh) { early_host_initialized_buf.get_access<access_mode::read_write>(cgh, acc::one_to_one{}); }, node_range);

	auto late_host_initialized_buf = tt.mbf.create_buffer(node_range, true);

	const auto late_writer_tid = test_utils::add_compute_task<class UKN(late_writer)>(
	    tt.tm, [&](handler& cgh) { late_host_initialized_buf.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, node_range);

	REQUIRE(tt.tm.has_task(init_tid));
	check_task_has_exact_dependencies("initial epoch task", init_tid, {});
	REQUIRE(tt.tm.has_task(writer_tid));
	check_task_has_exact_dependencies("writer", writer_tid, {{init_tid, dependency_kind::true_dep, dependency_origin::last_epoch}});
	REQUIRE(tt.tm.has_task(epoch_tid));
	check_task_has_exact_dependencies("epoch before", epoch_tid, {{writer_tid, dependency_kind::true_dep, dependency_origin::execution_front}});

	tt.tm.notify_epoch_reached(epoch_tid);

	const auto reader_tid = test_utils::add_compute_task<class UKN(reader)>(
	    tt.tm,
	    [&](handler& cgh) {
		    early_host_initialized_buf.get_access<access_mode::read>(cgh, acc::one_to_one{});
		    late_host_initialized_buf.get_access<access_mode::read>(cgh, acc::one_to_one{});
		    buf_written_from_kernel.get_access<access_mode::discard_write>(cgh, acc::one_to_one{});
	    },
	    node_range);

	CHECK(!tt.tm.has_task(init_tid));
	CHECK(!tt.tm.has_task(writer_tid));
	REQUIRE(tt.tm.has_task(epoch_tid));
	check_task_has_exact_dependencies("epoch after", epoch_tid, {});
	REQUIRE(tt.tm.has_task(reader_writer_tid));
	check_task_has_exact_dependencies("reader-writer", reader_writer_tid, {{epoch_tid, dependency_kind::true_dep, dependency_origin::dataflow}});
	REQUIRE(tt.tm.has_task(late_writer_tid));
	check_task_has_exact_dependencies("late writer", late_writer_tid, {{epoch_tid, dependency_kind::true_dep, dependency_origin::last_epoch}});
	REQUIRE(tt.tm.has_task(reader_tid));
	check_task_has_exact_dependencies("reader", reader_tid,
	    {
	        {epoch_tid, dependency_kind::anti_dep, dependency_origin::dataflow},
	        {reader_writer_tid, dependency_kind::true_dep, dependency_origin::dataflow},
	        {late_writer_tid, dependency_kind::true_dep, dependency_origin::dataflow},
	    });
}
