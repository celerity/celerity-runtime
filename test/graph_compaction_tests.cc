#include "unit_test_suite_celerity.h"

#include <array>
#include <optional>
#include <set>
#include <unordered_set>

#include <catch2/catch.hpp>

#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::one_to_one;

	template <typename T>
	std::ostream& operator<<(std::ostream& os, const std::optional<T>& v) {
		return v != std::nullopt ? (os << *v) : (os << "nullopt");
	}

	struct region_map_testspy {
		template <typename T>
		static size_t get_num_regions(const region_map<T>& map) {
			return map.region_values.size();
		}
		template <typename T>
		static void print_regions(const region_map<T>& map) {
			for(auto& reg : map.region_values) {
				fmt::print("{} -> {}\n", reg.first, reg.second);
			}
		}
	};

	struct graph_generator_testspy {
		static size_t get_buffer_states_num_regions(const graph_generator& ggen, const buffer_id bid) {
			if(auto* distr_state = std::get_if<graph_generator::distributed_state>(&ggen.buffer_states.at(bid))) {
				return region_map_testspy::get_num_regions(distr_state->region_sources);
			} else {
				return 1;
			}
		}
		static size_t get_buffer_last_writer_num_regions(const graph_generator& ggen, const buffer_id bid) {
			return region_map_testspy::get_num_regions(ggen.node_data.at(node_id{0}).buffer_last_writer.at(bid));
		}
		static void print_buffer_last_writers(const graph_generator& ggen, const buffer_id bid) {
			region_map_testspy::print_regions(ggen.node_data.at(node_id{0}).buffer_last_writer.at(bid));
		}
		static size_t get_command_buffer_reads_size(const graph_generator& ggen) { return ggen.command_buffer_reads.size(); }
		static const std::vector<horizon_command*>& get_current_horizon_cmds(const graph_generator& ggen) { return ggen.current_horizon_cmds; }
	};

	TEST_CASE("horizons prevent number of regions from growing indefinitely", "[horizon][command-graph]") {
		using namespace cl::sycl::access;

		constexpr int NUM_TIMESTEPS = 100;

		constexpr int NUM_NODES = 3;
		test_utils::cdag_test_context ctx(NUM_NODES);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto full_range = cl::sycl::range<1>(300);
		auto buf_a = mbf.create_buffer<2>(cl::sycl::range<2>(NUM_TIMESTEPS, full_range.size()));

		auto buf_a_region_map_size = [&ctx, &buf_a] {
			return graph_generator_testspy::get_buffer_states_num_regions(ctx.get_graph_generator(), buf_a.get_id());
		};
		auto buf_a_last_writer_map_size = [&ctx, &buf_a] {
			return graph_generator_testspy::get_buffer_last_writer_num_regions(ctx.get_graph_generator(), buf_a.get_id());
		};

		auto time_series_lambda = [&](bool growing_reads) {
			for(int timestep = 0; timestep < NUM_TIMESTEPS; ++timestep) {
				auto read_accessor = [t = timestep, grow = growing_reads](celerity::chunk<1> chnk) {
					celerity::subrange<2> ret;
					ret.range = cl::sycl::range<2>(grow ? t : 1, chnk.global_size.get(0));
					ret.offset = cl::sycl::id<2>(grow ? 0 : std::max(t - 1, 0), 0);
					return ret;
				};

				auto latest_write_accessor = [t = timestep](celerity::chunk<1> chnk) {
					celerity::subrange<2> ret;
					ret.range = cl::sycl::range<2>(1, chnk.range.size());
					ret.offset = cl::sycl::id<2>(t, chnk.offset.get(0));
					return ret;
				};

				test_utils::build_and_flush(ctx, NUM_NODES,
				    test_utils::add_compute_task<class growing_read_kernel>(
				        ctx.get_task_manager(),
				        [&](handler& cgh) {
					        buf_a.get_access<mode::read>(cgh, read_accessor);
					        buf_a.get_access<mode::discard_write>(cgh, latest_write_accessor);
				        },
				        full_range));
			}
		};

		SECTION("with horizon step size 1") {
			ctx.get_task_manager().set_horizon_step(1);

			SECTION("and a growing read pattern") { time_series_lambda(true); }
			SECTION("and a latest-only read pattern") { time_series_lambda(false); }

			CHECK(buf_a_region_map_size() <= NUM_NODES * 2);
			CHECK(buf_a_last_writer_map_size() <= NUM_NODES * 2);
			for(node_id n = 0; n < NUM_NODES; ++n) {
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() <= NUM_TIMESTEPS);
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() >= NUM_TIMESTEPS - 1);
			}

			// also check that unused commands are deleted
			CHECK(ctx.get_command_graph().command_count() <= NUM_NODES * 13);
			// and are removed from the read cache
			CHECK(graph_generator_testspy::get_command_buffer_reads_size(ctx.get_graph_generator()) < NUM_NODES * 9);
		}

		SECTION("with horizon step size 3") {
			ctx.get_task_manager().set_horizon_step(3);

			SECTION("and a growing read pattern") { time_series_lambda(true); }
			SECTION("and a latest-only read pattern") { time_series_lambda(false); }

			CHECK(buf_a_region_map_size() <= NUM_NODES * 2 * 3);
			CHECK(buf_a_last_writer_map_size() <= NUM_NODES * 2 * 3);
			for(node_id n = 0; n < NUM_NODES; ++n) {
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() <= NUM_TIMESTEPS / 3 + 1);
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() >= NUM_TIMESTEPS / 3 - 1);
			}

			// also check that unused commands are deleted
			CHECK(ctx.get_command_graph().command_count() <= NUM_NODES * 13 * 3);
			// and are removed from the read cache
			CHECK(graph_generator_testspy::get_command_buffer_reads_size(ctx.get_graph_generator()) < NUM_NODES * 9 * 3);
		}

		// graph_generator_testspy::print_buffer_last_writers(ctx.get_graph_generator(), buf_a.get_id());

		maybe_print_graphs(ctx);
	}

	TEST_CASE("horizons correctly deal with antidependencies", "[horizon][command-graph]") {
		using namespace cl::sycl::access;

		constexpr int NUM_NODES = 1;
		test_utils::cdag_test_context ctx(NUM_NODES);

		// For this test, we need to generate 2 horizons but still have the first one be relevant
		// after the second is generated -> use 2 buffers A and B, with a longer task chan on A, and write to B later
		// step size is set to ensure expected horizons
		ctx.get_task_manager().set_horizon_step(1);

		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto full_range = cl::sycl::range<1>(100);
		auto buf_a = mbf.create_buffer<1>(full_range);
		auto buf_b = mbf.create_buffer<1>(full_range);

		// write to buf_a and buf_b
		test_utils::build_and_flush(ctx, NUM_NODES,
		    test_utils::add_compute_task<class UKN(init_a_b)>(
		        ctx.get_task_manager(),
		        [&](handler& cgh) {
			        buf_a.get_access<mode::discard_write>(cgh, one_to_one{});
			        buf_b.get_access<mode::discard_write>(cgh, one_to_one{});
		        },
		        full_range));

		// then read from buf_b to later induce anti-dependence
		test_utils::build_and_flush(ctx, NUM_NODES,
		    test_utils::add_compute_task<class UKN(read_b_before_first_horizon)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::read>(cgh, one_to_one{}); }, full_range));

		// here, the first horizon should have been generated

		// do 1 more read/writes on buf_a to generate another horizon and apply the first one
		for(int i = 0; i < 1; ++i) {
			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(buf_a_rw)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, one_to_one{}); }, full_range));
		}

		// now, do a write on buf_b which should generate an anti-dependency on the first horizon

		auto write_b_after_first_horizon = test_utils::build_and_flush(ctx, NUM_NODES,
		    test_utils::add_compute_task<class UKN(write_b_after_first_horizon)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, one_to_one{}); }, full_range));

		// Now we need to check various graph properties

		auto cmds = inspector.get_commands(write_b_after_first_horizon, {}, {});
		CHECK(cmds.size() == 1);
		auto deps = inspector.get_dependencies(*cmds.cbegin());
		CHECK(deps.size() == 1);

		// check that the dependee is the first horizon
		auto horizon_cmds = inspector.get_commands({}, {}, command_type::HORIZON);
		CHECK(horizon_cmds.size() == 3);
		CHECK(deps[0] == *horizon_cmds.cbegin());

		// and that it's an anti-dependence
		auto write_b_cmd = ctx.get_command_graph().get(*cmds.cbegin());
		auto write_b_dependencies = write_b_cmd->get_dependencies();
		CHECK(!write_b_dependencies.empty());
		CHECK(write_b_dependencies.front().kind == dependency_kind::ANTI_DEP);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("horizons are flushed correctly even if not directly dependent on tasks", "[horizon][command-graph]") {
		using namespace cl::sycl::access;

		constexpr int NUM_NODES = 2;
		test_utils::cdag_test_context ctx(NUM_NODES);

		// For this test, we need to generate a horizon that attaches only
		// to an execution front of "push", without directly attaching to any computes
		// as such our minimum possible horizon step for testing this is 2

		ctx.get_task_manager().set_horizon_step(2);

		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto full_range = cl::sycl::range<1>(100);
		auto buf_a = mbf.create_buffer<1>(full_range);

		// write to buf_a on all nodes
		test_utils::build_and_flush(ctx, NUM_NODES,
		    test_utils::add_compute_task<class UKN(init_a)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, one_to_one{}); }, full_range));

		// perform another read-write step to ensure that horizons are generated as expected
		test_utils::build_and_flush(ctx, NUM_NODES,
		    test_utils::add_compute_task<class UKN(rw_a)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, one_to_one{}); }, full_range));

		// now for the actual test, read only on node 0
		test_utils::build_and_flush(
		    ctx, NUM_NODES, test_utils::add_host_task(ctx.get_task_manager(), on_master_node, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, all{}); }));

		// build some additional read/write steps so that we reach deletion
		for(int i = 0; i < 2; ++i) {
			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(rw_a)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, one_to_one{}); }, full_range));
		}

		// check that all horizons were flushed
		auto horizon_cmds = inspector.get_commands({}, {}, command_type::HORIZON);
		CHECK(horizon_cmds.size() == 4);

		maybe_print_graphs(ctx);
	}

	TEST_CASE("previous horizons are used as last writers for host-initialized buffers", "[graph_generator][horizon][command-graph]") {
		using namespace cl::sycl::access;

		constexpr int NUM_NODES = 2;
		test_utils::cdag_test_context ctx(NUM_NODES);

		ctx.get_task_manager().set_horizon_step(2);

		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		const auto buf_range = cl::sycl::range<1>(100);

		std::array<command_id, 2> initial_last_writer_ids = {-1, -1};
		{
			auto buf = mbf.create_buffer(buf_range, true);
			const auto tid = test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class UKN(access_host_init_buf)>(
			        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::read_write>(cgh, one_to_one{}); }, buf_range));
			const auto cmds = inspector.get_commands(tid, std::nullopt, std::nullopt);
			CHECK(cmds.size() == 2);
			std::transform(cmds.begin(), cmds.end(), initial_last_writer_ids.begin(), [&](auto cid) {
				// (Implementation detail: We can't use the inspector here b/c NOP commands are not flushed)
				const auto deps = ctx.get_command_graph().get(cid)->get_dependencies();
				REQUIRE(std::distance(deps.begin(), deps.end()) == 1);
				return deps.begin()->node->get_cid();
			});
		}

		// Create bunch of tasks to trigger horizon cleanup
		{
			auto buf = mbf.create_buffer(buf_range);
			task_id last_executed_horizon = 0;
			// We need 7 tasks to generate a pseudo-critical path length of 6 (3x2 horizon step size),
			// and another one that triggers the actual deferred deletion.
			for(int i = 0; i < 8; ++i) {
				const auto tid = test_utils::build_and_flush(ctx, NUM_NODES,
				    test_utils::add_compute_task<class UKN(generate_horizons)>(
				        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, one_to_one{}); }, buf_range));
				const auto* current_horizon = task_manager_testspy::get_current_horizon_task(ctx.get_task_manager());
				if(current_horizon != nullptr && current_horizon->get_id() > last_executed_horizon) {
					last_executed_horizon = current_horizon->get_id();
					ctx.get_task_manager().notify_horizon_executed(last_executed_horizon);
				}
			}
		}

		for(auto cid : initial_last_writer_ids) {
			INFO("initial last writer with id " << cid << " has been deleted")
			CHECK_FALSE(ctx.get_command_graph().has(cid));
		}

		auto buf = mbf.create_buffer(buf_range, true);
		const auto tid = test_utils::build_and_flush(ctx, NUM_NODES,
		    test_utils::add_compute_task<class UKN(access_host_init_buf)>(
		        ctx.get_task_manager(), [&](handler& cgh) { buf.get_access<mode::read_write>(cgh, one_to_one{}); }, buf_range));

		const auto cmds = inspector.get_commands(tid, std::nullopt, std::nullopt);
		std::array<command_id, 2> new_last_writer_ids = {-1, -1};
		CHECK(cmds.size() == 2);
		std::transform(cmds.begin(), cmds.end(), new_last_writer_ids.begin(), [&](auto cid) {
			const auto deps = inspector.get_dependencies(cid);
			REQUIRE(deps.size() == 1);
			return deps[0];
		});

		CHECK(isa<horizon_command>(ctx.get_command_graph().get(new_last_writer_ids[0])));
		CHECK(isa<horizon_command>(ctx.get_command_graph().get(new_last_writer_ids[1])));

		const auto& current_horizon_cmds = graph_generator_testspy::get_current_horizon_cmds(ctx.get_graph_generator());
		INFO("previous horizons are being used");
		CHECK(std::none_of(current_horizon_cmds.cbegin(), current_horizon_cmds.cend(),
		    [&](auto* cmd) { return cmd->get_cid() == new_last_writer_ids[0] || cmd->get_cid() == new_last_writer_ids[1]; }));

		maybe_print_graphs(ctx);
	}

	TEST_CASE("commands for collective host tasks do not order-depend on their predecessor if it is shadowed by a horizon",
	    "[graph_generator][command-graph][horizon]") {
		// Regression test: the order-dependencies between host tasks in the same collective group are built by tracking the last task command in each
		// collective group. Once a horizon is inserted, commands for new collective host tasks must order-depend on that horizon command instead.

		const size_t num_nodes = 1;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		tm.set_horizon_step(2);

		const auto first_collective = test_utils::build_and_flush(ctx, test_utils::add_host_task(tm, experimental::collective, [&](handler& cgh) {}));

		// generate exactly two horizons
		auto& ggen = ctx.get_graph_generator();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(range<1>(1));
		for(int i = 0; i < 5; ++i) {
			test_utils::build_and_flush(
			    ctx, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); }));
		}

		// This must depend on the first horizon, not first_collective
		const auto second_collective = test_utils::build_and_flush(ctx, test_utils::add_host_task(tm, experimental::collective, [&](handler& cgh) {}));

		const auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();
		const auto first_commands = inspector.get_commands(first_collective, std::nullopt, std::nullopt);
		const auto second_commands = inspector.get_commands(second_collective, std::nullopt, std::nullopt);
		for(const auto second_cid : second_commands) {
			for(const auto first_cid : first_commands) {
				CHECK(!inspector.has_dependency(second_cid, first_cid));
			}
			const auto second_deps = cdag.get(second_cid)->get_dependencies();
			CHECK(std::distance(second_deps.begin(), second_deps.end()) == 1);
			for(const auto& dep : second_deps) {
				CHECK(dep.kind == dependency_kind::ORDER_DEP);
				CHECK(dynamic_cast<const horizon_command*>(dep.node));
			}
		}

		maybe_print_graphs(ctx);
	}

	// TODO deduplicate from test case above
	TEST_CASE("side-effect dependencies are correctly subsumed by horizons", "[graph_generator][command-graph][horizon]") {
		const size_t num_nodes = 1;
		test_utils::cdag_test_context ctx(num_nodes);
		auto& tm = ctx.get_task_manager();
		tm.set_horizon_step(2);

		test_utils::mock_host_object_factory mhof;
		auto ho = mhof.create_host_object();
		const auto first_task = test_utils::build_and_flush(
		    ctx, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); }));

		// generate exactly two horizons
		auto& ggen = ctx.get_graph_generator();
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(range<1>(1));
		for(int i = 0; i < 5; ++i) {
			test_utils::build_and_flush(
			    ctx, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, all{}); }));
		}

		// This must depend on the first horizon, not first_task
		const auto second_task = test_utils::build_and_flush(
		    ctx, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) { ho.add_side_effect(cgh, experimental::side_effect_order::sequential); }));

		const auto& inspector = ctx.get_inspector();
		auto& cdag = ctx.get_command_graph();
		const auto first_commands = inspector.get_commands(first_task, std::nullopt, std::nullopt);
		const auto second_commands = inspector.get_commands(second_task, std::nullopt, std::nullopt);
		for(const auto second_cid : second_commands) {
			for(const auto first_cid : first_commands) {
				CHECK(!inspector.has_dependency(second_cid, first_cid));
			}
			const auto second_deps = cdag.get(second_cid)->get_dependencies();
			CHECK(std::distance(second_deps.begin(), second_deps.end()) == 1);
			for(const auto& dep : second_deps) {
				CHECK(dep.kind == dependency_kind::TRUE_DEP);
				CHECK(dynamic_cast<const horizon_command*>(dep.node));
			}
		}

		maybe_print_graphs(ctx);
	}

} // namespace detail
} // namespace celerity
