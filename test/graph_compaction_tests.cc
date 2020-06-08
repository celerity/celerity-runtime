#include "unit_test_suite_celerity.h"

#include <optional>
#include <set>
#include <unordered_set>

#include <catch2/catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

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
			return region_map_testspy::get_num_regions(ggen.buffer_states.at(bid));
		}
		static size_t get_buffer_last_writer_num_regions(const graph_generator& ggen, const buffer_id bid) {
			return region_map_testspy::get_num_regions(ggen.node_data.at(node_id{0}).buffer_last_writer.at(bid));
		}
		static void print_buffer_last_writers(const graph_generator& ggen, const buffer_id bid) {
			region_map_testspy::print_regions(ggen.node_data.at(node_id{0}).buffer_last_writer.at(bid));
		}
	};

	TEST_CASE("graph_compactor prevents number of regions from growing indefinitely", "[graph_compactor][command-graph]") {
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
			ctx.get_graph_generator().set_horizon_step_size(1);

			SECTION("and a growing read pattern") { time_series_lambda(true); }
			SECTION("and a latest-only read pattern") { time_series_lambda(false); }

			CHECK(buf_a_region_map_size() <= NUM_NODES * 2);
			CHECK(buf_a_last_writer_map_size() <= NUM_NODES * 2);
			for(node_id n = 0; n < NUM_NODES; ++n) {
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() <= NUM_TIMESTEPS);
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() >= NUM_TIMESTEPS - 1);
			}
		}

		SECTION("with horizon step size 3") {
			ctx.get_graph_generator().set_horizon_step_size(3);

			SECTION("and a growing read pattern") { time_series_lambda(true); }
			SECTION("and a latest-only read pattern") { time_series_lambda(false); }

			CHECK(buf_a_region_map_size() <= NUM_NODES * 2 * 3);
			CHECK(buf_a_last_writer_map_size() <= NUM_NODES * 2 * 3);
			for(node_id n = 0; n < NUM_NODES; ++n) {
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() <= NUM_TIMESTEPS / 3 + 1);
				CHECK(inspector.get_commands(std::nullopt, n, command_type::HORIZON).size() >= NUM_TIMESTEPS / 3 - 1);
			}
		}

		// graph_generator_testspy::print_buffer_last_writers(ctx.get_graph_generator(), buf_a.get_id());

		maybe_print_graphs(ctx);
	}


} // namespace detail
} // namespace celerity
