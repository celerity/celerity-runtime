#include "unit_test_suite_celerity.h"

#include <set>
#include <unordered_set>

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <catch2/catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "access_modes.h"

#include "test_utils.h"

namespace celerity {
namespace detail {

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

		constexpr int NUM_TIMESTEPS = 10;

		constexpr int NUM_NODES = 1;
		test_utils::cdag_test_context ctx(NUM_NODES);
		auto& inspector = ctx.get_inspector();
		test_utils::mock_buffer_factory mbf(ctx);
		auto full_range = cl::sycl::range<1>(300);
		auto buf_a = mbf.create_buffer<2>(cl::sycl::range<2>(NUM_TIMESTEPS, full_range.size()));

		fmt::print("buf_a region map size: {}\n", graph_generator_testspy::get_buffer_states_num_regions(ctx.get_graph_generator(), buf_a.get_id()));
		fmt::print("buf_a last writer map size: {}\n", graph_generator_testspy::get_buffer_last_writer_num_regions(ctx.get_graph_generator(), buf_a.get_id()));

		for(int timestep = 0; timestep < NUM_TIMESTEPS; ++timestep) {
			auto growing_read_accessor = [t = timestep](celerity::chunk<1> chnk) {
				celerity::subrange<2> ret;
				ret.range = cl::sycl::range<2>(t, chnk.global_size.get(0));
				ret.offset = cl::sycl::id<2>(0, 0);
				return ret;
			};

			auto latest_write_accessor = [t = timestep](celerity::chunk<1> chnk) {
				celerity::subrange<2> ret;
				ret.range = cl::sycl::range<2>(1, chnk.range.size());
				ret.offset = cl::sycl::id<2>(t, chnk.offset.get(0));
				return ret;
			};

			test_utils::build_and_flush(ctx, NUM_NODES,
			    test_utils::add_compute_task<class growing_read_kernel>(ctx.get_task_manager(),
			        [&](handler& cgh) {
				        buf_a.get_access<mode::read>(cgh, growing_read_accessor);
				        buf_a.get_access<mode::discard_write>(cgh, latest_write_accessor);
			        },
			        full_range));
		}

		fmt::print("buf_a region map size: {}\n", graph_generator_testspy::get_buffer_states_num_regions(ctx.get_graph_generator(), buf_a.get_id()));
		fmt::print("buf_a last writer map size: {}\n", graph_generator_testspy::get_buffer_last_writer_num_regions(ctx.get_graph_generator(), buf_a.get_id()));
		graph_generator_testspy::print_buffer_last_writers(ctx.get_graph_generator(), buf_a.get_id());

		// test_utils::build_and_flush(ctx, 1,
		//	test_utils::add_compute_task<class UKN(producer)>(ctx.get_task_manager(),
		//	    [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, full_range));

		// SECTION("when distributing a single reading task across nodes") {
		//	test_utils::build_and_flush(ctx, NUM_NODES,
		//		test_utils::add_compute_task<class UKN(producer)>(ctx.get_task_manager(),
		//			[&](handler& cgh) { buf_a.get_access<mode::read>(cgh, access::one_to_one<1>()); }, full_range));
		//}

		// SECTION("when distributing a single read-write task across nodes") {
		//	test_utils::build_and_flush(ctx, NUM_NODES,
		//		test_utils::add_compute_task<class UKN(producer)>(ctx.get_task_manager(),
		//			[&](handler& cgh) { buf_a.get_access<mode::read_write>(cgh, access::one_to_one<1>()); }, full_range));
		//}

		// SECTION("when running multiple reading task on separate nodes") {
		//	auto full_range_for_single_node = [=](node_id node) {
		//		return [=](chunk<1> chnk) -> subrange<1> {
		//			if(chnk.range == full_range) return chnk;
		//			if(chnk.offset[0] == (full_range.size() / NUM_NODES)*node) {
		//				return { 0, full_range };
		//			}
		//			return { 0, 0 };
		//		};
		//	};

		//	test_utils::build_and_flush(ctx, NUM_NODES,
		//		test_utils::add_compute_task<class UKN(producer)>(ctx.get_task_manager(),
		//			[&](handler& cgh) { buf_a.get_access<mode::read>(cgh, full_range_for_single_node(1)); }, full_range));

		//	test_utils::build_and_flush(ctx, NUM_NODES,
		//		test_utils::add_compute_task<class UKN(producer)>(ctx.get_task_manager(),
		//			[&](handler& cgh) { buf_a.get_access<mode::read>(cgh, full_range_for_single_node(2)); }, full_range));
		//}

		// IMPORTANT_CHECK(inspector.get_commands(boost::none, node_id(0), command_type::PUSH).size() == 2);
		// IMPORTANT_CHECK(inspector.get_commands(boost::none, node_id(1), command_type::PUSH).size() == 0);
		// IMPORTANT_CHECK(inspector.get_commands(boost::none, node_id(2), command_type::PUSH).size() == 0);
		// CHECK(inspector.get_commands(boost::none, node_id(1), command_type::AWAIT_PUSH).size() == 1);
		// CHECK(inspector.get_commands(boost::none, node_id(2), command_type::AWAIT_PUSH).size() == 1);

		maybe_print_graphs(ctx);
	}

} // namespace detail
} // namespace celerity
