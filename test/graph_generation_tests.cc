#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "graph_builder.h"
#include "graph_generator.h"

namespace celerity {

void compare_cmd_subrange(const command_subrange& sr, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) {
	REQUIRE(sr.offset[0] == offset[0]);
	REQUIRE(sr.offset[1] == offset[1]);
	REQUIRE(sr.offset[2] == offset[2]);
	REQUIRE(sr.range[0] == range[0]);
	REQUIRE(sr.range[1] == range[1]);
	REQUIRE(sr.range[2] == range[2]);
}

TEST_CASE("graph_buidler correctly handles command ids", "[graph_builder]") {
	command_dag cdag;
	REQUIRE(GRAPH_PROP(cdag, next_cmd_id) == 0);
	REQUIRE(GRAPH_PROP(cdag, command_vertices).empty());
	detail::graph_builder gb(cdag);
	const auto cid_0 = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, 0, command::NOP, command_data{}, "Foo");
	const auto cid_1 = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, 0, command::NOP, command_data{}, "Foo");
	REQUIRE(GRAPH_PROP(cdag, next_cmd_id) == 2);
	gb.commit();
	REQUIRE(GRAPH_PROP(cdag, command_vertices).count(cid_0) == 1);
	REQUIRE(GRAPH_PROP(cdag, command_vertices).count(cid_1) == 1);
}

TEST_CASE("graph_builder correctly splits commands", "[graph_builder]") {
	task_dag tdag;
	boost::add_vertex(tdag);
	tdag[0].label = "Foo Task";

	command_dag cdag;
	detail::graph_builder gb(cdag);

	cdag_vertex begin_task_cmd_v, end_task_cmd_v;
	std::tie(begin_task_cmd_v, end_task_cmd_v) = detail::create_task_commands(tdag, cdag, gb, 0);
	gb.commit();
	REQUIRE(cdag.m_vertices.size() == 2);

	command_data compute_data{};
	compute_data.compute.subrange = subrange<2>{cl::sycl::id<2>{64, 0}, cl::sycl::range<2>{192, 512}};
	const auto compute_cid = gb.add_command(begin_task_cmd_v, end_task_cmd_v, 0, 0, command::COMPUTE, compute_data);
	gb.commit();
	const auto first_chunk = chunk<3>{cl::sycl::id<3>{64, 0, 0}, cl::sycl::range<3>{64, 256, 1}, cl::sycl::range<3>{192, 512, 1}};
	const auto second_chunk = chunk<3>{cl::sycl::id<3>{128, 256, 0}, cl::sycl::range<3>{128, 256, 1}, cl::sycl::range<3>{192, 512, 1}};
	const std::vector<chunk<3>> split_chunks = {first_chunk, second_chunk};
	const std::vector<node_id> nodes = {3, 5};
	gb.split_command(compute_cid, split_chunks, nodes);
	gb.commit();

	// Verify that original command has been deleted
	REQUIRE(cdag.m_vertices.size() == 4);
	REQUIRE(GRAPH_PROP(cdag, command_vertices).count(compute_cid) == 0);

	// Check that new commands have been properly created
	const auto first_v = GRAPH_PROP(cdag, command_vertices).at(3);
	const auto& first_data = cdag[first_v];
	REQUIRE(first_data.cmd == command::COMPUTE);
	REQUIRE(first_data.tid == 0);
	REQUIRE(first_data.cid == 3);
	REQUIRE(first_data.nid == 3);
	compare_cmd_subrange(first_data.data.compute.subrange, {64, 0, 0}, {64, 256, 1});

	const auto second_v = GRAPH_PROP(cdag, command_vertices).at(4);
	const auto& second_data = cdag[second_v];
	REQUIRE(second_data.cmd == command::COMPUTE);
	REQUIRE(second_data.tid == 0);
	REQUIRE(second_data.cid == 4);
	REQUIRE(second_data.nid == 5);
	compare_cmd_subrange(second_data.data.compute.subrange, {128, 256, 0}, {128, 256, 1});
}

// NOTE: It's not ideal that we only run this test in debug builds, but the check is also only done in debug builds
#ifdef NDEBUG
std::cerr << "NOTE: Some tests only run in debug builds" << std::endl;
#else
TEST_CASE("graph_builder throws if split chunks don't add up to original chunk") {
	task_dag tdag;
	boost::add_vertex(tdag);
	tdag[0].label = "Foo Task";

	command_dag cdag;
	detail::graph_builder gb(cdag);

	cdag_vertex begin_task_cmd_v, end_task_cmd_v;
	std::tie(begin_task_cmd_v, end_task_cmd_v) = detail::create_task_commands(tdag, cdag, gb, 0);
	gb.commit();
	REQUIRE(cdag.m_vertices.size() == 2);

	command_data compute_data{};
	compute_data.compute.subrange = subrange<2>{cl::sycl::id<2>{64, 0}, cl::sycl::range<2>{192, 512}};
	const auto compute_cid = gb.add_command(begin_task_cmd_v, end_task_cmd_v, 0, 0, command::COMPUTE, compute_data);
	gb.commit();
	const auto first_chunk = chunk<3>{cl::sycl::id<3>{32, 0, 0}, cl::sycl::range<3>{64, 256, 1}, cl::sycl::range<3>{192, 512, 1}};
	const auto second_chunk = chunk<3>{cl::sycl::id<3>{128, 256, 0}, cl::sycl::range<3>{64, 128, 1}, cl::sycl::range<3>{192, 512, 1}};
	const std::vector<chunk<3>> split_chunks = {first_chunk, second_chunk};
	const std::vector<node_id> nodes = {3, 5};
	REQUIRE_THROWS_WITH(gb.split_command(compute_cid, split_chunks, nodes), Catch::Equals("Invalid split"));
}
#endif

} // namespace celerity
