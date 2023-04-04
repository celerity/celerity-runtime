#include <catch2/catch_test_macros.hpp>

#include "test_utils.h"


namespace celerity::detail {

using celerity::access::fixed;
using celerity::access::one_to_one;

TEST_CASE("task-graph printing is unchanged", "[print_graph][task-graph]") {
	task_manager tm{1, nullptr};
	test_utils::mock_buffer_factory mbf(tm);
	test_utils::mock_reduction_factory mrf;

	auto range = celerity::range<1>(64);
	auto buf_0 = mbf.create_buffer(range);
	auto buf_1 = mbf.create_buffer(celerity::range<1>(1));

	// graph copied from graph_gen_reduction_tests "graph_generator generates reduction command trees"

	test_utils::add_compute_task<class UKN(task_initialize)>(
	    tm, [&](handler& cgh) { buf_1.get_access<access_mode::discard_write>(cgh, one_to_one{}); }, range);
	test_utils::add_compute_task<class UKN(task_produce)>(
	    tm, [&](handler& cgh) { buf_0.get_access<access_mode::discard_write>(cgh, one_to_one{}); }, range);
	test_utils::add_compute_task<class UKN(task_reduce)>(
	    tm,
	    [&](handler& cgh) {
		    buf_0.get_access<access_mode::read>(cgh, one_to_one{});
		    test_utils::add_reduction(cgh, mrf, buf_1, true /* include_current_buffer_value */);
	    },
	    range);
	test_utils::add_compute_task<class UKN(task_consume)>(
	    tm,
	    [&](handler& cgh) {
		    buf_1.get_access<access_mode::read>(cgh, fixed<1>({0, 1}));
	    },
	    range);

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const auto expected =
	    "digraph G {label=\"Task Graph\" 0[shape=ellipse label=<T0<br/><b>epoch</b>>];1[shape=box style=rounded label=<T1 \"task_initialize_2\" "
	    "<br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>discard_write</i> B1 {[[0,0,0] - [1,1,1]]}>];0->1[color=orchid];2[shape=box style=rounded "
	    "label=<T2 \"task_produce_3\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>discard_write</i> B0 {[[0,0,0] - "
	    "[64,1,1]]}>];0->2[color=orchid];3[shape=box style=rounded label=<T3 \"task_reduce_4\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/>(R1) "
	    "<i>read_write</i> B1 {[[0,0,0] - [1,1,1]]}<br/><i>read</i> B0 {[[0,0,0] - [64,1,1]]}>];1->3[];2->3[];4[shape=box style=rounded label=<T4 "
	    "\"task_consume_5\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>read</i> B1 {[[0,0,0] - [1,1,1]]}>];3->4[];}";

	const auto dot = tm.print_graph(std::numeric_limits<size_t>::max()).value();
	CHECK(dot == expected);
}

TEST_CASE("command graph printing is unchanged", "[print_graph][command-graph]") {
	size_t num_nodes = 4;
	test_utils::cdag_test_context ctx(num_nodes);
	auto& tm = ctx.get_task_manager();
	auto& ggen = ctx.get_graph_generator();
	test_utils::mock_buffer_factory mbf(tm, ggen);
	test_utils::mock_reduction_factory mrf;

	auto buf_0 = mbf.create_buffer(range<1>{1});

	// graph copied from graph_gen_reduction_tests "graph_generator does not generate multiple reduction commands for redundant requirements"

	test_utils::build_and_flush(ctx, num_nodes,
	    test_utils::add_compute_task<class UKN(task_reduction)>(tm, [&](handler& cgh) { test_utils::add_reduction(cgh, mrf, buf_0, false); }, {num_nodes, 1}));
	test_utils::build_and_flush(ctx, num_nodes, test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
		buf_0.get_access<access_mode::read>(cgh, fixed<1>({0, 1}));
		buf_0.get_access<access_mode::read_write>(cgh, fixed<1>({0, 1}));
		buf_0.get_access<access_mode::write>(cgh, fixed<1>({0, 1}));
	}));

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const auto expected =
	    "digraph G{label=\"Command Graph\" subgraph cluster_2{label=<<font color=\"#606060\">T2 (master-node host)</font>>;color=darkgray;9[label=<C9 on "
	    "N0<br/><b>execution</b> [[0,0,0] - [0,0,0]]<br/><i>read</i> B0 {[[0,0,0] - [1,1,1]]}<br/><i>read_write</i> B0 {[[0,0,0] - "
	    "[1,1,1]]}<br/><i>write</i> "
	    "B0 {[[0,0,0] - [1,1,1]]}> fontcolor=black shape=box];}subgraph cluster_1{label=<<font color=\"#606060\">T1 \"task_reduction_8\" "
	    "(device-compute)</font>>;color=darkgray;5[label=<C5 on N0<br/><b>execution</b> [[0,0,0] - [1,1,1]]<br/>(R1) <i>discard_write</i> B0 {[[0,0,0] - "
	    "[1,1,1]]}> fontcolor=black shape=box];6[label=<C6 on N1<br/><b>execution</b> [[1,0,0] - [2,1,1]]<br/>(R1) <i>discard_write</i> B0 {[[0,0,0] - "
	    "[1,1,1]]}> fontcolor=crimson shape=box];7[label=<C7 on N2<br/><b>execution</b> [[2,0,0] - [3,1,1]]<br/>(R1) <i>discard_write</i> B0 {[[0,0,0] - "
	    "[1,1,1]]}> fontcolor=dodgerblue4 shape=box];8[label=<C8 on N3<br/><b>execution</b> [[3,0,0] - [4,1,1]]<br/>(R1) <i>discard_write</i> B0 {[[0,0,0] - "
	    "[1,1,1]]}> fontcolor=goldenrod shape=box];}subgraph cluster_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;0[label=<C0 on "
	    "N0<br/><b>epoch</b>> fontcolor=black shape=box];1[label=<C1 on N1<br/><b>epoch</b>> fontcolor=crimson shape=box];2[label=<C2 on N2<br/><b>epoch</b>> "
	    "fontcolor=dodgerblue4 shape=box];3[label=<C3 on N3<br/><b>epoch</b>> fontcolor=goldenrod shape=box];}16[label=<C16 on N0<br/>(R1) <b>await push</b> "
	    "from N3<br/> B0 [[0,0,0] - [1,1,1]]> fontcolor=black shape=ellipse];0->16[color=orchid];15->16[style=dashed color=gray40];15[label=<C15 on "
	    "N3<br/>(R1) "
	    "<b>push</b> to N0<br/> B0 [[0,0,0] - [1,1,1]]> fontcolor=goldenrod shape=ellipse];8->15[];14[label=<C14 on N0<br/>(R1) <b>await push</b> from "
	    "N2<br/> B0 [[0,0,0] - [1,1,1]]> fontcolor=black shape=ellipse];0->14[color=orchid];13->14[style=dashed color=gray40];13[label=<C13 on N2<br/>(R1) "
	    "<b>push</b> to N0<br/> B0 [[0,0,0] - [1,1,1]]> fontcolor=dodgerblue4 "
	    "shape=ellipse];7->13[];0->5[color=orchid];1->6[color=orchid];2->7[color=orchid];3->8[color=orchid];10->9[];10[label=<C10 on N0<br/><b>reduction</b> "
	    "R1<br/> B0 {[[0,0,0] - [1,1,1]]}> fontcolor=black shape=ellipse];5->10[];12->10[];14->10[];16->10[];11[label=<C11 on N1<br/>(R1) <b>push</b> to "
	    "N0<br/> B0 [[0,0,0] - [1,1,1]]> fontcolor=crimson shape=ellipse];6->11[];12[label=<C12 on N0<br/>(R1) <b>await push</b> from N1<br/> B0 [[0,0,0] - "
	    "[1,1,1]]> fontcolor=black shape=ellipse];0->12[color=orchid];11->12[style=dashed color=gray40];}";

	const auto dot = ctx.get_command_graph().print_graph(std::numeric_limits<size_t>::max(), tm, {}).value();
	CHECK(dot == expected);
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "Buffer debug names show up in the generated graph", "[print_graph][buffer_names]") {
	distr_queue q;
	celerity::range<1> range(16);
	celerity::buffer<int, 1> buff_a(range);
	std::string buff_name{"my_buffer"};
	celerity::debug::set_buffer_name(buff_a, buff_name);
	CHECK(celerity::debug::get_buffer_name(buff_a) == buff_name);

	q.submit([=](handler& cgh) {
		celerity::accessor acc{buff_a, cgh, celerity::access::all{}, celerity::write_only};
		cgh.parallel_for<class UKN(print_graph_buffer_name)>(range, [=](item<1> item) {});
	});

	// wait for commands to be generated in the scheduler thread
	q.slow_full_sync();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const auto expected =
	    "digraph G{label=\"Command Graph\" subgraph cluster_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;0[label=<C0 on "
	    "N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_1{label=<<font color=\"#606060\">T1 \"print_graph_buffer_name_11\" "
	    "(device-compute)</font>>;color=darkgray;2[label=<C2 on N0<br/><b>execution</b> [[0,0,0] - [16,1,1]]<br/><i>write</i> B0 \"my_buffer\" {[[0,0,0] - "
	    "[16,1,1]]}> fontcolor=black shape=box];}subgraph cluster_2{label=<<font color=\"#606060\">T2 (epoch)</font>>;color=darkgray;3[label=<C3 on "
	    "N0<br/><b>epoch</b> (barrier)> fontcolor=black shape=box];}2->3[color=orange];0->2[];}";

	const auto dot = runtime_testspy::print_graph(runtime::get_instance());
	CHECK(dot == expected);
}
} // namespace celerity::detail