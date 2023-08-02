#include <catch2/catch_test_macros.hpp>

#include "distributed_graph_generator_test_utils.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("task-graph printing is unchanged", "[print_graph][task-graph]") {
	task_manager tm{1, nullptr};
	test_utils::mock_buffer_factory mbf(tm);
	test_utils::mock_reduction_factory mrf;

	auto range = celerity::range<1>(64);
	auto buf_0 = mbf.create_buffer(range);
	auto buf_1 = mbf.create_buffer(celerity::range<1>(1));

	// graph copied from graph_gen_reduction_tests "distributed_graph_generator generates reduction command trees"

	test_utils::add_compute_task<class UKN(task_initialize)>(
	    tm, [&](handler& cgh) { buf_1.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);
	test_utils::add_compute_task<class UKN(task_produce)>(
	    tm, [&](handler& cgh) { buf_0.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);
	test_utils::add_compute_task<class UKN(task_reduce)>(
	    tm,
	    [&](handler& cgh) {
		    buf_0.get_access<access_mode::read>(cgh, acc::one_to_one{});
		    test_utils::add_reduction(cgh, mrf, buf_1, true /* include_current_buffer_value */);
	    },
	    range);
	test_utils::add_compute_task<class UKN(task_consume)>(
	    tm,
	    [&](handler& cgh) {
		    buf_1.get_access<access_mode::read>(cgh, acc::fixed<1>({0, 1}));
	    },
	    range);

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const auto expected =
	    "digraph G {label=\"Task Graph\" 0[shape=ellipse label=<T0<br/><b>epoch</b>>];1[shape=box style=rounded label=<T1 \"task_initialize_2\" "
	    "<br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>discard_write</i> B1 {[0,0,0] - [1,1,1]}>];0->1[color=orchid];2[shape=box style=rounded "
	    "label=<T2 \"task_produce_3\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>discard_write</i> B0 {[0,0,0] - "
	    "[64,1,1]}>];0->2[color=orchid];3[shape=box style=rounded label=<T3 \"task_reduce_4\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/>(R1) "
	    "<i>read_write</i> B1 {[0,0,0] - [1,1,1]}<br/><i>read</i> B0 {[0,0,0] - [64,1,1]}>];1->3[];2->3[];4[shape=box style=rounded label=<T4 "
	    "\"task_consume_5\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>read</i> B1 {[0,0,0] - [1,1,1]}>];3->4[];}";

	const auto dot = tm.print_graph(std::numeric_limits<size_t>::max()).value();
	CHECK(dot == expected);
}

TEST_CASE("command graph printing is unchanged", "[print_graph][command-graph]") {
	size_t num_nodes = 4;
	dist_cdag_test_context dctx(num_nodes);

	auto buf_0 = dctx.create_buffer(range<1>{1});

	dctx.device_compute<class UKN(reduce)>(range<1>(num_nodes)).reduce(buf_0, false).submit();
	dctx.device_compute<class UKN(consume)>(range<1>(num_nodes)).read(buf_0, acc::all{}).read_write(buf_0, acc::all{}).write(buf_0, acc::all{}).submit();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const auto expected =
	    "digraph G{label=\"Command Graph\" subgraph cluster_id_0_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;id_0_0[label=<C0 on "
	    "N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_id_0_1{label=<<font color=\"#606060\">T1 \"reduce_8\" "
	    "(device-compute)</font>>;color=darkgray;id_0_1[label=<C1 on N0<br/><b>execution</b> [0,0,0] - [1,1,1]<br/>(R1) <i>discard_write</i> B0 {[0,0,0] - "
	    "[1,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_2{label=<<font color=\"#606060\">T2 \"consume_9\" "
	    "(device-compute)</font>>;color=darkgray;id_0_2[label=<C2 on N0<br/><b>execution</b> [0,0,0] - [1,1,1]<br/><i>read</i> B0 {[0,0,0] - "
	    "[1,1,1]}<br/><i>read_write</i> B0 {[0,0,0] - [1,1,1]}<br/><i>write</i> B0 {[0,0,0] - [1,1,1]}> fontcolor=black shape=box];}id_0_7[label=<C7 on "
	    "N0<br/>(R1) <b>push</b> transfer 8589934595 to N3<br/>BB0 [0,0,0] - [1,1,1]> fontcolor=black shape=ellipse];id_0_1->id_0_7[];id_0_6[label=<C6 on "
	    "N0<br/>(R1) <b>push</b> transfer 8589934594 to N2<br/>BB0 [0,0,0] - [1,1,1]> fontcolor=black shape=ellipse];id_0_1->id_0_6[];id_0_5[label=<C5 on "
	    "N0<br/>(R1) <b>push</b> transfer 8589934593 to N1<br/>BB0 [0,0,0] - [1,1,1]> fontcolor=black shape=ellipse];id_0_1->id_0_5[];id_0_4[label=<C4 on "
	    "N0<br/>(R1) <b>await push</b> transfer 8589934592 <br/>BB0 {[0,0,0] - [1,1,1]}> fontcolor=black "
	    "shape=ellipse];id_0_0->id_0_4[color=orchid];id_0_3[label=<C3 on N0<br/><b>reduction</b> R1<br/> B0 {[0,0,0] - [1,1,1]}> fontcolor=black "
	    "shape=ellipse];id_0_1->id_0_3[];id_0_4->id_0_3[];id_0_3->id_0_2[];id_0_5->id_0_2[color=limegreen];id_0_6->id_0_2[color=limegreen];id_0_7->id_0_2["
	    "color=limegreen];id_0_0->id_0_1[color=orchid];}";

	// FIXME: We currently only print the graph for node 0.
	const auto dot = dctx.get_graph_generator(0).get_command_graph().print_graph(0, std::numeric_limits<size_t>::max(), dctx.get_task_manager(), {}).value();
	CHECK(dot == expected);
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "Buffer debug names show up in the generated graph", "[print_graph]") {
	distr_queue q;
	celerity::range<1> range(16);
	celerity::buffer<int, 1> buff_a(range);
	std::string buff_name{"my_buffer"};
	celerity::debug::set_buffer_name(buff_a, buff_name);
	CHECK(celerity::debug::get_buffer_name(buff_a) == buff_name);

	q.submit([&](handler& cgh) {
		celerity::accessor acc_a{buff_a, cgh, acc::all{}, celerity::write_only};
		cgh.parallel_for<class UKN(print_graph_buffer_name)>(range, [=](item<1> item) { (void)acc_a; });
	});

	// wait for commands to be generated in the scheduler thread
	q.slow_full_sync();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const auto expected =
	    "digraph G{label=\"Command Graph\" subgraph cluster_id_0_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;id_0_0[label=<C0 on "
	    "N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_id_0_1{label=<<font color=\"#606060\">T1 \"print_graph_buffer_name_12\" "
	    "(device-compute)</font>>;color=darkgray;id_0_1[label=<C1 on N0<br/><b>execution</b> [0,0,0] - [16,1,1]<br/><i>write</i> B0 \"my_buffer\" {[0,0,0] "
	    "- [16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_2{label=<<font color=\"#606060\">T2 (epoch)</font>>;color=darkgray;id_0_2[label=<C2 "
	    "on N0<br/><b>epoch</b> (barrier)> fontcolor=black shape=box];}id_0_1->id_0_2[color=orange];id_0_0->id_0_1[];}";

	const auto dot = runtime_testspy::print_graph(celerity::detail::runtime::get_instance());
	CHECK(dot == expected);
}