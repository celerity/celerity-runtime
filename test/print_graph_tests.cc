#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <libenvpp/env.hpp>

#include "distributed_graph_generator_test_utils.h"
#include "instruction_graph_test_utils.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

namespace acc = celerity::access;

TEST_CASE("task-graph printing is unchanged", "[print_graph][task-graph]") {
	auto tt = test_utils::task_test_context{};

	auto range = celerity::range<1>(64);
	auto buf_0 = tt.mbf.create_buffer(range);
	auto buf_1 = tt.mbf.create_buffer(celerity::range<1>(1));

	// graph copied from graph_gen_reduction_tests "distributed_graph_generator generates reduction command trees"

	test_utils::add_compute_task(
	    tt.tm, [&](handler& cgh) { buf_1.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);
	test_utils::add_compute_task(
	    tt.tm, [&](handler& cgh) { buf_0.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);
	test_utils::add_compute_task(
	    tt.tm,
	    [&](handler& cgh) {
		    buf_0.get_access<access_mode::read>(cgh, acc::one_to_one{});
		    test_utils::add_reduction(cgh, tt.mrf, buf_1, true /* include_current_buffer_value */);
	    },
	    range);
	test_utils::add_compute_task(
	    tt.tm,
	    [&](handler& cgh) {
		    buf_1.get_access<access_mode::read>(cgh, acc::fixed<1>({0, 1}));
	    },
	    range);

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const std::string expected =
	    "digraph G{label=<Task Graph>;pad=0.2;0[shape=ellipse label=<T0<br/><b>epoch</b>>];1[shape=box style=rounded label=<T1<br/><b>device-compute</b> "
	    "[0,0,0] + [64,1,1]<br/><i>discard_write</i> B1 {[0,0,0] - [1,1,1]}>];0->1[color=orchid];2[shape=box style=rounded label=<T2<br/><b>device-compute</b> "
	    "[0,0,0] + [64,1,1]<br/><i>discard_write</i> B0 {[0,0,0] - [64,1,1]}>];0->2[color=orchid];3[shape=box style=rounded "
	    "label=<T3<br/><b>device-compute</b> [0,0,0] + [64,1,1]<br/>(R1) <i>read_write</i> B1 {[0,0,0] - [1,1,1]}<br/><i>read</i> B0 {[0,0,0] - "
	    "[64,1,1]}>];1->3[];2->3[];4[shape=box style=rounded label=<T4<br/><b>device-compute</b> [0,0,0] + [64,1,1]<br/><i>read</i> B1 {[0,0,0] - "
	    "[1,1,1]}>];3->4[];}";

	const auto dot = print_task_graph(tt.trec);
	CHECK(dot == expected);
	if(dot != expected) { fmt::print("\n{}:\n\ngot:\n\n{}\n\nexpected:\n\n{}\n\n", Catch::getResultCapture().getCurrentTestName(), dot, expected); }
}

namespace {
int count_occurences(const std::string& str, const std::string& substr) {
	int occurrences = 0;
	std::string::size_type pos = 0;
	while((pos = str.find(substr, pos)) != std::string::npos) {
		++occurrences;
		pos += substr.length();
	}
	return occurrences;
}
} // namespace

TEST_CASE("command-graph printing is unchanged", "[print_graph][command-graph]") {
	size_t num_nodes = 4;
	dist_cdag_test_context dctx(num_nodes);

	auto buf_0 = dctx.create_buffer(range(1));
	auto buf_1 = dctx.create_buffer(range(num_nodes));

	dctx.device_compute(range<1>(num_nodes)) //
	    .discard_write(buf_1, acc::one_to_one{})
	    .reduce(buf_0, false)
	    .submit();
	dctx.device_compute(range<1>(num_nodes)).read(buf_0, acc::all{}).read_write(buf_1, acc::one_to_one{}).write(buf_1, acc::one_to_one{}).submit();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const std::string expected =
	    "digraph G{label=<<br/>Command Graph<br/><b>command-graph printing is unchanged</b>>;pad=0.2;subgraph cluster_id_0_0{label=<<font color=\"#606060\">T0 "
	    "(epoch)</font>>;color=darkgray;id_0_0[label=<C0 on N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_id_0_1{label=<<font "
	    "color=\"#606060\">T1 (device-compute)</font>>;color=darkgray;id_0_1[label=<C1 on N0<br/><b>execution</b> [0,0,0] + [1,1,1]<br/>(R1) "
	    "<i>discard_write</i> B0 {[0,0,0] - [1,1,1]}<br/><i>discard_write</i> B1 {[0,0,0] - [1,1,1]}> fontcolor=black shape=box];}subgraph "
	    "cluster_id_0_2{label=<<font color=\"#606060\">T2 (device-compute)</font>>;color=darkgray;id_0_2[label=<C2 on N0<br/><b>execution</b> [0,0,0] + "
	    "[1,1,1]<br/><i>write</i> B1 {[0,0,0] - [1,1,1]}<br/><i>read_write</i> B1 {[0,0,0] - [1,1,1]}<br/><i>read</i> B0 {[0,0,0] - [1,1,1]}> fontcolor=black "
	    "shape=box];}id_0_0->id_0_1[color=orchid];id_0_3->id_0_2[];id_0_1->id_0_2[];id_0_3[label=<C3 on N0<br/><b>reduction</b> R1<br/> B0 {[0,0,0] - "
	    "[1,1,1]}> fontcolor=black "
	    "shape=ellipse];id_0_1->id_0_3[];id_0_4->id_0_3[];id_0_5->id_0_3[color=limegreen];id_0_6->id_0_3[color=limegreen];id_0_7->id_0_3[color=limegreen];id_0_"
	    "4[label=<C4 on N0<br/>(R1) <b>await push</b> T2.B0.R1 <br/>B0 {[0,0,0] - [1,1,1]}> fontcolor=black "
	    "shape=ellipse];id_0_0->id_0_4[color=orchid];id_0_5[label=<C5 on N0<br/>(R1) <b>push</b> T2.B0.R1 to N1<br/>B0 [0,0,0] + [1,1,1]> fontcolor=black "
	    "shape=ellipse];id_0_1->id_0_5[];id_0_6[label=<C6 on N0<br/>(R1) <b>push</b> T2.B0.R1 to N2<br/>B0 [0,0,0] + [1,1,1]> fontcolor=black "
	    "shape=ellipse];id_0_1->id_0_6[];id_0_7[label=<C7 on N0<br/>(R1) <b>push</b> T2.B0.R1 to N3<br/>B0 [0,0,0] + [1,1,1]> fontcolor=black "
	    "shape=ellipse];id_0_1->id_0_7[];}";

	// fully check node 0
	const auto dot0 = dctx.print_command_graph(0);
	CHECK(dot0 == expected);
	if(dot0 != expected) { fmt::print("\n{}:\n\ngot:\n\n{}\n\nexpected:\n\n{}\n\n", Catch::getResultCapture().getCurrentTestName(), dot0, expected); }

	// only check the rough string length and occurence count of N1/N2... for other nodes
	const int expected_occurences = count_occurences(expected, "N0");
	for(node_id nid = 1; nid < num_nodes; ++nid) {
		CAPTURE(nid);
		const auto dot_n = dctx.print_command_graph(nid);
		CHECK_THAT(dot_n.size(), Catch::Matchers::WithinAbs(expected.size(), 50));
		CHECK(count_occurences(dot_n, fmt::format("N{}", nid)) == expected_occurences);
	}
}

TEST_CASE("instruction-graph printing is unchanged", "[print_graph][instruction-graph]") {
	const size_t num_nodes = 2;
	const node_id local_nid = 1;
	const size_t num_local_devices = 2;

	idag_test_context ictx(num_nodes, local_nid, num_local_devices);
	ictx.set_horizon_step(2);

	auto buf_0 = ictx.create_buffer(range(1), true /* host initialized */);
	auto buf_1 = ictx.create_buffer(range(num_nodes * num_local_devices), true /* host initialized */);
	auto buf_2 = ictx.create_buffer(range(1), true /* host initialized */);
	auto ho = ictx.create_host_object(true /* owns instance */);

	ictx.device_compute(range<1>(num_nodes * num_local_devices)) //
	    .read_write(buf_1, acc::one_to_one{})
	    .reduce(buf_0, false)
	    .submit();
	ictx.device_compute(range<1>(num_nodes * num_local_devices)) //
	    .read(buf_0, acc::all())
	    .read(buf_1, test_utils::access::reverse_one_to_one())
	    .write(buf_2, acc::one_to_one())
	    .submit();
	ictx.collective_host_task() //
	    .read(buf_0, acc::all())
	    .read(buf_2, acc::all())
	    .affect(ho)
	    .submit();
	ictx.fence(buf_0);
	ictx.fence(ho);
	ictx.finish();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.

	const std::string expected =
	    "digraph G{label=<<br/>Instruction Graph<br/><b>instruction-graph printing is unchanged</b><br/>for N1 out of 2 nodes, with 2 devices / "
	    "node>;pad=0.2;I0[color=black,shape=box,margin=0.2,style=rounded,label=<I0 (T0, "
	    "C0)<br/><b>epoch</b>>];I1[color=cyan3,shape=ellipse,label=<I1<br/>buffer <b>alloc</b> M1.A1<br/>for B0 [0,0,0] - [1,1,1]<br/>4%4 "
	    "bytes>];I2[color=cyan3,shape=ellipse,label=<I2<br/>buffer <b>alloc</b> M2.A1<br/>for B0 [0,0,0] - [1,1,1]<br/>4%4 "
	    "bytes>];I3[color=cyan3,shape=ellipse,label=<I3<br/>buffer <b>alloc</b> M3.A1<br/>for B0 [0,0,0] - [1,1,1]<br/>4%4 "
	    "bytes>];I4[color=cyan3,shape=ellipse,label=<I4<br/>buffer <b>alloc</b> M1.A2<br/>for B1 [2,0,0] - [4,1,1]<br/>8%4 "
	    "bytes>];I5[color=cyan3,shape=ellipse,label=<I5<br/>buffer <b>alloc</b> M2.A2<br/>for B1 [2,0,0] - [3,1,1]<br/>4%4 "
	    "bytes>];I6[color=cyan3,shape=ellipse,label=<I6<br/>buffer <b>alloc</b> M3.A2<br/>for B1 [3,0,0] - [4,1,1]<br/>4%4 "
	    "bytes>];I7[color=green3,shape=ellipse,label=<I7<br/>coherence <b>copy</b><br/>from M0.A2 ([0,0,0] - [4,1,1])<br/>to M1.A2 ([2,0,0] - [4,1,1])<br/>B1 "
	    "{[2,0,0] - [3,1,1]} x4 bytes>];I8[color=green3,shape=ellipse,label=<I8<br/>coherence <b>copy</b><br/>from M0.A2 ([0,0,0] - [4,1,1])<br/>to M1.A2 "
	    "([2,0,0] - [4,1,1])<br/>B1 {[3,0,0] - [4,1,1]} x4 bytes>];I9[color=green3,shape=ellipse,label=<I9<br/>coherence <b>copy</b><br/>from M1.A2 ([2,0,0] - "
	    "[4,1,1])<br/>to M2.A2 ([2,0,0] - [3,1,1])<br/>B1 {[2,0,0] - [3,1,1]} x4 bytes>];I10[color=green3,shape=ellipse,label=<I10<br/>coherence "
	    "<b>copy</b><br/>from M1.A2 ([2,0,0] - [4,1,1])<br/>to M3.A2 ([3,0,0] - [4,1,1])<br/>B1 {[3,0,0] - [4,1,1]} x4 "
	    "bytes>];I11[color=cyan3,shape=ellipse,label=<I11<br/>gather <b>alloc</b> M1.A3<br/>for B0 [0,0,0] - [1,1,1] x2<br/>8%4 "
	    "bytes>];I12[color=darkorange2,shape=box,margin=0.2,style=rounded,label=<I12 (device-compute T1, execution C1)<br/><b>device kernel</b><br/>on D0 "
	    "[2,0,0] - [3,1,1]<br/>+ access B1 [2,0,0] - [3,1,1]<br/>via M2.A2 [0,0,0] - [1,1,1]<br/>+ (R1) reduce into B0 [0,0,0] - [1,1,1]<br/>via M2.A1 [0,0,0] "
	    "- [1,1,1]>];I13[color=darkorange2,shape=box,margin=0.2,style=rounded,label=<I13 (device-compute T1, execution C1)<br/><b>device kernel</b><br/>on D1 "
	    "[3,0,0] - [4,1,1]<br/>+ access B1 [3,0,0] - [4,1,1]<br/>via M3.A2 [0,0,0] - [1,1,1]<br/>+ (R1) reduce into B0 [0,0,0] - [1,1,1]<br/>via M3.A1 [0,0,0] "
	    "- [1,1,1]>];I14[color=green3,shape=ellipse,label=<I14<br/>gather <b>copy</b><br/>from M2.A1 ([0,0,0] - [1,1,1])<br/>to M1.A3 ([0,0,0] - "
	    "[1,1,1])<br/>B0 {[0,0,0] - [1,1,1]} x4 bytes>];I15[color=green3,shape=ellipse,label=<I15<br/>gather <b>copy</b><br/>from M3.A1 ([0,0,0] - "
	    "[1,1,1])<br/>to M1.A3 + 4 bytes ([0,0,0] - [1,1,1])<br/>B0 {[0,0,0] - [1,1,1]} x4 bytes>];I16[color=blue,shape=ellipse,label=<I16<br/>local "
	    "<b>reduce</b> B0.R1<br/>B0 [0,0,0] - [1,1,1]<br/>from M1.A3 x2<br/>to M1.A1 x1>];I17[color=cyan3,shape=ellipse,label=<I17<br/><b>free</b> M1.A3 "
	    "<br/>8 bytes>];I18[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I18 (push C2)<br/><b>send</b> T2.B0.R1<br/>to N0 MSG0<br/>B0 [0,0,0] - "
	    "[1,1,1]<br/>via M1.A1 [0,0,0] - [1,1,1]<br/>[1,1,1]x4 bytes>];I19[color=green3,shape=ellipse,label=<I19<br/>coherence <b>copy</b><br/>from M2.A2 "
	    "([2,0,0] - [3,1,1])<br/>to M1.A2 ([2,0,0] - [4,1,1])<br/>B1 {[2,0,0] - [3,1,1]} x4 bytes>];I20[color=green3,shape=ellipse,label=<I20<br/>coherence "
	    "<b>copy</b><br/>from M3.A2 ([3,0,0] - [4,1,1])<br/>to M1.A2 ([2,0,0] - [4,1,1])<br/>B1 {[3,0,0] - [4,1,1]} x4 "
	    "bytes>];I21[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I21 (push C3)<br/><b>send</b> T2.B1<br/>to N0 MSG1<br/>B1 [2,0,0] - "
	    "[3,1,1]<br/>via M1.A2 [0,0,0] - [1,1,1]<br/>[1,1,1]x4 bytes>];I22[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I22 (push "
	    "C3)<br/><b>send</b> T2.B1<br/>to N0 MSG2<br/>B1 [3,0,0] - [4,1,1]<br/>via M1.A2 [1,0,0] - [2,1,1]<br/>[1,1,1]x4 "
	    "bytes>];I23[color=cyan3,shape=ellipse,label=<I23<br/>gather <b>alloc</b> M1.A4<br/>for B0 [0,0,0] - [1,1,1] x2<br/>8%4 "
	    "bytes>];I24[color=blue,shape=ellipse,label=<I24<br/><b>fill identity</b> for R1<br/>M1.A4 x2>];I25[color=green3,shape=ellipse,label=<I25<br/>gather "
	    "<b>copy</b><br/>from M1.A1 ([0,0,0] - [1,1,1])<br/>to M1.A4 + 4 bytes ([0,0,0] - [1,1,1])<br/>B0 {[0,0,0] - [1,1,1]} x4 "
	    "bytes>];I26[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I26 (await-push C6)<br/><b>gather receive</b> T2.B0.R1<br/>B0 [0,0,0] - [1,1,1] "
	    "x2<br/>into M1.A4>];I27[color=blue,shape=box,margin=0.2,style=rounded,label=<I27 (reduction C5)<br/>global <b>reduce</b> B0.R1<br/>B0 [0,0,0] - "
	    "[1,1,1]<br/>from M1.A4 x2<br/>to M1.A1 x1>];I28[color=cyan3,shape=ellipse,label=<I28<br/><b>free</b> M1.A4 <br/>8 "
	    "bytes>];I29[color=cyan3,shape=ellipse,label=<I29<br/>buffer <b>alloc</b> M1.A5<br/>for B1 [0,0,0] - [2,1,1]<br/>8%4 "
	    "bytes>];I30[color=cyan3,shape=ellipse,label=<I30<br/>buffer <b>alloc</b> M2.A3<br/>for B1 [1,0,0] - [2,1,1]<br/>4%4 "
	    "bytes>];I31[color=cyan3,shape=ellipse,label=<I31<br/>buffer <b>alloc</b> M3.A3<br/>for B1 [0,0,0] - [1,1,1]<br/>4%4 "
	    "bytes>];I32[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I32 (await-push C7)<br/><b>split receive</b> T2.B1<br/>B1 {[0,0,0] - "
	    "[2,1,1]}<br/>into M1.A5 (B1 [0,0,0] - [2,1,1])<br/>x4 bytes>];I33[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I33 (await-push "
	    "C7)<br/><b>await receive</b> T2.B1<br/>B1 {[1,0,0] - [2,1,1]}>];I34[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I34 (await-push "
	    "C7)<br/><b>await receive</b> T2.B1<br/>B1 {[0,0,0] - [1,1,1]}>];I35[color=green3,shape=ellipse,label=<I35<br/>coherence <b>copy</b><br/>from M1.A5 "
	    "([0,0,0] - [2,1,1])<br/>to M2.A3 ([1,0,0] - [2,1,1])<br/>B1 {[1,0,0] - [2,1,1]} x4 bytes>];I36[color=green3,shape=ellipse,label=<I36<br/>coherence "
	    "<b>copy</b><br/>from M1.A5 ([0,0,0] - [2,1,1])<br/>to M3.A3 ([0,0,0] - [1,1,1])<br/>B1 {[0,0,0] - [1,1,1]} x4 "
	    "bytes>];I37[color=green3,shape=ellipse,label=<I37<br/>coherence <b>copy</b><br/>from M1.A1 ([0,0,0] - [1,1,1])<br/>to M2.A1 ([0,0,0] - "
	    "[1,1,1])<br/>B0 {[0,0,0] - [1,1,1]} x4 bytes>];I38[color=green3,shape=ellipse,label=<I38<br/>coherence <b>copy</b><br/>from M1.A1 ([0,0,0] - "
	    "[1,1,1])<br/>to M3.A1 ([0,0,0] - [1,1,1])<br/>B0 {[0,0,0] - [1,1,1]} x4 bytes>];I39[color=darkorange2,shape=box,margin=0.2,style=rounded,label=<I39 "
	    "(device-compute T2, execution C4)<br/><b>device kernel</b><br/>on D0 [2,0,0] - [3,1,1]<br/>+ access B0 [0,0,0] - [1,1,1]<br/>via M2.A1 [0,0,0] - "
	    "[1,1,1]<br/>+ access B1 [1,0,0] - [2,1,1]<br/>via M2.A3 [0,0,0] - [1,1,1]<br/>+ access B2 [0,0,0] - [0,0,0]<br/>via null [0,0,0] - "
	    "[0,0,0]>];I40[color=darkorange2,shape=box,margin=0.2,style=rounded,label=<I40 (device-compute T2, execution C4)<br/><b>device kernel</b><br/>on D1 "
	    "[3,0,0] - [4,1,1]<br/>+ access B0 [0,0,0] - [1,1,1]<br/>via M3.A1 [0,0,0] - [1,1,1]<br/>+ access B1 [0,0,0] - [1,1,1]<br/>via M3.A3 [0,0,0] - "
	    "[1,1,1]<br/>+ access B2 [0,0,0] - [0,0,0]<br/>via null [0,0,0] - [0,0,0]>];I41[color=black,shape=box,margin=0.2,style=rounded,label=<I41 (T3, "
	    "C8)<br/><b>horizon</b><br/>collect R1>];I42[color=darkred,shape=ellipse,label=<I42<br/><b>clone collective group</b><br/>CG1 -&gt; "
	    "CG2>];I43[color=cyan3,shape=ellipse,label=<I43<br/>buffer <b>alloc</b> M1.A6<br/>for B2 [0,0,0] - [1,1,1]<br/>4%4 "
	    "bytes>];I44[color=deeppink2,shape=box,margin=0.2,style=rounded,label=<I44 (await-push C10)<br/><b>receive</b> T4.B2<br/>B2 {[0,0,0] - "
	    "[1,1,1]}<br/>into M1.A6 (B2 [0,0,0] - [1,1,1])<br/>x4 bytes>];I45[color=darkorange2,shape=box,margin=0.2,style=rounded,label=<I45 (CG2 "
	    "collective-host T4, execution C9)<br/><b>host task</b><br/>on host [1,0,0] - [2,1,1]<br/>+ access B0 [0,0,0] - [1,1,1]<br/>via M1.A1 [0,0,0] - "
	    "[1,1,1]<br/>+ access B2 [0,0,0] - [1,1,1]<br/>via M1.A6 [0,0,0] - [1,1,1]>];I46[color=green3,shape=ellipse,label=<I46<br/>fence <b>copy</b><br/>from "
	    "M1.A1 ([0,0,0] - [1,1,1])<br/>to M0.A4 ([0,0,0] - [1,1,1])<br/>B0 {[0,0,0] - [1,1,1]} x4 "
	    "bytes>];I47[color=darkorange,shape=box,margin=0.2,style=rounded,label=<I47 (T5, C11)<br/><b>fence</b><br/>B0 [0,0,0] - "
	    "[1,1,1]>];I48[color=darkorange,shape=box,margin=0.2,style=rounded,label=<I48 (T6, "
	    "C12)<br/><b>fence</b><br/>H0>];I49[color=black,shape=ellipse,label=<I49<br/><b>destroy</b> "
	    "H0>];I50[color=cyan3,shape=ellipse,label=<I50<br/><b>free</b> M1.A6<br/>B2 [0,0,0] - [1,1,1] <br/>4 "
	    "bytes>];I51[color=cyan3,shape=ellipse,label=<I51<br/><b>free</b> M1.A2<br/>B1 [2,0,0] - [4,1,1] <br/>8 "
	    "bytes>];I52[color=cyan3,shape=ellipse,label=<I52<br/><b>free</b> M1.A5<br/>B1 [0,0,0] - [2,1,1] <br/>8 "
	    "bytes>];I53[color=cyan3,shape=ellipse,label=<I53<br/><b>free</b> M2.A2<br/>B1 [2,0,0] - [3,1,1] <br/>4 "
	    "bytes>];I54[color=cyan3,shape=ellipse,label=<I54<br/><b>free</b> M2.A3<br/>B1 [1,0,0] - [2,1,1] <br/>4 "
	    "bytes>];I55[color=cyan3,shape=ellipse,label=<I55<br/><b>free</b> M3.A2<br/>B1 [3,0,0] - [4,1,1] <br/>4 "
	    "bytes>];I56[color=cyan3,shape=ellipse,label=<I56<br/><b>free</b> M3.A3<br/>B1 [0,0,0] - [1,1,1] <br/>4 "
	    "bytes>];I57[color=cyan3,shape=ellipse,label=<I57<br/><b>free</b> M1.A1<br/>B0 [0,0,0] - [1,1,1] <br/>4 "
	    "bytes>];I58[color=cyan3,shape=ellipse,label=<I58<br/><b>free</b> M2.A1<br/>B0 [0,0,0] - [1,1,1] <br/>4 "
	    "bytes>];I59[color=cyan3,shape=ellipse,label=<I59<br/><b>free</b> M3.A1<br/>B0 [0,0,0] - [1,1,1] <br/>4 "
	    "bytes>];I60[color=black,shape=box,margin=0.2,style=rounded,label=<I60 (T7, C13)<br/><b>epoch</b> (shutdown)<br/>collect M0.A4<br/>collect "
	    "M0.A3<br/>collect M0.A2<br/>collect "
	    "M0.A1>];I0->I1[color=orchid];I0->I2[color=orchid];I0->I3[color=orchid];I0->I4[color=orchid];I0->I5[color=orchid];I0->I6[color=orchid];I0->I7[];I0->I8["
	    "];I0->I11[color=orchid];I0->I23[color=orchid];I0->I29[color=orchid];I0->I30[color=orchid];I0->I31[color=orchid];I0->I42[color=blue];I0->I43[color="
	    "orchid];I0->I45[];I1->I16[color=cyan3];I2->I12[color=cyan3];I3->I13[color=cyan3];I4->I7[color=cyan3];I4->I8[color=cyan3];I5->I9[color=cyan3];I6->I10["
	    "color=cyan3];I7->I9[];I8->I10[];I9->I12[];I9->I19[color=limegreen];I10->I13[];I10->I20[color=limegreen];I11->I14[color=cyan3];I11->I15[color=cyan3];"
	    "I12->I14[];I12->I19[];I13->I15[];I13->I20[];I14->I16[];I14->I37[color=limegreen];I15->I16[];I15->I38[color=limegreen];I16->I17[color=cyan3];I16->I18[]"
	    ";I16->I25[];I17->I41[color=orange];I18->I27[color=limegreen];I19->I21[];I19->I53[color=cyan3];I20->I22[];I20->I55[color=cyan3];I21->I41[color=orange];"
	    "I21->I51[color=cyan3];I22->I41[color=orange];I22->I51[color=cyan3];I23->I24[color=cyan3];I24->I25[color=limegreen];I24->I26[color=limegreen];I25->I27["
	    "];I26->I27[];I27->I28[color=cyan3];I27->I37[];I27->I38[];I27->I45[];I27->I46[];I28->I41[color=orange];I29->I32[color=cyan3];I30->I35[color=cyan3];I31-"
	    ">I36[color=cyan3];I32->I33[color=gray];I32->I34[color=gray];I33->I35[];I34->I36[];I35->I39[];I35->I52[color=cyan3];I36->I40[];I36->I52[color=cyan3];"
	    "I37->I39[];I37->I57[color=cyan3];I38->I40[];I38->I57[color=cyan3];I39->I41[color=orange];I39->I54[color=cyan3];I39->I58[color=cyan3];I40->I41[color="
	    "orange];I40->I56[color=cyan3];I40->I59[color=cyan3];I41->I60[color=orange];I42->I45[color=blue];I43->I44[color=cyan3];I44->I45[];I45->I48[];I45->I50["
	    "color=cyan3];I45->I57[color=cyan3];I46->I47[];I46->I57[color=cyan3];I47->I60[color=orange];I48->I49[];I49->I60[color=orange];I50->I60[color=orange];"
	    "I51->I60[color=orange];I52->I60[color=orange];I53->I60[color=orange];I54->I60[color=orange];I55->I60[color=orange];I56->I60[color=orange];I57->I60["
	    "color=orange];I58->I60[color=orange];I59->I60[color=orange];P0[margin=0.2,shape=cds,color=\"#606060\",label=<<font color=\"#606060\"><b>pilot</b> to "
	    "N0 MSG0<br/>T2.B0.R1<br/>for B0 [0,0,0] - "
	    "[1,1,1]</font>>];P0->I18[dir=none,style=dashed,color=\"#606060\"];P1[margin=0.2,shape=cds,color=\"#606060\",label=<<font "
	    "color=\"#606060\"><b>pilot</b> to N0 MSG1<br/>T2.B1<br/>for B1 [2,0,0] - "
	    "[3,1,1]</font>>];P1->I21[dir=none,style=dashed,color=\"#606060\"];P2[margin=0.2,shape=cds,color=\"#606060\",label=<<font "
	    "color=\"#606060\"><b>pilot</b> to N0 MSG2<br/>T2.B1<br/>for B1 [3,0,0] - [4,1,1]</font>>];P2->I22[dir=none,style=dashed,color=\"#606060\"];}";

	const auto dot = ictx.print_instruction_graph();
	CHECK(dot == expected);
	if(dot != expected) { fmt::print("\n{}:\n\ngot:\n\n{}\n\nexpected:\n\n{}\n\n", Catch::getResultCapture().getCurrentTestName(), dot, expected); }
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer debug names show up in the generated graph", "[print_graph]") {
	env::scoped_test_environment tenv(print_graphs_env_setting);

	distr_queue q;
	celerity::range<1> range(16);
	celerity::buffer<int, 1> buff_a(range);
	std::string buff_name{"my_buffer"};
	celerity::debug::set_buffer_name(buff_a, buff_name);
	CHECK(celerity::debug::get_buffer_name(buff_a) == buff_name);

	q.submit([&](handler& cgh) {
		celerity::accessor acc_a{buff_a, cgh, acc::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(range, [=](item<1> item) { (void)acc_a; });
	});

	// wait for commands to be generated in the scheduler thread
	q.slow_full_sync();

	using Catch::Matchers::ContainsSubstring;
	const std::string expected_substring = "B0 \"my_buffer\"";
	SECTION("in the task graph") {
		const auto dot = runtime_testspy::print_task_graph(celerity::detail::runtime::get_instance());
		REQUIRE_THAT(dot, ContainsSubstring(expected_substring));
	}
	SECTION("in the command graph") {
		const auto dot = runtime_testspy::print_command_graph(0, celerity::detail::runtime::get_instance());
		REQUIRE_THAT(dot, ContainsSubstring(expected_substring));
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "full graph is printed if CELERITY_PRINT_GRAPHS is set", "[print_graph]") {
	env::scoped_test_environment tenv(print_graphs_env_setting);

	distr_queue q;
	celerity::range<1> range(16);
	std::vector<int> init(range.size());
	celerity::buffer<int, 1> buff_a(init.data(), range);

	// set small horizon step size so that we do not need to generate a very large graph to test this functionality
	auto& tm = celerity::detail::runtime::get_instance().get_task_manager();
	tm.set_horizon_step(1);

	for(int i = 0; i < 3; ++i) {
		q.submit([&](handler& cgh) {
			celerity::accessor acc_a{buff_a, cgh, acc::one_to_one{}, celerity::read_write_host_task};
			// we're launching distributed host tasks instead of device kernels so that the schedule is independent from available devices on the system
			cgh.host_task(range, [=](partition<1> item) { (void)acc_a; });
		});
	}

	q.slow_full_sync();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graphs are sane and
	// complete, and if so, replace the `expected` values with the new dot graph.

	SECTION("task graph") {
		const auto* expected =
		    "digraph G{label=<Task Graph>;pad=0.2;0[shape=ellipse label=<T0<br/><b>epoch</b>>];1[shape=box style=rounded label=<T1<br/><b>host-compute</b> "
		    "[0,0,0] + [16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}>];0->1[];2[shape=ellipse "
		    "label=<T2<br/><b>horizon</b>>];1->2[color=orange];3[shape=box style=rounded label=<T3<br/><b>host-compute</b> [0,0,0] + "
		    "[16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}>];1->3[];4[shape=ellipse "
		    "label=<T4<br/><b>horizon</b>>];3->4[color=orange];2->4[color=orange];5[shape=box style=rounded label=<T5<br/><b>host-compute</b> [0,0,0] + "
		    "[16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}>];3->5[];6[shape=ellipse "
		    "label=<T6<br/><b>horizon</b>>];5->6[color=orange];4->6[color=orange];7[shape=ellipse label=<T7<br/><b>epoch</b>>];6->7[color=orange];}";

		const auto dot = runtime_testspy::print_task_graph(celerity::detail::runtime::get_instance());
		CHECK(dot == expected);
		if(dot != expected) {
			fmt::print("\n{} (task graph):\n\ngot:\n\n{}\n\nexpected:\n\n{}\n\n", Catch::getResultCapture().getCurrentTestName(), dot, expected);
		};
	}

	SECTION("command graph") {
		const auto* expected =
		    "digraph G{label=<Command Graph>;pad=0.2;subgraph cluster_id_0_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;id_0_0[label=<C0 "
		    "on N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_id_0_1{label=<<font color=\"#606060\">T1 "
		    "(host-compute)</font>>;color=darkgray;id_0_1[label=<C1 on N0<br/><b>execution</b> [0,0,0] + [16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - "
		    "[16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_2{label=<<font color=\"#606060\">T2 "
		    "(horizon)</font>>;color=darkgray;id_0_2[label=<C2 on N0<br/><b>horizon</b>> fontcolor=black shape=box];}subgraph cluster_id_0_3{label=<<font "
		    "color=\"#606060\">T3 (host-compute)</font>>;color=darkgray;id_0_3[label=<C3 on N0<br/><b>execution</b> [0,0,0] + [16,1,1]<br/><i>read_write</i> "
		    "B0 {[0,0,0] - [16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_4{label=<<font color=\"#606060\">T4 "
		    "(horizon)</font>>;color=darkgray;id_0_4[label=<C4 on N0<br/><b>horizon</b>> fontcolor=black shape=box];}subgraph cluster_id_0_5{label=<<font "
		    "color=\"#606060\">T5 (host-compute)</font>>;color=darkgray;id_0_5[label=<C5 on N0<br/><b>execution</b> [0,0,0] + [16,1,1]<br/><i>read_write</i> "
		    "B0 {[0,0,0] - [16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_6{label=<<font color=\"#606060\">T6 "
		    "(horizon)</font>>;color=darkgray;id_0_6[label=<C6 on N0<br/><b>horizon</b>> fontcolor=black shape=box];}subgraph cluster_id_0_7{label=<<font "
		    "color=\"#606060\">T7 (epoch)</font>>;color=darkgray;id_0_7[label=<C7 on N0<br/><b>epoch</b> (barrier)> fontcolor=black "
		    "shape=box];}id_0_0->id_0_1[];id_0_1->id_0_2[color=orange];id_0_1->id_0_3[];id_0_3->id_0_4[color=orange];id_0_2->id_0_4[color=orange];id_0_3->id_0_"
		    "5[];id_0_5->id_0_6[color=orange];id_0_4->id_0_6[color=orange];id_0_6->id_0_7[color=orange];}";

		const auto dot = runtime_testspy::print_command_graph(0, celerity::detail::runtime::get_instance());
		CHECK(dot == expected);
		if(dot != expected) {
			fmt::print("\n{} (command graph):\n\ngot:\n\n{}\n\nexpected:\n\n{}\n\n", Catch::getResultCapture().getCurrentTestName(), dot, expected);
		};
	}
}

template <int X>
class name_class {};

TEST_CASE("task-graph names are escaped", "[print_graph][task-graph][task-name]") {
	auto tt = test_utils::task_test_context{};

	auto range = celerity::range<1>(64);
	auto buf = tt.mbf.create_buffer(range);

	test_utils::add_compute_task<name_class<5>>(
	    tt.tm, [&](handler& cgh) { buf.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);

	const auto* escaped_name = "\"name_class&lt;...&gt;\"";
	REQUIRE_THAT(print_task_graph(tt.trec), Catch::Matchers::ContainsSubstring(escaped_name));
}
