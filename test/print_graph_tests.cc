#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <libenvpp/env.hpp>

#include "distributed_graph_generator_test_utils.h"
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

	test_utils::add_compute_task<class UKN(task_initialize)>(
	    tt.tm, [&](handler& cgh) { buf_1.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);
	test_utils::add_compute_task<class UKN(task_produce)>(
	    tt.tm, [&](handler& cgh) { buf_0.get_access<access_mode::discard_write>(cgh, acc::one_to_one{}); }, range);
	test_utils::add_compute_task<class UKN(task_reduce)>(
	    tt.tm,
	    [&](handler& cgh) {
		    buf_0.get_access<access_mode::read>(cgh, acc::one_to_one{});
		    test_utils::add_reduction(cgh, tt.mrf, buf_1, true /* include_current_buffer_value */);
	    },
	    range);
	test_utils::add_compute_task<class UKN(task_consume)>(
	    tt.tm,
	    [&](handler& cgh) {
		    buf_1.get_access<access_mode::read>(cgh, acc::fixed<1>({0, 1}));
	    },
	    range);

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const std::string expected =
	    "digraph G {label=\"Task Graph\" 0[shape=ellipse label=<T0<br/><b>epoch</b>>];1[shape=box style=rounded label=<T1 \"task_initialize_2\" "
	    "<br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>discard_write</i> B1 {[0,0,0] - [1,1,1]}>];0->1[color=orchid];2[shape=box style=rounded "
	    "label=<T2 \"task_produce_3\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>discard_write</i> B0 {[0,0,0] - "
	    "[64,1,1]}>];0->2[color=orchid];3[shape=box style=rounded label=<T3 \"task_reduce_4\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/>(R1) "
	    "<i>read_write</i> B1 {[0,0,0] - [1,1,1]}<br/><i>read</i> B0 {[0,0,0] - [64,1,1]}>];1->3[];2->3[];4[shape=box style=rounded label=<T4 "
	    "\"task_consume_5\" <br/><b>device-compute</b> [0,0,0] - [64,1,1]<br/><i>read</i> B1 {[0,0,0] - [1,1,1]}>];3->4[];}";

	CHECK(print_task_graph(tt.trec) == expected);
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

TEST_CASE("command graph printing is unchanged", "[print_graph][command-graph]") {
	size_t num_nodes = 4;
	dist_cdag_test_context dctx(num_nodes);

	auto buf_0 = dctx.create_buffer(range<1>{1});

	dctx.device_compute<class UKN(reduce)>(range<1>(num_nodes)).reduce(buf_0, false).submit();
	dctx.device_compute<class UKN(consume)>(range<1>(num_nodes)).read(buf_0, acc::all{}).read_write(buf_0, acc::all{}).write(buf_0, acc::all{}).submit();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graph is sane and
	// replace the `expected` value with the new dot graph.
	const std::string expected =
	    "digraph G{label=\"Command Graph\" subgraph cluster_id_0_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;id_0_0[label=<C0 on "
	    "N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_id_0_1{label=<<font color=\"#606060\">T1 \"reduce_8\" "
	    "(device-compute)</font>>;color=darkgray;id_0_1[label=<C1 on N0<br/><b>execution</b> [0,0,0] - [1,1,1]<br/>(R1) <i>discard_write</i> B0 {[0,0,0] - "
	    "[1,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_2{label=<<font color=\"#606060\">T2 \"consume_9\" "
	    "(device-compute)</font>>;color=darkgray;id_0_2[label=<C2 on N0<br/><b>execution</b> [0,0,0] - [1,1,1]<br/><i>read</i> B0 {[0,0,0] - "
	    "[1,1,1]}<br/><i>read_write</i> B0 {[0,0,0] - [1,1,1]}<br/><i>write</i> B0 {[0,0,0] - [1,1,1]}> fontcolor=black "
	    "shape=box];}id_0_0->id_0_1[color=orchid];id_0_3->id_0_2[];id_0_5->id_0_2[color=limegreen];id_0_6->id_0_2[color=limegreen];id_0_7->id_0_2[color="
	    "limegreen];id_0_3[label=<C3 on N0<br/><b>reduction</b> R1<br/> B0 {[0,0,0] - [1,1,1]}> fontcolor=black "
	    "shape=ellipse];id_0_1->id_0_3[];id_0_4->id_0_3[];id_0_4[label=<C4 on N0<br/>(R1) <b>await push</b> transfer 8589934592 <br/>BB0 {[0,0,0] - "
	    "[1,1,1]}> fontcolor=black shape=ellipse];id_0_0->id_0_4[color=orchid];id_0_5[label=<C5 on N0<br/>(R1) <b>push</b> transfer 8589934593 to N1<br/>BB0 "
	    "[0,0,0] - [1,1,1]> fontcolor=black shape=ellipse];id_0_1->id_0_5[];id_0_6[label=<C6 on N0<br/>(R1) <b>push</b> transfer 8589934594 to N2<br/>BB0 "
	    "[0,0,0] - [1,1,1]> fontcolor=black shape=ellipse];id_0_1->id_0_6[];id_0_7[label=<C7 on N0<br/>(R1) <b>push</b> transfer 8589934595 to N3<br/>BB0 "
	    "[0,0,0] - [1,1,1]> fontcolor=black shape=ellipse];id_0_1->id_0_7[];}";

	// fully check node 0
	const auto dot0 = dctx.print_command_graph(0);
	CHECK(dot0 == expected);

	// only check the rough string length and occurence count of N1/N2... for other nodes
	const int expected_occurences = count_occurences(expected, "N0");
	for(size_t i = 1; i < num_nodes; ++i) {
		const auto dot_n = dctx.print_command_graph(i);
		REQUIRE_THAT(dot_n.size(), Catch::Matchers::WithinAbs(expected.size(), 50));
		CHECK(count_occurences(dot_n, fmt::format("N{}", i)) == expected_occurences);
	}
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer debug names show up in the generated graph", "[print_graph]") {
	env::scoped_test_environment tenv(recording_enabled_env_setting);

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

TEST_CASE_METHOD(test_utils::runtime_fixture, "full graph is printed if CELERITY_RECORDING is set", "[print_graph]") {
	env::scoped_test_environment tenv(recording_enabled_env_setting);

	distr_queue q;
	celerity::range<1> range(16);
	celerity::buffer<int, 1> buff_a(range);

	// set small horizon step size so that we do not need to generate a very large graph to test this functionality
	auto& tm = celerity::detail::runtime::get_instance().get_task_manager();
	tm.set_horizon_step(1);

	for(int i = 0; i < 3; ++i) {
		q.submit([&](handler& cgh) {
			celerity::accessor acc_a{buff_a, cgh, acc::one_to_one{}, celerity::read_write};
			cgh.parallel_for<class UKN(full_graph_printing)>(range, [=](item<1> item) { (void)acc_a; });
		});
	}

	q.slow_full_sync();

	// Smoke test: It is valid for the dot output to change with updates to graph generation. If this test fails, verify that the printed graphs are sane and
	// complete, and if so, replace the `expected` values with the new dot graph.

	SECTION("task graph") {
		const auto* expected =
		    "digraph G {label=\"Task Graph\" 0[shape=ellipse label=<T0<br/><b>epoch</b>>];1[shape=box style=rounded label=<T1 \"full_graph_printing_17\" "
		    "<br/><b>device-compute</b> [0,0,0] - [16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}>];0->1[color=orchid];2[shape=ellipse "
		    "label=<T2<br/><b>horizon</b>>];1->2[color=orange];3[shape=box style=rounded label=<T3 \"full_graph_printing_17\" <br/><b>device-compute</b> "
		    "[0,0,0] - [16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}>];1->3[];4[shape=ellipse "
		    "label=<T4<br/><b>horizon</b>>];3->4[color=orange];2->4[color=orange];5[shape=box style=rounded label=<T5 \"full_graph_printing_17\" "
		    "<br/><b>device-compute</b> [0,0,0] - [16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}>];3->5[];6[shape=ellipse "
		    "label=<T6<br/><b>horizon</b>>];5->6[color=orange];4->6[color=orange];7[shape=ellipse label=<T7<br/><b>epoch</b>>];6->7[color=orange];}";

		CHECK(runtime_testspy::print_task_graph(celerity::detail::runtime::get_instance()) == expected);
	}

	SECTION("command graph") {
		const auto* expected =
		    "digraph G{label=\"Command Graph\" subgraph cluster_id_0_0{label=<<font color=\"#606060\">T0 (epoch)</font>>;color=darkgray;id_0_0[label=<C0 on "
		    "N0<br/><b>epoch</b>> fontcolor=black shape=box];}subgraph cluster_id_0_1{label=<<font color=\"#606060\">T1 \"full_graph_printing_17\" "
		    "(device-compute)</font>>;color=darkgray;id_0_1[label=<C1 on N0<br/><b>execution</b> [0,0,0] - [16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - "
		    "[16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_2{label=<<font color=\"#606060\">T2 "
		    "(horizon)</font>>;color=darkgray;id_0_2[label=<C2 on N0<br/><b>horizon</b>> fontcolor=black shape=box];}subgraph cluster_id_0_3{label=<<font "
		    "color=\"#606060\">T3 \"full_graph_printing_17\" (device-compute)</font>>;color=darkgray;id_0_3[label=<C3 on N0<br/><b>execution</b> [0,0,0] - "
		    "[16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_4{label=<<font color=\"#606060\">T4 "
		    "(horizon)</font>>;color=darkgray;id_0_4[label=<C4 on N0<br/><b>horizon</b>> fontcolor=black shape=box];}subgraph cluster_id_0_5{label=<<font "
		    "color=\"#606060\">T5 \"full_graph_printing_17\" (device-compute)</font>>;color=darkgray;id_0_5[label=<C5 on N0<br/><b>execution</b> [0,0,0] - "
		    "[16,1,1]<br/><i>read_write</i> B0 {[0,0,0] - [16,1,1]}> fontcolor=black shape=box];}subgraph cluster_id_0_6{label=<<font color=\"#606060\">T6 "
		    "(horizon)</font>>;color=darkgray;id_0_6[label=<C6 on N0<br/><b>horizon</b>> fontcolor=black shape=box];}subgraph cluster_id_0_7{label=<<font "
		    "color=\"#606060\">T7 (epoch)</font>>;color=darkgray;id_0_7[label=<C7 on N0<br/><b>epoch</b> (barrier)> fontcolor=black "
		    "shape=box];}id_0_0->id_0_1[];id_0_1->id_0_2[color=orange];id_0_1->id_0_3[];id_0_3->id_0_4[color=orange];id_0_2->id_0_4[color=orange];id_0_3->id_0_"
		    "5[];id_0_5->id_0_6[color=orange];id_0_4->id_0_6[color=orange];id_0_6->id_0_7[color=orange];}";

		CHECK(runtime_testspy::print_command_graph(0, celerity::detail::runtime::get_instance()) == expected);
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
