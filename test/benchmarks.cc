#include <catch2/catch.hpp>

#include "intrusive_graph.h"

using namespace celerity::detail;

struct bench_graph_node : intrusive_graph_node<bench_graph_node> {};

template <int N>
void intrusive_graph_benchmark() {
	// note that bench_graph_nodes are created/destroyed *within* the BENCHMARK
	// in the first two cases while the latter 2 cases only operate on already
	// existing nodes -- this is intentional; both cases are relevant in practise

	BENCHMARK("creating nodes") {
		bench_graph_node nodes[N];
		return nodes[N - 1].get_pseudo_critical_path_length(); // trick the compiler
	};

	BENCHMARK("creating and adding dependencies") {
		bench_graph_node n0;
		bench_graph_node nodes[N];
		for(int i = 0; i < N; ++i) {
			n0.add_dependency({&nodes[i], dependency_kind::TRUE_DEP});
		}
		return n0.get_dependencies();
	};

	bench_graph_node n0;
	bench_graph_node nodes[N];
	BENCHMARK("adding and removing dependencies") {
		for(int i = 0; i < N; ++i) {
			n0.add_dependency({&nodes[i], dependency_kind::TRUE_DEP});
		}
		for(int i = 0; i < N; ++i) {
			n0.remove_dependency(&nodes[i]);
		}
		return n0.get_dependencies();
	};

	for(int i = 0; i < N; ++i) {
		n0.add_dependency({&nodes[i], dependency_kind::TRUE_DEP});
	}
	BENCHMARK("checking for dependencies") {
		int d = 0;
		for(int i = 0; i < N; ++i) {
			d += n0.has_dependency(&nodes[i]) ? 1 : 0;
		}
		return d;
	};
}

// try to cover the dependency counts we'll see in practice
TEST_CASE("benchmark intrusive graph dependency handling, N=1", "[benchmark]") {
	intrusive_graph_benchmark<1>();
}
TEST_CASE("benchmark intrusive graph dependency handling, N=10", "[benchmark]") {
	intrusive_graph_benchmark<10>();
}
TEST_CASE("benchmark intrusive graph dependency handling, N=100", "[benchmark]") {
	intrusive_graph_benchmark<100>();
}
