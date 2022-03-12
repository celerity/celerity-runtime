#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "intrusive_graph.h"
#include "task_manager.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

struct bench_graph_node : intrusive_graph_node<bench_graph_node> {};

// try to cover the dependency counts we'll see in practice
TEMPLATE_TEST_CASE_SIG("benchmark intrusive graph dependency handling with N nodes", "[benchmark][intrusive_graph_node]", ((int N), N), 1, 10, 100) {
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


struct task_manager_benchmark_context {
	task_manager tm{1, nullptr, nullptr};
	test_utils::mock_buffer_factory mbf{&tm};

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		tm.create_task([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};

struct graph_generator_benchmark_context {
	const size_t num_nodes;
	command_graph cdag;
	graph_serializer gsrlzr{cdag, [](node_id, command_pkg, const std::vector<command_id>&) {}};
	reduction_manager rm;
	task_manager tm{num_nodes, nullptr, &rm};
	graph_generator ggen{num_nodes, tm, rm, cdag};
	test_utils::mock_buffer_factory mbf{&tm, &ggen};

	explicit graph_generator_benchmark_context(size_t num_nodes) : num_nodes{num_nodes} {
		tm.register_task_callback([this](const task_id tid, const task_type) {
			naive_split_transformer transformer{this->num_nodes, this->num_nodes};
			ggen.build_task(tid, {&transformer});
			gsrlzr.flush(tid);
		});
	}

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		// note: This ignores communication overhead with the scheduler thread
		tm.create_task([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};

struct scheduler_benchmark_context {
	const size_t num_nodes;
	command_graph cdag;
	graph_serializer gsrlzr{cdag, [](node_id, command_pkg, const std::vector<command_id>&) {}};
	reduction_manager rm;
	task_manager tm{num_nodes, nullptr, &rm};
	graph_generator ggen{num_nodes, tm, rm, cdag};
	scheduler schdlr;
	test_utils::mock_buffer_factory mbf{&tm, &ggen};

	explicit scheduler_benchmark_context(background_thread& thrd, size_t num_nodes) : num_nodes{num_nodes}, schdlr{thrd, ggen, gsrlzr, num_nodes} {
		schdlr.startup();
	}

	~scheduler_benchmark_context() {
		// scheduler operates in a FIFO manner, so awaiting shutdown will await processing of all pending tasks first
		schdlr.shutdown();
	}

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		const auto tid = tm.create_task([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
		schdlr.notify_task_created(tid);
	}
};

// The generate_* methods are [[noinline]] to make them visible in a profiler.

// Artificial: large set of disconnected tasks, does not generate horizons
template <typename BenchmarkContext>
[[gnu::noinline]] void generate_soup_graph(BenchmarkContext&& ctx) {
	constexpr int num_tasks = 1000;
	const range<1> buffer_range{2048};

	for(int t = 0; t < num_tasks; ++t) {
		ctx.create_task(buffer_range, [](handler& cgh) {});
	}
}

// Artificial: Generate a linear chain of dependent tasks
template <typename BenchmarkContext>
[[gnu::noinline]] void generate_chain_graph(BenchmarkContext&& ctx) {
	constexpr int num_tasks = 200;
	const range<1> buffer_range{2048};

	test_utils::mock_buffer<1> buffer = ctx.mbf.create_buffer(buffer_range);
	for(int t = 0; t < num_tasks; ++t) {
		ctx.create_task(buffer_range, [&](handler& cgh) { buffer.get_access<access_mode::read_write>(cgh, celerity::access::one_to_one()); });
	}
}

// Artificial: Generate expanding (Map) or contracting (Reduce) tree of tasks
enum class TreeTopology { Map, Reduce };

template <TreeTopology Topology, typename BenchmarkContext>
[[gnu::noinline]] void generate_tree_graph(BenchmarkContext&& ctx) {
	constexpr int num_tasks = 100;
	const range<1> buffer_range{2048};

	test_utils::mock_buffer<1> buffer = ctx.mbf.create_buffer(buffer_range);
	test_utils::mock_buffer<1> buffer2 = ctx.mbf.create_buffer(buffer_range);

	int numEpochs = std::log2(num_tasks);
	int curEpochTasks = Topology == TreeTopology::Map ? 1 : 1 << numEpochs;
	int sentinelEpoch = Topology == TreeTopology::Map ? numEpochs - 1 : 0;
	int sentinelEpochMax = num_tasks - (curEpochTasks - 1); // how many tasks to generate at the last/first epoch to reach exactly numTasks

	for(int e = 0; e < numEpochs; ++e) {
		int taskCount = curEpochTasks;
		if(e == sentinelEpoch) taskCount = sentinelEpochMax;

		// build tasks for this epoch
		for(int t = 0; t < taskCount; ++t) {
			ctx.create_task(range<1>{1}, [&](celerity::handler& cgh) {
				// mappers constructed to build a binary (potentially inverted) tree
				auto read_mapper = [=](const celerity::chunk<1>& chunk) {
					return Topology == TreeTopology::Map ? celerity::subrange<1>(t / 2, 1) : celerity::subrange<1>(t * 2, 2);
				};
				auto write_mapper = [=](const celerity::chunk<1>& chunk) { return celerity::subrange<1>(t, 1); };
				buffer.get_access<access_mode::write>(cgh, write_mapper);
				buffer2.get_access<access_mode::read>(cgh, read_mapper);
			});
		}

		// get ready for the next epoch
		if(Topology == TreeTopology::Map) {
			curEpochTasks *= 2;
		} else {
			curEpochTasks /= 2;
		}
		std::swap(buffer, buffer2);
	}
}

// graphs identical to the wave_sim example
template <typename BenchmarkContext>
[[gnu::noinline]] void generate_wave_sim_graph(BenchmarkContext&& ctx) {
	constexpr int N = 512;
	constexpr float T = 20;
	constexpr float dt = 0.25f;

	const auto fill = [&](test_utils::mock_buffer<2> u) {
		ctx.create_task(u.get_range(), [&](celerity::handler& cgh) { u.get_access<access_mode::discard_write>(cgh, celerity::access::one_to_one{}); });
	};

	const auto step = [&](test_utils::mock_buffer<2> up, test_utils::mock_buffer<2> u) {
		ctx.create_task(up.get_range(), [&](celerity::handler& cgh) {
			up.get_access<access_mode::read_write>(cgh, celerity::access::one_to_one{});
			u.get_access<access_mode::read>(cgh, celerity::access::neighborhood{1, 1});
		});
	};

	auto up = ctx.mbf.create_buffer(celerity::range<2>(N, N));
	auto u = ctx.mbf.create_buffer(celerity::range<2>(N, N));

	fill(u);
	fill(up);
	step(up, u);

	auto t = 0.0;
	size_t i = 0;
	while(t < T) {
		step(up, u);
		std::swap(u, up);
		t += dt;
	}
}

// Graph of a simple iterative Jacobi solver
template <typename BenchmarkContext>
[[gnu::noinline]] void generate_jacobi_graph(BenchmarkContext&& ctx) {
	constexpr int N = 1024;
	constexpr int steps = 50;

	// Naming scheme from https://en.wikipedia.org/wiki/Jacobi_method#Python_example
	test_utils::mock_buffer<2> A = ctx.mbf.create_buffer(range<2>{N, N}, true /* host initialized */);
	test_utils::mock_buffer<1> b = ctx.mbf.create_buffer(range<1>{N}, true /* host initialized */);
	test_utils::mock_buffer<1> x = ctx.mbf.create_buffer(range<1>{N});
	test_utils::mock_buffer<1> x_new = ctx.mbf.create_buffer(range<1>{N});

	// initial guess zero
	ctx.create_task(range<1>{N}, [&](handler& cgh) { x.get_access<access_mode::discard_write>(cgh, celerity::access::one_to_one{}); });

	constexpr auto one_to_one = celerity::access::one_to_one{};
	constexpr auto rows = [](const chunk<2>& chnk) { return subrange<1>{chnk.offset[0], chnk.range[0]}; };
	constexpr auto columns = [](const chunk<2>& chnk) { return subrange<1>{chnk.offset[1], chnk.range[1]}; };

	for(int k = 0; k < steps; ++k) {
		ctx.create_task(range<2>{N, N}, [&](handler& cgh) {
			A.get_access<access_mode::read>(cgh, one_to_one);
			b.get_access<access_mode::read>(cgh, rows);
			x.get_access<access_mode::read>(cgh, columns);
			x_new.get_access<access_mode::discard_write>(cgh, rows); // dependent on dim0 split
		});
		std::swap(x, x_new);
	}
}

template <typename BenchmarkContextFactory>
void run_benchmarks(BenchmarkContextFactory&& make_ctx) {
	BENCHMARK("soup topology") { generate_soup_graph(make_ctx()); };
	BENCHMARK("chain topology") { generate_chain_graph(make_ctx()); };
	BENCHMARK("map topology") { generate_tree_graph<TreeTopology::Map>(make_ctx()); };
	BENCHMARK("reduce topology") { generate_tree_graph<TreeTopology::Reduce>(make_ctx()); };
	BENCHMARK("wave_sim topology") { generate_wave_sim_graph(make_ctx()); };
	BENCHMARK("jacobi topology") { generate_jacobi_graph(make_ctx()); };
}

TEST_CASE("generating large task graphs", "[benchmark][task-graph]") {
	run_benchmarks([] { return task_manager_benchmark_context{}; });
}

TEMPLATE_TEST_CASE_SIG("generating large command graphs for N nodes", "[benchmark][command-graph]", ((size_t NumNodes), NumNodes), 1, 4, 16) {
	run_benchmarks([] { return graph_generator_benchmark_context{NumNodes}; });
}

TEMPLATE_TEST_CASE_SIG("processing large graphs with a scheduler thread for N nodes", "[benchmark][scheduler]", ((size_t NumNodes), NumNodes), 1, 4, 16) {
	background_thread thrd;
	run_benchmarks([&] { return scheduler_benchmark_context{thrd, NumNodes}; });
}
