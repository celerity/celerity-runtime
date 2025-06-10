#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "command_graph.h"
#include "command_graph_generator.h"
#include "instruction_graph_generator.h"
#include "intrusive_graph.h"
#include "task_manager.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace std::chrono_literals;

struct bench_graph_node : intrusive_graph_node<bench_graph_node> {};

// try to cover the dependency counts we'll see in practice
TEMPLATE_TEST_CASE_SIG("benchmark intrusive graph dependency handling with N nodes", "[benchmark][group:graph-nodes]", ((int N), N), 1, 10, 100) {
	test_utils::benchmark_thread_pinner pinner;
	// note that bench_graph_nodes are created/destroyed *within* the BENCHMARK
	// in the first two cases while the latter 2 cases only operate on already
	// existing nodes -- this is intentional; both cases are relevant in practise

	BENCHMARK("creating nodes") {
		const bench_graph_node nodes[N];
		return nodes[N - 1].get_pseudo_critical_path_length(); // trick the compiler
	};

	BENCHMARK("creating and adding dependencies") {
		bench_graph_node n0;
		bench_graph_node nodes[N];
		for(int i = 0; i < N; ++i) {
			n0.add_dependency({&nodes[i], dependency_kind::true_dep, dependency_origin::dataflow});
		}
		return n0.get_dependencies();
	};

	bench_graph_node n0;
	bench_graph_node nodes[N];
	BENCHMARK("adding and removing dependencies") {
		for(int i = 0; i < N; ++i) {
			n0.add_dependency({&nodes[i], dependency_kind::true_dep, dependency_origin::dataflow});
		}
		for(int i = 0; i < N; ++i) {
			n0.remove_dependency(&nodes[i]);
		}
		return n0.get_dependencies();
	};

	for(int i = 0; i < N; ++i) {
		n0.add_dependency({&nodes[i], dependency_kind::true_dep, dependency_origin::dataflow});
	}
	BENCHMARK("checking for dependencies") {
		int d = 0;
		for(int i = 0; i < N; ++i) {
			d += n0.has_dependency(&nodes[i]) ? 1 : 0;
		}
		return d;
	};
}

TEST_CASE("benchmark task handling", "[benchmark][group:task-graph]") {
	test_utils::benchmark_thread_pinner pinner;
	constexpr int N = 10000;
	constexpr int report_interval = 10;

	BENCHMARK("generating and deleting tasks") {
		task_graph tdag;
		task_manager tm(1, tdag, nullptr, nullptr);
		// we use this trick to force horizon creation without introducing dependency overhead in this microbenchmark
		tm.set_horizon_step(0);

		tm.generate_epoch_task(epoch_action::init);
		for(int i = 0; i < N; ++i) {
			// create simplest possible host task
			auto cg = invoke_command_group_function([](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
			const auto highest_tid = tm.generate_command_group_task(std::move(cg));
			// start notifying once we've built some tasks
			if(i % report_interval == 0 && i / report_interval > 2) {
				// every other generated task is always a horizon (step size 0)
				tdag.erase_before_epoch(highest_tid - 1);
			}
		}
	};
}


// these policies are equivalent to the ones used by `runtime` (except that we throw exceptions here for benchmark-debugging purposes)
static constexpr task_manager::policy_set benchmark_task_manager_policy = {
    /* uninitialized_read_error */ CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::panic : error_policy::ignore,
};
static constexpr command_graph_generator::policy_set benchmark_command_graph_generator_policy{
    /* uninitialized_read_error */ error_policy::ignore, // uninitialized reads already detected by task manager
    /* overlapping_write_error */ CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::panic : error_policy::ignore,
};
static constexpr instruction_graph_generator::policy_set benchmark_instruction_graph_generator_policy{
    /* uninitialized_read_error */ error_policy::ignore, // uninitialized reads already detected by task manager
    /* overlapping_write_error */ CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::panic : error_policy::ignore,
};

struct dag_benchmark_context {
	dag_benchmark_context() = default;
	virtual ~dag_benchmark_context() = 0;
	CELERITY_DETAIL_UTILS_NON_COPYABLE(dag_benchmark_context);
	CELERITY_DETAIL_UTILS_NON_MOVABLE(dag_benchmark_context);

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		m_command_groups.push_back(invoke_command_group_function([=](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for(global_range, [](item<KernelDims>) {});
		}));
	}

  protected:
	std::vector<raw_command_group> m_command_groups;
};

dag_benchmark_context::~dag_benchmark_context() = default;

struct tdag_benchmark_context : dag_benchmark_context, private task_manager::delegate {
	const size_t num_nodes;
	task_graph tdag;
	task_recorder trec;
	task_manager tm{num_nodes, tdag, test_utils::g_print_graphs ? &trec : nullptr, static_cast<task_manager::delegate*>(this), benchmark_task_manager_policy};
	test_utils::mock_buffer_factory mbf{tm};

	explicit tdag_benchmark_context(const size_t num_nodes = 1) : num_nodes(num_nodes) {}

	void task_created(const task* tsk) override { m_tasks.push_back(tsk); }

	// is called before the initialization of benchmark tasks
	void initialize() { tm.generate_epoch_task(celerity::detail::epoch_action::init); }

	// is called after the initialization of benchmark tasks
	void prepare() { m_tasks.reserve(m_command_groups.size()); }

	// executes everything that is actually benchmarked (measured)
	void execute() { create_all_tasks(); }

	// finalizes the benchmark context, e.g. generates shutdown epoch, and clears the command groups
	void finalize() {
		tm.generate_epoch_task(celerity::detail::epoch_action::shutdown);
		m_command_groups.clear();
	}

  protected:
	std::vector<const task*> m_tasks; // for use in derived classes

	void create_all_tasks() {
		for(auto& cg : m_command_groups) {
			tm.generate_command_group_task(std::move(cg));
		}
	}
};

struct cdag_benchmark_context : tdag_benchmark_context {
	command_graph cdag;
	command_recorder crec;
	command_graph_generator cggen{num_nodes, 0 /* local_nid */, cdag, test_utils::g_print_graphs ? &crec : nullptr, benchmark_command_graph_generator_policy};
	test_utils::mock_buffer_factory mbf{tm, cggen};

	explicit cdag_benchmark_context(const size_t num_nodes) : tdag_benchmark_context(num_nodes) {}

	void initialize() {
		tdag_benchmark_context::initialize();
		create_all_commands(); // used to initialize the m_command_batches even if empty
		m_tasks.clear();
	}

	void prepare() {
		create_all_tasks();
		m_command_batches.reserve(m_tasks.size());
	}

	void execute() { create_all_commands(); }

	void finalize() {
		m_tasks.clear();
		tdag_benchmark_context::finalize();
		create_all_commands();
	}

  protected:
	std::vector<std::vector<const command*>> m_command_batches; // for use in derived classes

	void create_all_commands() {
		for(const auto* tsk : m_tasks) {
			m_command_batches.push_back(cggen.build_task(*tsk));
		}
	}
};

struct idag_benchmark_context : cdag_benchmark_context {
	const size_t num_devices;
	const bool supports_d2d_copies;
	instruction_recorder irec;
	instruction_graph idag;
	instruction_graph_generator iggen{num_nodes, 0 /* local nid */, test_utils::make_system_info(num_devices, supports_d2d_copies), idag,
	    nullptr /* delegate */, test_utils::g_print_graphs ? &irec : nullptr, benchmark_instruction_graph_generator_policy};
	test_utils::mock_buffer_factory mbf{tm, cggen, iggen};

	explicit idag_benchmark_context(const size_t num_nodes, const size_t num_devices, const bool supports_d2d_copies = true)
	    : cdag_benchmark_context(num_nodes), num_devices(num_devices), supports_d2d_copies(supports_d2d_copies) {}

	void initialize() {
		cdag_benchmark_context::initialize();
		create_all_instructions(); // used to initialize the m_command_batches even if empty
		m_command_batches.clear();
	}

	void prepare() {
		create_all_tasks();
		create_all_commands();
	}

	void execute() { create_all_instructions(); }

	void finalize() {
		m_command_batches.clear();
		cdag_benchmark_context::finalize();
		create_all_instructions();
	}

  protected:
	void create_all_instructions() {
		for(const auto& batch : m_command_batches) {
			for(const auto* cmd : batch) {
				iggen.compile(*cmd);
			}
		}
	}
};

/// Like idag_benchmark_context, but measures construction of all three graphs
struct all_dags_benchmark_context : idag_benchmark_context {
	all_dags_benchmark_context(const size_t num_nodes, const size_t num_devices, const bool supports_d2d_copies = true)
	    : idag_benchmark_context(num_nodes, num_devices, supports_d2d_copies) {}

	void initialize() {
		tdag_benchmark_context::initialize();
		create_all_commands();
		create_all_instructions();
		m_tasks.clear();
		m_command_batches.clear();
	}

	void prepare() { /* no-op */ }

	void execute() {
		create_all_tasks();
		create_all_commands();
		create_all_instructions();
	}
};

// Keeps an OS thread around between benchmark iterations to avoid creating thousands of threads
class restartable_scheduler_thread {
	struct empty {};
	struct shutdown {};

  public:
	using thread_func = std::function<void()>;

	restartable_scheduler_thread() = default;
	restartable_scheduler_thread(const restartable_scheduler_thread&) = delete;
	restartable_scheduler_thread(restartable_scheduler_thread&&) = delete;
	restartable_scheduler_thread& operator=(const restartable_scheduler_thread&) = delete;
	restartable_scheduler_thread& operator=(restartable_scheduler_thread&&) = delete;

	~restartable_scheduler_thread() {
		{
			std::unique_lock lk{m_mutex};
			wait(lk);
			m_next = shutdown{};
			m_update.notify_one();
		}
		m_thread.join();
	}

	void start(std::function<void()> thread_func) {
		std::unique_lock lk{m_mutex};
		wait(lk);
		m_next = std::move(thread_func);
		m_update.notify_all();
	}

	void join() {
		std::unique_lock lk{m_mutex};
		wait(lk);
	}

  private:
	std::mutex m_mutex;
	std::variant<empty, thread_func, shutdown> m_next;
	std::condition_variable m_update;
	std::thread m_thread{&restartable_scheduler_thread::main, this};

	void main() {
		// This thread is used for scheduling, so pin it to the scheduler core
		name_and_pin_and_order_this_thread(named_threads::thread_type::scheduler);
		std::unique_lock lk{m_mutex};
		for(;;) {
			m_update.wait(lk, [this] { return !std::holds_alternative<empty>(m_next); });
			if(std::holds_alternative<shutdown>(m_next)) break;
			std::get<thread_func>(m_next)();
			m_next = empty{};
			m_update.notify_all();
		}
	}

	void wait(std::unique_lock<std::mutex>& lk) {
		m_update.wait(lk, [this] {
			assert(!std::holds_alternative<shutdown>(m_next));
			return std::holds_alternative<empty>(m_next);
		});
	}
};

struct scheduler_benchmark_context : tdag_benchmark_context {
	restartable_scheduler_thread* thread;
	scheduler schdlr;
	test_utils::mock_buffer_factory mbf;

	explicit scheduler_benchmark_context(restartable_scheduler_thread& thrd, const size_t num_nodes, const size_t num_devices_per_node)
	    : tdag_benchmark_context(num_nodes), thread(&thrd), //
	      schdlr(scheduler_testspy::make_threadless_scheduler(num_nodes, 0 /* local_nid */,
	          test_utils::make_system_info(num_devices_per_node, true /* supports d2d copies */), nullptr /* delegate */, nullptr /* crec */,
	          nullptr /* irec */)),
	      mbf(tm, schdlr) {}

	void task_created(const task* tsk) override { schdlr.notify_task_created(tsk); }

	void execute() {
		thread->start([this] { scheduler_testspy::run_scheduling_loop(schdlr); });
		create_all_tasks();
		const auto tid = tm.generate_epoch_task(celerity::detail::epoch_action::shutdown);
		// There is no executor thread and notifications are processed in-order, so we can immediately notify the scheduler about shutdown-epoch completion
		schdlr.notify_epoch_reached(tid);
		thread->join();
	}

	void finalize() { /* no-op */ }
};

// Artificial: large set of disconnected tasks, does not generate horizons
template <typename BenchmarkContext>
void generate_soup_graph(BenchmarkContext& ctx, const size_t num_tasks) {
	test_utils::mock_buffer<2> buf = ctx.mbf.create_buffer(range<2>{ctx.num_nodes, num_tasks}, true /* host_initialized */);
	for(size_t t = 0; t < num_tasks; ++t) {
		ctx.create_task(range<1>{ctx.num_nodes},
		    [&](handler& cgh) { buf.get_access<access_mode::read_write>(cgh, [=](chunk<1> ck) { return subrange<2>{{ck.offset[0], t}, {ck.range[0], 1}}; }); });
	}
}

// Artificial: Linear chain of dependent tasks, with all-to-all communication
template <typename BenchmarkContext>
void generate_chain_graph(BenchmarkContext& ctx, const size_t num_tasks) {
	const range<2> global_range{ctx.num_nodes, ctx.num_nodes};
	test_utils::mock_buffer<2> buf = ctx.mbf.create_buffer(global_range, true /* host initialized */);
	for(size_t t = 0; t < num_tasks; ++t) {
		ctx.create_task(global_range, [&](handler& cgh) {
			buf.get_access<access_mode::read>(cgh, [=](chunk<2> ck) { return subrange<2>{{ck.offset[1], ck.offset[0]}, {ck.range[1], ck.range[0]}}; });
			buf.get_access<access_mode::write>(cgh, celerity::access::one_to_one{});
		});
	}
}

// Artificial: Generate expanding or contracting tree of tasks, with gather/scatter communication
enum class tree_topology { expanding, contracting };

template <tree_topology Topology, typename BenchmarkContext>
void generate_tree_graph(BenchmarkContext& ctx, const size_t target_num_tasks) {
	const size_t tree_breadth = static_cast<int>(pow(2, ceil(log2(target_num_tasks + 1)) - 1));
	test_utils::mock_buffer<2> buf = ctx.mbf.create_buffer(range<2>{ctx.num_nodes, tree_breadth}, true /* host initialized */);

	for(size_t exp_step = 1; exp_step <= tree_breadth; exp_step *= 2) {
		const auto sr_range = Topology == tree_topology::expanding ? tree_breadth / exp_step : exp_step;
		for(size_t sr_off = 0; sr_off < tree_breadth; sr_off += sr_range) {
			ctx.create_task(range<1>{ctx.num_nodes}, [&](handler& cgh) {
				buf.get_access<access_mode::read>(cgh, [=](chunk<1> ck) { return subrange<2>{{0, sr_off}, {ck.global_size[0], sr_range}}; });
				buf.get_access<access_mode::write>(cgh, [=](chunk<1> ck) { return subrange<2>{{ck.offset[0], sr_off}, {ck.range[0], sr_range}}; });
			});
		}
	}
}

// graphs identical to the wave_sim example
template <typename BenchmarkContext>
void generate_wave_sim_graph(BenchmarkContext& ctx, const float T) {
	constexpr int N = 512;
	constexpr float dt = 0.25f;

	const auto fill = [&](test_utils::mock_buffer<2> u) {
		ctx.create_task(u.get_range(), [&](celerity::handler& cgh) { u.get_access<access_mode::discard_write>(cgh, celerity::access::one_to_one{}); });
	};

	const auto step = [&](test_utils::mock_buffer<2> up, test_utils::mock_buffer<2> u) {
		ctx.create_task(up.get_range(), [&](celerity::handler& cgh) {
			up.get_access<access_mode::read_write>(cgh, celerity::access::one_to_one{});
			u.get_access<access_mode::read>(cgh, celerity::access::neighborhood{{1, 1}, celerity::neighborhood_shape::along_axes});
		});
	};

	auto up = ctx.mbf.create_buffer(range<2>(N, N));
	auto u = ctx.mbf.create_buffer(range<2>(N, N));

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
void generate_jacobi_graph(BenchmarkContext& ctx, const int steps) {
	constexpr int N = 1024;

	// Naming scheme from https://en.wikipedia.org/wiki/Jacobi_method#Python_example
	test_utils::mock_buffer<2> A = ctx.mbf.create_buffer(range<2>{N, N}, true /* host initialized */);
	test_utils::mock_buffer<1> b = ctx.mbf.create_buffer(range<1>{N}, true /* host initialized */);
	test_utils::mock_buffer<1> x = ctx.mbf.create_buffer(range<1>{N});
	test_utils::mock_buffer<1> x_new = ctx.mbf.create_buffer(range<1>{N});

	// initial guess zero
	ctx.create_task(range<1>{N}, [&](handler& cgh) { x.get_access<access_mode::discard_write>(cgh, celerity::access::one_to_one{}); });

	constexpr auto one_to_one = celerity::access::one_to_one{};
	constexpr auto rows = [](const chunk<2>& ck) { return subrange<1>{ck.offset[0], ck.range[0]}; };
	constexpr auto columns = [](const chunk<2>& ck) { return subrange<1>{ck.offset[1], ck.range[1]}; };

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

template <typename BenchmarkContext, typename... ContextArgs>
void run_benchmarks(ContextArgs&&... args) {
	const auto run = [&](Catch::Benchmark::Chronometer& meter, const auto& cb) {
		std::vector<std::unique_ptr<BenchmarkContext>> contexts; // unique_ptr because contexts are non-movable
		for(int i = 0; i < meter.runs(); ++i) {
			contexts.emplace_back(std::make_unique<BenchmarkContext>(std::forward<ContextArgs>(args)...));
			contexts.back()->initialize();
			cb(*contexts.back());
			contexts.back()->prepare();
		}
		meter.measure([&](const int i) { contexts[i]->execute(); });
		for(auto& ctx : contexts) {
			ctx->finalize();
		}
	};
	BENCHMARK_ADVANCED("soup topology")(Catch::Benchmark::Chronometer meter) {
		run(meter, [](auto& ctx) { generate_soup_graph(ctx, 100); });
	};
	BENCHMARK_ADVANCED("chain topology")(Catch::Benchmark::Chronometer meter) {
		run(meter, [](auto& ctx) { generate_chain_graph(ctx, 30); });
	};
	BENCHMARK_ADVANCED("expanding tree topology")(Catch::Benchmark::Chronometer meter) {
		run(meter, [](auto& ctx) { generate_tree_graph<tree_topology::expanding>(ctx, 30); });
	};
	BENCHMARK_ADVANCED("contracting tree topology")(Catch::Benchmark::Chronometer meter) {
		run(meter, [](auto& ctx) { generate_tree_graph<tree_topology::contracting>(ctx, 30); });
	};
	BENCHMARK_ADVANCED("wave_sim topology")(Catch::Benchmark::Chronometer meter) {
		run(meter, [](auto& ctx) { generate_wave_sim_graph(ctx, 50); });
	};
	BENCHMARK_ADVANCED("jacobi topology")(Catch::Benchmark::Chronometer meter) {
		run(meter, [](auto& ctx) { generate_jacobi_graph(ctx, 50); });
	};
}

TEST_CASE("generating large task graphs", "[benchmark][group:task-graph]") {
	test_utils::benchmark_thread_pinner pinner;
	run_benchmarks<tdag_benchmark_context>();
}

TEMPLATE_TEST_CASE_SIG("generating large command graphs for N nodes", "[benchmark][group:command-graph]", ((size_t NumNodes), NumNodes), 1, 4, 16) {
	test_utils::benchmark_thread_pinner pinner;
	run_benchmarks<cdag_benchmark_context>(NumNodes);
}

TEMPLATE_TEST_CASE_SIG(
    "generating large instruction graphs for N devices", "[benchmark][group:instruction-graph]", ((size_t NumDevices), NumDevices), 1, 4, 16) {
	test_utils::benchmark_thread_pinner pinner;
	constexpr static size_t num_nodes = 2;
	run_benchmarks<idag_benchmark_context>(num_nodes, NumDevices);
}

TEMPLATE_TEST_CASE_SIG("generating large instruction graphs for N devices without d2d copy support", "[benchmark][group:instruction-graph]",
    ((size_t NumDevices), NumDevices), 1, 4, 16) {
	test_utils::benchmark_thread_pinner pinner;
	constexpr static size_t num_nodes = 2;
	run_benchmarks<idag_benchmark_context>(num_nodes, NumDevices, false /* supports_d2d_copies */);
}

TEMPLATE_TEST_CASE_SIG("building command- and instruction graphs in a dedicated scheduler thread for N nodes", "[benchmark][group:scheduler]",
    ((size_t NumNodes), NumNodes), 1, 4) //
{
	test_utils::benchmark_thread_pinner pinner;
	constexpr static size_t num_devices = 1;
	SECTION("reference: single-threaded graph generation") { //
		run_benchmarks<all_dags_benchmark_context>(NumNodes, num_devices);
	}
	SECTION("using a dedicated scheduler thread") {
		restartable_scheduler_thread thrd;
		run_benchmarks<scheduler_benchmark_context>(thrd, NumNodes, num_devices);
	}
}

template <typename BenchmarkContext, typename BenchmarkContextConsumer, typename... ContextArgs>
void debug_graphs(BenchmarkContextConsumer&& debug_ctx, ContextArgs&&... args) {
	const auto run = [&](const auto& cb) {
		BenchmarkContext ctx(std::forward<ContextArgs>(args)...);
		ctx.initialize();
		cb(ctx);
		ctx.prepare();
		ctx.execute();
		ctx.finalize();
		debug_ctx(ctx);
	};
	run([](auto& ctx) { generate_soup_graph(ctx, 10); });
	run([](auto& ctx) { generate_chain_graph(ctx, 5); });
	run([](auto& ctx) { generate_tree_graph<tree_topology::expanding>(ctx, 7); });
	run([](auto& ctx) { generate_tree_graph<tree_topology::contracting>(ctx, 7); });
	run([](auto& ctx) { generate_wave_sim_graph(ctx, 2); });
	run([](auto& ctx) { generate_jacobi_graph(ctx, 5); });
}

TEST_CASE("printing benchmark task graphs", "[.][debug-graphs][task-graph]") {
	REQUIRE(test_utils::g_print_graphs); // requires --print-graphs
	debug_graphs<tdag_benchmark_context>([](auto&& ctx) { fmt::print("{}\n\n", detail::print_task_graph(ctx.trec)); });
}

TEST_CASE("printing benchmark command graphs", "[.][debug-graphs][command-graph]") {
	REQUIRE(test_utils::g_print_graphs); // requires --print-graphs
	debug_graphs<cdag_benchmark_context>([](auto&& ctx) { fmt::print("{}\n\n", detail::print_command_graph(0, ctx.crec)); }, 2 /* num_nodes */);
}

TEST_CASE("printing benchmark instruction graphs", "[.][debug-graphs][instruction-graph]") {
	REQUIRE(test_utils::g_print_graphs); // requires --print-graphs
	debug_graphs<idag_benchmark_context>(
	    [](auto&& ctx) { fmt::print("{}\n\n", detail::print_instruction_graph(ctx.irec, ctx.crec, ctx.trec)); }, 2 /* num_nodes */, 2 /* num_devices */);
}
