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
	using namespace std::chrono_literals;
	std::unique_ptr<task_manager> tm;

	constexpr int N = 10000;
	constexpr int report_interval = 10;

	std::atomic<int> highest_tid = 0;
	std::atomic<int> highest_tid_to_delete = 0;

	auto initialization_lambda = [&] {
		highest_tid = 0;
		tm = std::make_unique<task_manager>(1, nullptr);
		// we use this trick to force horizon creation without introducing dependency overhead in this microbenchmark
		tm->set_horizon_step(0);
	};

	auto task_creation_lambda = [&](bool with_sync = false) {
		for(int i = 0; i < N; ++i) {
			// create simplest possible host task
			highest_tid = tm->submit_command_group([](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
			// start notifying once we've built some tasks
			if(i % report_interval == 0 && i / report_interval > 1) {
				while(with_sync && highest_tid_to_delete.load() < highest_tid.load())
					; // need to potentially wait for task lookup thread to catch up
				// every other generated task is always a horizon (step size 0)
				tm->notify_horizon_reached(highest_tid + 1);
			}
		}
	};

	SECTION("without access thread") {
		highest_tid_to_delete = N * 2; // set sufficiently high to just run through
		BENCHMARK("generating and deleting tasks") {
			initialization_lambda();
			task_creation_lambda();
		};
	}

	SECTION("with access thread") {
		constexpr auto delay_per_call = 20us;
		auto last_call = std::chrono::steady_clock::now();
		std::atomic<bool> run_lookups = false;
		std::atomic<bool> run_thread = true;

		std::thread task_lookup([&] {
			while(run_thread.load()) {
				while(run_lookups.load()) {
					auto htid = highest_tid.load();
					while((run_lookups.load() && htid > 0 && htid <= highest_tid_to_delete.load()) //
					      || std::chrono::steady_clock::now() - last_call < delay_per_call)
						htid = highest_tid.load(); // wait until we are synchronized, and across the minimum delay interval
					tm->get_task(htid);
					highest_tid_to_delete = htid;
				}
				highest_tid_to_delete = 0;
			}
		});

		BENCHMARK("generating and deleting tasks with access thread") {
			initialization_lambda();
			highest_tid_to_delete = 0;
			run_lookups = true;
			task_creation_lambda(true);
			run_lookups = false;
			while(highest_tid_to_delete.load() != 0)
				; // wait for task lookup thread to finish
		};

		run_thread = false;
		task_lookup.join();
	}
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


struct task_manager_benchmark_context {
	const size_t num_nodes = 1;
	task_recorder trec;
	task_manager tm{1, test_utils::g_print_graphs ? &trec : nullptr, benchmark_task_manager_policy};
	test_utils::mock_buffer_factory mbf{tm};

	task_manager_benchmark_context() = default;

	task_manager_benchmark_context(const task_manager_benchmark_context&) = delete;
	task_manager_benchmark_context(task_manager_benchmark_context&&) = delete;
	task_manager_benchmark_context& operator=(const task_manager_benchmark_context&) = delete;
	task_manager_benchmark_context& operator=(task_manager_benchmark_context&&) = delete;

	~task_manager_benchmark_context() { tm.generate_epoch_task(celerity::detail::epoch_action::shutdown); }

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};


struct command_graph_generator_benchmark_context {
	const size_t num_nodes;
	command_graph cdag;
	task_recorder trec;
	task_manager tm{num_nodes, test_utils::g_print_graphs ? &trec : nullptr, benchmark_task_manager_policy};
	command_recorder crec;
	command_graph_generator cggen{
	    num_nodes, 0 /* local_nid */, cdag, tm, test_utils::g_print_graphs ? &crec : nullptr, benchmark_command_graph_generator_policy};
	test_utils::mock_buffer_factory mbf{tm, cggen};

	explicit command_graph_generator_benchmark_context(const size_t num_nodes) : num_nodes(num_nodes) {
		tm.register_task_callback([this](const task* tsk) { const auto cmds = cggen.build_task(*tsk); });
	}

	command_graph_generator_benchmark_context(const command_graph_generator_benchmark_context&) = delete;
	command_graph_generator_benchmark_context(command_graph_generator_benchmark_context&&) = delete;
	command_graph_generator_benchmark_context& operator=(const command_graph_generator_benchmark_context&) = delete;
	command_graph_generator_benchmark_context& operator=(command_graph_generator_benchmark_context&&) = delete;

	~command_graph_generator_benchmark_context() { tm.generate_epoch_task(celerity::detail::epoch_action::shutdown); }

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		// note: This ignores communication overhead with the scheduler thread
		tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for(global_range, [](item<KernelDims>) {});
		});
	}
};

struct instruction_graph_generator_benchmark_context {
	const size_t num_nodes;
	const size_t num_devices;
	const bool supports_d2d_copies;
	command_graph cdag;
	task_recorder trec;
	task_manager tm{num_nodes, test_utils::g_print_graphs ? &trec : nullptr, benchmark_task_manager_policy};
	command_recorder crec;
	command_graph_generator cggen{
	    num_nodes, 0 /* local_nid */, cdag, tm, test_utils::g_print_graphs ? &crec : nullptr, benchmark_command_graph_generator_policy};
	instruction_recorder irec;
	instruction_graph idag;
	instruction_graph_generator iggen{tm, num_nodes, 0 /* local nid */, test_utils::make_system_info(num_devices, supports_d2d_copies), idag,
	    nullptr /* delegate */, test_utils::g_print_graphs ? &irec : nullptr, benchmark_instruction_graph_generator_policy};
	test_utils::mock_buffer_factory mbf{tm, cggen, iggen};

	explicit instruction_graph_generator_benchmark_context(const size_t num_nodes, const size_t num_devices, const bool supports_d2d_copies = true)
	    : num_nodes(num_nodes), num_devices(num_devices), supports_d2d_copies(supports_d2d_copies) {
		tm.register_task_callback([this](const task* tsk) {
			for(const auto cmd : cggen.build_task(*tsk)) {
				iggen.compile(*cmd);
			}
		});
	}

	instruction_graph_generator_benchmark_context(const instruction_graph_generator_benchmark_context&) = delete;
	instruction_graph_generator_benchmark_context(instruction_graph_generator_benchmark_context&&) = delete;
	instruction_graph_generator_benchmark_context& operator=(const instruction_graph_generator_benchmark_context&) = delete;
	instruction_graph_generator_benchmark_context& operator=(instruction_graph_generator_benchmark_context&&) = delete;

	~instruction_graph_generator_benchmark_context() { tm.generate_epoch_task(celerity::detail::epoch_action::shutdown); }

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		// note: This ignores communication overhead with the scheduler thread
		tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};

// Keeps an OS thread around between benchmark iterations to avoid measuring thread creation overhead
class restartable_thread {
	struct empty {};
	struct shutdown {};

  public:
	using thread_func = std::function<void()>;

	~restartable_thread() {
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
	std::thread m_thread{&restartable_thread::main, this};

	void main() {
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

class benchmark_scheduler final : public abstract_scheduler {
  public:
	benchmark_scheduler(restartable_thread& thread, const size_t num_nodes, const node_id local_node_id, const system_info& system_info, const task_manager& tm,
	    delegate* const delegate, command_recorder* const crec, instruction_recorder* const irec)
	    : abstract_scheduler(num_nodes, local_node_id, system_info, tm, delegate, crec, irec), m_thread(&thread) {
		m_thread->start([this] { schedule(); });
	}

	benchmark_scheduler(const benchmark_scheduler&) = delete;
	benchmark_scheduler(benchmark_scheduler&&) = delete;
	benchmark_scheduler& operator=(const benchmark_scheduler&) = delete;
	benchmark_scheduler& operator=(benchmark_scheduler&&) = delete;

	~benchmark_scheduler() override {
		// schedule() will exit as soon as it has acknowledged the shutdown epoch
		m_thread->join();
	}

  private:
	restartable_thread* m_thread;
};

struct scheduler_benchmark_context {
	const size_t num_nodes;
	command_graph cdag;
	task_manager tm{num_nodes, nullptr, benchmark_task_manager_policy};
	benchmark_scheduler schdlr;
	test_utils::mock_buffer_factory mbf;

	explicit scheduler_benchmark_context(restartable_thread& thrd, const size_t num_nodes, const size_t num_devices_per_node)
	    : num_nodes(num_nodes), schdlr(thrd, num_nodes, 0 /* local_nid */, test_utils::make_system_info(num_devices_per_node, true /* supports d2d copies */),
	                                tm, nullptr /* delegate */, nullptr /* crec */, nullptr /* irec */),
	      mbf(tm, schdlr) //
	{
		tm.register_task_callback([this](const task* tsk) { schdlr.notify_task_created(tsk); });
	}

	scheduler_benchmark_context(const scheduler_benchmark_context&) = delete;
	scheduler_benchmark_context(scheduler_benchmark_context&&) = delete;
	scheduler_benchmark_context& operator=(const scheduler_benchmark_context&) = delete;
	scheduler_benchmark_context& operator=(scheduler_benchmark_context&&) = delete;

	~scheduler_benchmark_context() {
		const auto tid = tm.generate_epoch_task(celerity::detail::epoch_action::shutdown);
		// There is no executor thread and notifications are processed in-order, so we can immediately notify the scheduler about shutdown-epoch completion
		schdlr.notify_epoch_reached(tid);
	}

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		tm.submit_command_group([=](handler& cgh) {
			cgf(cgh);
			cgh.host_task(global_range, [](partition<KernelDims>) {});
		});
	}
};

template <typename BaseBenchmarkContext>
struct submission_throttle_benchmark_context : public BaseBenchmarkContext {
	const std::chrono::steady_clock::duration delay_per_submission;
	std::chrono::steady_clock::time_point last_submission{};

	template <typename... BaseCtorParams>
	explicit submission_throttle_benchmark_context(std::chrono::steady_clock::duration delay_per_submission, BaseCtorParams&&... args)
	    : BaseBenchmarkContext{std::forward<BaseCtorParams>(args)...}, delay_per_submission{delay_per_submission} {}

	template <int KernelDims, typename CGF>
	void create_task(range<KernelDims> global_range, CGF cgf) {
		// "busy sleep" because system timer resolution is not high enough to get down to 10 us intervals
		while(std::chrono::steady_clock::now() - last_submission < delay_per_submission)
			;

		BaseBenchmarkContext::create_task(global_range, cgf);
		last_submission = std::chrono::steady_clock::now();
	}
};


// The generate_* methods are [[noinline]] to make them visible in a profiler.

// Artificial: large set of disconnected tasks, does not generate horizons
template <typename BenchmarkContext>
[[gnu::noinline]] BenchmarkContext&& generate_soup_graph(BenchmarkContext&& ctx, const size_t num_tasks) {
	test_utils::mock_buffer<2> buf = ctx.mbf.create_buffer(range<2>{ctx.num_nodes, num_tasks}, true /* host_initialized */);
	for(size_t t = 0; t < num_tasks; ++t) {
		ctx.create_task(range<1>{ctx.num_nodes}, [&](handler& cgh) {
			buf.get_access<access_mode::read_write>(cgh, [=](chunk<1> ck) { return subrange<2>{{ck.offset[0], t}, {ck.range[0], 1}}; });
		});
	}

	return std::forward<BenchmarkContext>(ctx);
}

// Artificial: Linear chain of dependent tasks, with all-to-all communication
template <typename BenchmarkContext>
[[gnu::noinline]] BenchmarkContext&& generate_chain_graph(BenchmarkContext&& ctx, const size_t num_tasks) {
	const range<2> global_range{ctx.num_nodes, ctx.num_nodes};
	test_utils::mock_buffer<2> buf = ctx.mbf.create_buffer(global_range, true /* host initialized */);
	for(size_t t = 0; t < num_tasks; ++t) {
		ctx.create_task(global_range, [&](handler& cgh) {
			buf.get_access<access_mode::read>(cgh, [=](chunk<2> ck) { return subrange<2>{{ck.offset[1], ck.offset[0]}, {ck.range[1], ck.range[0]}}; });
			buf.get_access<access_mode::write>(cgh, celerity::access::one_to_one{});
		});
	}

	return std::forward<BenchmarkContext>(ctx);
}

// Artificial: Generate expanding or contracting tree of tasks, with gather/scatter communication
enum class tree_topology { expanding, contracting };

template <tree_topology Topology, typename BenchmarkContext>
[[gnu::noinline]] BenchmarkContext&& generate_tree_graph(BenchmarkContext&& ctx, const size_t target_num_tasks) {
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

	return std::forward<BenchmarkContext>(ctx);
}

// graphs identical to the wave_sim example
template <typename BenchmarkContext>
[[gnu::noinline]] BenchmarkContext&& generate_wave_sim_graph(BenchmarkContext&& ctx, const float T) {
	constexpr int N = 512;
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

	return std::forward<BenchmarkContext>(ctx);
}

// Graph of a simple iterative Jacobi solver
template <typename BenchmarkContext>
[[gnu::noinline]] BenchmarkContext&& generate_jacobi_graph(BenchmarkContext&& ctx, const int steps) {
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

	return std::forward<BenchmarkContext>(ctx);
}

template <typename BenchmarkContextFactory>
void run_benchmarks(BenchmarkContextFactory&& make_ctx) {
	BENCHMARK("soup topology") { generate_soup_graph(make_ctx(), 100); };
	BENCHMARK("chain topology") { generate_chain_graph(make_ctx(), 30); };
	BENCHMARK("expanding tree topology") { generate_tree_graph<tree_topology::expanding>(make_ctx(), 30); };
	BENCHMARK("contracting tree topology") { generate_tree_graph<tree_topology::contracting>(make_ctx(), 30); };
	BENCHMARK("wave_sim topology") { generate_wave_sim_graph(make_ctx(), 50); };
	BENCHMARK("jacobi topology") { generate_jacobi_graph(make_ctx(), 50); };
}

TEST_CASE("generating large task graphs", "[benchmark][group:task-graph]") {
	run_benchmarks([] { return task_manager_benchmark_context{}; });
}

TEMPLATE_TEST_CASE_SIG("generating large command graphs for N nodes", "[benchmark][group:command-graph]", ((size_t NumNodes), NumNodes), 1, 4, 16) {
	run_benchmarks([] { return command_graph_generator_benchmark_context{NumNodes}; });
}

TEMPLATE_TEST_CASE_SIG(
    "generating large instruction graphs for N devices", "[benchmark][group:instruction-graph]", ((size_t NumDevices), NumDevices), 1, 4, 16) {
	constexpr static size_t num_nodes = 2;
	run_benchmarks([] { return instruction_graph_generator_benchmark_context(num_nodes, NumDevices); });
}

TEMPLATE_TEST_CASE_SIG("generating large instruction graphs for N devices without d2d copy support", "[benchmark][group:instruction-graph]",
    ((size_t NumDevices), NumDevices), 1, 4, 16) {
	constexpr static size_t num_nodes = 2;
	run_benchmarks([] { return instruction_graph_generator_benchmark_context(num_nodes, NumDevices, false /* supports_d2d_copies */); });
}

TEMPLATE_TEST_CASE_SIG("building command- and instruction graphs in a dedicated scheduler thread for N nodes", "[benchmark][group:scheduler]",
    ((size_t NumNodes), NumNodes), 1, 4) //
{
	constexpr static size_t num_devices = 1;
	SECTION("reference: single-threaded immediate graph generation") {
		run_benchmarks([&] { return command_graph_generator_benchmark_context(NumNodes); });
	}
	SECTION("immediate submission to a scheduler thread") {
		restartable_thread thrd;
		run_benchmarks([&] { return scheduler_benchmark_context(thrd, NumNodes, num_devices); });
	}
	SECTION("reference: throttled single-threaded graph generation at 10 us per task") {
		run_benchmarks([] { return submission_throttle_benchmark_context<command_graph_generator_benchmark_context>(10us, NumNodes); });
	}
	SECTION("throttled submission to a scheduler thread at 10 us per task") {
		restartable_thread thrd;
		run_benchmarks([&] { return submission_throttle_benchmark_context<scheduler_benchmark_context>(10us, thrd, NumNodes, num_devices); });
	}
}

template <typename BenchmarkContextFactory, typename BenchmarkContextConsumer>
void debug_graphs(BenchmarkContextFactory&& make_ctx, BenchmarkContextConsumer&& debug_ctx) {
	debug_ctx(generate_soup_graph(make_ctx(), 10));
	debug_ctx(generate_chain_graph(make_ctx(), 5));
	debug_ctx(generate_tree_graph<tree_topology::expanding>(make_ctx(), 7));
	debug_ctx(generate_tree_graph<tree_topology::contracting>(make_ctx(), 7));
	debug_ctx(generate_wave_sim_graph(make_ctx(), 2));
	debug_ctx(generate_jacobi_graph(make_ctx(), 5));
}

TEST_CASE("printing benchmark task graphs", "[.][debug-graphs][task-graph]") {
	debug_graphs([] { return task_manager_benchmark_context{}; }, [](auto&& ctx) { fmt::print("{}\n\n", detail::print_task_graph(ctx.trec)); });
}

TEST_CASE("printing benchmark command graphs", "[.][debug-graphs][command-graph]") {
	debug_graphs(
	    [] { return command_graph_generator_benchmark_context{2}; }, [](auto&& ctx) { fmt::print("{}\n\n", detail::print_command_graph(0, ctx.crec)); });
}

TEST_CASE("printing benchmark instruction graphs", "[.][debug-graphs][instruction-graph]") {
	debug_graphs([] { return instruction_graph_generator_benchmark_context(2, 2); },
	    [](auto&& ctx) { fmt::print("{}\n\n", detail::print_instruction_graph(ctx.irec, ctx.crec, ctx.trec)); });
}
