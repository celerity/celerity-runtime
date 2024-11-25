#pragma once

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <catch2/benchmark/catch_optimizer.hpp> // for keep_memory()
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <celerity.h>

#include "async_event.h"
#include "backend/sycl_backend.h"
#include "command_graph.h"
#include "command_graph_generator.h"
#include "named_threads.h"
#include "print_graph.h"
#include "range_mapper.h"
#include "region_map.h"
#include "runtime.h"
#include "runtime_impl.h"
#include "scheduler.h"
#include "system_info.h"
#include "task_manager.h"
#include "types.h"

// To avoid having to come up with tons of unique kernel names, we simply use the CPP counter.
// This is non-standard but widely supported.
#define _UKN_CONCAT2(x, y) x##_##y
#define _UKN_CONCAT(x, y) _UKN_CONCAT2(x, y)
#define UKN(name) _UKN_CONCAT(name, __COUNTER__)

/**
 * REQUIRE_LOOP is a utility macro for performing Catch2 REQUIRE assertions inside of loops.
 * The advantage over using a regular REQUIRE is that the number of reported assertions is much lower,
 * as only the first iteration is actually passed on to Catch2 (useful when showing successful assertions with `-s`).
 * If an expression result is false, it will also be forwarded to Catch2.
 *
 * NOTE: Since the checked expression will be evaluated twice, it must be idempotent!
 */
#define REQUIRE_LOOP(...) CELERITY_DETAIL_REQUIRE_LOOP(__VA_ARGS__)

namespace celerity {
namespace detail {

	const std::unordered_map<std::string, std::string> print_graphs_env_setting{{"CELERITY_PRINT_GRAPHS", "1"}};

	struct graph_testspy {
		template <GraphNode Node, typename Predicate>
		static size_t count_nodes_if(const graph<Node>& dag, const Predicate& p) {
			size_t count = 0;
			for(const auto& epoch : dag.m_epochs) {
				for(const auto& node : epoch.nodes) {
					if(p(*node)) count++;
				}
			}
			return count;
		}

		template <GraphNode Node, typename Predicate>
		static const Node* find_node_if(const graph<Node>& dag, const Predicate& p) {
			for(const auto& epoch : dag.m_epochs) {
				for(const auto& node : epoch.nodes) {
					if(p(*node)) { return node.get(); }
				}
			}
			return nullptr;
		}

		template <GraphNode Node>
		static size_t get_live_node_count(const graph<Node>& dag) {
			size_t count = 0;
			for(const auto& epoch : dag.m_epochs) {
				count += epoch.nodes.size();
			}
			return count;
		}
	};

	struct scheduler_testspy {
		using test_state = scheduler_detail::test_state;

		class threadless_scheduler : public scheduler {
		  public:
			threadless_scheduler(const auto&... params) : scheduler(test_threadless_tag(), params...) {}
			void scheduling_loop() { test_scheduling_loop(); }
		};

		template <typename F>
		static auto inspect_thread(scheduler& schdlr, F&& f) {
			using return_t = std::invoke_result_t<F, const test_state&>;
			std::promise<return_t> channel;
			schdlr.test_inspect([&](const scheduler_detail::test_state& state) {
				if constexpr(std::is_void_v<return_t>) {
					f(state), channel.set_value();
				} else {
					channel.set_value(f(state));
				}
			});
			return channel.get_future().get();
		}

		static size_t get_live_command_count(scheduler& schdlr) {
			return inspect_thread(schdlr, [](const test_state& state) { return graph_testspy::get_live_node_count(*state.cdag); });
		}

		static size_t get_live_instruction_count(scheduler& schdlr) {
			return inspect_thread(schdlr, [](const test_state& state) { return graph_testspy::get_live_node_count(*state.idag); });
		}

		static experimental::lookahead get_lookahead(scheduler& schdlr) {
			return inspect_thread(schdlr, [](const test_state& state) { return state.lookahead; });
		}
	};

	struct runtime_testspy {
		static const runtime_impl& impl(const runtime& rt) { return dynamic_cast<const runtime_impl&>(rt); }
		static runtime_impl& impl(runtime& rt) { return dynamic_cast<runtime_impl&>(rt); }

		static node_id get_local_nid(const runtime& rt) { return impl(rt).m_local_nid; }
		static size_t get_num_nodes(const runtime& rt) { return impl(rt).m_num_nodes; }
		static size_t get_num_local_devices(const runtime& rt) { return impl(rt).m_num_local_devices; }

		static task_graph& get_task_graph(runtime& rt) { return impl(rt).m_tdag; }
		static task_manager& get_task_manager(runtime& rt) { return *impl(rt).m_task_mngr; }
		static scheduler& get_schdlr(runtime& rt) { return *impl(rt).m_schdlr; }
		static executor& get_exec(runtime& rt) { return *impl(rt).m_exec; }

		static task_id get_latest_epoch_reached(const runtime& rt) { return impl(rt).m_latest_epoch_reached.load(std::memory_order_relaxed); }

		static std::string print_task_graph(runtime& rt) {
			return detail::print_task_graph(*impl(rt).m_task_recorder); // task recorder is mutated by task manager (application / test thread)
		}

		static std::string print_command_graph(const node_id local_nid, runtime& rt) {
			// command_recorder is mutated by scheduler thread
			return scheduler_testspy::inspect_thread(
			    get_schdlr(rt), [&](const auto&) { return detail::print_command_graph(local_nid, *impl(rt).m_command_recorder); });
		}

		static std::string print_instruction_graph(runtime& rt) {
			// instruction recorder is mutated by scheduler thread
			return scheduler_testspy::inspect_thread(get_schdlr(rt), [&](const auto&) {
				return detail::print_instruction_graph(*impl(rt).m_instruction_recorder, *impl(rt).m_command_recorder, *impl(rt).m_task_recorder);
			});
		}
	};

	struct task_manager_testspy {
		inline static constexpr task_id initial_epoch_task = task_manager::initial_epoch_task;

		static const task* get_epoch_for_new_tasks(const task_manager& tm) { return tm.m_epoch_for_new_tasks; }

		static const task* get_current_horizon(const task_manager& tm) { return tm.m_current_horizon; }

		static const region_map<task*>& get_last_writer(const task_manager& tm, const buffer_id bid) { return tm.m_buffers.at(bid).last_writers; }

		static int get_max_pseudo_critical_path_length(const task_manager& tm) { return tm.m_max_pseudo_critical_path_length; }

		static const std::unordered_set<task*>& get_execution_front(const task_manager& tm) { return tm.m_execution_front; }
	};

	struct range_mapper_testspy {
		template <int Dims>
		static bool neighborhood_equals(const celerity::access::neighborhood<Dims>& lhs, const celerity::access::neighborhood<Dims>& rhs) {
			return lhs.m_extent == rhs.m_extent && lhs.m_shape == rhs.m_shape;
		}
	};

} // namespace detail

namespace test_utils {

	// Pin the benchmark threads (even in absence of a runtime) for more consistent results
	struct benchmark_thread_pinner {
		benchmark_thread_pinner() {
			const detail::thread_pinning::runtime_configuration cfg{
			    .enabled = true,
			    .use_backend_device_submission_threads = false,
			};
			m_thread_pinner.emplace(cfg);
			name_and_pin_and_order_this_thread(detail::named_threads::thread_type::application);
		}

		std::optional<detail::thread_pinning::thread_pinner> m_thread_pinner;
	};

	inline const detail::task* find_task(const detail::task_graph& tdag, const detail::task_id tid) {
		return detail::graph_testspy::find_node_if(tdag, [tid](const detail::task& tsk) { return tsk.get_id() == tid; });
	}

	inline bool has_task(const detail::task_graph& tdag, const detail::task_id tid) { return find_task(tdag, tid) != nullptr; }

	inline const detail::task* get_task(const detail::task_graph& tdag, const detail::task_id tid) {
		const auto tsk = find_task(tdag, tid);
		REQUIRE(tsk != nullptr);
		return tsk;
	}

	inline size_t get_num_live_horizons(const detail::task_graph& tdag) {
		return detail::graph_testspy::count_nodes_if(tdag, [](const detail::task& tsk) { return tsk.get_type() == detail::task_type::horizon; });
	}

	inline bool has_dependency(const detail::task_graph& tdag, detail::task_id dependent, detail::task_id dependency,
	    detail::dependency_kind kind = detail::dependency_kind::true_dep) {
		for(auto dep : get_task(tdag, dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency && dep.kind == kind) return true;
		}
		return false;
	}

	inline bool has_any_dependency(const detail::task_graph& tdag, detail::task_id dependent, detail::task_id dependency) {
		for(auto dep : get_task(tdag, dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency) return true;
		}
		return false;
	}

	class require_loop_assertion_registry {
	  public:
		static require_loop_assertion_registry& get_instance() {
			if(instance == nullptr) { instance = std::make_unique<require_loop_assertion_registry>(); }
			return *instance;
		}

		void reset() { m_logged_lines.clear(); }

		bool should_log(std::string line_info) {
			auto [_, is_new] = m_logged_lines.emplace(std::move(line_info));
			return is_new;
		}

	  private:
		inline static std::unique_ptr<require_loop_assertion_registry> instance;
		std::unordered_set<std::string> m_logged_lines{};
	};

#define CELERITY_DETAIL_REQUIRE_LOOP(...)                                                                                                                      \
	if(celerity::test_utils::require_loop_assertion_registry::get_instance().should_log(std::string(__FILE__) + std::to_string(__LINE__))) {                   \
		REQUIRE(__VA_ARGS__);                                                                                                                                  \
	} else if(!(__VA_ARGS__)) {                                                                                                                                \
		REQUIRE(__VA_ARGS__);                                                                                                                                  \
	}

	/// By default, tests fail if their log contains a warning, error or critical message. This function allows tests to pass when higher-severity messages
	/// are expected. The property is re-set at the beginning of each test-case run (even when it is re-entered due to a generator or section).
	void allow_max_log_level(detail::log_level level);

	/// Like allow_max_log_level(), but only applies to messages that match a regex. This is used in test fixtures to allow common system-dependent messages.
	void allow_higher_level_log_messages(detail::log_level level, const std::string& text_regex);

	/// Returns whether the log of the current test so far contains a message that exactly equals the given log level and message. Time stamps and the log level
	/// are not part of the text, but any active log_context is.
	bool log_contains_exact(detail::log_level level, const std::string& text);

	/// Returns whether the log of the current test so far contains a message with exactly the given log level and a message that contains `substring`.
	bool log_contains_substring(detail::log_level level, const std::string& substring);

	template <int Dims, typename F>
	void for_each_in_range(range<Dims> range, id<Dims> offset, F&& f) {
		const auto range3 = detail::range_cast<3>(range);
		id<3> index;
		for(index[0] = 0; index[0] < range3[0]; ++index[0]) {
			for(index[1] = 0; index[1] < range3[1]; ++index[1]) {
				for(index[2] = 0; index[2] < range3[2]; ++index[2]) {
					f(offset + detail::id_cast<Dims>(index));
				}
			}
		}
	}

	template <int Dims, typename F>
	void for_each_in_range(range<Dims> range, F&& f) {
		for_each_in_range(range, {}, f);
	}

	class mock_buffer_factory;
	class mock_host_object_factory;
	class cdag_test_context;
	class idag_test_context;

	template <int Dims>
	class mock_buffer {
	  public:
		template <sycl::access::mode Mode, typename Functor>
		void get_access(handler& cgh, Functor rmfn) {
			(void)detail::add_requirement(cgh, m_id, Mode, std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, m_size));
		}

		detail::buffer_id get_id() const { return m_id; }

		range<Dims> get_range() const { return m_size; }

	  private:
		friend class mock_buffer_factory;
		friend class cdag_test_context;
		friend class idag_test_context;
		friend class scheduler_test_context;

		detail::buffer_id m_id;
		range<Dims> m_size;

		mock_buffer(detail::buffer_id id, range<Dims> size) : m_id(id), m_size(size) {}
	};

	class mock_host_object {
	  public:
		void add_side_effect(handler& cgh, const experimental::side_effect_order order) { (void)detail::add_requirement(cgh, m_id, order, true); }

		detail::host_object_id get_id() const { return m_id; }

	  private:
		friend class mock_host_object_factory;
		friend class cdag_test_context;
		friend class idag_test_context;
		friend class scheduler_test_context;

		detail::host_object_id m_id;

	  public:
		explicit mock_host_object(detail::host_object_id id) : m_id(id) {}
	};

	class mock_buffer_factory {
	  public:
		explicit mock_buffer_factory() = default;
		explicit mock_buffer_factory(detail::task_manager& tm) : m_task_mngr(&tm) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::command_graph_generator& cggen) : m_task_mngr(&tm), m_cggen(&cggen) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::command_graph_generator& cggen, detail::instruction_graph_generator& iggen)
		    : m_task_mngr(&tm), m_cggen(&cggen), m_iggen(&iggen) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::scheduler& schdlr) : m_task_mngr(&tm), m_schdlr(&schdlr) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = m_next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			const auto user_allocation_id =
			    mark_as_host_initialized ? detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++) : detail::null_allocation_id;
			if(m_task_mngr != nullptr) { m_task_mngr->notify_buffer_created(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(m_schdlr != nullptr) { m_schdlr->notify_buffer_created(bid, detail::range_cast<3>(size), sizeof(int), alignof(int), user_allocation_id); }
			if(m_cggen != nullptr) { m_cggen->notify_buffer_created(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(m_iggen != nullptr) { m_iggen->notify_buffer_created(bid, detail::range_cast<3>(size), sizeof(int), alignof(int), user_allocation_id); }
			return buf;
		}

	  private:
		detail::task_manager* m_task_mngr = nullptr;
		detail::scheduler* m_schdlr = nullptr;
		detail::command_graph_generator* m_cggen = nullptr;
		detail::instruction_graph_generator* m_iggen = nullptr;
		detail::buffer_id m_next_buffer_id = 0;
		detail::raw_allocation_id m_next_user_allocation_id = 1;
	};

	class mock_host_object_factory {
	  public:
		explicit mock_host_object_factory() = default;
		explicit mock_host_object_factory(detail::task_manager& tm) : m_task_mngr(&tm) {}
		explicit mock_host_object_factory(detail::task_manager& tm, detail::scheduler& schdlr) : m_task_mngr(&tm), m_schdlr(&schdlr) {}

		mock_host_object create_host_object(bool owns_instance = true) {
			const detail::host_object_id hoid = m_next_id++;
			if(m_task_mngr != nullptr) { m_task_mngr->notify_host_object_created(hoid); }
			if(m_schdlr != nullptr) { m_schdlr->notify_host_object_created(hoid, owns_instance); }
			return mock_host_object(hoid);
		}

	  private:
		detail::task_manager* m_task_mngr = nullptr;
		detail::scheduler* m_schdlr = nullptr;
		detail::host_object_id m_next_id = 0;
	};

	template <typename KernelName = detail::unnamed_kernel, typename CGF, int KernelDims = 2>
	detail::task_id add_compute_task(detail::task_manager& tm, CGF cgf, range<KernelDims> global_size = {1, 1}, id<KernelDims> global_offset = {}) {
		// Here and below: Using these functions will cause false-positive CGF diagnostic errors, b/c we are not capturing any accessors.
		// TODO: For many test cases using these functions it may actually be preferable to circumvent the whole handler mechanism entirely.
		detail::cgf_diagnostics::teardown();
		auto cg = detail::invoke_command_group_function([&, gs = global_size, go = global_offset](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(gs, go, [](id<KernelDims>) {});
		});
		return tm.generate_command_group_task(std::move(cg));
		detail::cgf_diagnostics::make_available();
	}

	template <typename KernelName = detail::unnamed_kernel, typename CGF, int KernelDims = 2>
	detail::task_id add_nd_range_compute_task(detail::task_manager& tm, CGF cgf, celerity::nd_range<KernelDims> execution_range = {{1, 1}, {1, 1}}) {
		// (See above).
		detail::cgf_diagnostics::teardown();
		auto cg = detail::invoke_command_group_function([&, er = execution_range](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(er, [](nd_item<KernelDims>) {});
		});
		return tm.generate_command_group_task(std::move(cg));
		detail::cgf_diagnostics::make_available();
	}

	template <typename Spec, typename CGF>
	detail::task_id add_host_task(detail::task_manager& tm, Spec spec, CGF cgf) {
		// (See above).
		detail::cgf_diagnostics::teardown();
		auto cg = detail::invoke_command_group_function([&](handler& cgh) {
			cgf(cgh);
			cgh.host_task(spec, [](auto...) {});
		});
		return tm.generate_command_group_task(std::move(cg));
		detail::cgf_diagnostics::make_available();
	}

	inline detail::task_id add_fence_task(detail::task_manager& tm, mock_host_object ho) {
		const detail::host_object_effect effect{ho.get_id(), experimental::side_effect_order::sequential};
		return tm.generate_fence_task(effect, nullptr);
	}

	template <int Dims>
	inline detail::task_id add_fence_task(detail::task_manager& tm, mock_buffer<Dims> buf, subrange<Dims> sr) {
		detail::buffer_access access{buf.get_id(), access_mode::read,
		    std::make_unique<detail::range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), buf.get_range())};
		return tm.generate_fence_task(std::move(access), nullptr);
	}

	template <int Dims>
	inline detail::task_id add_fence_task(detail::task_manager& tm, mock_buffer<Dims> buf) {
		return add_fence_task(tm, buf, {{}, buf.get_range()});
	}

	class mock_reduction_factory {
	  public:
		detail::reduction_info create_reduction(const detail::buffer_id bid, const bool include_current_buffer_value) {
			return detail::reduction_info{m_next_id++, bid, include_current_buffer_value};
		}

	  private:
		detail::reduction_id m_next_id = 1;
	};

	template <int Dims>
	void add_reduction(handler& cgh, mock_reduction_factory& mrf, const mock_buffer<Dims>& vars, bool include_current_buffer_value) {
		detail::add_reduction(cgh, mrf.create_reduction(vars.get_id(), include_current_buffer_value));
	}

	detail::system_info make_system_info(const size_t num_devices, const bool supports_d2d_copies);

	// This fixture (or a subclass) must be used by all tests that transitively use MPI.
	class mpi_fixture {
	  public:
		mpi_fixture() { detail::runtime::test_require_mpi(); }
		mpi_fixture(const mpi_fixture&) = delete;
		mpi_fixture(mpi_fixture&&) = delete;
		mpi_fixture& operator=(const mpi_fixture&) = delete;
		mpi_fixture& operator=(mpi_fixture&&) = delete;
		~mpi_fixture() = default;
	};

	// Allow "falling back to generic backend" warnings to appear in log
	void allow_backend_fallback_warnings();

	// Allow "fence in dry run" warning to appear in log
	void allow_dry_run_executor_warnings();

	// This fixture (or a subclass) must be used by all tests that transitively instantiate the runtime.
	class runtime_fixture : public mpi_fixture {
	  public:
		runtime_fixture();
		runtime_fixture(const runtime_fixture&) = delete;
		runtime_fixture(runtime_fixture&&) = delete;
		runtime_fixture& operator=(const runtime_fixture&) = delete;
		runtime_fixture& operator=(runtime_fixture&&) = delete;
		~runtime_fixture();
	};

	template <int>
	struct runtime_fixture_dims : test_utils::runtime_fixture {};

	class sycl_queue_fixture {
	  public:
		sycl_queue_fixture() {
			try {
				m_queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
			} catch(sycl::exception&) { SKIP("no GPUs available"); }
		}

		sycl::queue& get_sycl_queue() { return m_queue; }

		// Convenience function for submitting parallel_for with global offset without having to create a CGF
		template <int Dims, typename KernelFn>
		void parallel_for(const range<Dims>& global_range, const id<Dims>& global_offset, KernelFn fn) {
			m_queue.submit([=](sycl::handler& cgh) {
				cgh.parallel_for(sycl::range<Dims>{global_range}, detail::bind_simple_kernel(fn, global_range, global_offset, global_offset));
			});
			m_queue.wait_and_throw();
		}

	  private:
		sycl::queue m_queue;
	};

	// Printing of graphs can be enabled using the "--print-graphs" command line flag
	extern bool g_print_graphs;

	std::string make_test_graph_title(const std::string& type);
	std::string make_test_graph_title(const std::string& type, size_t num_nodes, detail::node_id local_nid);
	std::string make_test_graph_title(const std::string& type, size_t num_nodes, detail::node_id local_nid, size_t num_devices_per_node);

	struct task_test_context {
		detail::task_graph tdag;
		detail::task_recorder trec;
		detail::task_manager tm;
		mock_buffer_factory mbf;
		mock_host_object_factory mhof;
		mock_reduction_factory mrf;
		detail::task_id initial_epoch_task;

		explicit task_test_context(const detail::task_manager::policy_set& policy = {})
		    : tm(1, tdag, &trec, nullptr /* delegate */, policy), mbf(tm), mhof(tm), initial_epoch_task(tm.generate_epoch_task(detail::epoch_action::init)) {}

		task_test_context(const task_test_context&) = delete;
		task_test_context(task_test_context&&) = delete;
		task_test_context& operator=(const task_test_context&) = delete;
		task_test_context& operator=(task_test_context&&) = delete;
		~task_test_context();
	};

	// explicitly invoke a copy constructor without repeating the type
	template <typename T>
	T copy(const T& v) {
		return v;
	}

	template <typename T>
	void black_hole(T&& v) {
		Catch::Benchmark::keep_memory(&v);
	}

	// truncate_*(): unchecked versions of *_cast() with signatures friendly to parameter type inference

	template <int Dims>
	constexpr range<Dims> truncate_range(const range<3>& r3) {
		static_assert(Dims <= 3);
		range<Dims> r = detail::zeros;
		for(int d = 0; d < Dims; ++d) {
			r[d] = r3[d];
		}
		return r;
	}

	template <int Dims>
	constexpr id<Dims> truncate_id(const id<3>& i3) {
		static_assert(Dims <= 3);
		id<Dims> i;
		for(int d = 0; d < Dims; ++d) {
			i[d] = i3[d];
		}
		return i;
	}

	template <int Dims>
	subrange<Dims> truncate_subrange(const subrange<3>& sr3) {
		return subrange<Dims>(truncate_id<Dims>(sr3.offset), truncate_range<Dims>(sr3.range));
	}

	template <int Dims>
	chunk<Dims> truncate_chunk(const chunk<3>& ck3) {
		return chunk<Dims>(truncate_id<Dims>(ck3.offset), truncate_range<Dims>(ck3.range), truncate_range<Dims>(ck3.global_size));
	}

	template <int Dims>
	detail::box<Dims> truncate_box(const detail::box<3>& b3) {
		return detail::box<Dims>(truncate_id<Dims>(b3.get_min()), truncate_id<Dims>(b3.get_max()));
	}

	template <typename T>
	class vector_generator final : public Catch::Generators::IGenerator<T> {
	  public:
		explicit vector_generator(std::vector<T>&& values) : m_values(std::move(values)) {}
		const T& get() const override { return m_values[m_idx]; }
		bool next() override { return ++m_idx < m_values.size(); }

	  private:
		std::vector<T> m_values;
		size_t m_idx = 0;
	};

	template <typename T>
	Catch::Generators::GeneratorWrapper<T> from_vector(std::vector<T> values) {
		return Catch::Generators::GeneratorWrapper<T>(Catch::Detail::make_unique<vector_generator<T>>(std::move(values)));
	}

	inline void* await(const celerity::detail::async_event& evt) {
		while(!evt.is_complete()) {}
		return evt.get_result();
	}

} // namespace test_utils
} // namespace celerity

namespace celerity::test_utils::access {

struct reverse_one_to_one {
	template <int Dims>
	subrange<Dims> operator()(chunk<Dims> ck) const {
		subrange<Dims> sr;
		for(int d = 0; d < Dims; ++d) {
			sr.offset[d] = ck.global_size[d] - ck.range[d] - ck.offset[d];
			sr.range[d] = ck.range[d];
		}
		return sr;
	}
};

} // namespace celerity::test_utils::access

namespace Catch {

template <typename A, typename B>
struct StringMaker<std::pair<A, B>> {
	static std::string convert(const std::pair<A, B>& v) {
		return fmt::format("({}, {})", Catch::Detail::stringify(v.first), Catch::Detail::stringify(v.second));
	}
};

template <typename T>
struct StringMaker<std::optional<T>> {
	static std::string convert(const std::optional<T>& v) { return v.has_value() ? Catch::Detail::stringify(*v) : "null"; }
};

#define CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER(Type)                                                                                                 \
	template <>                                                                                                                                                \
	struct StringMaker<Type> {                                                                                                                                 \
		static std::string convert(const Type& v) { return fmt::format("{}", v); }                                                                             \
	};

CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER(celerity::detail::allocation_id)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER(celerity::detail::transfer_id)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER(celerity::detail::sycl_backend_type)

#define CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(Type)                                                                                        \
	template <int Dims>                                                                                                                                        \
	struct StringMaker<Type<Dims>> {                                                                                                                           \
		static std::string convert(const Type<Dims>& v) { return fmt::format("{}", v); }                                                                       \
	};

CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(celerity::id)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(celerity::range)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(celerity::subrange)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(celerity::chunk)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(celerity::detail::box)
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER_FOR_DIMS(celerity::detail::region)

template <>
struct StringMaker<sycl::device> {
	static std::string convert(const sycl::device& d) {
		return fmt::format("sycl::device(vendor_id={}, name=\"{}\")", d.get_info<sycl::info::device::vendor_id>(), d.get_info<sycl::info::device::name>());
	}
};

template <>
struct StringMaker<sycl::platform> {
	static std::string convert(const sycl::platform& d) {
		return fmt::format("sycl::platform(vendor=\"{}\", name=\"{}\")", d.get_info<sycl::info::platform::vendor>(), d.get_info<sycl::info::platform::name>());
	}
};

template <>
struct StringMaker<celerity::detail::linearized_layout> {
	static std::string convert(const celerity::detail::linearized_layout& v) { return fmt::format("linearized_layout({})", v.offset_bytes); }
};

template <>
struct StringMaker<celerity::detail::strided_layout> {
	static std::string convert(const celerity::detail::strided_layout& v) { return fmt::format("strided_layout({})", v.allocation); }
};

template <>
struct StringMaker<celerity::detail::region_layout> {
	static std::string convert(const celerity::detail::region_layout& v) {
		return matchbox::match(v, [](const auto& a) { return StringMaker<std::decay_t<decltype(a)>>::convert(a); });
	}
};

} // namespace Catch
