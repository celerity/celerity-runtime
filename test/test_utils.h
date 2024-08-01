#pragma once

#include <future>
#include <memory>
#include <string>
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
#include "command.h"
#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "print_graph.h"
#include "range_mapper.h"
#include "region_map.h"
#include "runtime.h"
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

	struct scheduler_testspy {
		static std::thread& get_thread(scheduler& schdlr) { return schdlr.m_thread; }

		template <typename F>
		static auto inspect_thread(scheduler& schdlr, F&& f) {
			using return_t = decltype(f());
			std::promise<return_t> channel;
			schdlr.notify(scheduler::event_test_inspect{[&] { channel.set_value(f()); }});
			return channel.get_future().get();
		}

		static size_t get_command_count(scheduler& schdlr) {
			return inspect_thread(schdlr, [&] { return schdlr.m_cdag->command_count(); });
		}

		static size_t get_live_instruction_count(scheduler& schdlr) {
			return inspect_thread(schdlr, [&] { return schdlr.m_idag->get_live_instruction_count(); });
		}
	};

	struct runtime_testspy {
		static node_id get_local_nid(const runtime& rt) { return rt.m_local_nid; }
		static size_t get_num_nodes(const runtime& rt) { return rt.m_num_nodes; }
		static size_t get_num_local_devices(const runtime& rt) { return rt.m_num_local_devices; }

		static scheduler& get_schdlr(runtime& rt) { return *rt.m_schdlr; }
		static executor& get_exec(runtime& rt) { return *rt.m_exec; }

		static std::string print_task_graph(runtime& rt) {
			return detail::print_task_graph(*rt.m_task_recorder); // task recorder is mutated by task manager (application / test thread)
		}

		static std::string print_command_graph(const node_id local_nid, runtime& rt) {
			return scheduler_testspy::inspect_thread(get_schdlr(rt), [&] { // command_recorder is mutated by scheduler thread
				return detail::print_command_graph(local_nid, *rt.m_command_recorder);
			});
		}

		static std::string print_instruction_graph(runtime& rt) {
			return scheduler_testspy::inspect_thread(get_schdlr(rt), [&] { // instruction recorder is mutated by scheduler thread
				return detail::print_instruction_graph(*rt.m_instruction_recorder, *rt.m_command_recorder, *rt.m_task_recorder);
			});
		}
	};

	struct task_ring_buffer_testspy {
		static void create_task_slot(task_ring_buffer& trb) { trb.m_number_of_deleted_tasks += 1; }
	};

	struct task_manager_testspy {
		static std::optional<task_id> get_current_horizon(task_manager& tm) { return tm.m_current_horizon; }

		static std::optional<task_id> get_latest_horizon_reached(task_manager& tm) { return tm.m_latest_horizon_reached; }

		static int get_num_horizons(task_manager& tm) {
			int horizon_counter = 0;
			for(auto task_ptr : tm.m_task_buffer) {
				if(task_ptr->get_type() == task_type::horizon) { horizon_counter++; }
			}
			return horizon_counter;
		}

		static const region_map<std::optional<task_id>>& get_last_writer(task_manager& tm, const buffer_id bid) { return tm.m_buffers.at(bid).last_writers; }

		static int get_max_pseudo_critical_path_length(task_manager& tm) { return tm.get_max_pseudo_critical_path_length(); }

		static auto get_execution_front(task_manager& tm) { return tm.get_execution_front(); }

		static void create_task_slot(task_manager& tm) { task_ring_buffer_testspy::create_task_slot(tm.m_task_buffer); }
	};

	inline bool has_dependency(const task_manager& tm, task_id dependent, task_id dependency, dependency_kind kind = dependency_kind::true_dep) {
		for(auto dep : tm.get_task(dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency && dep.kind == kind) return true;
		}
		return false;
	}

	inline bool has_any_dependency(const task_manager& tm, task_id dependent, task_id dependency) {
		for(auto dep : tm.get_task(dependent)->get_dependencies()) {
			if(dep.node->get_id() == dependency) return true;
		}
		return false;
	}
} // namespace detail

namespace test_utils {
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
	class dist_cdag_test_context;
	class idag_test_context;

	template <int Dims>
	class mock_buffer {
	  public:
		template <sycl::access::mode Mode, typename Functor>
		void get_access(handler& cgh, Functor rmfn) {
			(void)detail::add_requirement(cgh, m_id, std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, m_size));
		}

		detail::buffer_id get_id() const { return m_id; }

		range<Dims> get_range() const { return m_size; }

	  private:
		friend class mock_buffer_factory;
		friend class dist_cdag_test_context;
		friend class idag_test_context;

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
		friend class dist_cdag_test_context;
		friend class idag_test_context;

		detail::host_object_id m_id;

	  public:
		explicit mock_host_object(detail::host_object_id id) : m_id(id) {}
	};

	class mock_buffer_factory {
	  public:
		explicit mock_buffer_factory() = default;
		explicit mock_buffer_factory(detail::task_manager& tm) : m_task_mngr(&tm) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::distributed_graph_generator& dggen) : m_task_mngr(&tm), m_dggen(&dggen) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::distributed_graph_generator& dggen, detail::instruction_graph_generator& iggen)
		    : m_task_mngr(&tm), m_dggen(&dggen), m_iggen(&iggen) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::abstract_scheduler& schdlr) : m_task_mngr(&tm), m_schdlr(&schdlr) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = m_next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			const auto user_allocation_id =
			    mark_as_host_initialized ? detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++) : detail::null_allocation_id;
			if(m_task_mngr != nullptr) { m_task_mngr->notify_buffer_created(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(m_schdlr != nullptr) { m_schdlr->notify_buffer_created(bid, detail::range_cast<3>(size), sizeof(int), alignof(int), user_allocation_id); }
			if(m_dggen != nullptr) { m_dggen->notify_buffer_created(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(m_iggen != nullptr) { m_iggen->notify_buffer_created(bid, detail::range_cast<3>(size), sizeof(int), alignof(int), user_allocation_id); }
			return buf;
		}

	  private:
		detail::task_manager* m_task_mngr = nullptr;
		detail::abstract_scheduler* m_schdlr = nullptr;
		detail::distributed_graph_generator* m_dggen = nullptr;
		detail::instruction_graph_generator* m_iggen = nullptr;
		detail::buffer_id m_next_buffer_id = 0;
		detail::raw_allocation_id m_next_user_allocation_id = 1;
	};

	class mock_host_object_factory {
	  public:
		explicit mock_host_object_factory() = default;
		explicit mock_host_object_factory(detail::task_manager& tm) : m_task_mngr(&tm) {}
		explicit mock_host_object_factory(detail::task_manager& tm, detail::abstract_scheduler& schdlr) : m_task_mngr(&tm), m_schdlr(&schdlr) {}

		mock_host_object create_host_object(bool owns_instance = true) {
			const detail::host_object_id hoid = m_next_id++;
			if(m_task_mngr != nullptr) { m_task_mngr->notify_host_object_created(hoid); }
			if(m_schdlr != nullptr) { m_schdlr->notify_host_object_created(hoid, owns_instance); }
			return mock_host_object(hoid);
		}

	  private:
		detail::task_manager* m_task_mngr = nullptr;
		detail::abstract_scheduler* m_schdlr = nullptr;
		detail::host_object_id m_next_id = 0;
	};

	template <typename KernelName = detail::unnamed_kernel, typename CGF, int KernelDims = 2>
	detail::task_id add_compute_task(detail::task_manager& tm, CGF cgf, range<KernelDims> global_size = {1, 1}, id<KernelDims> global_offset = {}) {
		// Here and below: Using these functions will cause false-positive CGF diagnostic errors, b/c we are not capturing any accessors.
		// TODO: For many test cases using these functions it may actually be preferable to circumvent the whole handler mechanism entirely.
		detail::cgf_diagnostics::teardown();
		return tm.submit_command_group([&, gs = global_size, go = global_offset](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(gs, go, [](id<KernelDims>) {});
		});
		detail::cgf_diagnostics::make_available();
	}

	template <typename KernelName = detail::unnamed_kernel, typename CGF, int KernelDims = 2>
	detail::task_id add_nd_range_compute_task(detail::task_manager& tm, CGF cgf, celerity::nd_range<KernelDims> execution_range = {{1, 1}, {1, 1}}) {
		// (See above).
		detail::cgf_diagnostics::teardown();
		return tm.submit_command_group([&, er = execution_range](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(er, [](nd_item<KernelDims>) {});
		});
		detail::cgf_diagnostics::make_available();
	}

	template <typename Spec, typename CGF>
	detail::task_id add_host_task(detail::task_manager& tm, Spec spec, CGF cgf) {
		// (See above).
		detail::cgf_diagnostics::teardown();
		return tm.submit_command_group([&](handler& cgh) {
			cgf(cgh);
			cgh.host_task(spec, [](auto...) {});
		});
		detail::cgf_diagnostics::make_available();
	}

	inline detail::task_id add_fence_task(detail::task_manager& tm, mock_host_object ho) {
		detail::side_effect_map side_effects;
		side_effects.add_side_effect(ho.get_id(), experimental::side_effect_order::sequential);
		return tm.generate_fence_task({}, std::move(side_effects), nullptr);
	}

	template <int Dims>
	inline detail::task_id add_fence_task(detail::task_manager& tm, mock_buffer<Dims> buf, subrange<Dims> sr) {
		detail::buffer_access_map access_map;
		access_map.add_access(buf.get_id(),
		    std::make_unique<detail::range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), access_mode::read, buf.get_range()));
		return tm.generate_fence_task(std::move(access_map), {}, nullptr);
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
		detail::task_recorder trec;
		detail::task_manager tm;
		mock_buffer_factory mbf;
		mock_host_object_factory mhof;
		mock_reduction_factory mrf;

		explicit task_test_context(const detail::task_manager::policy_set& policy = {}) : tm(1, &trec, policy), mbf(tm), mhof(tm) {}
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

template <int Dims>
static celerity::access::neighborhood<Dims> make_neighborhood(const size_t border) {
	if constexpr(Dims == 1) {
		return celerity::access::neighborhood<1>(border);
	} else if constexpr(Dims == 2) {
		return celerity::access::neighborhood<2>(border, border);
	} else if constexpr(Dims == 3) {
		return celerity::access::neighborhood<3>(border, border, border);
	}
}

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
CELERITY_TEST_UTILS_IMPLEMENT_CATCH_STRING_MAKER(celerity::detail::allocation_with_offset)
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

} // namespace Catch
