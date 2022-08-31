#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <celerity.h>

#include "command.h"
#include "command_graph.h"
#include "device_queue.h"
#include "graph_generator.h"
#include "graph_serializer.h"
#include "range_mapper.h"
#include "runtime.h"
#include "scheduler.h"
#include "task_manager.h"
#include "transformers/naive_split.h"
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

	struct task_ring_buffer_testspy {
		static void create_task_slot(task_ring_buffer& trb) { trb.m_number_of_deleted_tasks += 1; }
	};

	struct task_manager_testspy {
		static std::optional<task_id> get_current_horizon(task_manager& tm) { return tm.m_current_horizon; }

		static int get_num_horizons(task_manager& tm) {
			int horizon_counter = 0;
			for(auto task_ptr : tm.m_task_buffer) {
				if(task_ptr->get_type() == task_type::horizon) { horizon_counter++; }
			}
			return horizon_counter;
		}

		static region_map<std::optional<task_id>> get_last_writer(task_manager& tm, const buffer_id bid) { return tm.m_buffers_last_writers.at(bid); }

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

	template <int Dims, typename F>
	void for_each_in_range(sycl::range<Dims> range, sycl::id<Dims> offset, F&& f) {
		const auto range3 = detail::range_cast<3>(range);
		sycl::id<3> index;
		for(index[0] = 0; index[0] < range3[0]; ++index[0]) {
			for(index[1] = 0; index[1] < range3[1]; ++index[1]) {
				for(index[2] = 0; index[2] < range3[2]; ++index[2]) {
					f(offset + detail::id_cast<Dims>(index));
				}
			}
		}
	}

	template <int Dims, typename F>
	void for_each_in_range(sycl::range<Dims> range, F&& f) {
		for_each_in_range(range, {}, f);
	}

	class mock_buffer_factory;
	class mock_host_object_factory;

	template <int Dims>
	class mock_buffer {
	  public:
		template <cl::sycl::access::mode Mode, typename Functor>
		void get_access(handler& cgh, Functor rmfn) {
			if(detail::is_prepass_handler(cgh)) {
				auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh); // No live pass in tests
				prepass_cgh.add_requirement(m_id, std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, m_size));
			}
		}

		detail::buffer_id get_id() const { return m_id; }

		range<Dims> get_range() const { return m_size; }

	  private:
		friend class mock_buffer_factory;

		detail::buffer_id m_id;
		cl::sycl::range<Dims> m_size;

		mock_buffer(detail::buffer_id id, cl::sycl::range<Dims> size) : m_id(id), m_size(size) {}
	};

	class mock_host_object {
	  public:
		void add_side_effect(handler& cgh, const experimental::side_effect_order order) {
			if(detail::is_prepass_handler(cgh)) {
				auto& prepass_cgh = static_cast<detail::prepass_handler&>(cgh);
				prepass_cgh.add_requirement(m_id, order);
			}
		}

	  private:
		friend class mock_host_object_factory;

		detail::host_object_id m_id;

	  public:
		explicit mock_host_object(detail::host_object_id id) : m_id(id) {}
	};

	class cdag_inspector {
	  public:
		auto get_cb() {
			return [this](detail::node_id nid, detail::unique_frame_ptr<detail::command_frame> frame) {
#ifndef NDEBUG
				for(const auto dcid : frame->iter_dependencies()) {
					// Sanity check: All dependencies must have already been flushed
					assert(m_commands.count(dcid) == 1);
				}
#endif

				const detail::command_id cid = frame->pkg.cid;
				m_commands[cid] = {nid, frame->pkg, std::vector(frame->iter_dependencies().begin(), frame->iter_dependencies().end())};
				if(const auto tid = frame->pkg.get_tid()) { m_by_task[*tid].insert(cid); }
				m_by_node[nid].insert(cid);
			};
		}

		std::set<detail::command_id> get_commands(
		    std::optional<detail::task_id> tid, std::optional<detail::node_id> nid, std::optional<detail::command_type> cmd) const {
			// Sanity check: Not all commands have an associated task id
			assert(tid == std::nullopt
			       || (cmd == std::nullopt || cmd == detail::command_type::execution || cmd == detail::command_type::horizon
			           || cmd == detail::command_type::epoch));

			std::set<detail::command_id> result;
			std::transform(m_commands.cbegin(), m_commands.cend(), std::inserter(result, result.begin()), [](auto p) { return p.first; });

			if(tid != std::nullopt) {
				auto& task_set = m_by_task.at(*tid);
				std::set<detail::command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), task_set.cbegin(), task_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(nid != std::nullopt) {
				auto& node_set = m_by_node.at(*nid);
				std::set<detail::command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), node_set.cbegin(), node_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(cmd != std::nullopt) {
				std::set<detail::command_id> new_result;
				std::copy_if(result.cbegin(), result.cend(), std::inserter(new_result, new_result.begin()),
				    [this, cmd](detail::command_id cid) { return m_commands.at(cid).pkg.get_command_type() == cmd; });
				result = std::move(new_result);
			}

			return result;
		}

		bool has_dependency(detail::command_id dependent, detail::command_id dependency) const {
			const auto& deps = m_commands.at(dependent).dependencies;
			return std::find(deps.cbegin(), deps.cend(), dependency) != deps.cend();
		}

		size_t get_dependency_count(detail::command_id dependent) const { return m_commands.at(dependent).dependencies.size(); }

		std::vector<detail::command_id> get_dependencies(detail::command_id dependent) const { return m_commands.at(dependent).dependencies; }

	  private:
		struct cmd_info {
			detail::node_id nid;
			detail::command_pkg pkg;
			std::vector<detail::command_id> dependencies;
		};

		std::map<detail::command_id, cmd_info> m_commands;
		std::map<detail::task_id, std::set<detail::command_id>> m_by_task;
		std::map<experimental::bench::detail::node_id, std::set<detail::command_id>> m_by_node;
	};

	class cdag_test_context {
	  public:
		cdag_test_context(size_t num_nodes) {
			m_rm = std::make_unique<detail::reduction_manager>();
			m_tm = std::make_unique<detail::task_manager>(1 /* num_nodes */, nullptr /* host_queue */, m_rm.get());
			m_cdag = std::make_unique<detail::command_graph>();
			m_ggen = std::make_unique<detail::graph_generator>(num_nodes, *m_rm, *m_cdag);
			m_gser = std::make_unique<detail::graph_serializer>(*m_cdag, m_inspector.get_cb());
			this->m_num_nodes = num_nodes;
		}

		detail::reduction_manager& get_reduction_manager() { return *m_rm; }
		detail::task_manager& get_task_manager() { return *m_tm; }
		detail::command_graph& get_command_graph() { return *m_cdag; }
		detail::graph_generator& get_graph_generator() { return *m_ggen; }
		cdag_inspector& get_inspector() { return m_inspector; }
		detail::graph_serializer& get_graph_serializer() { return *m_gser; }

		detail::task_id build_task_horizons() {
			const auto most_recently_generated_task_horizon = detail::task_manager_testspy::get_current_horizon(get_task_manager());
			if(most_recently_generated_task_horizon != m_most_recently_built_task_horizon) {
				m_most_recently_built_task_horizon = most_recently_generated_task_horizon;
				if(m_most_recently_built_task_horizon) {
					// naive_split does not really do anything for horizons, but this mirrors the behavior of scheduler::schedule exactly.
					detail::naive_split_transformer naive_split(m_num_nodes, m_num_nodes);
					get_graph_generator().build_task(*m_tm->get_task(*m_most_recently_built_task_horizon), {&naive_split});
					return *m_most_recently_built_task_horizon;
				}
			}
			return 0;
		}

	  private:
		std::unique_ptr<detail::reduction_manager> m_rm;
		std::unique_ptr<detail::task_manager> m_tm;
		std::unique_ptr<detail::command_graph> m_cdag;
		std::unique_ptr<detail::graph_generator> m_ggen;
		cdag_inspector m_inspector;
		std::unique_ptr<detail::graph_serializer> m_gser;
		size_t m_num_nodes;
		std::optional<detail::task_id> m_most_recently_built_task_horizon;
	};

	class mock_buffer_factory {
	  public:
		explicit mock_buffer_factory() = default;
		explicit mock_buffer_factory(detail::task_manager& tm) : m_task_mngr(&tm) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::graph_generator& ggen) : m_task_mngr(&tm), m_ggen(&ggen) {}
		explicit mock_buffer_factory(detail::task_manager& tm, detail::abstract_scheduler& schdlr) : m_task_mngr(&tm), m_schdlr(&schdlr) {}
		explicit mock_buffer_factory(cdag_test_context& ctx) : m_task_mngr(&ctx.get_task_manager()), m_ggen(&ctx.get_graph_generator()) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(cl::sycl::range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = m_next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			if(m_task_mngr != nullptr) { m_task_mngr->add_buffer(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(m_schdlr != nullptr) { m_schdlr->notify_buffer_registered(bid, detail::range_cast<3>(size)); }
			if(m_ggen != nullptr) { m_ggen->add_buffer(bid, detail::range_cast<3>(size)); }
			return buf;
		}

	  private:
		detail::task_manager* m_task_mngr = nullptr;
		detail::abstract_scheduler* m_schdlr = nullptr;
		detail::graph_generator* m_ggen = nullptr;
		detail::buffer_id m_next_buffer_id = 0;
	};

	class mock_host_object_factory {
	  public:
		mock_host_object create_host_object() { return mock_host_object{m_next_id++}; }

	  private:
		detail::host_object_id m_next_id = 0;
	};

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_compute_task(
	    detail::task_manager& tm, CGF cgf, cl::sycl::range<KernelDims> global_size = {1, 1}, cl::sycl::id<KernelDims> global_offset = {}) {
		return tm.submit_command_group([&, gs = global_size, go = global_offset](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(gs, go, [](cl::sycl::id<KernelDims>) {});
		});
	}

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_nd_range_compute_task(detail::task_manager& tm, CGF cgf, celerity::nd_range<KernelDims> execution_range = {{1, 1}, {1, 1}}) {
		return tm.submit_command_group([&, er = execution_range](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(er, [](nd_item<KernelDims>) {});
		});
	}

	template <typename Spec, typename CGF>
	detail::task_id add_host_task(detail::task_manager& tm, Spec spec, CGF cgf) {
		return tm.submit_command_group([&](handler& cgh) {
			cgf(cgh);
			cgh.host_task(spec, [](auto...) {});
		});
	}

	inline detail::task_id build_and_flush(cdag_test_context& ctx, size_t num_nodes, size_t num_chunks, detail::task_id tid) {
		detail::naive_split_transformer transformer{num_chunks, num_nodes};
		ctx.get_graph_generator().build_task(*ctx.get_task_manager().get_task(tid), {&transformer});
		ctx.get_graph_serializer().flush(tid);
		if(const auto htid = ctx.build_task_horizons()) { ctx.get_graph_serializer().flush(htid); }
		return tid;
	}

	// Defaults to the same number of chunks as nodes
	inline detail::task_id build_and_flush(cdag_test_context& ctx, size_t num_nodes, detail::task_id tid) {
		return build_and_flush(ctx, num_nodes, num_nodes, tid);
	}

	// Defaults to one node and chunk
	inline detail::task_id build_and_flush(cdag_test_context& ctx, detail::task_id tid) { return build_and_flush(ctx, 1, 1, tid); }

	template <int Dims>
	void add_reduction(handler& cgh, detail::reduction_manager& rm, const mock_buffer<Dims>& vars, bool include_current_buffer_value) {
		auto bid = vars.get_id();
		auto rid = rm.create_reduction<int, Dims>(
		    bid, [](int a, int b) { return a + b; }, 0, include_current_buffer_value);
		static_cast<detail::prepass_handler&>(cgh).add_reduction<Dims>(rid);
	}

	// This fixture (or a subclass) must be used by all tests that transitively use MPI.
	class mpi_fixture {
	  public:
		mpi_fixture() { detail::runtime::test_require_mpi(); }

		mpi_fixture(const mpi_fixture&) = delete;
		mpi_fixture& operator=(const mpi_fixture&) = delete;
	};

	// This fixture (or a subclass) must be used by all tests that transitively instantiate the runtime.
	class runtime_fixture : public mpi_fixture {
	  public:
		runtime_fixture() { detail::runtime::test_case_enter(); }

		runtime_fixture(const runtime_fixture&) = delete;
		runtime_fixture& operator=(const runtime_fixture&) = delete;

		~runtime_fixture() {
			if(!detail::runtime::test_runtime_was_instantiated()) { WARN("Test specified a runtime_fixture, but did not end up instantiating the runtime"); }
			detail::runtime::test_case_exit();
		}
	};

	class device_queue_fixture : public mpi_fixture { // mpi_fixture for config
	  public:
		~device_queue_fixture() { get_device_queue().get_sycl_queue().wait_and_throw(); }

		detail::device_queue& get_device_queue() {
			if(!m_dq) {
				m_cfg = std::make_unique<detail::config>(nullptr, nullptr);
				m_dq = std::make_unique<detail::device_queue>();
				m_dq->init(*m_cfg, detail::auto_select_device{});
			}
			return *m_dq;
		}

	  private:
		std::unique_ptr<detail::config> m_cfg;
		std::unique_ptr<detail::device_queue> m_dq;
	};

	// Printing of graphs can be enabled using the "--print-graphs" command line flag
	inline bool print_graphs = false;

	inline void maybe_print_graph(celerity::detail::task_manager& tm) {
		if(print_graphs) {
			const auto graph_str = tm.print_graph(std::numeric_limits<size_t>::max());
			assert(graph_str.has_value());
			CELERITY_INFO("Task graph:\n\n{}\n", *graph_str);
		}
	}

	inline void maybe_print_graph(
	    celerity::detail::command_graph& cdag, const celerity::detail::task_manager& tm, const celerity::detail::reduction_manager& rm) {
		if(print_graphs) {
			const auto graph_str = cdag.print_graph(std::numeric_limits<size_t>::max(), tm, rm);
			assert(graph_str.has_value());
			CELERITY_INFO("Command graph:\n\n{}\n", *graph_str);
		}
	}

	inline void maybe_print_graphs(celerity::test_utils::cdag_test_context& ctx) {
		if(print_graphs) {
			maybe_print_graph(ctx.get_task_manager());
			maybe_print_graph(ctx.get_command_graph(), ctx.get_task_manager(), ctx.get_reduction_manager());
		}
	}

} // namespace test_utils
} // namespace celerity


namespace Catch {

template <int Dims>
struct StringMaker<cl::sycl::id<Dims>> {
	static std::string convert(const cl::sycl::id<Dims>& value) {
		switch(Dims) {
		case 1: return fmt::format("{{{}}}", value[0]);
		case 2: return fmt::format("{{{}, {}}}", value[0], value[1]);
		case 3: return fmt::format("{{{}, {}, {}}}", value[0], value[1], value[2]);
		default: return {};
		}
	}
};

template <int Dims>
struct StringMaker<cl::sycl::range<Dims>> {
	static std::string convert(const cl::sycl::range<Dims>& value) {
		switch(Dims) {
		case 1: return fmt::format("{{{}}}", value[0]);
		case 2: return fmt::format("{{{}, {}}}", value[0], value[1]);
		case 3: return fmt::format("{{{}, {}, {}}}", value[0], value[1], value[2]);
		default: return {};
		}
	}
};

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