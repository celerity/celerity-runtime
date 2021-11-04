#pragma once

#include <map>
#include <ostream>

#include <catch2/catch.hpp>

#include <celerity.h>
#include <memory>

#include "command.h"
#include "command_graph.h"
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

namespace celerity {

namespace detail {

	struct task_manager_testspy {
		static task* get_previous_horizon_task(task_manager& tm) { return tm.previous_horizon_task; }

		static int get_num_horizons(task_manager& tm) {
			int horizon_counter = 0;
			for(auto& [_, task_ptr] : tm.task_map) {
				if(task_ptr->get_type() == task_type::HORIZON) { horizon_counter++; }
			}
			return horizon_counter;
		}

		static region_map<std::optional<task_id>> get_last_writer(task_manager& tm, const buffer_id bid) { return tm.buffers_last_writers.at(bid); }

		static int num_tasks(task_manager& tm) { return tm.task_map.size(); }

		static int get_max_pseudo_critical_path_length(task_manager& tm) { return tm.get_max_pseudo_critical_path_length(); }

		static auto get_execution_front(task_manager& tm) { return tm.get_execution_front(); }
	};
} // namespace detail

namespace test_utils {


	class mock_buffer_factory;

	template <int Dims>
	class mock_buffer {
	  public:
		template <cl::sycl::access::mode Mode, typename Functor>
		void get_access(handler& cgh, Functor rmfn) {
			if(detail::is_prepass_handler(cgh)) {
				auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh); // No live pass in tests
				prepass_cgh.add_requirement(id, std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, size));
			}
		}

		detail::buffer_id get_id() const { return id; }

	  private:
		friend class mock_buffer_factory;

		detail::buffer_id id;
		cl::sycl::range<Dims> size;

		mock_buffer(detail::buffer_id id, cl::sycl::range<Dims> size) : id(id), size(size) {}
	};

	class cdag_inspector {
	  public:
		auto get_cb() {
			return [this](detail::node_id nid, detail::command_pkg pkg, const std::vector<detail::command_id>& dependencies) {
				for(detail::command_id dep : dependencies) {
					// Sanity check: All dependencies must have already been flushed
					(void)dep;
					assert(commands.count(dep) == 1);
				}

				const detail::command_id cid = pkg.cid;
				commands[cid] = {nid, pkg, dependencies};
				if(pkg.cmd == detail::command_type::TASK) { by_task[std::get<detail::task_data>(pkg.data).tid].insert(cid); }
				by_node[nid].insert(cid);
			};
		}

		std::set<detail::command_id> get_commands(
		    std::optional<detail::task_id> tid, std::optional<detail::node_id> nid, std::optional<detail::command_type> cmd) const {
			// Sanity check: Not all commands have an associated task id
			assert(tid == std::nullopt || (cmd == std::nullopt || cmd == detail::command_type::TASK));

			std::set<detail::command_id> result;
			std::transform(commands.cbegin(), commands.cend(), std::inserter(result, result.begin()), [](auto p) { return p.first; });

			if(tid != std::nullopt) {
				auto& task_set = by_task.at(*tid);
				std::set<detail::command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), task_set.cbegin(), task_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(nid != std::nullopt) {
				auto& node_set = by_node.at(*nid);
				std::set<detail::command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), node_set.cbegin(), node_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(cmd != std::nullopt) {
				std::set<detail::command_id> new_result;
				std::copy_if(result.cbegin(), result.cend(), std::inserter(new_result, new_result.begin()),
				    [this, cmd](detail::command_id cid) { return commands.at(cid).pkg.cmd == cmd; });
				result = std::move(new_result);
			}

			return result;
		}

		bool has_dependency(detail::command_id dependent, detail::command_id dependency) {
			const auto& deps = commands.at(dependent).dependencies;
			return std::find(deps.cbegin(), deps.cend(), dependency) != deps.cend();
		}

		size_t get_dependency_count(detail::command_id dependent) const { return commands.at(dependent).dependencies.size(); }

		std::vector<detail::command_id> get_dependencies(detail::command_id dependent) const { return commands.at(dependent).dependencies; }

	  private:
		struct cmd_info {
			detail::node_id nid;
			detail::command_pkg pkg;
			std::vector<detail::command_id> dependencies;
		};

		std::map<detail::command_id, cmd_info> commands;
		std::map<detail::task_id, std::set<detail::command_id>> by_task;
		std::map<experimental::bench::detail::node_id, std::set<detail::command_id>> by_node;
	};

	class cdag_test_context {
	  public:
		cdag_test_context(size_t num_nodes) {
			rm = std::make_unique<detail::reduction_manager>();
			tm = std::make_unique<detail::task_manager>(1 /* num_nodes */, nullptr /* host_queue */, rm.get());
			cdag = std::make_unique<detail::command_graph>();
			ggen = std::make_unique<detail::graph_generator>(num_nodes, *tm, *rm, *cdag);
			gsrlzr = std::make_unique<detail::graph_serializer>(*cdag, inspector.get_cb());
		}

		detail::reduction_manager& get_reduction_manager() { return *rm; }
		detail::task_manager& get_task_manager() { return *tm; }
		detail::command_graph& get_command_graph() { return *cdag; }
		detail::graph_generator& get_graph_generator() { return *ggen; }
		cdag_inspector& get_inspector() { return inspector; }
		detail::graph_serializer& get_graph_serializer() { return *gsrlzr; }

		void build_task_horizons() {
			auto most_recently_generated_task_horizon = detail::task_manager_testspy::get_previous_horizon_task(get_task_manager());
			if(most_recently_generated_task_horizon != most_recently_built_task_horizon) {
				most_recently_built_task_horizon = most_recently_generated_task_horizon;
				if(most_recently_built_task_horizon != nullptr) {
					auto htid = most_recently_built_task_horizon->get_id();
					get_graph_generator().build_task(htid, {nullptr});
				}
			}
		}

	  private:
		std::unique_ptr<detail::reduction_manager> rm;
		std::unique_ptr<detail::task_manager> tm;
		std::unique_ptr<detail::command_graph> cdag;
		std::unique_ptr<detail::graph_generator> ggen;
		cdag_inspector inspector;
		std::unique_ptr<detail::graph_serializer> gsrlzr;
		detail::task* most_recently_built_task_horizon = nullptr;
	};

	class mock_buffer_factory {
	  public:
		mock_buffer_factory(detail::task_manager* tm = nullptr, detail::graph_generator* ggen = nullptr) : task_mngr(tm), ggen(ggen) {}
		mock_buffer_factory(cdag_test_context& ctx) : task_mngr(&ctx.get_task_manager()), ggen(&ctx.get_graph_generator()) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(cl::sycl::range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			if(task_mngr != nullptr) { task_mngr->add_buffer(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(ggen != nullptr) { ggen->add_buffer(bid, detail::range_cast<3>(size)); }
			return buf;
		}

	  private:
		detail::task_manager* task_mngr;
		detail::graph_generator* ggen;
		detail::buffer_id next_buffer_id = 0;
	};

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_compute_task(
	    detail::task_manager& tm, CGF cgf, cl::sycl::range<KernelDims> global_size = {1, 1}, cl::sycl::id<KernelDims> global_offset = {}) {
		return tm.create_task([&, gs = global_size, go = global_offset](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(gs, go, [](cl::sycl::id<KernelDims>) {});
		});
	}

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_nd_range_compute_task(detail::task_manager& tm, CGF cgf, celerity::nd_range<KernelDims> execution_range = {{1, 1}, {1, 1}}) {
		return tm.create_task([&, er = execution_range](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(er, [](nd_item<KernelDims>) {});
		});
	}

	template <typename Spec, typename CGF>
	detail::task_id add_host_task(detail::task_manager& tm, Spec spec, CGF cgf) {
		return tm.create_task([&](handler& cgh) {
			cgf(cgh);
			cgh.host_task(spec, [](auto...) {});
		});
	}

	inline detail::task_id build_and_flush(cdag_test_context& ctx, size_t num_nodes, size_t num_chunks, detail::task_id tid) {
		detail::naive_split_transformer transformer{num_chunks, num_nodes};
		ctx.get_graph_generator().build_task(tid, {&transformer});
		ctx.get_graph_serializer().flush(tid);
		ctx.build_task_horizons();
		ctx.get_graph_serializer().flush_horizons();
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

} // namespace Catch