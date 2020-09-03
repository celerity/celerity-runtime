#pragma once

#include <catch2/catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "command.h"
#include "command_graph.h"
#include "graph_generator.h"
#include "graph_serializer.h"
#include "range_mapper.h"
#include "task_manager.h"
#include "transformers/naive_split.h"
#include "types.h"

// To avoid having to come up with tons of unique kernel names, we simply use the CPP counter.
// This is non-standard but widely supported.
#define _UKN_CONCAT2(x, y) x##_##y
#define _UKN_CONCAT(x, y) _UKN_CONCAT2(x, y)
#define UKN(name) _UKN_CONCAT(name, __COUNTER__)

namespace celerity {
namespace test_utils {

	class mock_buffer_factory;

	template <int Dims>
	class mock_buffer {
	  public:
		template <cl::sycl::access::mode Mode, typename Functor>
		void get_access(handler& cgh, Functor rmfn) {
			using rmfn_traits = allscale::utils::lambda_traits<Functor>;
			static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
			if(detail::is_prepass_handler(cgh)) {
				auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh);
				prepass_cgh.add_requirement(id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, size));
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
			tm = std::make_unique<detail::task_manager>(1 /* num_nodes */, nullptr /* host_queue */, true /* is_master */);
			cdag = std::make_unique<detail::command_graph>();
			ggen = std::make_unique<detail::graph_generator>(num_nodes, *tm, *cdag);
			gsrlzr = std::make_unique<detail::graph_serializer>(*cdag, inspector.get_cb());
		}

		detail::task_manager& get_task_manager() { return *tm; }
		detail::command_graph& get_command_graph() { return *cdag; }
		detail::graph_generator& get_graph_generator() { return *ggen; }
		cdag_inspector& get_inspector() { return inspector; }
		detail::graph_serializer& get_graph_serializer() { return *gsrlzr; }

	  private:
		std::unique_ptr<detail::task_manager> tm;
		std::unique_ptr<detail::command_graph> cdag;
		std::unique_ptr<detail::graph_generator> ggen;
		cdag_inspector inspector;
		std::unique_ptr<detail::graph_serializer> gsrlzr;
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
		ctx.get_graph_serializer().flush_horizons();
		return tid;
	}

	// Defaults to the same number of chunks as nodes
	inline detail::task_id build_and_flush(cdag_test_context& ctx, size_t num_nodes, detail::task_id tid) {
		return build_and_flush(ctx, num_nodes, num_nodes, tid);
	}

	// Defaults to one node and chunk
	inline detail::task_id build_and_flush(cdag_test_context& ctx, detail::task_id tid) { return build_and_flush(ctx, 1, 1, tid); }

} // namespace test_utils
} // namespace celerity
