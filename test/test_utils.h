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
		static task* get_current_horizon_task(task_manager& tm) { return tm.current_horizon_task; }

		static int get_num_horizons(task_manager& tm) {
			int horizon_counter = 0;
			for(auto& [_, task_ptr] : tm.task_map) {
				if(task_ptr->get_type() == task_type::HORIZON) { horizon_counter++; }
			}
			return horizon_counter;
		}

		static region_map<std::optional<task_id>> get_last_writer(task_manager& tm, const buffer_id bid) { return tm.buffers_last_writers.at(bid); }

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
			auto most_recently_generated_task_horizon = detail::task_manager_testspy::get_current_horizon_task(get_task_manager());
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

	class buffer_manager_fixture {
	  public:
		enum class access_target { HOST, DEVICE };

		~buffer_manager_fixture() { get_device_queue().get_sycl_queue().wait_and_throw(); }

		void initialize(detail::buffer_manager::buffer_lifecycle_callback cb = [](detail::buffer_manager::buffer_lifecycle_event, detail::buffer_id) {}) {
			l = std::make_unique<detail::logger>("test", detail::log_level::warn);
			cfg = std::make_unique<detail::config>(nullptr, nullptr, *l);
			dq = std::make_unique<detail::device_queue>(*l);
			dq->init(*cfg, nullptr);
			bm = std::make_unique<detail::buffer_manager>(*dq, cb);
			bm->enable_test_mode();
			initialized = true;
		}

		detail::buffer_manager& get_buffer_manager() {
			if(!initialized) initialize();
			return *bm;
		}

		detail::device_queue& get_device_queue() {
			if(!initialized) initialize();
			return *dq;
		}

		static access_target get_other_target(access_target tgt) {
			if(tgt == access_target::HOST) return access_target::DEVICE;
			return access_target::HOST;
		}

		template <typename DataT, int Dims>
		cl::sycl::range<Dims> get_backing_buffer_range(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset) {
			if(tgt == access_target::HOST) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, detail::range_cast<3>(range), detail::id_cast<3>(offset));
				return info.buffer.get_range();
			}
			auto info = bm->get_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return info.buffer.get_range();
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename KernelName = class buffer_for_each, typename Callback>
		void buffer_for_each(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, Callback cb) {
			const auto range3 = detail::range_cast<3>(range);
			const auto offset3 = detail::id_cast<3>(offset);

			if(tgt == access_target::HOST) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_range = detail::range_cast<3>(info.buffer.get_range());
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - detail::id_cast<3>(info.offset);
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							cb(detail::id_cast<Dims>(global_idx), info.buffer.get_pointer()[linear_idx]);
						}
					}
				}
			}

			if(tgt == access_target::DEVICE) {
				auto info = bm->get_device_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_offset = info.offset;
				dq->get_sycl_queue()
				    .submit([&](cl::sycl::handler& cgh) {
					    auto acc = info.buffer.template get_access<Mode>(cgh);
					    cgh.parallel_for<detail::bind_kernel_name<KernelName>>(range, offset, [=](cl::sycl::id<Dims> global_idx) {
						    const auto local_idx = global_idx - buf_offset;
						    cb(global_idx, acc[local_idx]);
					    });
				    })
				    .wait();
			}
		}

		template <typename DataT, int Dims, typename KernelName = class buffer_reduce, typename ReduceT, typename Operation>
		ReduceT buffer_reduce(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, ReduceT init, Operation op) {
			const auto range3 = detail::range_cast<3>(range);
			const auto offset3 = detail::id_cast<3>(offset);

			if(tgt == access_target::HOST) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
				const auto buf_range = detail::range_cast<3>(info.buffer.get_range());
				ReduceT result = init;
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - detail::id_cast<3>(info.offset);
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							result = op(detail::id_cast<Dims>(global_idx), result, info.buffer.get_pointer()[linear_idx]);
						}
					}
				}
				return result;
			}

			auto info = bm->get_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
			const auto buf_offset = info.offset;
			cl::sycl::buffer<ReduceT, 1> result_buf(1); // Use 1-dimensional instead of 0-dimensional since it's NYI in hipSYCL as of 0.8.1
			// Simply do a serial reduction on the device as well
			dq->get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = info.buffer.template get_access<cl::sycl::access::mode::read>(cgh);
				    auto result_acc = result_buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
				    cgh.single_task<detail::bind_kernel_name<KernelName>>([=]() {
					    result_acc[0] = init;
					    for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
						    for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
							    for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
								    const auto global_idx = cl::sycl::id<3>(i, j, k);
								    const cl::sycl::id<3> local_idx = global_idx - detail::id_cast<3>(buf_offset);
								    result_acc[0] = op(detail::id_cast<Dims>(global_idx), result_acc[0], acc[detail::id_cast<Dims>(local_idx)]);
							    }
						    }
					    }
				    });
			    })
			    .wait();

			ReduceT result;
			dq->get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = result_buf.template get_access<cl::sycl::access::mode::read>(cgh);
				    cgh.copy(acc, &result);
			    })
			    .wait();

			return result;
		}

		template <typename DataT, int Dims, access_mode Mode>
		accessor<DataT, Dims, Mode, target::device> get_device_accessor(
		    detail::live_pass_device_handler& cgh, detail::buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = bm->get_device_buffer<DataT, Dims>(bid, Mode, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::make_device_accessor<DataT, Dims, Mode>(
			    cgh.get_eventual_sycl_cgh(), subrange<Dims>(offset, range), buf_info.buffer, buf_info.offset);
		}

		template <typename DataT, int Dims, access_mode Mode>
		accessor<DataT, Dims, Mode, target::host_task> get_host_accessor(
		    detail::buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = bm->get_host_buffer<DataT, Dims>(bid, Mode, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::make_host_accessor<DataT, Dims, Mode>(
			    subrange<Dims>(offset, range), buf_info.buffer, buf_info.offset, detail::range_cast<Dims>(bm->get_buffer_info(bid).range));
		}

	  private:
		bool initialized = false;
		std::unique_ptr<detail::logger> l;
		std::unique_ptr<detail::config> cfg;
		std::unique_ptr<detail::device_queue> dq;
		std::unique_ptr<detail::buffer_manager> bm;
	};

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