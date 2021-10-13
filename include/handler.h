#pragma once

#include <regex>
#include <type_traits>
#include <typeinfo>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "buffer.h"
#include "device_queue.h"
#include "host_queue.h"
#include "item.h"
#include "range_mapper.h"
#include "ranges.h"
#include "reduction_manager.h"
#include "task.h"
#include "types.h"
#include "workaround.h"

#if !defined(_MSC_VER)
// Required for kernel name demangling in Clang
#include <cxxabi.h>
#endif

namespace celerity {

class handler;

namespace detail {
	class device_queue;
	class task_manager;
	class prepass_handler;

	inline bool is_prepass_handler(const handler& cgh);
	inline execution_target get_handler_execution_target(const handler& cgh);

	template <typename Name>
	std::string kernel_debug_name() {
		std::string name = typeid(Name*).name();
		// TODO: On Windows, returned names are still a bit messy. Consider improving this.
#if !defined(_MSC_VER)
		const std::unique_ptr<char, void (*)(void*)> demangled(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr), std::free);
		const std::string demangled_s(demangled.get());
		if(size_t lastc; (lastc = demangled_s.rfind(":")) != std::string::npos) {
			name = demangled_s.substr(lastc + 1, demangled_s.length() - lastc - 1);
		} else {
			name = demangled_s;
		}
#endif
		return name.substr(0, name.length() - 1);
	}
} // namespace detail

/**
 * Tag type marking a `handler::host_task` as a master-node task. Do not construct this type directly, but use `celerity::on_master_node`.
 */
class on_master_node_tag {};

/**
 * Pass to `handler::host_task` to select the master-node task overload.
 */
inline constexpr on_master_node_tag on_master_node;

namespace experimental {
	class collective_tag_factory;

	/**
	 * Each collective host task is executed within a collective group. If multiple host tasks are scheduled within the same collective group, they are
	 * guaranteed to execute in the same order on every node and within a single thread per node. Each group has its own MPI communicator spanning all
	 * participating nodes, so MPI operations the user invokes from different collective groups do not race.
	 */
	class collective_group {
	  public:
		/// Creates a new collective group with a globally unique id. This must only be called from the main thread.
		collective_group() noexcept : cgid(next_cgid++) {}

	  private:
		friend class collective_tag_factory;
		detail::collective_group_id cgid;
		inline static size_t next_cgid = 1;
	};

	/**
	 * Tag type marking a `handler::host_task` as a collective task. Do not construct this type directly, but use `celerity::experimental::collective`
	 * or `celerity::experimental::collective(group)`.
	 */
	class collective_tag {
	  private:
		friend class collective_tag_factory;
		friend class celerity::handler;
		collective_tag(detail::collective_group_id cgid) : cgid(cgid) {}
		detail::collective_group_id cgid;
	};

	/**
	 * The collective group used in collective host tasks when no group is specified explicitly.
	 */
	inline const collective_group default_collective_group;

	/**
	 * Tag type construction helper. Do not construct this type directly, use `celerity::experimental::collective` instead.
	 */
	class collective_tag_factory {
	  public:
		operator experimental::collective_tag() const { return default_collective_group.cgid; }
		experimental::collective_tag operator()(experimental::collective_group cg) const { return cg.cgid; }
	};

	/**
	 * Pass to `handler::host_task` to select the collective host task overload.
	 *
	 * Either as a value to schedule with the `default_collective_group`:
	 * ```c++
	 * cgh.host_task(celerity::experimental::collective, []...);
	 * ```
	 *
	 * Or by specifying a collective group explicitly:
	 * ```c++
	 * celerity::experimental::collective_group my_group;
	 * ...
	 * cgh.host_task(celerity::experimental::collective(my_group), []...);
	 * ```
	 */
	inline constexpr collective_tag_factory collective;
} // namespace experimental

class handler {
  public:
	virtual ~handler() = default;

	template <typename Name, int Dims, typename... ReductionsAndKernel>
	void parallel_for(cl::sycl::range<Dims> global_range, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<Name, Dims, ReductionsAndKernel...>(
		    global_range, cl::sycl::id<Dims>(), std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{}, reductions_and_kernel...);
	}

	template <typename Name, int Dims, typename... ReductionsAndKernel>
	void parallel_for(cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<Name, Dims, ReductionsAndKernel...>(
		    global_range, global_offset, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{}, reductions_and_kernel...);
	}

	/**
	 * Schedules `kernel` to execute on the master node only. Call via `cgh.host_task(celerity::on_master_node, []...)`. The kernel is assumed to be invocable
	 * with the signature `void(const celerity::partition<0> &)` or `void()`.
	 *
	 * The kernel is executed in a background thread pool and multiple master node tasks may be executed concurrently if they are independent in the
	 * task graph, so proper synchronization must be ensured.
	 *
	 * **Compatibility note:** This replaces master-access tasks from Celerity 0.1 which were executed on the master node's main thread, so this implementation
	 * may require different lifetimes for captures. See `celerity::allow_by_ref` for more information on this topic.
	 */
	template <typename Functor>
	void host_task(on_master_node_tag, Functor kernel);

	/**
	 * Schedules `kernel` to be executed collectively on all nodes participating in the specified collective group. Call via
	 * `cgh.host_task(celerity::experimental::collective, []...)` or  `cgh.host_task(celerity::experimental::collective(group), []...)`.
	 * The kernel is assumed to be invocable with the signature `void(const celerity::experimental::collective_partition&)`
	 * or `void(const celerity::partition<1>&)`.
	 *
	 * This provides framework to use arbitrary collective MPI operations in a host task, such as performing collective I/O with parallel HDF5.
	 * The local node id,t the number of participating nodes as well as the group MPI communicator can be obtained from the `collective_partition` passed into
	 * the kernel.
	 *
	 * All collective tasks within a collective group are guaranteed to be executed in the same order on all nodes, additionally, all internal MPI operations
	 * and all host kernel invocations are executed in a single thread on each host.
	 */
	template <typename Functor>
	void host_task(experimental::collective_tag tag, Functor kernel);

	/**
	 * Schedules a distributed execution of `kernel` by splitting the iteration space in a runtime-defined manner. The kernel is assumed to be invocable
	 * with the signature `void(const celerity::partition<Dims>&)`.
	 *
	 * The kernel is executed in a background thread pool with multiple host tasks being run concurrently if they are independent in the task graph,
	 * so proper synchronization must be ensured. The partition passed into the kernel describes the split each host receives. It may be used with accessors
	 * to obtain the per-node portion of a buffer en-bloc, see `celerity::accessor::get_allocation_window` for details.
	 *
	 * There are no guarantees with respect to the split size and the order in which host tasks are re-orered between nodes other than
	 * the restrictions imposed by dependencies in the task graph. Also, the kernel may be invoked multiple times on one node and not be scheduled on
	 * another node. If you need guarantees about execution order
	 */
	template <int Dims, typename Functor>
	void host_task(cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset, Functor kernel);

	/**
	 * Like `host_task(cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset, Functor kernel)`, but with a `global_offset` of zero.
	 */
	template <int Dims, typename Functor>
	void host_task(cl::sycl::range<Dims> global_range, Functor task) {
		host_task(global_range, {}, task);
	}

  protected:
	friend bool detail::is_prepass_handler(const handler& cgh);
	friend detail::execution_target detail::get_handler_execution_target(const handler& cgh);

	handler() = default;

	virtual bool is_prepass() const = 0;

	virtual const detail::task& get_task() const = 0;

	virtual void create_collective_task(detail::collective_group_id cgid) {
		std::terminate(); // unimplemented
	}

	virtual void create_host_compute_task(int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset) {
		std::terminate(); // unimplemented
	}

	virtual void create_device_compute_task(int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset, std::string debug_name) {
		std::terminate(); // unimplemented
	}

	virtual void create_master_node_task() {
		std::terminate(); // unimplemented
	}

  private:
	template <typename Name, int Dims, typename... ReductionsAndKernel, size_t... ReductionIndices>
	void parallel_for_reductions_and_kernel(cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset,
	    std::index_sequence<ReductionIndices...> indices, ReductionsAndKernel&... kernel_and_reductions) {
		auto args_tuple = std::forward_as_tuple(kernel_and_reductions...);
		auto& kernel = std::get<sizeof...(kernel_and_reductions) - 1>(args_tuple);
		parallel_for_kernel_and_reductions<Name>(global_range, global_offset, kernel, std::get<ReductionIndices>(args_tuple)...);
	}

	template <typename Name, int Dims, typename Kernel, typename... Reductions>
	void parallel_for_kernel_and_reductions(cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset, Kernel& kernel, Reductions&... reductions);
};

namespace detail {

	inline bool is_prepass_handler(const handler& cgh) { return cgh.is_prepass(); }
	inline execution_target get_handler_execution_target(const handler& cgh) { return cgh.get_task().get_execution_target(); }

	class prepass_handler final : public handler {
	  public:
		explicit prepass_handler(task_id tid, std::unique_ptr<command_group_storage_base> cgf, size_t num_collective_nodes)
		    : tid(tid), cgf(std::move(cgf)), num_collective_nodes(num_collective_nodes) {}

		void add_requirement(buffer_id bid, std::unique_ptr<range_mapper_base> rm) {
			assert(task == nullptr);
			access_map.add_access(bid, std::move(rm));
		}

		template <int Dims>
		void add_reduction(reduction_id rid) {
			reductions.push_back(rid);
		}

		void create_host_compute_task(int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset) override {
			assert(task == nullptr);
			task = detail::task::make_host_compute(tid, dimensions, global_size, global_offset, std::move(cgf), std::move(access_map), std::move(reductions));
		}

		void create_device_compute_task(int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset, std::string debug_name) override {
			assert(task == nullptr);
			task = detail::task::make_device_compute(
			    tid, dimensions, global_size, global_offset, std::move(cgf), std::move(access_map), std::move(reductions), std::move(debug_name));
		}

		void create_collective_task(collective_group_id cgid) override {
			assert(task == nullptr);
			task = detail::task::make_collective(tid, cgid, num_collective_nodes, std::move(cgf), std::move(access_map));
		}

		void create_master_node_task() override {
			assert(task == nullptr);
			task = detail::task::make_master_node(tid, std::move(cgf), std::move(access_map));
		}

		std::shared_ptr<class task> into_task() && { return std::move(task); }

	  protected:
		bool is_prepass() const override { return true; }

		const class task& get_task() const override {
			assert(task != nullptr);
			return *task;
		}

	  private:
		task_id tid;
		std::unique_ptr<command_group_storage_base> cgf;
		buffer_access_map access_map;
		std::vector<reduction_id> reductions;
		std::shared_ptr<class task> task = nullptr;
		size_t num_collective_nodes;
	};

	class live_pass_handler : public handler {
	  public:
		bool is_prepass() const final { return false; }

		const class task& get_task() const final { return *task; }

		template <int BufferDims, typename RangeMapper>
		subrange<BufferDims> apply_range_mapper(RangeMapper rm, const cl::sycl::range<BufferDims>& buffer_range) const {
			return invoke_range_mapper(task->get_dimensions(), rm, chunk{sr.offset, sr.range, task->get_global_size()}, buffer_range);
		}

		subrange<3> get_iteration_range() { return sr; }

		bool is_reduction_initializer() const { return initialize_reductions; }

	  protected:
		live_pass_handler(std::shared_ptr<const class task> task, subrange<3> sr, bool initialize_reductions)
		    : task(std::move(task)), sr(sr), initialize_reductions(initialize_reductions) {}

		std::shared_ptr<const class task> task = nullptr;

		// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
		subrange<3> sr;

		bool initialize_reductions;
	};

	class live_pass_host_handler final : public live_pass_handler {
	  public:
		live_pass_host_handler(std::shared_ptr<const class task> task, subrange<3> sr, bool initialize_reductions, host_queue& queue)
		    : live_pass_handler(std::move(task), sr, initialize_reductions), queue(&queue) {}

		template <int Dims, typename Kernel>
		void schedule(Kernel kernel) {
			static_assert(Dims >= 0);
			future = queue->submit(task->get_collective_group_id(), [kernel, global_size = task->get_global_size(), sr = sr](MPI_Comm) {
				if constexpr(Dims > 0) {
					const auto part = make_partition<Dims>(range_cast<Dims>(global_size), subrange_cast<Dims>(sr));
					kernel(part);
				} else if constexpr(std::is_invocable_v<Kernel, const partition<0>&>) {
					(void)sr;
					const auto part = make_0d_partition();
					kernel(part);
				} else {
					(void)sr;
					kernel();
				}
			});
		}

		template <typename Kernel>
		void schedule_collective(Kernel kernel) {
			future = queue->submit(task->get_collective_group_id(), [kernel, global_size = task->get_global_size(), sr = sr](MPI_Comm comm) {
				const auto part = make_collective_partition(range_cast<1>(global_size), subrange_cast<1>(sr), comm);
				kernel(part);
			});
		}

		std::future<host_queue::execution_info> into_future() { return std::move(future); }

	  private:
		host_queue* queue;
		std::future<host_queue::execution_info> future;
	};

	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel_with_celerity_item(const Kernel& kernel, const cl::sycl::id<Dims>& s_id, const cl::sycl::range<Dims>& global_range,
	    const cl::sycl::id<Dims>& global_offset, const cl::sycl::id<Dims>& chunk_offset, Reducers&... reducers) {
		kernel(make_item<Dims>(s_id + chunk_offset, global_offset, global_range), reducers...);
	}

	template <typename Kernel, int Dims, typename... Reducers>
	[[deprecated("Support for kernels receiving cl::sycl::item<Dims> will be removed in the future, change parameter type to celerity::item<Dims>")]] //
	inline void
	invoke_kernel_with_sycl_item(const Kernel& kernel, const cl::sycl::item<Dims>& s_item, Reducers&... reducers) {
		kernel(s_item, reducers...);
	}

	template <typename Kernel, int Dims>
	auto bind_kernel(
	    const Kernel& kernel, const cl::sycl::range<Dims>& global_range, const cl::sycl::id<Dims>& global_offset, const cl::sycl::id<Dims>& chunk_offset) {
		// The current mechanism for hydrating the SYCL placeholder accessors inside Celerity accessors requires that the kernel functor
		// capturing those accessors is copied at least once during submission (see also live_pass_device_handler::submit_to_sycl).
		// As of SYCL 2020 kernel functors are passed as const references, so we explicitly capture by value here.
		return [=](auto s_item_or_id, auto&... reducers) {
			if constexpr(std::is_invocable_v<Kernel, celerity::item<Dims>, decltype(reducers)...>) {
				if constexpr(WORKAROUND_DPCPP && std::is_same_v<cl::sycl::id<Dims>, decltype(s_item_or_id)>) {
					// WORKAROUND: DPC++ passes a sycl::id instead of a sycl::item to kernels alongside reductions
					invoke_kernel_with_celerity_item(kernel, s_item_or_id, global_range, global_offset, chunk_offset, reducers...);
				} else {
					// Explicit item constructor: ComputeCpp does not pass a sycl::item, but an implicitly convertible sycl::item_base (?) which does not have
					// `sycl::id<> get_id()`
					invoke_kernel_with_celerity_item(
					    kernel, cl::sycl::item<Dims>{s_item_or_id}.get_id(), global_range, global_offset, chunk_offset, reducers...);
				}
			} else if constexpr(std::is_invocable_v<Kernel, cl::sycl::item<Dims>, decltype(reducers)...>) {
				invoke_kernel_with_sycl_item(kernel, cl::sycl::item<Dims>{s_item_or_id}, reducers...);
			} else {
				static_assert(constexpr_false<decltype(reducers)...>,
				    "Kernel function must be invocable with celerity::item<Dims> (or cl::sycl::item<Dims>, deprecated) and as "
				    "many reducer objects as reductions passed to parallel_for");
			}
		};
	}

	class live_pass_device_handler final : public live_pass_handler {
	  public:
		live_pass_device_handler(std::shared_ptr<const class task> task, subrange<3> sr, bool initialize_reductions, device_queue& d_queue)
		    : live_pass_handler(std::move(task), sr, initialize_reductions), d_queue(&d_queue) {}

		template <typename CGF>
		void submit_to_sycl(CGF&& cgf) {
			event = d_queue->submit([&](cl::sycl::handler& cgh) {
				this->eventual_cgh = &cgh;
				std::forward<CGF>(cgf)(cgh);
				this->eventual_cgh = nullptr;
			});
		}

		cl::sycl::event get_submission_event() const { return event; }

		cl::sycl::handler* const* get_eventual_sycl_cgh() const { return &eventual_cgh; }

	  private:
		device_queue* d_queue;
		cl::sycl::handler* eventual_cgh = nullptr;
		cl::sycl::event event;
	};

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	class reduction_descriptor;

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	auto make_sycl_reduction(cl::sycl::handler& sycl_cgh, const reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>& d) {
#if WORKAROUND_COMPUTECPP || (WORKAROUND_HIPSYCL && !CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)
		static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
		cl::sycl::property_list props;
		if(!d.include_current_buffer_value) { props = {cl::sycl::property::reduction::initialize_to_identity{}}; }
		if constexpr(WithExplicitIdentity) {
			return cl::sycl::reduction(*d.sycl_buffer, sycl_cgh, d.identity, d.op, props);
		} else {
			return cl::sycl::reduction(*d.sycl_buffer, sycl_cgh, d.op, props);
		}
#endif
	}

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, false /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(
		    buffer_id bid, BinaryOperation combiner, DataT /* identity */, bool include_current_buffer_value, cl::sycl::buffer<DataT, Dims>* sycl_buffer)
		    : bid(bid), op(combiner), include_current_buffer_value(include_current_buffer_value), sycl_buffer(sycl_buffer) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, false>(cl::sycl::handler&, const reduction_descriptor&);

		buffer_id bid;
		BinaryOperation op;
		bool include_current_buffer_value;
		cl::sycl::buffer<DataT, Dims>* sycl_buffer;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, true /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(
		    buffer_id bid, BinaryOperation combiner, DataT identity, bool include_current_buffer_value, cl::sycl::buffer<DataT, Dims>* sycl_buffer)
		    : bid(bid), op(combiner), identity(identity), include_current_buffer_value(include_current_buffer_value), sycl_buffer(sycl_buffer) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, true>(cl::sycl::handler&, const reduction_descriptor&);

		buffer_id bid;
		BinaryOperation op;
		DataT identity{};
		bool include_current_buffer_value;
		cl::sycl::buffer<DataT, Dims>* sycl_buffer;
	};

	template <bool WithExplicitIdentity, typename DataT, int Dims, typename BinaryOperation>
	auto make_reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation op, DataT identity, const cl::sycl::property_list& prop_list) {
#if WORKAROUND_COMPUTECPP || (WORKAROUND_HIPSYCL && !CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)
		static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
		if(vars.get_range().size() != 1) {
			// Like SYCL 2020, Celerity only supports reductions to unit-sized buffers. This allows us to avoid tracking different parts of the buffer
			// as distributed_state and pending_reduction_state.
			throw std::runtime_error("Only unit-sized buffers can be reduction targets");
		}

		auto bid = detail::get_buffer_id(vars);
		auto include_current_buffer_value = !prop_list.has_property<cl::sycl::property::reduction::initialize_to_identity>();
		cl::sycl::buffer<DataT, Dims>* sycl_buffer = nullptr;

		if(detail::is_prepass_handler(cgh)) {
			auto rid = detail::runtime::get_instance().get_reduction_manager().create_reduction<DataT, Dims>(bid, op, identity, include_current_buffer_value);
			static_cast<detail::prepass_handler&>(cgh).add_reduction<Dims>(rid);
		} else {
			include_current_buffer_value &= static_cast<detail::live_pass_handler&>(cgh).is_reduction_initializer();

			auto mode = cl::sycl::access_mode::discard_write;
			if(include_current_buffer_value) { mode = cl::sycl::access_mode::read_write; }
			sycl_buffer = &runtime::get_instance()
			                   .get_buffer_manager()
			                   .get_device_buffer<DataT, Dims>(bid, mode, cl::sycl::range<3>{1, 1, 1}, cl::sycl::id<3>{}) //
			                   .buffer;
		}
		return detail::reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>{bid, op, identity, include_current_buffer_value, sycl_buffer};
#endif
	}

} // namespace detail

template <typename Name, int Dims, typename Kernel, typename... Reductions>
void handler::parallel_for_kernel_and_reductions(
    cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset, Kernel& kernel, Reductions&... reductions) {
	if(is_prepass()) {
		return create_device_compute_task(Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset), detail::kernel_debug_name<Name>());
	}

	auto& device_handler = dynamic_cast<detail::live_pass_device_handler&>(*this);
	const auto sr = device_handler.get_iteration_range();

	device_handler.submit_to_sycl([&](cl::sycl::handler& cgh) {
		// ComputeCpp does not support reductions at all, but users cannot create reductions without triggering a static_assert in that case anyway.
		if constexpr(WORKAROUND_DPCPP && sizeof...(reductions) > 1) {
			static_assert(detail::constexpr_false<Kernel>, "DPC++ currently does not support more than one reduction variable per kernel");
		} else {
			cgh.parallel_for<Name>(detail::range_cast<Dims>(sr.range), detail::make_sycl_reduction(cgh, reductions)...,
			    detail::bind_kernel(kernel, global_range, global_offset, detail::id_cast<Dims>(sr.offset)));
		}
	});
}

template <typename Functor>
void handler::host_task(on_master_node_tag, Functor kernel) {
	if(is_prepass()) {
		create_master_node_task();
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule<0>(kernel);
	}
}

template <typename Functor>
void handler::host_task(experimental::collective_tag tag, Functor kernel) {
	if(is_prepass()) {
		create_collective_task(tag.cgid);
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule_collective(kernel);
	}
}

template <int Dims, typename Functor>
void handler::host_task(cl::sycl::range<Dims> global_range, cl::sycl::id<Dims> global_offset, Functor kernel) {
	if(is_prepass()) {
		create_host_compute_task(Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset));
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule<Dims>(kernel);
	}
}

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation combiner, const cl::sycl::property_list& prop_list = {}) {
#if WORKAROUND_COMPUTECPP || (WORKAROUND_HIPSYCL && !CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)
	static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
#if WORKAROUND_DPCPP
	static_assert(Dims == 1, "DPC++ currently does not support reductions to buffers with dimensionality != 1");
#endif
	static_assert(cl::sycl::has_known_identity_v<BinaryOperation, DataT>,
	    "Celerity does not currently support reductions without an identity. Either specialize "
	    "cl::sycl::known_identity or use the reduction() overload taking an identity at runtime");
	return detail::make_reduction<false>(vars, cgh, combiner, cl::sycl::known_identity_v<BinaryOperation, DataT>, prop_list);
#endif
}

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(const buffer<DataT, Dims>& vars, handler& cgh, const DataT identity, BinaryOperation combiner, const cl::sycl::property_list& prop_list = {}) {
#if WORKAROUND_COMPUTECPP || (WORKAROUND_HIPSYCL && !CELERITY_HIPSYCL_SUPPORTS_REDUCTIONS)
	static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
	static_assert(!cl::sycl::has_known_identity_v<BinaryOperation, DataT>, "Identity is known to SYCL, remove the identity parameter from reduction()");
	return detail::make_reduction<true>(vars, cgh, combiner, identity, prop_list);
#endif
}

} // namespace celerity
