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
#if !defined(_MSC_VER)
		const std::unique_ptr<char, void (*)(void*)> demangled(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, nullptr), std::free);
		const std::string demangled_s(demangled.get());
		if(size_t lastc = demangled_s.rfind(':'); lastc != std::string::npos) {
			name = demangled_s.substr(lastc + 1, demangled_s.length() - lastc - 1);
		} else {
			name = demangled_s;
		}
#elif defined(_MSC_VER)
		if(size_t lastc, id_end; (lastc = name.rfind(":")) != std::string::npos && (id_end = name.find(" ", lastc)) != std::string::npos) {
			name = name.substr(lastc + 1, id_end - lastc);
		}
#endif
		return name.substr(0, name.length() - 1);
	}

	struct unnamed_kernel {};

	template <typename KernelName>
	constexpr bool is_unnamed_kernel = std::is_same_v<KernelName, unnamed_kernel>;

#if CELERITY_DETAIL_IS_OLD_COMPUTECPP_COMPILER
	template <typename KernelName>
	struct kernel_name_wrapper;
#endif

	template <typename KernelName>
	struct bound_kernel_name {
		static_assert(!is_unnamed_kernel<KernelName>);
#if CELERITY_DETAIL_IS_OLD_COMPUTECPP_COMPILER
		using type = kernel_name_wrapper<KernelName>; // Suppress -Rsycl-kernel-naming diagnostic for local types
#else
		using type = KernelName;
#endif
	};

	template <typename KernelName>
	using bind_kernel_name = typename bound_kernel_name<KernelName>::type;

	struct simple_kernel_flavor {};
	struct nd_range_kernel_flavor {};

	template <typename Flavor, int Dims>
	struct kernel_flavor_traits;

	struct no_local_size {};

	template <int Dims>
	struct kernel_flavor_traits<simple_kernel_flavor, Dims> {
		inline static constexpr bool has_local_size = false;
		using local_size_type = no_local_size;
	};

	template <int Dims>
	struct kernel_flavor_traits<nd_range_kernel_flavor, Dims> {
		inline static constexpr bool has_local_size = true;
		using local_size_type = range<Dims>;
	};
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
		collective_group() noexcept : m_cgid(next_cgid++) {}

	  private:
		friend class collective_tag_factory;
		detail::collective_group_id m_cgid;
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
		collective_tag(detail::collective_group_id cgid) : m_cgid(cgid) {}
		detail::collective_group_id m_cgid;
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
		operator experimental::collective_tag() const { return default_collective_group.m_cgid; }
		experimental::collective_tag operator()(experimental::collective_group cg) const { return cg.m_cgid; }
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

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(range<Dims> global_range, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::simple_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(
		    global_range, id<Dims>(), detail::no_local_size{}, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{}, reductions_and_kernel...);
	}

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(range<Dims> global_range, id<Dims> global_offset, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::simple_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(
		    global_range, global_offset, detail::no_local_size{}, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{}, reductions_and_kernel...);
	}

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(celerity::nd_range<Dims> execution_range, ReductionsAndKernel... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::nd_range_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(execution_range.get_global_range(),
		    execution_range.get_offset(), execution_range.get_local_range(), std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{},
		    reductions_and_kernel...);
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
	void host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel);

	/**
	 * Like `host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel)`, but with a `global_offset` of zero.
	 */
	template <int Dims, typename Functor>
	void host_task(range<Dims> global_range, Functor task) {
		host_task(global_range, {}, task);
	}

  protected:
	friend bool detail::is_prepass_handler(const handler& cgh);
	friend detail::execution_target detail::get_handler_execution_target(const handler& cgh);

	handler() = default;

	virtual bool is_prepass() const = 0;

	virtual const detail::task& get_task() const = 0;

  private:
	template <typename KernelFlavor, typename KernelName, int Dims, typename... ReductionsAndKernel, size_t... ReductionIndices>
	void parallel_for_reductions_and_kernel(range<Dims> global_range, id<Dims> global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_size, std::index_sequence<ReductionIndices...> indices,
	    ReductionsAndKernel&... kernel_and_reductions) {
		auto args_tuple = std::forward_as_tuple(kernel_and_reductions...);
		auto& kernel = std::get<sizeof...(kernel_and_reductions) - 1>(args_tuple);
		parallel_for_kernel_and_reductions<KernelFlavor, KernelName>(
		    global_range, global_offset, local_size, kernel, std::get<ReductionIndices>(args_tuple)...);
	}

	template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, typename... Reductions>
	void parallel_for_kernel_and_reductions(range<Dims> global_range, id<Dims> global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel& kernel, Reductions&... reductions);
};

namespace detail {

	inline bool is_prepass_handler(const handler& cgh) { return cgh.is_prepass(); }
	inline execution_target get_handler_execution_target(const handler& cgh) { return cgh.get_task().get_execution_target(); }

	class prepass_handler final : public handler {
	  public:
		explicit prepass_handler(task_id tid, std::unique_ptr<command_group_storage_base> cgf, size_t num_collective_nodes)
		    : m_tid(tid), m_cgf(std::move(cgf)), m_num_collective_nodes(num_collective_nodes) {}

		void add_requirement(buffer_id bid, std::unique_ptr<range_mapper_base> rm) {
			assert(m_task == nullptr);
			m_access_map.add_access(bid, std::move(rm));
		}

		void add_requirement(const host_object_id hoid, const experimental::side_effect_order order) {
			assert(m_task == nullptr);
			m_side_effects.add_side_effect(hoid, order);
		}

		void add_reduction(const reduction_info& rinfo) { m_reductions.push_back(rinfo); }

		void create_host_compute_task(task_geometry geometry) {
			assert(m_task == nullptr);
			if(geometry.global_size.size() == 0) {
				// TODO this can be easily supported by not creating a task in case the execution range is empty
				throw std::runtime_error{"The execution range of distributed host tasks must have at least one item"};
			}
			m_task =
			    detail::task::make_host_compute(m_tid, geometry, std::move(m_cgf), std::move(m_access_map), std::move(m_side_effects), std::move(m_reductions));
		}

		void create_device_compute_task(task_geometry geometry, std::string debug_name) {
			assert(m_task == nullptr);
			if(geometry.global_size.size() == 0) {
				// TODO unless reductions are involved, this can be easily supported by not creating a task in case the execution range is empty.
				// Edge case: If the task includes reductions that specify property::reduction::initialize_to_identity, we need to create a task that sets
				// the buffer state to an empty pending_reduction_state in the graph_generator. This will cause a trivial reduction_command to be generated on
				// each node that reads from the reduction output buffer, initializing it to the identity value locally.
				throw std::runtime_error{"The execution range of device tasks must have at least one item"};
			}
			if(!m_side_effects.empty()) { throw std::runtime_error{"Side effects cannot be used in device kernels"}; }
			m_task =
			    detail::task::make_device_compute(m_tid, geometry, std::move(m_cgf), std::move(m_access_map), std::move(m_reductions), std::move(debug_name));
		}

		void create_collective_task(collective_group_id cgid) {
			assert(m_task == nullptr);
			m_task = detail::task::make_collective(m_tid, cgid, m_num_collective_nodes, std::move(m_cgf), std::move(m_access_map), std::move(m_side_effects));
		}

		void create_master_node_task() {
			assert(m_task == nullptr);
			m_task = detail::task::make_master_node(m_tid, std::move(m_cgf), std::move(m_access_map), std::move(m_side_effects));
		}

		std::unique_ptr<class task> into_task() && { return std::move(m_task); }

	  protected:
		bool is_prepass() const override { return true; }

		const class task& get_task() const override {
			assert(m_task != nullptr);
			return *m_task;
		}

	  private:
		task_id m_tid;
		std::unique_ptr<command_group_storage_base> m_cgf;
		buffer_access_map m_access_map;
		side_effect_map m_side_effects;
		reduction_set m_reductions;
		std::unique_ptr<class task> m_task = nullptr;
		size_t m_num_collective_nodes;
	};

	class live_pass_handler : public handler {
	  public:
		bool is_prepass() const final { return false; }

		const class task& get_task() const final { return *m_task; }

		template <int BufferDims, typename RangeMapper>
		subrange<BufferDims> apply_range_mapper(RangeMapper rm, const range<BufferDims>& buffer_range) const {
			return invoke_range_mapper(m_task->get_dimensions(), rm, chunk{m_sr.offset, m_sr.range, m_task->get_global_size()}, buffer_range);
		}

		subrange<3> get_iteration_range() { return m_sr; }

		bool is_reduction_initializer() const { return m_initialize_reductions; }

	  protected:
		live_pass_handler(const class task* task, subrange<3> sr, bool initialize_reductions)
		    : m_task(task), m_sr(sr), m_initialize_reductions(initialize_reductions) {}

		const class task* m_task = nullptr;

		// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
		subrange<3> m_sr;

		bool m_initialize_reductions;
	};

	class live_pass_host_handler final : public live_pass_handler {
	  public:
		live_pass_host_handler(const class task* task, subrange<3> sr, bool initialize_reductions, host_queue& queue)
		    : live_pass_handler(task, sr, initialize_reductions), m_queue(&queue) {}

		template <int Dims, typename Kernel>
		void schedule(Kernel kernel) {
			static_assert(Dims >= 0);
			m_future = m_queue->submit(m_task->get_collective_group_id(), [kernel, global_size = m_task->get_global_size(), sr = m_sr](MPI_Comm) {
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
			m_future = m_queue->submit(m_task->get_collective_group_id(), [kernel, global_size = m_task->get_global_size(), sr = m_sr](MPI_Comm comm) {
				const auto part = make_collective_partition(range_cast<1>(global_size), subrange_cast<1>(sr), comm);
				kernel(part);
			});
		}

		std::future<host_queue::execution_info> into_future() { return std::move(m_future); }

	  private:
		host_queue* m_queue;
		std::future<host_queue::execution_info> m_future;
	};

	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel(const Kernel& kernel, const id<Dims>& s_id, const range<Dims>& global_range, const id<Dims>& global_offset,
	    const id<Dims>& chunk_offset, Reducers&... reducers) {
		kernel(make_item<Dims>(s_id + chunk_offset, global_offset, global_range), reducers...);
	}

	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel(const Kernel& kernel, const cl::sycl::nd_item<Dims>& s_item, const range<Dims>& global_range, const id<Dims>& global_offset,
	    const id<Dims>& chunk_offset, const range<Dims>& group_range, const id<Dims>& group_offset, Reducers&... reducers) {
		kernel(make_nd_item<Dims>(s_item, global_range, global_offset, chunk_offset, group_range, group_offset), reducers...);
	}

	template <typename Kernel, int Dims>
	auto bind_simple_kernel(const Kernel& kernel, const range<Dims>& global_range, const id<Dims>& global_offset, const id<Dims>& chunk_offset) {
		// The current mechanism for hydrating the SYCL placeholder accessors inside Celerity accessors requires that the kernel functor
		// capturing those accessors is copied at least once during submission (see also live_pass_device_handler::submit_to_sycl).
		// As of SYCL 2020 kernel functors are passed as const references, so we explicitly capture by value here.
		return [=](auto s_item_or_id, auto&... reducers) {
			static_assert(std::is_invocable_v<Kernel, celerity::item<Dims>, decltype(reducers)...>,
			    "Kernel function must be invocable with celerity::item<Dims> and as many reducer objects as reductions passed to parallel_for");
			if constexpr(CELERITY_WORKAROUND(DPCPP) && std::is_same_v<id<Dims>, decltype(s_item_or_id)>) {
				// CELERITY_WORKAROUND_LESS_OR_EQUAL: DPC++ passes a sycl::id instead of a sycl::item to kernels alongside reductions
				invoke_kernel(kernel, s_item_or_id, global_range, global_offset, chunk_offset, reducers...);
			} else {
				// Explicit item constructor: ComputeCpp does not pass a sycl::item, but an implicitly convertible sycl::item_base (?) which does not have
				// `sycl::id<> get_id()`
				invoke_kernel(kernel, cl::sycl::item<Dims>{s_item_or_id}.get_id(), global_range, global_offset, chunk_offset, reducers...);
			}
		};
	}

	template <typename Kernel, int Dims>
	auto bind_nd_range_kernel(const Kernel& kernel, const range<Dims>& global_range, const id<Dims>& global_offset, const id<Dims> chunk_offset,
	    const range<Dims>& group_range, const id<Dims>& group_offset) {
		return [=](cl::sycl::nd_item<Dims> s_item, auto&... reducers) {
			static_assert(std::is_invocable_v<Kernel, celerity::nd_item<Dims>, decltype(reducers)...>,
			    "Kernel function must be invocable with celerity::nd_item<Dims> or and as many reducer objects as reductions passed to parallel_for");
			invoke_kernel(kernel, s_item, global_range, global_offset, chunk_offset, group_range, group_offset, reducers...);
		};
	}

	template <typename KernelName, typename... Params>
	inline void invoke_sycl_parallel_for(cl::sycl::handler& cgh, Params&&... args) {
		static_assert(CELERITY_FEATURE_UNNAMED_KERNELS || !is_unnamed_kernel<KernelName>,
		    "Your SYCL implementation does not support unnamed kernels, add a kernel name template parameter to this parallel_for invocation");
		if constexpr(detail::is_unnamed_kernel<KernelName>) {
#if CELERITY_FEATURE_UNNAMED_KERNELS // see static_assert above
			cgh.parallel_for(std::forward<Params>(args)...);
#endif
		} else {
			cgh.parallel_for<detail::bind_kernel_name<KernelName>>(std::forward<Params>(args)...);
		}
	}

	class live_pass_device_handler final : public live_pass_handler {
	  public:
		live_pass_device_handler(const class task* task, subrange<3> sr, bool initialize_reductions, device_queue& d_queue)
		    : live_pass_handler(task, sr, initialize_reductions), m_d_queue(&d_queue) {}

		template <typename CGF>
		void submit_to_sycl(CGF&& cgf) {
			m_event = m_d_queue->submit([&](cl::sycl::handler& cgh) {
				this->m_eventual_cgh = &cgh;
				std::forward<CGF>(cgf)(cgh);
				this->m_eventual_cgh = nullptr;
			});
		}

		cl::sycl::event get_submission_event() const { return m_event; }

		cl::sycl::handler* const* get_eventual_sycl_cgh() const { return &m_eventual_cgh; }

	  private:
		device_queue* m_d_queue;
		cl::sycl::handler* m_eventual_cgh = nullptr;
		cl::sycl::event m_event;
	};

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	class reduction_descriptor;

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	auto make_sycl_reduction(const reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>& d) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
		static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
		cl::sycl::property_list props;
		if(!d.m_include_current_buffer_value) { props = {cl::sycl::property::reduction::initialize_to_identity{}}; }
		if constexpr(WithExplicitIdentity) {
			return sycl::reduction(d.m_device_ptr, d.m_identity, d.m_op, props);
		} else {
			return sycl::reduction(d.m_device_ptr, d.m_op, props);
		}
#endif
	}

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, false /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(buffer_id bid, BinaryOperation combiner, DataT /* identity */, bool include_current_buffer_value, DataT* device_ptr)
		    : m_bid(bid), m_op(combiner), m_include_current_buffer_value(include_current_buffer_value), m_device_ptr(device_ptr) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, false>(const reduction_descriptor&);

		buffer_id m_bid;
		BinaryOperation m_op;
		bool m_include_current_buffer_value;
		DataT* m_device_ptr;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, true /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(buffer_id bid, BinaryOperation combiner, DataT identity, bool include_current_buffer_value, DataT* device_ptr)
		    : m_bid(bid), m_op(combiner), m_identity(identity), m_include_current_buffer_value(include_current_buffer_value), m_device_ptr(device_ptr) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, true>(const reduction_descriptor&);

		buffer_id m_bid;
		BinaryOperation m_op;
		DataT m_identity{};
		bool m_include_current_buffer_value;
		DataT* m_device_ptr;
	};

	template <bool WithExplicitIdentity, typename DataT, int Dims, typename BinaryOperation>
	auto make_reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation op, DataT identity, const cl::sycl::property_list& prop_list) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
		static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
		if(vars.get_range().size() != 1) {
			// Like SYCL 2020, Celerity only supports reductions to unit-sized buffers. This allows us to avoid tracking different parts of the buffer
			// as distributed_state and pending_reduction_state.
			throw std::runtime_error("Only unit-sized buffers can be reduction targets");
		}

		auto bid = detail::get_buffer_id(vars);
		auto include_current_buffer_value = !prop_list.has_property<celerity::property::reduction::initialize_to_identity>();
		DataT* device_ptr = nullptr;

		if(detail::is_prepass_handler(cgh)) {
			auto rid = detail::runtime::get_instance().get_reduction_manager().create_reduction<DataT, Dims>(bid, op, identity);
			static_cast<detail::prepass_handler&>(cgh).add_reduction(reduction_info{rid, bid, include_current_buffer_value});
		} else {
			auto& device_handler = static_cast<detail::live_pass_device_handler&>(cgh);
			include_current_buffer_value &= device_handler.is_reduction_initializer();

			auto mode = cl::sycl::access_mode::discard_write;
			if(include_current_buffer_value) { mode = cl::sycl::access_mode::read_write; }
			device_ptr = static_cast<DataT*>(
			    runtime::get_instance().get_buffer_manager().access_device_buffer<DataT, Dims>(bid, mode, subrange_cast<Dims>(subrange<3>{{}, {1, 1, 1}})).ptr);
		}
		return detail::reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>{bid, op, identity, include_current_buffer_value, device_ptr};
#endif
	}

} // namespace detail

template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, typename... Reductions>
void handler::parallel_for_kernel_and_reductions(range<Dims> global_range, id<Dims> global_offset,
    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel& kernel, Reductions&... reductions) {
	if(is_prepass()) {
		range<3> granularity = {1, 1, 1};
		if constexpr(detail::kernel_flavor_traits<KernelFlavor, Dims>::has_local_size) {
			for(int d = 0; d < Dims; ++d) {
				granularity[d] = local_range[d];
			}
		}
		const detail::task_geometry geometry{Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset), granularity};
		return dynamic_cast<detail::prepass_handler&>(*this).create_device_compute_task(geometry, detail::kernel_debug_name<KernelName>());
	}

	auto& device_handler = dynamic_cast<detail::live_pass_device_handler&>(*this);
	const auto sr = device_handler.get_iteration_range();
	auto chunk_range = detail::range_cast<Dims>(sr.range);
	auto chunk_offset = detail::id_cast<Dims>(sr.offset);

	device_handler.submit_to_sycl([&](cl::sycl::handler& cgh) {
		if constexpr(!CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS && sizeof...(reductions) > 0) {
			static_assert(detail::constexpr_false<Kernel>, "Reductions are not supported by your SYCL implementation");
		} else if constexpr(!CELERITY_FEATURE_SCALAR_REDUCTIONS && sizeof...(reductions) > 1) {
			static_assert(detail::constexpr_false<Kernel>, "DPC++ currently does not support more than one reduction variable per kernel");
		} else if constexpr(std::is_same_v<KernelFlavor, detail::simple_kernel_flavor>) {
			detail::invoke_sycl_parallel_for<KernelName>(
			    cgh, chunk_range, detail::make_sycl_reduction(reductions)..., detail::bind_simple_kernel(kernel, global_range, global_offset, chunk_offset));
		} else if constexpr(std::is_same_v<KernelFlavor, detail::nd_range_kernel_flavor>) {
			detail::invoke_sycl_parallel_for<KernelName>(cgh, cl::sycl::nd_range{chunk_range, local_range}, detail::make_sycl_reduction(reductions)...,
			    detail::bind_nd_range_kernel(kernel, global_range, global_offset, chunk_offset, global_range / local_range, chunk_offset / local_range));
		} else {
			static_assert(detail::constexpr_false<KernelFlavor>);
		}
	});
}

template <typename Functor>
void handler::host_task(on_master_node_tag, Functor kernel) {
	if(is_prepass()) {
		dynamic_cast<detail::prepass_handler&>(*this).create_master_node_task();
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule<0>(kernel);
	}
}

template <typename Functor>
void handler::host_task(experimental::collective_tag tag, Functor kernel) {
	if(is_prepass()) {
		dynamic_cast<detail::prepass_handler&>(*this).create_collective_task(tag.m_cgid);
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule_collective(kernel);
	}
}

template <int Dims, typename Functor>
void handler::host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel) {
	if(is_prepass()) {
		const detail::task_geometry geometry{Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset), {1, 1, 1}};
		dynamic_cast<detail::prepass_handler&>(*this).create_host_compute_task(geometry);
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule<Dims>(kernel);
	}
}

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation combiner, const cl::sycl::property_list& prop_list = {}) {
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
	static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
#if CELERITY_WORKAROUND(DPCPP)
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
#if !CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS
	static_assert(detail::constexpr_false<BinaryOperation>, "Reductions are not supported by your SYCL implementation");
#else
	static_assert(!cl::sycl::has_known_identity_v<BinaryOperation, DataT>, "Identity is known to SYCL, remove the identity parameter from reduction()");
	return detail::make_reduction<true>(vars, cgh, combiner, identity, prop_list);
#endif
}

} // namespace celerity
