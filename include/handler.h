#pragma once

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <fmt/format.h>
#include <sycl/sycl.hpp>

#include "buffer.h"
#include "cgf_diagnostics.h"
#include "item.h"
#include "partition.h"
#include "range_mapper.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "version.h"
#include "workaround.h"

namespace celerity {
class handler;
}

namespace celerity::experimental {

/**
 * Constrains the granularity at which a task's global range can be split into chunks.
 *
 * In some situations an output buffer access is only guaranteed to write to non-overlapping subranges
 * if the task is split in a certain way. For example when computing the row-wise sum of a 2D matrix into
 * a 1D vector, a split constraint is required to ensure that each element of the vector is written by
 * exactly one chunk.
 *
 * Another use case is for performance optimization, for example when the creation of lots of small chunks
 * would result in hardware under-utilization and excessive data transfers.
 *
 * Since ND-range parallel_for kernels are already constrained to be split with group size granularity,
 * adding an additional constraint on top results in an effective constraint of LCM(group size, constraint).
 *
 * The constraint (or effective constraint) must evenly divide the global range.
 * This function has no effect when called for a task without a user-provided global range.
 */
template <int Dims>
void constrain_split(handler& cgh, const range<Dims>& constraint);

} // namespace celerity::experimental

namespace celerity {

namespace detail {
	class task_manager;

	handler make_command_group_handler(const task_id tid, const size_t num_collective_nodes);
	std::unique_ptr<task> into_task(handler&& cgh);
	hydration_id add_requirement(handler& cgh, const buffer_id bid, std::unique_ptr<range_mapper_base> rm);
	void add_requirement(handler& cgh, const host_object_id hoid, const experimental::side_effect_order order, const bool is_void);
	void add_reduction(handler& cgh, const reduction_info& rinfo);

	void set_task_name(handler& cgh, const std::string& debug_name);

	struct unnamed_kernel {};

	template <typename KernelName>
	constexpr bool is_unnamed_kernel = std::is_same_v<KernelName, unnamed_kernel>;

	template <typename KernelName>
	std::string kernel_debug_name() {
		return !is_unnamed_kernel<KernelName> ? utils::get_simplified_type_name<KernelName>() : std::string{};
	}

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
		collective_group() noexcept : m_cgid(s_next_cgid++) {}

	  private:
		friend class collective_tag_factory;
		detail::collective_group_id m_cgid;
		inline static detail::collective_group_id s_next_cgid = detail::root_collective_group_id + 1;
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

namespace detail {
	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel(const Kernel& kernel, const sycl::id<std::max(1, Dims)>& s_id, const range<Dims>& global_range, const id<Dims>& global_offset,
	    const id<Dims>& chunk_offset, Reducers&... reducers) {
		kernel(make_item<Dims>(id_cast<Dims>(id<std::max(1, Dims)>(s_id)) + chunk_offset, global_offset, global_range), reducers...);
	}

	template <typename Kernel, int Dims, typename... Reducers>
	inline void invoke_kernel(const Kernel& kernel, const sycl::nd_item<std::max(1, Dims)>& s_item, const range<Dims>& global_range,
	    const id<Dims>& global_offset, const id<Dims>& chunk_offset, const range<Dims>& group_range, const id<Dims>& group_offset, Reducers&... reducers) {
		kernel(make_nd_item<Dims>(s_item, global_range, global_offset, chunk_offset, group_range, group_offset), reducers...);
	}

	template <typename Kernel, int Dims>
	auto bind_simple_kernel(const Kernel& kernel, const range<Dims>& global_range, const id<Dims>& global_offset, const id<Dims>& chunk_offset) {
		return [=](auto s_item_or_id, auto&... reducers) {
			static_assert(std::is_invocable_v<Kernel, celerity::item<Dims>, decltype(reducers)...>,
			    "Kernel function must be invocable with celerity::item<Dims> and as many reducer objects as reductions passed to parallel_for");
			if constexpr(CELERITY_WORKAROUND(DPCPP) && std::is_same_v<sycl::id<Dims>, decltype(s_item_or_id)>) {
				// CELERITY_WORKAROUND_LESS_OR_EQUAL: DPC++ passes a sycl::id instead of a sycl::item to kernels alongside reductions
				invoke_kernel(kernel, s_item_or_id, global_range, global_offset, chunk_offset, reducers...);
			} else {
				invoke_kernel(kernel, s_item_or_id.get_id(), global_range, global_offset, chunk_offset, reducers...);
			}
		};
	}

	template <typename Kernel, int Dims>
	auto bind_nd_range_kernel(const Kernel& kernel, const range<Dims>& global_range, const id<Dims>& global_offset, const id<Dims> chunk_offset,
	    const range<Dims>& group_range, const id<Dims>& group_offset) {
		return [=](sycl::nd_item<std::max(1, Dims)> s_item, auto&... reducers) {
			static_assert(std::is_invocable_v<Kernel, celerity::nd_item<Dims>, decltype(reducers)...>,
			    "Kernel function must be invocable with celerity::nd_item<Dims> or and as many reducer objects as reductions passed to parallel_for");
			invoke_kernel(kernel, s_item, global_range, global_offset, chunk_offset, group_range, group_offset, reducers...);
		};
	}

	template <typename KernelName, typename... Params>
	inline void invoke_sycl_parallel_for(sycl::handler& cgh, Params&&... args) {
		static_assert(CELERITY_FEATURE_UNNAMED_KERNELS || !is_unnamed_kernel<KernelName>,
		    "Your SYCL implementation does not support unnamed kernels, add a kernel name template parameter to this parallel_for invocation");
		if constexpr(detail::is_unnamed_kernel<KernelName>) {
#if CELERITY_FEATURE_UNNAMED_KERNELS // see static_assert above
			cgh.parallel_for(std::forward<Params>(args)...);
#endif
		} else {
			cgh.parallel_for<KernelName>(std::forward<Params>(args)...);
		}
	}

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	class reduction_descriptor;

	template <typename DataT, int Dims, typename BinaryOperation, bool WithExplicitIdentity>
	auto make_sycl_reduction(const reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>& d, void* ptr) {
		if constexpr(WithExplicitIdentity) {
			return sycl::reduction(static_cast<DataT*>(ptr), d.m_identity, d.m_op, sycl::property_list{sycl::property::reduction::initialize_to_identity{}});
		} else {
			return sycl::reduction(static_cast<DataT*>(ptr), d.m_op, sycl::property_list{sycl::property::reduction::initialize_to_identity{}});
		}
	}

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, false /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(buffer_id bid, BinaryOperation combiner, DataT /* identity */, bool include_current_buffer_value)
		    : m_bid(bid), m_op(combiner), m_include_current_buffer_value(include_current_buffer_value) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, false>(const reduction_descriptor&, void*);

		buffer_id m_bid;
		BinaryOperation m_op;
		bool m_include_current_buffer_value;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class reduction_descriptor<DataT, Dims, BinaryOperation, true /* WithExplicitIdentity */> {
	  public:
		reduction_descriptor(buffer_id bid, BinaryOperation combiner, DataT identity, bool include_current_buffer_value)
		    : m_bid(bid), m_op(combiner), m_identity(identity), m_include_current_buffer_value(include_current_buffer_value) {}

	  private:
		friend auto make_sycl_reduction<DataT, Dims, BinaryOperation, true>(const reduction_descriptor&, void*);

		buffer_id m_bid;
		BinaryOperation m_op;
		DataT m_identity{};
		bool m_include_current_buffer_value;
	};

	template <bool WithExplicitIdentity, typename DataT, int Dims, typename BinaryOperation>
	auto make_reduction(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation op, DataT identity, const sycl::property_list& prop_list) {
		if(vars.get_range().size() != 1) {
			// Like SYCL 2020, Celerity only supports reductions to unit-sized buffers. This allows us to avoid tracking different parts of the buffer
			// as distributed_state and pending_reduction_state.
			throw std::runtime_error("Only unit-sized buffers can be reduction targets");
		}

		const auto bid = detail::get_buffer_id(vars);
		const auto include_current_buffer_value = !prop_list.has_property<celerity::property::reduction::initialize_to_identity>();

		const auto rid = detail::runtime::get_instance().create_reduction(detail::make_reducer(op, identity));
		add_reduction(cgh, reduction_info{rid, bid, include_current_buffer_value});

		return detail::reduction_descriptor<DataT, Dims, BinaryOperation, WithExplicitIdentity>{bid, op, identity, include_current_buffer_value};
	}

} // namespace detail

class handler {
  public:
	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(range<Dims> global_range, ReductionsAndKernel&&... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::simple_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(global_range, id<Dims>(),
		    detail::no_local_size{}, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{},
		    std::forward<ReductionsAndKernel>(reductions_and_kernel)...);
	}

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(range<Dims> global_range, id<Dims> global_offset, ReductionsAndKernel&&... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::simple_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(global_range, global_offset,
		    detail::no_local_size{}, std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{},
		    std::forward<ReductionsAndKernel>(reductions_and_kernel)...);
	}

	template <typename KernelName = detail::unnamed_kernel, int Dims, typename... ReductionsAndKernel>
	void parallel_for(celerity::nd_range<Dims> execution_range, ReductionsAndKernel&&... reductions_and_kernel) {
		static_assert(sizeof...(reductions_and_kernel) > 0, "No kernel given");
		parallel_for_reductions_and_kernel<detail::nd_range_kernel_flavor, KernelName, Dims, ReductionsAndKernel...>(execution_range.get_global_range(),
		    execution_range.get_offset(), execution_range.get_local_range(), std::make_index_sequence<sizeof...(reductions_and_kernel) - 1>{},
		    std::forward<ReductionsAndKernel>(reductions_and_kernel)...);
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
	void host_task(on_master_node_tag /* tag */, Functor&& kernel) {
		auto launcher = make_host_task_launcher<0, false>(detail::zeros, 0, std::forward<Functor>(kernel));
		create_master_node_task(std::move(launcher));
	}

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
	void host_task(experimental::collective_tag tag, Functor&& kernel) {
		// FIXME: We should not have to know how the global range is determined for collective tasks to create the launcher
		auto launcher = make_host_task_launcher<1, true>(range<3>{m_num_collective_nodes, 1, 1}, tag.m_cgid, std::forward<Functor>(kernel));
		create_collective_task(tag.m_cgid, std::move(launcher));
	}

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
	void host_task(range<Dims> global_range, id<Dims> global_offset, Functor&& kernel) {
		const detail::task_geometry geometry{
		    Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset), get_constrained_granularity(global_range, range<Dims>(detail::ones))};
		auto launcher = make_host_task_launcher<Dims, false>(detail::range_cast<3>(global_range), 0, std::forward<Functor>(kernel));
		create_host_compute_task(geometry, std::move(launcher));
	}

	/**
	 * Like `host_task(range<Dims> global_range, id<Dims> global_offset, Functor kernel)`, but with a `global_offset` of zero.
	 */
	template <int Dims, typename Functor>
	void host_task(range<Dims> global_range, Functor&& kernel) {
		host_task(global_range, {}, std::forward<Functor>(kernel));
	}

  private:
	friend handler detail::make_command_group_handler(const detail::task_id tid, const size_t num_collective_nodes);
	friend std::unique_ptr<detail::task> detail::into_task(handler&& cgh);
	friend detail::hydration_id detail::add_requirement(handler& cgh, const detail::buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm);
	friend void detail::add_requirement(handler& cgh, const detail::host_object_id hoid, const experimental::side_effect_order order, const bool is_void);
	friend void detail::add_reduction(handler& cgh, const detail::reduction_info& rinfo);
	template <int Dims>
	friend void experimental::constrain_split(handler& cgh, const range<Dims>& constraint);
	template <typename Hint>
	friend void experimental::hint(handler& cgh, Hint&& hint);
	friend void detail::set_task_name(handler& cgh, const std::string& debug_name);

	detail::task_id m_tid;
	detail::buffer_access_map m_access_map;
	detail::side_effect_map m_side_effects;
	size_t m_non_void_side_effects_count = 0;
	detail::reduction_set m_reductions;
	std::unique_ptr<detail::task> m_task = nullptr;
	size_t m_num_collective_nodes;
	detail::hydration_id m_next_accessor_hydration_id = 1;
	std::optional<std::string> m_usr_def_task_name;
	range<3> m_split_constraint = detail::ones;
	std::vector<std::unique_ptr<detail::hint_base>> m_hints;

	handler(detail::task_id tid, size_t num_collective_nodes) : m_tid(tid), m_num_collective_nodes(num_collective_nodes) {}

	template <typename KernelFlavor, typename KernelName, int Dims, typename... ReductionsAndKernel, size_t... ReductionIndices>
	void parallel_for_reductions_and_kernel(range<Dims> global_range, id<Dims> global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_size, std::index_sequence<ReductionIndices...> indices,
	    ReductionsAndKernel&&... kernel_and_reductions) {
		auto args_tuple = std::forward_as_tuple(kernel_and_reductions...);
		auto&& kernel = std::get<sizeof...(kernel_and_reductions) - 1>(args_tuple);
		parallel_for_kernel_and_reductions<KernelFlavor, KernelName>(
		    global_range, global_offset, local_size, std::forward<decltype(kernel)>(kernel), std::get<ReductionIndices>(args_tuple)...);
	}

	template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, typename... Reductions>
	void parallel_for_kernel_and_reductions(range<Dims> global_range, id<Dims> global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel&& kernel, Reductions&... reductions) {
		range<3> granularity = {1, 1, 1};
		if constexpr(detail::kernel_flavor_traits<KernelFlavor, Dims>::has_local_size) {
			for(int d = 0; d < Dims; ++d) {
				granularity[d] = local_range[d];
			}
		}
		const detail::task_geometry geometry{Dims, detail::range_cast<3>(global_range), detail::id_cast<3>(global_offset),
		    get_constrained_granularity(global_range, detail::range_cast<Dims>(granularity))};
		auto launcher = make_device_kernel_launcher<KernelFlavor, KernelName, Dims>(
		    global_range, global_offset, local_range, std::forward<Kernel>(kernel), std::index_sequence_for<Reductions...>(), reductions...);
		create_device_compute_task(geometry, detail::kernel_debug_name<KernelName>(), std::move(launcher));
	}

	[[nodiscard]] detail::hydration_id add_requirement(const detail::buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) {
		assert(m_task == nullptr);
		m_access_map.add_access(bid, std::move(rm));
		return m_next_accessor_hydration_id++;
	}

	void add_requirement(const detail::host_object_id hoid, const experimental::side_effect_order order, const bool is_void) {
		assert(m_task == nullptr);
		m_side_effects.add_side_effect(hoid, order);
		if(!is_void) { m_non_void_side_effects_count++; }
	}

	void add_reduction(const detail::reduction_info& rinfo) {
		assert(m_task == nullptr);
		m_reductions.push_back(rinfo);
	}

	template <int Dims>
	void experimental_constrain_split(const range<Dims>& constraint) {
		assert(m_task == nullptr);
		m_split_constraint = detail::range_cast<3>(constraint);
	}

	template <typename Hint>
	void experimental_hint(Hint&& hint) {
		static_assert(std::is_base_of_v<detail::hint_base, std::decay_t<Hint>>, "Hint must extend hint_base");
		static_assert(std::is_move_constructible_v<Hint>, "Hint must be move-constructible");
		for(auto& h : m_hints) {
			// We currently don't allow more than one hint of the same type for simplicity; this could be loosened in the future.
			auto& hr = *h; // Need to do this here to avoid -Wpotentially-evaluated-expression
			if(typeid(hr) == typeid(hint)) { throw std::runtime_error("Providing more than one hint of the same type is not allowed"); }
			h->validate(hint);
		}
		m_hints.emplace_back(std::make_unique<std::decay_t<Hint>>(std::forward<Hint>(hint)));
	}

	template <int Dims>
	range<3> get_constrained_granularity(const range<Dims>& global_size, const range<Dims>& granularity) const {
		range<3> result = detail::range_cast<3>(granularity);
		for(int i = 0; i < Dims; ++i) {
			const auto lcm = std::lcm(granularity[i], m_split_constraint[i]);
			if(lcm == 0) { throw std::runtime_error("Split constraint cannot be 0"); }
			result[i] = lcm;
		}
		if(global_size % detail::range_cast<Dims>(result) != range<Dims>(detail::zeros)) {
			throw std::runtime_error(fmt::format("The{}split constraint {} does not evenly divide the kernel global size {}",
			    granularity.size() > 1 ? " effective " : " ", detail::range_cast<Dims>(result), global_size));
		}
		return result;
	}

	void create_host_compute_task(const detail::task_geometry& geometry, detail::host_task_launcher launcher) {
		assert(m_task == nullptr);
		if(geometry.global_size.size() == 0) {
			// TODO this can be easily supported by not creating a task in case the execution range is empty
			throw std::runtime_error{"The execution range of distributed host tasks must have at least one item"};
		}
		m_task =
		    detail::task::make_host_compute(m_tid, geometry, std::move(launcher), std::move(m_access_map), std::move(m_side_effects), std::move(m_reductions));

		m_task->set_debug_name(m_usr_def_task_name.value_or(""));
	}

	void create_device_compute_task(const detail::task_geometry& geometry, const std::string& debug_name, detail::device_kernel_launcher launcher) {
		assert(m_task == nullptr);
		if(geometry.global_size.size() == 0) {
			// TODO unless reductions are involved, this can be easily supported by not creating a task in case the execution range is empty.
			// Edge case: If the task includes reductions that specify property::reduction::initialize_to_identity, we need to create a task that sets
			// the buffer state to an empty pending_reduction_state in the graph_generator. This will cause a trivial reduction_command to be generated on
			// each node that reads from the reduction output buffer, initializing it to the identity value locally.
			throw std::runtime_error{"The execution range of device tasks must have at least one item"};
		}
		// Note that cgf_diagnostics has a similar check, but we don't catch void side effects there.
		if(!m_side_effects.empty()) { throw std::runtime_error{"Side effects cannot be used in device kernels"}; }
		m_task = detail::task::make_device_compute(m_tid, geometry, std::move(launcher), std::move(m_access_map), std::move(m_reductions));

		m_task->set_debug_name(m_usr_def_task_name.value_or(debug_name));
	}

	void create_collective_task(const detail::collective_group_id cgid, detail::host_task_launcher launcher) {
		assert(m_task == nullptr);
		m_task = detail::task::make_collective(m_tid, cgid, m_num_collective_nodes, std::move(launcher), std::move(m_access_map), std::move(m_side_effects));

		m_task->set_debug_name(m_usr_def_task_name.value_or(""));
	}

	void create_master_node_task(detail::host_task_launcher launcher) {
		assert(m_task == nullptr);
		m_task = detail::task::make_master_node(m_tid, std::move(launcher), std::move(m_access_map), std::move(m_side_effects));

		m_task->set_debug_name(m_usr_def_task_name.value_or(""));
	}

	template <typename KernelFlavor, typename KernelName, int Dims, typename Kernel, size_t... ReductionIndices, typename... Reductions>
	detail::device_kernel_launcher make_device_kernel_launcher(const range<Dims>& global_range, const id<Dims>& global_offset,
	    typename detail::kernel_flavor_traits<KernelFlavor, Dims>::local_size_type local_range, Kernel&& kernel,
	    std::index_sequence<ReductionIndices...> /* indices */, Reductions... reductions) {
		static_assert(std::is_copy_constructible_v<std::decay_t<Kernel>>, "Kernel functor must be copyable"); // Required for hydration

		// Check whether all accessors are being captured by value etc.
		// Although the diagnostics should always be available, we currently disable them for some test cases.
		if(detail::cgf_diagnostics::is_available()) { detail::cgf_diagnostics::get_instance().check<target::device>(kernel, m_access_map); }

		return [=](sycl::handler& sycl_cgh, const detail::box<3>& execution_range, const std::vector<void*>& reduction_ptrs) {
			constexpr int sycl_dims = std::max(1, Dims);
			if constexpr(std::is_same_v<KernelFlavor, detail::simple_kernel_flavor>) {
				const auto sycl_global_range = sycl::range<sycl_dims>(detail::range_cast<sycl_dims>(execution_range.get_range()));
				detail::invoke_sycl_parallel_for<KernelName>(sycl_cgh, sycl_global_range,
				    detail::make_sycl_reduction(reductions, reduction_ptrs[ReductionIndices])...,
				    detail::bind_simple_kernel(kernel, global_range, global_offset, detail::id_cast<Dims>(execution_range.get_offset())));
			} else if constexpr(std::is_same_v<KernelFlavor, detail::nd_range_kernel_flavor>) {
				const auto sycl_global_range = sycl::range<sycl_dims>(detail::range_cast<sycl_dims>(execution_range.get_range()));
				const auto sycl_local_range = sycl::range<sycl_dims>(detail::range_cast<sycl_dims>(local_range));
				detail::invoke_sycl_parallel_for<KernelName>(sycl_cgh, sycl::nd_range{sycl_global_range, sycl_local_range},
				    detail::make_sycl_reduction(reductions, reduction_ptrs[ReductionIndices])...,
				    detail::bind_nd_range_kernel(kernel, global_range, global_offset, detail::id_cast<Dims>(execution_range.get_offset()),
				        global_range / local_range, detail::id_cast<Dims>(execution_range.get_offset()) / local_range));
			} else {
				static_assert(detail::constexpr_false<KernelFlavor>);
			}
		};
	}

	template <int Dims, bool Collective, typename Kernel>
	detail::host_task_launcher make_host_task_launcher(const range<3>& global_range, const detail::collective_group_id cgid, Kernel&& kernel) {
		static_assert(Collective || std::is_invocable_v<Kernel> || std::is_invocable_v<Kernel, const partition<Dims>>,
		    "Kernel for host task must be invocable with either no arguments or a celerity::partition<Dims>");
		static_assert(!Collective || std::is_invocable_v<Kernel> || std::is_invocable_v<Kernel, const experimental::collective_partition>,
		    "Kernel for collective host task must be invocable with either no arguments or a celerity::experimental::collective_partition");
		static_assert(std::is_copy_constructible_v<std::decay_t<Kernel>>, "Kernel functor must be copyable"); // Required for hydration
		static_assert(Dims >= 0);

		// Check whether all accessors are being captured by value etc.
		// Although the diagnostics should always be available, we currently disable them for some test cases.
		if(detail::cgf_diagnostics::is_available()) {
			detail::cgf_diagnostics::get_instance().check<target::host_task>(kernel, m_access_map, m_non_void_side_effects_count);
		}

		return [kernel, global_range](const detail::box<3>& execution_range, const detail::communicator* collective_comm) {
			(void)global_range;
			(void)collective_comm;
			if constexpr(Dims > 0) {
				if constexpr(Collective) {
					static_assert(Dims == 1);
					assert(collective_comm != nullptr);
					const auto part =
					    detail::make_collective_partition(detail::range_cast<1>(global_range), detail::box_cast<1>(execution_range), *collective_comm);
					kernel(part);
				} else {
					const auto part = detail::make_partition<Dims>(detail::range_cast<Dims>(global_range), detail::box_cast<Dims>(execution_range));
					kernel(part);
				}
			} else if constexpr(std::is_invocable_v<Kernel, const partition<0>&>) {
				(void)execution_range;
				const auto part = detail::make_partition<0>(range<0>(), subrange<0>());
				kernel(part);
			} else {
				(void)execution_range;
				kernel();
			}
		};
	}

	std::unique_ptr<detail::task> into_task() && {
		assert(m_task != nullptr);
		for(auto& h : m_hints) {
			m_task->add_hint(std::move(h));
		}
		return std::move(m_task);
	}
};

namespace detail {

	inline handler make_command_group_handler(const detail::task_id tid, const size_t num_collective_nodes) { return handler(tid, num_collective_nodes); }

	inline std::unique_ptr<detail::task> into_task(handler&& cgh) { return std::move(cgh).into_task(); }

	[[nodiscard]] inline hydration_id add_requirement(handler& cgh, const buffer_id bid, std::unique_ptr<range_mapper_base> rm) {
		return cgh.add_requirement(bid, std::move(rm));
	}

	inline void add_requirement(handler& cgh, const host_object_id hoid, const experimental::side_effect_order order, const bool is_void) {
		return cgh.add_requirement(hoid, order, is_void);
	}

	inline void add_reduction(handler& cgh, const detail::reduction_info& rinfo) { return cgh.add_reduction(rinfo); }

	inline void set_task_name(handler& cgh, const std::string& debug_name) { cgh.m_usr_def_task_name = {debug_name}; }

	// TODO: The _impl functions in detail only exist during the grace period for deprecated reductions on const buffers; move outside again afterwards.
	template <typename DataT, int Dims, typename BinaryOperation>
	auto reduction_impl(const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation combiner, const sycl::property_list& prop_list = {}) {
		static_assert(sycl::has_known_identity_v<BinaryOperation, DataT>,
		    "Celerity does not currently support reductions without an identity. Either specialize "
		    "sycl::known_identity or use the reduction() overload taking an identity at runtime");
		return detail::make_reduction<false>(vars, cgh, combiner, sycl::known_identity_v<BinaryOperation, DataT>, prop_list);
	}

	template <typename DataT, int Dims, typename BinaryOperation>
	auto reduction_impl(
	    const buffer<DataT, Dims>& vars, handler& cgh, const DataT identity, BinaryOperation combiner, const sycl::property_list& prop_list = {}) {
		static_assert(!sycl::has_known_identity_v<BinaryOperation, DataT>, "Identity is known to SYCL, remove the identity parameter from reduction()");
		return detail::make_reduction<true>(vars, cgh, combiner, identity, prop_list);
	}

} // namespace detail

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation combiner, const sycl::property_list& prop_list = {}) {
	return detail::reduction_impl(vars, cgh, combiner, prop_list);
}

template <typename DataT, int Dims, typename BinaryOperation>
auto reduction(buffer<DataT, Dims>& vars, handler& cgh, const DataT identity, BinaryOperation combiner, const sycl::property_list& prop_list = {}) {
	return detail::reduction_impl(vars, cgh, identity, combiner, prop_list);
}

template <typename DataT, int Dims, typename BinaryOperation>
[[deprecated("Creating reduction from const buffer is deprecated, capture buffer by reference instead")]] auto reduction(
    const buffer<DataT, Dims>& vars, handler& cgh, BinaryOperation combiner, const sycl::property_list& prop_list = {}) {
	return detail::reduction_impl(vars, cgh, combiner, prop_list);
}

template <typename DataT, int Dims, typename BinaryOperation>
[[deprecated("Creating reduction from const buffer is deprecated, capture buffer by reference instead")]] auto reduction(
    const buffer<DataT, Dims>& vars, handler& cgh, const DataT identity, BinaryOperation combiner, const sycl::property_list& prop_list = {}) {
	return detail::reduction_impl(vars, cgh, identity, combiner, prop_list);
}

} // namespace celerity

namespace celerity::experimental {
template <int Dims>
void constrain_split(handler& cgh, const range<Dims>& constraint) {
	cgh.experimental_constrain_split(constraint);
}

template <typename Hint>
void hint(handler& cgh, Hint&& hint) {
	cgh.experimental_hint(std::forward<Hint>(hint));
}
} // namespace celerity::experimental
