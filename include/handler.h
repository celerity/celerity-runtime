#pragma once

#include <regex>
#include <type_traits>
#include <typeinfo>

#include <CL/sycl.hpp>
#include <spdlog/fmt/fmt.h>

#include "device_queue.h"
#include "host_queue.h"
#include "range_mapper.h"
#include "ranges.h"
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

	template <typename Name, int Dims, typename Functor>
	void parallel_for(cl::sycl::range<Dims> global_size, const Functor& kernel) {
		parallel_for<Name, Dims, Functor>(global_size, cl::sycl::id<Dims>(), kernel);
	}

	template <typename Name, int Dims, typename Functor>
	void parallel_for(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, const Functor& kernel);

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
	void host_task(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel);

	/**
	 * Like `host_task(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel)`, but with a `global_offset` of zero.
	 */
	template <int Dims, typename Functor>
	void host_task(cl::sycl::range<Dims> global_size, Functor task) {
		host_task(global_size, {}, task);
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

		void create_host_compute_task(int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset) override {
			assert(task == nullptr);
			task = detail::task::make_host_compute(tid, dimensions, global_size, global_offset, std::move(cgf), std::move(access_map));
		}

		void create_device_compute_task(int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset, std::string debug_name) override {
			assert(task == nullptr);
			task = detail::task::make_device_compute(tid, dimensions, global_size, global_offset, std::move(cgf), std::move(access_map), std::move(debug_name));
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

	  protected:
		live_pass_handler(std::shared_ptr<const class task> task, subrange<3> sr) : task(std::move(task)), sr(sr) {}

		std::shared_ptr<const class task> task = nullptr;

		// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
		subrange<3> sr;
	};

	class live_pass_host_handler final : public live_pass_handler {
	  public:
		live_pass_host_handler(std::shared_ptr<const class task> task, subrange<3> sr, host_queue& queue)
		    : live_pass_handler(std::move(task), sr), queue(&queue) {}

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

	// The current mechanism for hydrating the SYCL placeholder accessors inside Celerity accessors requires that the kernel functor
	// capturing those accessors is copied at least once during submission (see also live_pass_device_handler::submit_to_sycl).
	// As of SYCL 2020 kernel functors are passed as const references, so we have to do this manually here.
	template <typename Functor>
	Functor copy_kernel(const Functor& kernel) {
		return Functor{kernel};
	}

	class live_pass_device_handler final : public live_pass_handler {
	  public:
		live_pass_device_handler(std::shared_ptr<const class task> task, subrange<3> sr, device_queue& d_queue)
		    : live_pass_handler(std::move(task), sr), d_queue(&d_queue) {}

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

} // namespace detail

template <typename Name, int Dims, typename Functor>
void handler::parallel_for(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, const Functor& kernel) {
	if(is_prepass()) {
		return create_device_compute_task(Dims, detail::range_cast<3>(global_size), detail::id_cast<3>(global_offset), detail::kernel_debug_name<Name>());
	}

	auto& device_handler = dynamic_cast<detail::live_pass_device_handler&>(*this);
	const auto sr = device_handler.get_iteration_range();

	device_handler.submit_to_sycl([&](cl::sycl::handler& cgh) {
		cgh.parallel_for<Name>(detail::range_cast<Dims>(sr.range), detail::id_cast<Dims>(sr.offset), detail::copy_kernel(kernel));
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
void handler::host_task(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel) {
	if(is_prepass()) {
		create_host_compute_task(Dims, detail::range_cast<3>(global_size), detail::id_cast<3>(global_offset));
	} else {
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule<Dims>(kernel);
	}
}

} // namespace celerity
