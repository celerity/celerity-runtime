#pragma once

#include <regex>
#include <type_traits>

#include <CL/sycl.hpp>
#include <boost/type_index.hpp>
#include <spdlog/fmt/fmt.h>

#include "host_queue.h"
#include "range_mapper.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "workaround.h"

namespace celerity {

template <int Dims>
class host_chunk;
class handler;

namespace detail {
	class device_queue;
	class task_manager;
	class collective_spec;
	class prepass_handler;

	template <class Name, bool NdRange>
	class wrapped_kernel_name {};

	inline bool is_prepass_handler(const handler& cgh);
	inline execution_target get_handler_execution_target(const handler& cgh);

	template <typename Name>
	std::string kernel_debug_name() {
		// DEBUG: Find nice name for kernel (regex is probably not super portable)
		auto qualified_name = boost::typeindex::type_id<Name*>().pretty_name();
		std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
		std::smatch matches;
		std::regex_search(qualified_name, matches, name_regex);
		return matches.size() > 0 ? matches[1] : qualified_name;
	}
} // namespace detail

namespace experimental {
	class collective_group {
	  public:
		collective_group() : cgid(next_cgid++) {}

	  private:
		friend class detail::collective_spec;
		detail::collective_group_id cgid;
		inline static size_t next_cgid = 2;
	};

	class collective_tag {
	  private:
		friend class detail::collective_spec;
		friend class celerity::handler;
		collective_tag(detail::collective_group_id cgid) : cgid(cgid) {}
		detail::collective_group_id cgid;
	};
} // namespace experimental

class on_master_node_tag {};

class detail::collective_spec {
  public:
	operator experimental::collective_tag() const { return {1}; }
	experimental::collective_tag operator()(experimental::collective_group cg) const { return cg.cgid; }
};

namespace experimental {
	inline constexpr detail::collective_spec collective;
}
inline constexpr on_master_node_tag on_master_node;

class handler {
  public:
	virtual ~handler() = default;

	template <typename Name, int Dims, typename Functor>
	void parallel_for(cl::sycl::range<Dims> global_size, Functor kernel) {
		parallel_for<Name, Dims, Functor>(global_size, cl::sycl::id<Dims>(), kernel);
	}

	template <typename Name, int Dims, typename Functor>
	void parallel_for(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel);

	template <int Dims, typename Functor>
	void host_task(cl::sycl::range<Dims> global_size, Functor task) {
		host_task(global_size, {}, task);
	}

	template <typename Functor>
	void host_task(on_master_node_tag, Functor kernel);

	template <typename Functor>
	void host_task(experimental::collective_tag tag, Functor kernel);

	template <int Dims, typename Functor>
	void host_task(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel);

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
			switch(task->get_dimensions()) {
			case 0:
				[[fallthrough]]; // cl::sycl::range is not defined for the 0d case, but since only constant range mappers are useful in the 0d-kernel case
				                 // anyway,
				                 // we require range mappers to take at least 1d subranges
			case 1:
				return invoke_range_mapper(
				    rm, chunk<1>(detail::id_cast<1>(sr.offset), detail::range_cast<1>(sr.range), detail::range_cast<1>(task->get_global_size())), buffer_range);
			case 2:
				return invoke_range_mapper(
				    rm, chunk<2>(detail::id_cast<2>(sr.offset), detail::range_cast<2>(sr.range), detail::range_cast<2>(task->get_global_size())), buffer_range);
			case 3:
				return invoke_range_mapper(
				    rm, chunk<3>(detail::id_cast<3>(sr.offset), detail::range_cast<3>(sr.range), detail::range_cast<3>(task->get_global_size())), buffer_range);
			default: assert(false);
			}
			return {};
		}

		subrange<3> get_iteration_range() { return sr; }

	  protected:
		live_pass_handler(std::shared_ptr<const class task> task, subrange<3> sr) : task(std::move(task)), sr(sr) { assert(this->task != nullptr); }

		// The handler does not take ownership of the sycl_handler, but expects it to
		// exist for the duration of it's lifetime.
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
			future = queue->submit(task->get_collective_group_id(), [kernel, global_size = task->get_global_size(), sr = sr](MPI_Comm comm) {
				if constexpr(Dims > 0 || std::is_invocable_v<Kernel, const partition<0>&>) {
					const auto part = make_partition<Dims>(global_size, sr, comm);
					kernel(part);
				} else {
					kernel();
				}
			});
		}

		std::future<host_queue::execution_info> into_future() { return std::move(future); }

	  private:
		host_queue* queue;
		std::future<host_queue::execution_info> future;
	};

	class live_pass_device_handler final : public live_pass_handler {
	  public:
		// The handler does not take ownership of the sycl_handler, but expects it to
		// exist for the duration of it's lifetime.
		live_pass_device_handler(std::shared_ptr<const class task> task, subrange<3> sr, cl::sycl::handler& sycl_handler, size_t forced_work_group_size)
		    : live_pass_handler(std::move(task), sr), sycl_handler(&sycl_handler), forced_work_group_size(forced_work_group_size) {}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::access::target Target, cl::sycl::access::placeholder IsPlaceholder>
		void require_accessor(cl::sycl::accessor<DataT, Dims, Mode, Target, IsPlaceholder>& accessor) {
			sycl_handler->require(accessor);
		}

		cl::sycl::handler& get_sycl_handler() const { return *sycl_handler; }

		// This is a workaround until we get proper nd_item overloads for parallel_for into the API.
		size_t get_forced_work_group_size() { return forced_work_group_size; }

	  private:
		cl::sycl::handler* sycl_handler = nullptr;
		// This is a workaround until we get proper nd_item overloads for parallel_for into the API.
		size_t forced_work_group_size = 0;
	};

} // namespace detail

template <typename Name, int Dims, typename Functor>
void handler::parallel_for(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel) {
	if(is_prepass()) {
		return create_device_compute_task(Dims, detail::range_cast<3>(global_size), detail::id_cast<3>(global_offset), detail::kernel_debug_name<Name>());
	}

	auto& device_handler = dynamic_cast<detail::live_pass_device_handler&>(*this);
	auto& sycl_handler = device_handler.get_sycl_handler();
	const auto fwgs = device_handler.get_forced_work_group_size();
	const auto sr = device_handler.get_iteration_range();

#if WORKAROUND_COMPUTECPP
	// As of ComputeCpp 1.1.5 the PTX backend has problems with kernel invocations that have an offset.
	// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-98 (psalz).
	// To work around this, instead of passing an offset to SYCL, we simply add it to the item that is passed to the kernel.
	const cl::sycl::id<Dims> ccpp_ptx_workaround_offset = {};
#else
	const cl::sycl::id<Dims> ccpp_ptx_workaround_offset = detail::id_cast<Dims>(sr.offset);
#endif
	if(fwgs == 0) {
#if WORKAROUND_COMPUTECPP
		sycl_handler.parallel_for<detail::wrapped_kernel_name<Name, false>>(
		    detail::range_cast<Dims>(sr.range), ccpp_ptx_workaround_offset, [=](cl::sycl::item<Dims> item) {
			    const cl::sycl::id<Dims> ptx_workaround_id = detail::range_cast<Dims>(item.get_id()) + detail::id_cast<Dims>(sr.offset);
			    const auto item_base = cl::sycl::detail::item_base(ptx_workaround_id, sr.range, ccpp_ptx_workaround_offset);
			    const auto offset_item = cl::sycl::item<Dims, true>(item_base);
			    kernel(offset_item);
		    });
#else
		sycl_handler.parallel_for<detail::wrapped_kernel_name<Name, false>>(detail::range_cast<Dims>(sr.range), detail::id_cast<Dims>(sr.offset), kernel);
#endif
	} else {
		const auto nd_range = cl::sycl::nd_range<Dims>(detail::range_cast<Dims>(sr.range),
		    detail::range_cast<Dims>(cl::sycl::range<3>(fwgs, Dims > 1 ? fwgs : 1, Dims == 3 ? fwgs : 1)), ccpp_ptx_workaround_offset);
		sycl_handler.parallel_for<detail::wrapped_kernel_name<Name, true>>(nd_range, [=](cl::sycl::nd_item<Dims> item) {
#if WORKAROUND_HIPSYCL
			kernel(cl::sycl::item<Dims>(
			    cl::sycl::detail::make_item<Dims>(item.get_global_id(), detail::range_cast<Dims>(sr.range), detail::id_cast<Dims>(sr.offset))));
#elif WORKAROUND_COMPUTECPP
			const cl::sycl::id<Dims> ptx_workaround_id = detail::range_cast<Dims>(item.get_global_id()) + detail::id_cast<Dims>(sr.offset);
			const auto item_base = cl::sycl::detail::item_base(ptx_workaround_id, sr.range, ccpp_ptx_workaround_offset);
			const auto offset_item = cl::sycl::item<Dims, true>(item_base);
			kernel(offset_item);
#else
#error Unsupported SYCL implementation
#endif
		});
	}
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
		dynamic_cast<detail::live_pass_host_handler&>(*this).schedule<1>(kernel);
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
