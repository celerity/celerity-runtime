#pragma once

#include <regex>
#include <string>

#include <CL/sycl.hpp>
#include <boost/type_index.hpp>
#include <spdlog/fmt/fmt.h>

#include "range_mapper.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "workaround.h"

namespace celerity {

namespace detail {
	class device_queue;
	class task_manager;
} // namespace detail

class compute_prepass_handler {
  public:
	compute_prepass_handler(detail::compute_task& task) : task(task) { debug_name = fmt::format("task{}", task.get_id()); }

	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims> global_size, const Functor& kernel) {
		parallel_for<Name, Functor, Dims>(global_size, cl::sycl::id<Dims>(), kernel);
	}

	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, const Functor&) {
		task.set_dimensions(Dims);
		task.set_global_size(detail::range_cast<3>(global_size));
		task.set_global_offset(detail::id_cast<3>(global_offset));

		// DEBUG: Find nice name for kernel (regex is probably not super portable)
		auto qualified_name = boost::typeindex::type_id<Name*>().pretty_name();
		std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
		std::smatch matches;
		std::regex_search(qualified_name, matches, name_regex);
		debug_name = matches.size() > 0 ? matches[1] : qualified_name;
		task.set_debug_name(debug_name);
	}

	/**
	 * @internal
	 */
	void require(detail::buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) const;

  private:
	detail::compute_task& task;
	std::string debug_name;
	cl::sycl::range<3> global_size;
};

namespace detail {
	template <class Name, bool NdRange>
	class wrapped_kernel_name {};
} // namespace detail

class compute_livepass_handler {
  public:
	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims> global_size, Functor kernel) {
		parallel_for<Name, Functor, Dims>(global_size, cl::sycl::id<Dims>(), kernel);
	}

	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims>, cl::sycl::id<Dims>, Functor kernel) {
#if WORKAROUND_COMPUTECPP
		// As of ComputeCpp 1.0.2 the PTX backend has problems with kernel invocations that have an offset.
		// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-98 (psalz).
		// To work around this, instead of passing an offset to SYCL, we simply add it to the item that is passed to the kernel.
		const cl::sycl::id<Dims> ccpp_ptx_workaround_offset = {};
#else
		const cl::sycl::id<Dims> ccpp_ptx_workaround_offset = detail::id_cast<Dims>(sr.offset);
#endif
		if(forced_work_group_size == 0) {
#if WORKAROUND_COMPUTECPP
			sycl_handler->parallel_for<detail::wrapped_kernel_name<Name, false>>(
			    detail::range_cast<Dims>(sr.range), ccpp_ptx_workaround_offset, [=, sr = this->sr](cl::sycl::item<Dims> item) {
				    const cl::sycl::id<Dims> ptx_workaround_id = detail::range_cast<Dims>(item.get_id()) + detail::id_cast<Dims>(sr.offset);
				    const auto item_base = cl::sycl::detail::item_base(ptx_workaround_id, sr.range, ccpp_ptx_workaround_offset);
				    const auto offset_item = cl::sycl::item<Dims, true>(item_base);
				    kernel(offset_item);
			    });
#else
			sycl_handler->parallel_for<detail::wrapped_kernel_name<Name, false>>(detail::range_cast<Dims>(sr.range), detail::id_cast<Dims>(sr.offset), kernel);
#endif
		} else {
			const auto fwgs = forced_work_group_size;
			const auto nd_range = cl::sycl::nd_range<Dims>(detail::range_cast<Dims>(sr.range),
			    detail::range_cast<Dims>(cl::sycl::range<3>(fwgs, Dims > 1 ? fwgs : 1, Dims == 3 ? fwgs : 1)), ccpp_ptx_workaround_offset);
			sycl_handler->parallel_for<detail::wrapped_kernel_name<Name, true>>(nd_range, [=, sr = this->sr](cl::sycl::nd_item<Dims> item) {
#if WORKAROUND_HIPSYCL
				kernel(cl::sycl::item<Dims>(cl::sycl::detail::item_impl<Dims>(item.get_global())));
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

	cl::sycl::handler& get_sycl_handler() const { return *sycl_handler; }

	template <int BufferDims, typename RangeMapper>
	subrange<BufferDims> apply_range_mapper(RangeMapper rm) const {
		switch(task.get_dimensions()) {
		case 1: return rm(chunk<1>(detail::id_cast<1>(sr.offset), detail::range_cast<1>(sr.range), detail::range_cast<1>(task.get_global_size())));
		case 2: return rm(chunk<2>(detail::id_cast<2>(sr.offset), detail::range_cast<2>(sr.range), detail::range_cast<2>(task.get_global_size())));
		case 3: return rm(chunk<3>(detail::id_cast<3>(sr.offset), detail::range_cast<3>(sr.range), detail::range_cast<3>(task.get_global_size())));
		default: assert(false);
		}
		return {};
	}

  private:
	friend class detail::device_queue;

	const detail::compute_task& task;
	// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
	subrange<3> sr;
	cl::sycl::handler* sycl_handler;
	// This is a workaround until we get proper nd_item overloads for parallel_for into the API.
	size_t forced_work_group_size;

	// The handler does not take ownership of the sycl_handler, but expects it to
	// exist for the duration of it's lifetime.
	compute_livepass_handler(const detail::compute_task& task, subrange<3> sr, cl::sycl::handler* sycl_handler, size_t forced_work_group_size)
	    : task(task), sr(sr), sycl_handler(sycl_handler), forced_work_group_size(forced_work_group_size) {}
};

class master_access_prepass_handler {
  public:
	master_access_prepass_handler(detail::master_access_task& task) : task(task) {}

	template <typename MAF>
	void run(MAF maf) const {
		// nop
	}

	/**
	 * @internal
	 */
	void require(cl::sycl::access::mode mode, detail::buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) const;

  private:
	detail::master_access_task& task;
};

class master_access_livepass_handler {
  public:
	template <typename MAF>
	void run(MAF maf) const {
		maf();
	}
};

} // namespace celerity
