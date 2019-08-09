#pragma once

#include <regex>
#include <type_traits>

#include <CL/sycl.hpp>
#include <boost/type_index.hpp>
#define FMT_HEADER_ONLY
#include <spdlog/fmt/fmt.h>

#include "range_mapper.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "workaround.h"

namespace celerity {

class handler;

namespace detail {
	class device_queue;
	class task_manager;
	class master_access_job;

	template <class Name, bool NdRange>
	class wrapped_kernel_name {};

	inline bool is_prepass_handler(const handler& cgh);
	inline task_type get_handler_type(const handler& cgh);

	// Helper type so we can transfer all required information from a compute_task_handler
	// to the base class handler without having to do multiple virtual function calls.
	// (Not pretty but it works...)
	struct compute_task_exec_context {
		cl::sycl::handler* sycl_handler;
		subrange<3> sr;
		size_t forced_work_group_size;
	};

} // namespace detail

class handler {
  public:
	virtual ~handler() = default;

	template <typename Name, int Dims, typename Functor>
	void parallel_for(cl::sycl::range<Dims> global_size, Functor kernel) {
		assert(task_type == detail::task_type::COMPUTE);
		parallel_for<Name, Dims, Functor>(global_size, cl::sycl::id<Dims>(), kernel);
	}

	template <typename Name, int Dims, typename Functor>
	void parallel_for(cl::sycl::range<Dims> global_size, cl::sycl::id<Dims> global_offset, Functor kernel) {
		assert(task_type == detail::task_type::COMPUTE);
		if(is_prepass()) {
			// DEBUG: Find nice name for kernel (regex is probably not super portable)
			auto qualified_name = boost::typeindex::type_id<Name*>().pretty_name();
			std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
			std::smatch matches;
			std::regex_search(qualified_name, matches, name_regex);
			auto debug_name = matches.size() > 0 ? matches[1] : qualified_name;
			set_compute_task_data(Dims, detail::range_cast<3>(global_size), detail::id_cast<3>(global_offset), debug_name);
			return;
		}

		auto exec_ctx = get_compute_task_exec_context();
		const auto sycl_handler = exec_ctx.sycl_handler;
		const auto fwgs = exec_ctx.forced_work_group_size;
		const auto sr = exec_ctx.sr;

#if WORKAROUND_COMPUTECPP
		// As of ComputeCpp 1.0.2 the PTX backend has problems with kernel invocations that have an offset.
		// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-98 (psalz).
		// To work around this, instead of passing an offset to SYCL, we simply add it to the item that is passed to the kernel.
		const cl::sycl::id<Dims> ccpp_ptx_workaround_offset = {};
#else
		const cl::sycl::id<Dims> ccpp_ptx_workaround_offset = detail::id_cast<Dims>(sr.offset);
#endif
		if(fwgs == 0) {
#if WORKAROUND_COMPUTECPP
			sycl_handler->parallel_for<detail::wrapped_kernel_name<Name, false>>(
			    detail::range_cast<Dims>(sr.range), ccpp_ptx_workaround_offset, [=](cl::sycl::item<Dims> item) {
				    const cl::sycl::id<Dims> ptx_workaround_id = detail::range_cast<Dims>(item.get_id()) + detail::id_cast<Dims>(sr.offset);
				    const auto item_base = cl::sycl::detail::item_base(ptx_workaround_id, sr.range, ccpp_ptx_workaround_offset);
				    const auto offset_item = cl::sycl::item<Dims, true>(item_base);
				    kernel(offset_item);
			    });
#else
			sycl_handler->parallel_for<detail::wrapped_kernel_name<Name, false>>(detail::range_cast<Dims>(sr.range), detail::id_cast<Dims>(sr.offset), kernel);
#endif
		} else {
			const auto nd_range = cl::sycl::nd_range<Dims>(detail::range_cast<Dims>(sr.range),
			    detail::range_cast<Dims>(cl::sycl::range<3>(fwgs, Dims > 1 ? fwgs : 1, Dims == 3 ? fwgs : 1)), ccpp_ptx_workaround_offset);
			sycl_handler->parallel_for<detail::wrapped_kernel_name<Name, true>>(nd_range, [=](cl::sycl::nd_item<Dims> item) {
#if WORKAROUND_HIPSYCL
				kernel(cl::sycl::item<Dims>(cl::sycl::detail::make_item<Dims>(
				    item.get_global_id() - detail::id_cast<Dims>(sr.offset), detail::range_cast<Dims>(sr.range), detail::id_cast<Dims>(sr.offset))));
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

	template <typename MAF>
	void run(MAF maf) const {
		assert(task_type == detail::task_type::MASTER_ACCESS);
		if(!is_prepass()) { maf(); }
	}

  protected:
	friend bool detail::is_prepass_handler(const handler& cgh);
	friend detail::task_type detail::get_handler_type(const handler& cgh);

	handler(detail::task_type type) : task_type(type) {}

	virtual bool is_prepass() const = 0;

	virtual void set_compute_task_data(
	    int dimensions, const cl::sycl::range<3>& global_size, const cl::sycl::id<3>& global_offset, const std::string& debug_name) = 0;

	virtual detail::compute_task_exec_context get_compute_task_exec_context() const = 0;

  private:
	detail::task_type task_type;
};

namespace detail {

	inline bool is_prepass_handler(const handler& cgh) { return cgh.is_prepass(); }
	inline task_type get_handler_type(const handler& cgh) { return cgh.task_type; }

	template <bool IsPrepass>
	class compute_task_handler : public handler {
	  public:
		// The handler does not take ownership of the sycl_handler, but expects it to
		// exist for the duration of it's lifetime.
		template <bool IP = IsPrepass, typename = std::enable_if_t<IP == false>>
		compute_task_handler(std::shared_ptr<const compute_task> task, subrange<3> sr, cl::sycl::handler* sycl_handler, size_t forced_work_group_size)
		    : handler(task_type::COMPUTE), const_task(task), sr(sr), sycl_handler(sycl_handler), forced_work_group_size(forced_work_group_size) {}

		template <bool IP = IsPrepass, typename = std::enable_if_t<IP>>
		compute_task_handler(std::shared_ptr<compute_task> task) : handler(task_type::COMPUTE), task(task) {}

		template <bool IP = IsPrepass, typename = std::enable_if_t<IP>>
		void add_requirement(buffer_id bid, std::unique_ptr<range_mapper_base> rm) {
			task->add_range_mapper(bid, std::move(rm));
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode, bool IP = IsPrepass, typename = std::enable_if_t<IP == false>>
		void require_accessor(cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>& accessor) {
			sycl_handler->require(accessor);
		}

		template <int BufferDims, typename RangeMapper, bool IP = IsPrepass, typename = std::enable_if_t<IP == false>>
		subrange<BufferDims> apply_range_mapper(RangeMapper rm) const {
			switch(const_task->get_dimensions()) {
			case 1: return rm(chunk<1>(detail::id_cast<1>(sr.offset), detail::range_cast<1>(sr.range), detail::range_cast<1>(const_task->get_global_size())));
			case 2: return rm(chunk<2>(detail::id_cast<2>(sr.offset), detail::range_cast<2>(sr.range), detail::range_cast<2>(const_task->get_global_size())));
			case 3: return rm(chunk<3>(detail::id_cast<3>(sr.offset), detail::range_cast<3>(sr.range), detail::range_cast<3>(const_task->get_global_size())));
			default: assert(false);
			}
			return {};
		}

	  protected:
		bool is_prepass() const override { return IsPrepass; }

		void set_compute_task_data(
		    int dimensions, const cl::sycl::range<3>& global_size, const cl::sycl::id<3>& global_offset, const std::string& debug_name) override {
			assert(IsPrepass);
			task->set_dimensions(dimensions);
			task->set_global_size(global_size);
			task->set_global_offset(global_offset);
			task->set_debug_name(debug_name);
		}

		compute_task_exec_context get_compute_task_exec_context() const override {
			assert(!IsPrepass);
			return {sycl_handler, sr, forced_work_group_size};
		}

	  private:
		// We store two pointers, one non-const and one const, for usage during pre-pass and live-pass, respectively.
		std::shared_ptr<compute_task> task = nullptr;
		std::shared_ptr<const compute_task> const_task = nullptr;

		// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
		subrange<3> sr;
		cl::sycl::handler* sycl_handler = nullptr;
		// This is a workaround until we get proper nd_item overloads for parallel_for into the API.
		size_t forced_work_group_size = 0;
	};

	template <bool IsPrepass>
	class master_access_task_handler : public handler {
	  public:
		template <bool IP = IsPrepass, typename = std::enable_if_t<IP == false>>
		master_access_task_handler() : handler(task_type::MASTER_ACCESS) {}

		template <bool IP = IsPrepass, typename = std::enable_if_t<IP>>
		master_access_task_handler(std::shared_ptr<master_access_task> task) : handler(task_type::MASTER_ACCESS), task(task) {}

		template <bool IP = IsPrepass, typename = std::enable_if_t<IP>>
		void add_requirement(cl::sycl::access::mode mode, buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) {
			task->add_buffer_access(bid, mode, subrange<3>(offset, range));
		}

	  protected:
		bool is_prepass() const override { return IsPrepass; }

		void set_compute_task_data(
		    int dimensions, const cl::sycl::range<3>& global_size, const cl::sycl::id<3>& global_offset, const std::string& debug_name) override {
			throw std::runtime_error("Illegal usage of master access handler");
		}

		compute_task_exec_context get_compute_task_exec_context() const override { throw std::runtime_error("Illegal usage of master access handler"); }

	  private:
		std::shared_ptr<master_access_task> task;
	};

} // namespace detail

} // namespace celerity
