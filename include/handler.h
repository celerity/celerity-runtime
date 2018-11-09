#pragma once

#include <regex>
#include <string>

#include <SYCL/sycl.hpp>
#include <boost/type_index.hpp>
#include <spdlog/fmt/fmt.h>

#include "range_mapper.h"
#include "ranges.h"
#include "task.h"
#include "types.h"

namespace celerity {

namespace detail {
	class task_manager;
}

class distr_queue;

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
		task.set_global_size(cl::sycl::range<3>(global_size));
		task.set_global_offset(cl::sycl::id<3>(global_offset));

		// DEBUG: Find nice name for kernel (regex is probably not super portable)
		auto qualified_name = boost::typeindex::type_id<Name*>().pretty_name();
		std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
		std::smatch matches;
		std::regex_search(qualified_name, matches, name_regex);
		debug_name = matches.size() > 0 ? matches[1] : qualified_name;
		task.set_debug_name(debug_name);
	}

	void require(buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) const;

  private:
	detail::compute_task& task;
	std::string debug_name;
	cl::sycl::range<3> global_size;
};

class compute_livepass_handler {
  public:
	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims> global_size, const Functor& kernel) {
		parallel_for<Name, Functor, Dims>(global_size, cl::sycl::id<Dims>(), kernel);
	}

	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims>, cl::sycl::id<Dims>, const Functor& kernel) {
		switch(Dims) {
		case 0: {
			sycl_handler->parallel_for<Name>(cl::sycl::range<1>(sr.range), cl::sycl::id<1>(sr.offset), kernel);
		} break;
		case 1: {
			sycl_handler->parallel_for<Name>(cl::sycl::range<2>(sr.range), cl::sycl::id<2>(sr.offset), kernel);
		} break;
		case 2: {
			sycl_handler->parallel_for<Name>(cl::sycl::range<3>(sr.range), cl::sycl::id<3>(sr.offset), kernel);
		} break;
		default: assert(false);
		}
	}

	cl::sycl::handler& get_sycl_handler() const { return *sycl_handler; }

	template <int BufferDims, typename RangeMapper>
	subrange<BufferDims> apply_range_mapper(RangeMapper rm) const {
		switch(task.get_dimensions()) {
		case 1: return rm(chunk<1>(cl::sycl::id<1>(sr.offset), cl::sycl::range<1>(sr.range), cl::sycl::range<1>(task.get_global_size())));
		case 2: return rm(chunk<2>(cl::sycl::id<2>(sr.offset), cl::sycl::range<2>(sr.range), cl::sycl::range<2>(task.get_global_size())));
		case 3: return rm(chunk<3>(cl::sycl::id<3>(sr.offset), cl::sycl::range<3>(sr.range), cl::sycl::range<3>(task.get_global_size())));
		default: assert(false);
		}
		return {};
	}

  private:
	friend class distr_queue;

	const detail::compute_task& task;
	// The subrange, when combined with the tasks global size, defines the chunk this handler executes.
	subrange<3> sr;
	cl::sycl::handler* sycl_handler;

	// The handler does not take ownership of the sycl_handler, but expects it to
	// exist for the duration of it's lifetime.
	compute_livepass_handler(const detail::compute_task& task, subrange<3> sr, cl::sycl::handler* sycl_handler)
	    : task(task), sr(sr), sycl_handler(sycl_handler) {}
};

class master_access_prepass_handler {
  public:
	master_access_prepass_handler(detail::master_access_task& task) : task(task) {}

	template <typename MAF>
	void run(MAF maf) const {
		// nop
	}

	void require(cl::sycl::access::mode mode, buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) const;

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
