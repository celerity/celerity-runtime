#pragma once

#include <functional>
#include <regex>
#include <string>

#include <SYCL/sycl.hpp>
#include <boost/format.hpp>
#include <boost/variant.hpp>

#include "range_mapper.h"
#include "subrange.h"
#include "types.h"

namespace celerity {

class distr_queue;

class compute_prepass_handler {
  public:
	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims> global_size, const Functor& kernel) {
		this->global_size = global_size;
		// DEBUG: Find nice name for kernel (regex is probably not super portable)
		auto qualified_name = boost::typeindex::type_id<Name*>().pretty_name();
		std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
		std::smatch matches;
		std::regex_search(qualified_name, matches, name_regex);
		debug_name = matches.size() > 0 ? matches[1] : qualified_name;
	}

	void require(cl::sycl::access::mode mode, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm);

	~compute_prepass_handler();

  private:
	friend class distr_queue;
	distr_queue& queue;
	task_id tid;
	std::string debug_name;
	any_range global_size;

	compute_prepass_handler(distr_queue& q, task_id tid) : queue(q), tid(tid) { debug_name = (boost::format("task%d") % tid).str(); }
};

class compute_livepass_handler {
  public:
	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims>, const Functor& kernel) {
		assert(asr.which() == Dims - 1);
		switch(asr.which()) {
		case 0: {
			auto& sr = boost::get<subrange<1>>(asr);
			sycl_handler->parallel_for<Name>(sr.range, cl::sycl::id<1>(sr.start), kernel);
		} break;
		case 1: {
			auto& sr = boost::get<subrange<2>>(asr);
			sycl_handler->parallel_for<Name>(sr.range, cl::sycl::id<2>(sr.start), kernel);
		} break;
		case 2: {
			auto& sr = boost::get<subrange<3>>(asr);
			sycl_handler->parallel_for<Name>(sr.range, cl::sycl::id<3>(sr.start), kernel);
		} break;
		default: assert(false);
		}
	}

	cl::sycl::handler& get_sycl_handler() { return *sycl_handler; }

  private:
	friend class distr_queue;
	using any_subrange = boost::variant<subrange<1>, subrange<2>, subrange<3>>;

	distr_queue& queue;
	cl::sycl::handler* sycl_handler;
	task_id tid;
	any_subrange asr;

	// The handler does not take ownership of the sycl_handler, but expects it to
	// exist for the duration of it's lifetime.
	compute_livepass_handler(distr_queue& q, task_id tid, any_subrange asr, cl::sycl::handler* sycl_handler)
	    : queue(q), tid(tid), asr(asr), sycl_handler(sycl_handler) {}
};

class master_access_prepass_handler {
  public:
	master_access_prepass_handler(distr_queue& queue, task_id tid) : queue(queue), tid(tid) {}

	void run(std::function<void()> fun) const {
		// nop
	}

	void require(cl::sycl::access::mode mode, buffer_id bid, any_range range, any_range offset) const;

  private:
	distr_queue& queue;
	task_id tid;
};

class master_access_livepass_handler {
  public:
	void run(std::function<void()> fun) const { fun(); }
};

} // namespace celerity
