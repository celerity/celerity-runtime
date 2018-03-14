#pragma once

#include <regex>
#include <string>

#include <SYCL/sycl.hpp>
#include <boost/format.hpp>
#include <boost/variant.hpp>

#include "accessor.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

enum class is_prepass { true_t, false_t };

template <is_prepass>
class handler {};

class distr_queue;

template <>
class handler<is_prepass::true_t> {
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

	template <cl::sycl::access::mode Mode>
	void require(prepass_accessor<Mode> a, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm);

	~handler();

  private:
	friend class distr_queue;
	distr_queue& queue;
	task_id tid;
	std::string debug_name;
	boost::variant<cl::sycl::range<1>, cl::sycl::range<2>, cl::sycl::range<3>> global_size;

	handler(distr_queue& q, task_id tid) : queue(q), tid(tid) { debug_name = (boost::format("task%d") % tid).str(); }
};

template <>
class handler<is_prepass::false_t> {
  public:
	template <typename Name, typename Functor, int Dims>
	void parallel_for(cl::sycl::range<Dims> range, const Functor& kernel) {
		sycl_handler->parallel_for<Name>(range, kernel);
	}

	template <cl::sycl::access::mode Mode>
	void require(accessor<Mode> a, buffer_id bid) {
		// TODO: Is there any need for this?
	}

	cl::sycl::handler& get_sycl_handler() { return *sycl_handler; }

  private:
	friend class distr_queue;
	distr_queue& queue;
	cl::sycl::handler* sycl_handler;
	task_id tid;

	// The handler does not take ownership of the sycl_handler, but expects it to
	// exist for the duration of it's lifetime.
	handler(distr_queue& q, task_id tid, cl::sycl::handler* sycl_handler) : queue(q), tid(tid), sycl_handler(sycl_handler) {
		this->sycl_handler = sycl_handler;
	}
};

template <>
void handler<is_prepass::true_t>::require(prepass_accessor<cl::sycl::access::mode::read> a, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm);

template <>
void handler<is_prepass::true_t>::require(prepass_accessor<cl::sycl::access::mode::write> a, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm);

} // namespace celerity
