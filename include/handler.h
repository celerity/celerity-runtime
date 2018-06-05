#pragma once

#include <functional>
#include <regex>
#include <string>

#include <SYCL/sycl.hpp>
#include <boost/variant.hpp>
#include <spdlog/fmt/fmt.h>

#include "grid.h"
#include "range_mapper.h"
#include "subrange.h"
#include "task.h"
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

	compute_prepass_handler(distr_queue& q, task_id tid) : queue(q), tid(tid) { debug_name = fmt::format("task{}", tid); }
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

	cl::sycl::handler& get_sycl_handler() const { return *sycl_handler; }

	/**
	 * Computes the range and offset required for a given buffer and access mode
	 * by applying the corresponding range mapper(s) to the chunk this livepass has
	 * been designated to process.
	 *
	 * We currently use this to determine the required size for accessors on worker nodes.
	 * However, since there currently exists not good way of tying a particular accessor
	 * back to a range mapper, we consider ALL range mappers for every accessor (with
	 * compatible access modes). This means that
	 * FIXME: Multiple accessors with compatible modes all access the bounding box of the union of their respective requests
	 * => We'll have to determine how common the scenario of having multiple compatible accessors on the same buffer really is
	 * => At best this is a performance issue, at worst it corrupts data inside the non-written regions (for writing accessors)
	 */
	template <int BufferDims>
	std::pair<cl::sycl::range<BufferDims>, cl::sycl::id<BufferDims>> get_buffer_range_offset(buffer_id bid, cl::sycl::access::mode mode) const {
		const auto& rms = task->get_range_mappers();

		GridRegion<BufferDims> reqs;
		assert(rms.count(bid) != 0);
		for(auto& rm : rms.at(bid)) {
			const auto m = rm->get_access_mode();
			assert(m == cl::sycl::access::mode::read || m == cl::sycl::access::mode::write);
			if(m != mode) continue;

			subrange<BufferDims> req = apply_range_mapper<BufferDims>(*rm);
			reqs = GridRegion<BufferDims>::merge(reqs, detail::subrange_to_grid_region(req));
		}

		const subrange<BufferDims> bb = detail::grid_box_to_subrange(reqs.boundingBox());
		return std::make_pair(bb.range, cl::sycl::id<BufferDims>(bb.start));
	}

  private:
	friend class distr_queue;
	using any_subrange = boost::variant<subrange<1>, subrange<2>, subrange<3>>;

	distr_queue& queue;
	cl::sycl::handler* sycl_handler;
	task_id tid;
	std::shared_ptr<compute_task> task;
	any_subrange asr;

	// The handler does not take ownership of the sycl_handler, but expects it to
	// exist for the duration of it's lifetime.
	compute_livepass_handler(distr_queue& q, task_id tid, std::shared_ptr<compute_task> task, any_subrange asr, cl::sycl::handler* sycl_handler)
	    : queue(q), tid(tid), task(task), asr(asr), sycl_handler(sycl_handler) {}

	template <int BufferDims>
	subrange<BufferDims> apply_range_mapper(const detail::range_mapper_base& rm) const;
};

template <>
inline subrange<1> compute_livepass_handler::apply_range_mapper(const detail::range_mapper_base& rm) const {
	switch(asr.which()) {
	default: // suppress warnings
	case 0: return rm.map_1(boost::get<subrange<1>>(asr));
	case 1: return rm.map_1(boost::get<subrange<2>>(asr));
	case 2: return rm.map_1(boost::get<subrange<3>>(asr));
	}
}

template <>
inline subrange<2> compute_livepass_handler::apply_range_mapper(const detail::range_mapper_base& rm) const {
	switch(asr.which()) {
	default: // suppress warnings
	case 0: return rm.map_2(boost::get<subrange<1>>(asr));
	case 1: return rm.map_2(boost::get<subrange<2>>(asr));
	case 2: return rm.map_2(boost::get<subrange<3>>(asr));
	}
}

template <>
inline subrange<3> compute_livepass_handler::apply_range_mapper(const detail::range_mapper_base& rm) const {
	switch(asr.which()) {
	default: // suppress warnings
	case 0: return rm.map_3(boost::get<subrange<1>>(asr));
	case 1: return rm.map_3(boost::get<subrange<2>>(asr));
	case 2: return rm.map_3(boost::get<subrange<3>>(asr));
	}
}

class master_access_prepass_handler {
  public:
	master_access_prepass_handler(distr_queue& queue, task_id tid) : queue(queue), tid(tid) {}

	void run(std::function<void()> fun) const {
		// nop
	}

	void require(cl::sycl::access::mode mode, buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) const;

  private:
	distr_queue& queue;
	task_id tid;
};

class master_access_livepass_handler {
  public:
	void run(std::function<void()> fun) const { fun(); }
};

} // namespace celerity
