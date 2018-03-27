#pragma once

#include <SYCL/sycl.hpp>

#include "accessor.h"
#include "grid.h"
#include "handler.h"
#include "range_mapper.h"
#include "runtime.h"

namespace celerity {

template <typename DataT, int Dims>
class buffer {
  public:
	buffer(DataT* host_ptr, cl::sycl::range<Dims> size) : size(size), sycl_buffer(host_ptr, size) {
		id = runtime::get_instance().register_buffer(size, sycl_buffer);
	}

	buffer(const buffer&) = delete;
	buffer(buffer&&) = delete;

	~buffer() { runtime::get_instance().unregister_buffer(id); }

	template <cl::sycl::access::mode Mode>
	prepass_accessor<Mode> get_access(handler<is_prepass::true_t> handler, detail::range_mapper_fn<Dims> rmfn) {
		prepass_accessor<Mode> a;
		handler.require(a, id, std::make_unique<detail::range_mapper<Dims>>(rmfn, Mode));
		return a;
	}

	template <cl::sycl::access::mode Mode>
	accessor<Mode> get_access(handler<is_prepass::false_t> handler, detail::range_mapper_fn<Dims> rmfn) {
		auto a = accessor<Mode>(sycl_buffer, handler.get_sycl_handler());
		handler.require(a, id);
		return a;
	}

	size_t get_id() { return id; }

	// FIXME Host-size access should block
	DataT operator[](size_t idx) { return 1.f; }

  private:
	friend distr_queue;
	buffer_id id;
	cl::sycl::range<Dims> size;
	cl::sycl::buffer<float, Dims> sycl_buffer;
};


} // namespace celerity
