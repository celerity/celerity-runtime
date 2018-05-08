#pragma once

#include <SYCL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "handler.h"
#include "prepass_accessor.h"
#include "range_mapper.h"
#include "runtime.h"

namespace celerity {

template <typename DataT, int Dims>
class buffer {
  public:
	// TODO: We may want to experiment with allocating smaller buffers on each worker node.
	// However this either requires knowledge of the entire buffer range that will be used over the buffer's lifetime,
	// or that buffers can be resized further down the line.
	// A big advantage of going that route is that it would enable buffers much larger than the per-worker device(s) would otherwise allow
	buffer(DataT* host_ptr, cl::sycl::range<Dims> size) : size(size), sycl_buffer(host_ptr, size) {
		id = runtime::get_instance().register_buffer(size, sycl_buffer);
	}

	buffer(const buffer&) = delete;
	buffer(buffer&&) = delete;

	~buffer() { runtime::get_instance().unregister_buffer(id); }

	template <cl::sycl::access::mode Mode, typename Functor>
	prepass_accessor<DataT, Dims, Mode> get_access(compute_prepass_handler& handler, Functor rmfn) {
		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
		handler.require(Mode, id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode));
		return prepass_accessor<DataT, Dims, Mode>();
	}

	template <cl::sycl::access::mode Mode, typename Functor>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_access(compute_livepass_handler& handler, Functor rmfn) {
		const auto range_offset = handler.get_buffer_range_offset<Dims>(id, Mode);
		// Sanity check
		assert(range_offset.second[0] + range_offset.first[0] <= sycl_buffer.get_range()[0]
		       && range_offset.second[1] + range_offset.first[1] <= sycl_buffer.get_range()[1]
		       && range_offset.second[2] + range_offset.first[2] <= sycl_buffer.get_range()[2]);
		auto a = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>(
		    sycl_buffer, handler.get_sycl_handler(), range_offset.first, range_offset.second);
		return a;
	}

	template <cl::sycl::access::mode Mode>
	prepass_accessor<DataT, Dims, Mode> get_access(master_access_prepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		// FIXME: This should also take a sycl::id for offset
		handler.require(Mode, id, range, cl::sycl::range<Dims>(offset));
		return prepass_accessor<DataT, Dims, Mode>();
	}

	template <cl::sycl::access::mode Mode>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> get_access(
	    master_access_livepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		return sycl_buffer.template get_access<Mode>(range, offset);
	}

	// TODO: Should we support this?
	// (Currently only used for branching demo code)
	DataT operator[](size_t idx) { return 1.f; }

	size_t get_id() const { return id; }

  private:
	friend distr_queue;
	buffer_id id;
	cl::sycl::range<Dims> size;
	cl::sycl::buffer<DataT, Dims> sycl_buffer;
};


} // namespace celerity
