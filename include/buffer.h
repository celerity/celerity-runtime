#pragma once

#include <SYCL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "accessor.h"
#include "buffer_storage.h"
#include "handler.h"
#include "range_mapper.h"

namespace celerity {

// We have to jump through some hoops to resolve a circular dependency with runtime
class buffer_base {
  protected:
	buffer_id register_with_runtime(cl::sycl::range<3> range, std::shared_ptr<detail::buffer_storage_base> buffer_storage) const;
	void unregister_with_runtime(buffer_id id) const;
};

template <typename DataT, int Dims>
class buffer : public buffer_base {
  public:
	// TODO: We may want to experiment with allocating smaller buffers on each worker node.
	// However this either requires knowledge of the entire buffer range that will be used over the buffer's lifetime,
	// or that buffers can be resized further down the line.
	// A big advantage of going that route is that it would enable buffers much larger than the per-worker device(s) would otherwise allow
	buffer(DataT* host_ptr, cl::sycl::range<Dims> range) : range(range) {
		buffer_storage = std::make_shared<detail::buffer_storage<DataT, Dims>>(range);
		id = register_with_runtime(cl::sycl::range<3>(range), buffer_storage);

		// TODO: Get rid of this functionality. Add high-level interface for explicit transfers instead.
		// --> Most of the time we'd not want a backing host-buffer, since it would contain only partial results (i.e. chunks
		//		computed on that particular worker node) in the end anyway.
		// --> Note that we're currently not even transfering data back to the host_ptr, but the inteface looks like the SYCL
		//		interface that does just that!.
		if(host_ptr != nullptr) { buffer_storage->set_data(detail::raw_data_handle{host_ptr, cl::sycl::range<3>(range), cl::sycl::id<3>{}}); }
	}

	buffer(const buffer&) = delete;
	buffer(buffer&&) = delete;

	~buffer() { unregister_with_runtime(id); }

	template <cl::sycl::access::mode Mode, typename Functor>
	prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_access(compute_prepass_handler& handler, Functor rmfn) {
		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
		handler.require(Mode, id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode));
		return prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>();
	}

	template <cl::sycl::access::mode Mode, typename Functor>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_access(compute_livepass_handler& handler, Functor rmfn) {
		const auto range_offset = handler.get_buffer_range_offset<Dims>(id, Mode);
		// Sanity check
		assert(range_offset.second[0] + range_offset.first[0] <= range[0] && range_offset.second[1] + range_offset.first[1] <= range[1]
		       && range_offset.second[2] + range_offset.first[2] <= range[2]);
		auto& sycl_buffer = buffer_storage->get_sycl_buffer();
		auto a = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>(
		    sycl_buffer, handler.get_sycl_handler(), range_offset.first, range_offset.second);
		return a;
	}

	template <cl::sycl::access::mode Mode>
	prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> get_access(
	    master_access_prepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		handler.require(Mode, id, cl::sycl::range<3>(range), cl::sycl::id<3>(offset));
		return prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer>();
	}

	template <cl::sycl::access::mode Mode>
	host_accessor<DataT, Dims, Mode> get_access(master_access_livepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		return host_accessor<DataT, Dims, Mode>(buffer_storage, range, offset);
	}

	size_t get_id() const { return id; }

	cl::sycl::range<Dims> get_range() const { return range; }

  private:
	buffer_id id;
	cl::sycl::range<Dims> range;
	std::shared_ptr<detail::buffer_storage<DataT, Dims>> buffer_storage;
};


} // namespace celerity
