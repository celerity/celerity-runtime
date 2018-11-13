#pragma once

#include <SYCL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "accessor.h"
#include "buffer_storage.h"
#include "handler.h"
#include "range_mapper.h"

namespace celerity {

namespace detail {

	// We have to jump through some hoops to resolve a circular dependency with runtime
	class buffer_base {
	  protected:
		buffer_type get_type() const;
		buffer_id register_with_runtime(cl::sycl::range<3> range, std::shared_ptr<detail::buffer_storage_base> buffer_storage, bool host_initialized) const;
		void unregister_with_runtime(buffer_id id) const;
	};

} // namespace detail

template <typename DataT, int Dims>
class buffer : public detail::buffer_base {
  public:
	// TODO: We may want to experiment with allocating smaller buffers on each worker node.
	// However this either requires knowledge of the entire buffer range that will be used over the buffer's lifetime,
	// or that buffers can be resized further down the line.
	// A big advantage of going that route is that it would enable buffers much larger than the per-worker device(s) would otherwise allow
	buffer(DataT* host_ptr, cl::sycl::range<Dims> range) : range(range) {
		const bool host_initialized = host_ptr != nullptr;

		buffer_storage = std::make_shared<detail::buffer_storage<DataT, Dims>>(range);
		buffer_storage->set_type(get_type());

		// TODO: Get rid of this functionality. Add high-level interface for explicit transfers instead.
		// --> Most of the time we'd not want a backing host-buffer, since it would contain only partial results (i.e. chunks
		//		computed on that particular worker node) in the end anyway.
		// --> Note that we're currently not even transferring data back to the host_ptr, but the interface looks like the SYCL
		//		interface that does just that!.
		if(host_initialized) { buffer_storage->set_data(detail::raw_data_handle{host_ptr, cl::sycl::range<3>(range), cl::sycl::id<3>{}}); }

		// It's important that we register the buffer AFTER we transferred the initial data (if any):
		// As soon as the buffer is registered, incoming transfers can be written to it.
		// In rare cases this might happen before the initial transfer is finished, causing a data race.
		id = register_with_runtime(cl::sycl::range<3>(range), buffer_storage, host_initialized);
	}

	buffer(cl::sycl::range<Dims> range) : buffer(nullptr, range) {}

	buffer(const buffer&) = delete;
	buffer(buffer&&) = delete;

	~buffer() { unregister_with_runtime(id); }

	template <cl::sycl::access::mode Mode, typename Functor>
	prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_access(compute_prepass_handler& handler, Functor rmfn) {
		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
		handler.require(id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, range));
		return prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>();
	}

	template <cl::sycl::access::mode Mode, typename Functor>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_access(compute_livepass_handler& handler, Functor rmfn) {
		// It's difficult to figure out which stored range mapper corresponds to this get_access call, which is why we just call the raw mapper manually.
		// This also means that we have to clamp the subrange ourselves here, which is not ideal from an encapsulation standpoint.
		const auto sr = detail::clamp_subrange_to_buffer_size(handler.apply_range_mapper<Dims>(rmfn), range);
		auto& sycl_buffer = buffer_storage->get_sycl_buffer();
		auto a = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>(sycl_buffer, handler.get_sycl_handler(), sr.range, sr.offset);
		return a;
	}

	template <cl::sycl::access::mode Mode>
	prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> get_access(
	    master_access_prepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		static_assert(Mode != cl::sycl::access::mode::atomic, "Atomic access not supported on host buffers");
		handler.require(Mode, id, cl::sycl::range<3>(range), cl::sycl::id<3>(offset));
		return prepass_accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer>();
	}

	template <cl::sycl::access::mode Mode>
	host_accessor<DataT, Dims, Mode> get_access(master_access_livepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		static_assert(Mode != cl::sycl::access::mode::atomic, "Atomic access not supported on host buffers");
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
