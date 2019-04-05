#pragma once

#include <CL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "buffer_storage.h"
#include "handler.h"
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
	buffer(const DataT* host_ptr, cl::sycl::range<Dims> range) : range(range) {
		const bool host_initialized = host_ptr != nullptr;

		buffer_storage = std::make_shared<detail::buffer_storage<DataT, Dims>>(range);

		// TODO: Get rid of this functionality. Add high-level interface for explicit transfers instead.
		// --> Most of the time we'd not want a backing host-buffer, since it would contain only partial results (i.e. chunks
		//		computed on that particular worker node) in the end anyway.
		// --> Note that we're currently not even transferring data back to the host_ptr, but the interface looks like the SYCL
		//		interface that does just that!.
		// FIXME: It's not ideal that we have a const_cast here. Solve this at raw_data_handle instead.
		if(host_initialized) {
			auto queue = detail::runtime::get_instance().get_device_queue().get_sycl_queue();
			buffer_storage->set_data(queue, detail::raw_data_handle{const_cast<DataT*>(host_ptr), detail::range_cast<3>(range), cl::sycl::id<3>{}});
		}

		// It's important that we register the buffer AFTER we transferred the initial data (if any):
		// As soon as the buffer is registered, incoming transfers can be written to it.
		// In rare cases this might happen before the initial transfer is finished, causing a data race.
		id = detail::runtime::get_instance().register_buffer(detail::range_cast<3>(range), buffer_storage, host_initialized);
	}

	buffer(cl::sycl::range<Dims> range) : buffer(nullptr, range) {}

	buffer(const buffer&) = delete;
	buffer(buffer&&) = delete;

	~buffer() { detail::runtime::get_instance().unregister_buffer(id); }

	template <cl::sycl::access::mode Mode, typename Functor>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> get_access(
	    handler& cgh, Functor rmfn) {
		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
		if(detail::get_handler_type(cgh) != detail::task_type::COMPUTE) {
			throw std::runtime_error("This get_access overload is only allowed in compute tasks");
		}
		auto& sycl_buffer = buffer_storage->get_sycl_buffer();
		if(detail::is_prepass_handler(cgh)) {
			auto compute_cgh = dynamic_cast<detail::compute_task_handler<true>&>(cgh);
			compute_cgh.add_requirement(id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, range));
			return cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>(sycl_buffer);
		}

		auto compute_cgh = dynamic_cast<detail::compute_task_handler<false>&>(cgh);
		// It's difficult to figure out which stored range mapper corresponds to this get_access call, which is why we just call the raw mapper manually.
		// This also means that we have to clamp the subrange ourselves here, which is not ideal from an encapsulation standpoint.
		const auto sr = detail::clamp_subrange_to_buffer_size(compute_cgh.apply_range_mapper<Dims>(rmfn), range);
		auto a = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>(
		    sycl_buffer, sr.range, sr.offset);
		compute_cgh.require_accessor(a);
		return a;
	}


	template <cl::sycl::access::mode Mode>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> get_access(
	    handler& cgh, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
		if(detail::get_handler_type(cgh) != detail::task_type::MASTER_ACCESS) {
			throw std::runtime_error("This get_access overload is only allowed in master access tasks");
		}
		if(detail::is_prepass_handler(cgh)) {
			auto ma_cgh = dynamic_cast<detail::master_access_task_handler<true>&>(cgh);
			ma_cgh.add_requirement(Mode, id, detail::range_cast<3>(range), detail::id_cast<3>(offset));
		}

		// Unfortunately we cannot return a placeholder accessor in this case (as host accessors cannot be placeholders).
		// Instead, we simply return a real host accessor during prepass as well.
		// The idea is that the master node will never do any work on a device anyway, making this a cheap operation.
		return buffer_storage->get_sycl_buffer().template get_access<Mode>(range, offset);
	}

	size_t get_id() const { return id; }

	cl::sycl::range<Dims> get_range() const { return range; }

  private:
	detail::buffer_id id;
	cl::sycl::range<Dims> range;
	std::shared_ptr<detail::buffer_storage<DataT, Dims>> buffer_storage;
};

} // namespace celerity
