#pragma once

#include <memory>
#include <utility>

#include <CL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "buffer_storage.h"
#include "handler.h"
#include "range_mapper.h"
#include "ranges.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	struct master_proxy_buffer_base {
		virtual ~master_proxy_buffer_base() = default;
	};

	/**
	 * In order to allow the master node to participate in computations as a regular worker,
	 * we have to ensure that master access tasks do not incur any unnecessary data transfers
	 * during the pre-pass. This is a concern because unlike for compute tasks, we cannot
	 * return a placeholder accessors during the pre-pass. Instead, each buffer existing on
	 * the master node also wraps a single element proxy SYCL buffer, for which it returns
	 * a regular accessor during the pre-pass. Since the proxy buffer will never be used
	 * during a live pass, it will never incur any host <-> device data transfers.
	 *
	 * TODO: Can we share these among all buffers with same DataT/Dims?
	 * FIXME: Ultimately find a cleaner solution for this issue.
	 */
	template <typename DataT, int Dims>
	struct master_proxy_buffer : master_proxy_buffer_base {
		master_proxy_buffer() {
			assert(runtime::get_instance().is_master_node());
			proxy_buf = std::make_unique<cl::sycl::buffer<DataT, Dims>>(range_cast<Dims>(cl::sycl::range<3>(1, 1, 1)));
		}

		template <cl::sycl::access::mode Mode>
		auto get_access() {
			return proxy_buf->template get_access<Mode>();
		}

	  private:
		std::unique_ptr<cl::sycl::buffer<DataT, Dims>> proxy_buf;
	};

	class buffer_impl {
	  public:
		buffer_impl(std::shared_ptr<buffer_storage_base> storage, std::unique_ptr<master_proxy_buffer_base> master_proxy_buf, const cl::sycl::range<3>& range,
		    bool is_host_initialized)
		    : storage(storage), master_proxy_buf(std::move(master_proxy_buf)), range(range) {
			id = runtime::get_instance().register_buffer(range, storage, is_host_initialized);
		}

		cl::sycl::range<3> get_range() const { return range; }

		buffer_id get_id() const { return id; }

		buffer_storage_base& get_buffer_storage() const { return *storage; }

		master_proxy_buffer_base& get_master_proxy_buffer() const {
			assert(master_proxy_buf != nullptr);
			return *master_proxy_buf;
		}

		~buffer_impl() noexcept { runtime::get_instance().unregister_buffer(id); }

	  private:
		const std::shared_ptr<buffer_storage_base> storage;
		const std::unique_ptr<master_proxy_buffer_base> master_proxy_buf;
		const cl::sycl::range<3> range;
		buffer_id id;
	};

} // namespace detail

template <typename DataT, int Dims>
class buffer {
  public:
	buffer(const DataT* host_ptr, cl::sycl::range<Dims> range) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }

		const bool is_host_initialized = host_ptr != nullptr;
		auto buf_storage = std::make_shared<detail::buffer_storage<DataT, Dims>>(range);

		// TODO: Get rid of this functionality. Add high-level interface for explicit transfers instead.
		// --> Most of the time we'd not want a backing host-buffer, since it would contain only partial results (i.e. chunks
		//		computed on that particular worker node) in the end anyway.
		// --> Note that we're currently not even transferring data back to the host_ptr, but the interface looks like the SYCL
		//		interface that does just that!.
		// FIXME: It's not ideal that we have a const_cast here. Solve this at raw_data_handle instead.
		if(is_host_initialized) {
			auto queue = detail::runtime::get_instance().get_device_queue().get_sycl_queue();
			buf_storage->set_data(queue, detail::raw_data_handle{const_cast<DataT*>(host_ptr), detail::range_cast<3>(range), cl::sycl::id<3>{}});
		}

		std::unique_ptr<detail::master_proxy_buffer_base> master_proxy_buf = nullptr;
		if(detail::runtime::get_instance().is_master_node()) { master_proxy_buf = std::make_unique<detail::master_proxy_buffer<DataT, Dims>>(); }

		// It's important that we register the buffer AFTER we transferred the initial data (if any):
		// As soon as the buffer is registered, incoming transfers can be written to it.
		// In rare cases this might happen before the initial transfer is finished, causing a data race.
		pimpl = std::make_shared<detail::buffer_impl>(buf_storage, std::move(master_proxy_buf), detail::range_cast<3>(range), is_host_initialized);
	}

	buffer(cl::sycl::range<Dims> range) : buffer(nullptr, range) {}

	buffer(const buffer&) = default;
	buffer(buffer&&) = default;

	buffer<DataT, Dims>& operator=(const buffer&) = default;
	buffer<DataT, Dims>& operator=(buffer&&) = default;

	~buffer() {}

	template <cl::sycl::access::mode Mode, typename Functor>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> get_access(
	    handler& cgh, Functor rmfn) const {
		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
		if(detail::get_handler_type(cgh) != detail::task_type::COMPUTE) {
			throw std::runtime_error("This get_access overload is only allowed in compute tasks");
		}
		auto& sycl_buffer = get_buffer_storage().get_sycl_buffer();
		if(detail::is_prepass_handler(cgh)) {
			auto compute_cgh = dynamic_cast<detail::compute_task_handler<true>&>(cgh);
			compute_cgh.add_requirement(get_id(), std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, get_range()));
			return cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>(sycl_buffer);
		}

		auto compute_cgh = dynamic_cast<detail::compute_task_handler<false>&>(cgh);
		// It's difficult to figure out which stored range mapper corresponds to this get_access call, which is why we just call the raw mapper manually.
		// This also means that we have to clamp the subrange ourselves here, which is not ideal from an encapsulation standpoint.
		const auto sr = detail::clamp_subrange_to_buffer_size(compute_cgh.apply_range_mapper<Dims>(rmfn), get_range());
		auto a = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>(
		    sycl_buffer, sr.range, sr.offset);
		compute_cgh.require_accessor(a);
		return a;
	}

	template <cl::sycl::access::mode Mode>
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> get_access(
	    handler& cgh, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) const {
		if(detail::get_handler_type(cgh) != detail::task_type::MASTER_ACCESS) {
			throw std::runtime_error("This get_access overload is only allowed in master access tasks");
		}
		if(detail::is_prepass_handler(cgh)) {
			auto ma_cgh = dynamic_cast<detail::master_access_task_handler<true>&>(cgh);
			ma_cgh.add_requirement(Mode, get_id(), detail::range_cast<3>(range), detail::id_cast<3>(offset));
			// Since we unfortunately cannot return a placeholder accessor for host accesses,
			// we instead return an accessor to the proxy buffer, which should be a cheap operation.
			return dynamic_cast<detail::master_proxy_buffer<DataT, Dims>&>(pimpl->get_master_proxy_buffer()).template get_access<Mode>();
		}

		return get_buffer_storage().get_sycl_buffer().template get_access<Mode>(range, offset);
	}

	cl::sycl::range<Dims> get_range() const { return detail::range_cast<Dims>(pimpl->get_range()); }

  private:
	std::shared_ptr<detail::buffer_impl> pimpl;

	size_t get_id() const { return pimpl->get_id(); }
	detail::buffer_storage<DataT, Dims>& get_buffer_storage() const { return dynamic_cast<detail::buffer_storage<DataT, Dims>&>(pimpl->get_buffer_storage()); }
};

} // namespace celerity
