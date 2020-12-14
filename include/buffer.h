#pragma once

#include <memory>

#include <CL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "accessor.h"
#include "buffer_manager.h"
#include "handler.h"
#include "range_mapper.h"
#include "ranges.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	struct buffer_lifetime_tracker {
		buffer_lifetime_tracker() = default;
		template <typename DataT, int Dims>
		buffer_id initialize(cl::sycl::range<3> range, const DataT* host_init_ptr) {
			id = runtime::get_instance().get_buffer_manager().register_buffer<DataT, Dims>(range, host_init_ptr);
			return id;
		}
		buffer_lifetime_tracker(const buffer_lifetime_tracker&) = delete;
		buffer_lifetime_tracker(buffer_lifetime_tracker&&) = delete;
		~buffer_lifetime_tracker() noexcept { runtime::get_instance().get_buffer_manager().unregister_buffer(id); }
		buffer_id id;
	};

} // namespace detail

template <typename DataT, int Dims>
class buffer {
  public:
	static_assert(Dims > 0, "0-dimensional buffers NYI");

	buffer(const DataT* host_ptr, cl::sycl::range<Dims> range)
	    : range(range), faux_buf(new cl::sycl::buffer<DataT, Dims>(detail::range_cast<Dims>(cl::sycl::range<3>{1, 1, 1}))) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }

		lifetime_tracker = std::make_shared<detail::buffer_lifetime_tracker>();
		id = lifetime_tracker->initialize<DataT, Dims>(detail::range_cast<3>(range), host_ptr);
	}

	buffer(cl::sycl::range<Dims> range) : buffer(nullptr, range) {}

	buffer(const buffer&) = default;
	buffer(buffer&&) = default;

	buffer<DataT, Dims>& operator=(const buffer&) = default;
	buffer<DataT, Dims>& operator=(buffer&&) = default;

	~buffer() {}

	template <cl::sycl::access::mode Mode, typename Functor>
	accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> get_access(handler& cgh, Functor rmfn) const {
		return get_access<Mode, cl::sycl::access::target::global_buffer>(cgh, rmfn);
	}

	template <cl::sycl::access::mode Mode, cl::sycl::access::target Target, typename Functor>
	accessor<DataT, Dims, Mode, Target> get_access(handler& cgh, Functor rmfn) const {
		static_assert(!std::is_same_v<Functor, cl::sycl::range<Dims>>, "The buffer::get_access overload for master-access tasks (now called 'host tasks') has "
		                                                               "been removed with Celerity 0.2.0. Please provide a range mapper instead.");

		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");

		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, get_range()));
			if constexpr(Target == cl::sycl::access::target::host_buffer) {
				return detail::make_host_accessor<DataT, Dims, Mode>();
			} else {
				return detail::make_device_accessor<DataT, Dims, Mode>(*faux_buf);
			}
		}

		// It's difficult to figure out which stored range mapper corresponds to this get_access call, which is why we just call the raw mapper manually.
		// This also means that we have to clamp the subrange ourselves here, which is not ideal from an encapsulation standpoint.
		if constexpr(Target == cl::sycl::access::target::host_buffer) {
			if(detail::get_handler_execution_target(cgh) != detail::execution_target::HOST) {
				throw std::runtime_error("Calling buffer::get_access with sycl::access::target::host_buffer is only allowed in host tasks.");
			}
			auto& live_cgh = dynamic_cast<detail::live_pass_host_handler&>(cgh);
			const auto sr = detail::clamp_subrange_to_buffer_size(live_cgh.apply_range_mapper<Dims>(rmfn, get_range()), get_range());
			auto access_info = detail::runtime::get_instance().get_buffer_manager().get_host_buffer<DataT, Dims>(
			    id, Mode, detail::range_cast<3>(sr.range), detail::id_cast<3>(sr.offset));
			return detail::make_host_accessor<DataT, Dims, Mode>(sr, access_info.buffer, access_info.offset, range);
		} else {
			if(detail::get_handler_execution_target(cgh) != detail::execution_target::DEVICE) {
				throw std::runtime_error(
				    "Calling buffer::get_access on device buffers is only allowed in compute tasks. "
				    "If you want to access this buffer from within a host task, please specialize the call using sycl::access::target::host_buffer.");
			}
			auto& live_cgh = dynamic_cast<detail::live_pass_device_handler&>(cgh);
			const auto sr = detail::clamp_subrange_to_buffer_size(live_cgh.apply_range_mapper<Dims>(rmfn, get_range()), get_range());
			auto access_info = detail::runtime::get_instance().get_buffer_manager().get_device_buffer<DataT, Dims>(
			    id, Mode, detail::range_cast<3>(sr.range), detail::id_cast<3>(sr.offset));
			return detail::make_device_accessor<DataT, Dims, Mode>(live_cgh.get_eventual_sycl_cgh(), access_info.buffer, sr.range, access_info.offset);
		}
	}

	cl::sycl::range<Dims> get_range() const { return range; }

  private:
	std::shared_ptr<detail::buffer_lifetime_tracker> lifetime_tracker = nullptr;
	cl::sycl::range<Dims> range;
	detail::buffer_id id;

	// Unfortunately, as of SYCL 1.2.1 Rev 6, there is now way of creating a
	// SYCL accessor without at least a buffer reference (i.e., there is no
	// default ctor, even for placeholder accessors). During the pre-pass, we
	// not only don't have access to a SYCL command group handler, but also
	// don't know the backing buffer yet (it might not even exist at that
	// point). For calls to get_access() we however still have to construct a
	// SYCL accessor to return inside the Celerity accessor. For this, we use
	// this faux buffer. It has size 1 in all dimensions, so the allocation
	// overhead should be minimal. Hopefully the runtime overhead is also
	// negligible.
	//
	// (The reason why we make this a shared_ptr is so that Celerity buffers
	// still satisfy StandardLayoutType, which we use as a crude safety check;
	// see distr_queue::submit).
	std::shared_ptr<cl::sycl::buffer<DataT, Dims>> faux_buf;
};

} // namespace celerity
