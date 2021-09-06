#pragma once

#include <memory>

#include <CL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "buffer_manager.h"
#include "range_mapper.h"
#include "ranges.h"
#include "runtime.h"
#include "sycl_wrappers.h"

namespace celerity {

template <typename DataT, int Dims>
class buffer;

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

	template <typename T, int D>
	buffer_id get_buffer_id(const buffer<T, D>& buff);

} // namespace detail

template <typename DataT, int Dims, access_mode Mode, target Target>
class accessor;

template <typename DataT, int Dims>
class buffer {
  public:
	static_assert(Dims > 0, "0-dimensional buffers NYI");

	buffer(const DataT* host_ptr, cl::sycl::range<Dims> range) : range(range) {
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

	template <access_mode Mode, typename Functor>
	accessor<DataT, Dims, Mode, target::device> get_access(handler& cgh, Functor rmfn) const {
		return get_access<Mode, target::device, Functor>(cgh, rmfn);
	}


	template <access_mode Mode, target Target, typename Functor>
	accessor<DataT, Dims, Mode, Target> get_access(handler& cgh, Functor rmfn) const {
		return accessor<DataT, Dims, Mode, Target>(*this, cgh, rmfn);
	}

	cl::sycl::range<Dims> get_range() const { return range; }

  private:
	std::shared_ptr<detail::buffer_lifetime_tracker> lifetime_tracker = nullptr;
	cl::sycl::range<Dims> range;
	detail::buffer_id id;

	template <typename T, int D>
	friend detail::buffer_id detail::get_buffer_id(const buffer<T, D>& buff);
};

namespace detail {

	template <typename T, int D>
	buffer_id get_buffer_id(const buffer<T, D>& buff) {
		return buff.id;
	}

} // namespace detail

} // namespace celerity
