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

template <typename DataT, int Dims = 1>
class buffer;

namespace detail {

	template <typename T, int D>
	buffer_id get_buffer_id(const buffer<T, D>& buff);

	template <typename DataT, int Dims>
	void set_buffer_name(const celerity::buffer<DataT, Dims>& buff, const std::string& debug_name) {
		buff.m_impl->debug_name = debug_name;
	};
	template <typename DataT, int Dims>
	std::string get_buffer_name(const celerity::buffer<DataT, Dims>& buff) {
		return buff.m_impl->debug_name;
	};

} // namespace detail

template <typename DataT, int Dims, access_mode Mode, target Target>
class accessor;

template <typename DataT, int Dims>
class buffer {
  public:
	static_assert(Dims > 0, "0-dimensional buffers NYI");

	buffer(const DataT* host_ptr, celerity::range<Dims> range) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }
		m_impl = std::make_shared<impl>(range, host_ptr);
	}

	buffer(celerity::range<Dims> range) : buffer(nullptr, range) {}

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

	celerity::range<Dims> get_range() const { return m_impl->range; }

  private:
	struct impl {
		impl(celerity::range<Dims> rng, const DataT* host_init_ptr) : range(rng) {
			id = detail::runtime::get_instance().get_buffer_manager().register_buffer<DataT, Dims>(detail::range_cast<3>(range), host_init_ptr);
		}
		impl(const impl&) = delete;
		impl(impl&&) = delete;
		~impl() noexcept { detail::runtime::get_instance().get_buffer_manager().unregister_buffer(id); }
		detail::buffer_id id;
		celerity::range<Dims> range;
		std::string debug_name;
	};

	std::shared_ptr<impl> m_impl = nullptr;

	template <typename T, int D>
	friend detail::buffer_id detail::get_buffer_id(const buffer<T, D>& buff);
	template <typename T, int D>
	friend void detail::set_buffer_name(const celerity::buffer<T, D>& buff, const std::string& debug_name);
	template <typename T, int D>
	friend std::string detail::get_buffer_name(const celerity::buffer<T, D>& buff);
};

namespace detail {

	template <typename T, int D>
	buffer_id get_buffer_id(const buffer<T, D>& buff) {
		return buff.m_impl->id;
	}

} // namespace detail

} // namespace celerity
