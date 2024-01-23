#pragma once

#include <memory>

#include <CL/sycl.hpp>

#include "buffer_manager.h"
#include "lifetime_extending_state.h"
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
class buffer final : public detail::lifetime_extending_state_wrapper {
  public:
	static_assert(Dims <= 3);

	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	buffer() : buffer(nullptr, {}) {}

	explicit buffer(const DataT* host_ptr, range<Dims> range) : m_impl(std::make_shared<impl>(range, host_ptr)) {}

	explicit buffer(range<Dims> range) : buffer(nullptr, range) {}

	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	buffer(const DataT& value) : buffer(&value, {}) {}

	buffer(const buffer&) = default;
	buffer(buffer&&) = default;

	buffer<DataT, Dims>& operator=(const buffer&) = default;
	buffer<DataT, Dims>& operator=(buffer&&) = default;

	~buffer() {}

	template <access_mode Mode, typename Functor, int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	accessor<DataT, Dims, Mode, target::device> get_access(handler& cgh, Functor rmfn) {
		return get_access<Mode, target::device, Functor>(cgh, rmfn);
	}

	template <access_mode Mode, typename Functor, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor<DataT, Dims, Mode, target::device> get_access(handler& cgh) {
		return get_access<Mode, target::device, Functor>(cgh);
	}

	template <access_mode Mode, target Target, typename Functor, int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	accessor<DataT, Dims, Mode, Target> get_access(handler& cgh, Functor rmfn) {
		return accessor<DataT, Dims, Mode, Target>(*this, cgh, rmfn);
	}

	template <access_mode Mode, target Target, typename Functor, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor<DataT, Dims, Mode, Target> get_access(handler& cgh) {
		return accessor<DataT, Dims, Mode, Target>(*this, cgh);
	}

	template <access_mode Mode, typename Functor, int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	[[deprecated("Calling get_access on a const buffer is deprecated")]] accessor<DataT, Dims, Mode, target::device> get_access(
	    handler& cgh, Functor rmfn) const {
		return get_access<Mode, target::device, Functor>(cgh, rmfn);
	}

	template <access_mode Mode, target Target, typename Functor, int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	[[deprecated("Calling get_access on a const buffer is deprecated")]] accessor<DataT, Dims, Mode, Target> get_access(handler& cgh, Functor rmfn) const {
		return accessor<DataT, Dims, Mode, Target>(*this, cgh, rmfn);
	}

	const range<Dims>& get_range() const { return m_impl->range; }

  protected:
	std::shared_ptr<detail::lifetime_extending_state> get_lifetime_extending_state() const override { return m_impl; }

  private:
	struct impl final : public detail::lifetime_extending_state {
		impl(celerity::range<Dims> rng, const DataT* host_init_ptr) : range(rng) {
			if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }
			id = detail::runtime::get_instance().get_buffer_manager().register_buffer<DataT, Dims>(detail::range_cast<3>(range), host_init_ptr);
		}
		impl(const impl&) = delete;
		impl(impl&&) = delete;
		impl& operator=(const impl&) = delete;
		impl& operator=(impl&&) = delete;
		~impl() override { detail::runtime::get_instance().get_buffer_manager().unregister_buffer(id); }
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
