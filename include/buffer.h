#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "ranges.h"
#include "runtime.h"
#include "sycl_wrappers.h"
#include "tracy.h"


namespace celerity {

template <typename DataT, int Dims = 1>
class buffer;

}

namespace celerity::detail {

template <typename T, int D>
buffer_id get_buffer_id(const buffer<T, D>& buff) {
	assert(buff.m_tracker != nullptr);
	return buff.m_tracker->id;
}

template <typename DataT, int Dims>
void set_buffer_name(const celerity::buffer<DataT, Dims>& buff, const std::string& debug_name) {
	assert(buff.m_tracker != nullptr);
	buff.m_tracker->debug_name = debug_name;
}

template <typename DataT, int Dims>
std::string get_buffer_name(const celerity::buffer<DataT, Dims>& buff) {
	assert(buff.m_tracker != nullptr);
	return buff.m_tracker->debug_name;
}

} // namespace celerity::detail

namespace celerity {

template <typename DataT, int Dims, access_mode Mode, target Target>
class accessor;

template <typename DataT, int Dims>
class buffer {
  public:
	static_assert(Dims <= 3);

	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	buffer() : buffer(nullptr, {}) {}

	explicit buffer(const DataT* host_ptr, const range<Dims>& range) : m_tracker(std::make_shared<tracker>(range, host_ptr)) {}

	explicit buffer(const range<Dims>& range) : buffer(nullptr, range) {}

	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	buffer(const DataT& value) : buffer(&value, {}) {}

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

	const range<Dims>& get_range() const {
		assert(m_tracker != nullptr);
		return m_tracker->range;
	}

  private:
	/// A `tacker` instance is shared by all shallow copies of this `buffer` via a `std::shared_ptr` to implement (SYCL) reference semantics.
	/// It notifies the runtime of buffer creation and destruction and also persists changes of the buffer debug name.
	struct tracker {
		tracker(const celerity::range<Dims>& range, const void* const host_init_ptr) : range(range) {
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("buffer::buffer", DarkSlateBlue);

			if(!detail::runtime::has_instance()) { detail::runtime::init(nullptr, nullptr); }
			auto user_aid = detail::null_allocation_id;
			if(host_init_ptr != nullptr) {
				const auto user_ptr = const_cast<void*>(host_init_ptr); // promise: instruction_graph_generator will never issue a write to this allocation
				user_aid = detail::runtime::get_instance().create_user_allocation(user_ptr);
			}
			id = detail::runtime::get_instance().create_buffer(detail::range_cast<3>(range), sizeof(DataT), alignof(DataT), user_aid);
		}

		tracker(const tracker&) = delete;
		tracker(tracker&&) = delete;
		tracker& operator=(const tracker&) = delete;
		tracker& operator=(tracker&&) = delete;

		~tracker() {
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("buffer::~buffer", DarkCyan);
			detail::runtime::get_instance().destroy_buffer(id);
		}

		detail::buffer_id id;
		celerity::range<Dims> range;
		std::string debug_name;
	};

	std::shared_ptr<tracker> m_tracker;

	template <typename T, int D>
	friend detail::buffer_id detail::get_buffer_id(const buffer<T, D>& buff);
	template <typename T, int D>
	friend void detail::set_buffer_name(const celerity::buffer<T, D>& buff, const std::string& debug_name);
	template <typename T, int D>
	friend std::string detail::get_buffer_name(const celerity::buffer<T, D>& buff);
};

} // namespace celerity
