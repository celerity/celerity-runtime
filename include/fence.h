#pragma once

#include <future>
#include <memory>
#include <type_traits>
#include <vector>

#include "buffer.h"
#include "host_object.h"
#include "range_mapper.h"
#include "runtime.h"
#include "sycl_wrappers.h"
#include "task.h"
#include "task_manager.h"
#include "tracy.h"

namespace celerity::detail {

template <typename DataT, int Dims>
class buffer_fence_promise;

} // namespace celerity::detail

namespace celerity {

/**
 * Owned representation of buffer contents as captured by celerity::queue::fence.
 */
template <typename T, int Dims>
class buffer_snapshot {
  public:
	buffer_snapshot() = default;

	buffer_snapshot(buffer_snapshot&& other) noexcept : m_subrange(other.m_subrange), m_data(std::move(other.m_data)) { other.m_subrange = {}; }

	buffer_snapshot& operator=(buffer_snapshot&& other) noexcept {
		m_subrange = other.m_subrange, other.m_subrange = {};
		m_data = std::move(other.m_data);
	}

	id<Dims> get_offset() const { return m_subrange.offset; }

	range<Dims> get_range() const { return m_subrange.range; }

	subrange<Dims> get_subrange() const { return m_subrange; }

	const T* get_data() const { return m_data.get(); }

	inline const T& operator[](const id<Dims> index) const { return m_data[detail::get_linear_index(m_subrange.range, index)]; }

	template <int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t index) const {
		return detail::subscript<Dims>(*this, index);
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	inline const T& operator*() const {
		return m_data[0];
	}

  private:
	template <typename U, int Dims2>
	friend class detail::buffer_fence_promise;

	subrange<Dims> m_subrange;
	std::unique_ptr<T[]> m_data; // cannot use std::vector here because of vector<bool> m(

	explicit buffer_snapshot(subrange<Dims> sr, std::unique_ptr<T[]> data) : m_subrange(sr), m_data(std::move(data)) {}
};

} // namespace celerity

namespace celerity::detail {

template <typename T>
class host_object_fence_promise final : public detail::task_promise {
  public:
	explicit host_object_fence_promise(const T* instance) : m_instance(instance) {}

	std::future<T> get_future() { return m_promise.get_future(); }

	void fulfill() override { m_promise.set_value(*m_instance); }

	allocation_id get_user_allocation_id() override { utils::panic("host_object_fence_promise::get_user_allocation_id"); }

  private:
	const T* m_instance;
	std::promise<T> m_promise;
};

template <typename DataT, int Dims>
class buffer_fence_promise final : public detail::task_promise {
  public:
	explicit buffer_fence_promise(const subrange<Dims>& sr)
	    : m_subrange(sr), m_data(std::make_unique<DataT[]>(sr.range.size())), m_aid(runtime::get_instance().create_user_allocation(m_data.get())) {}

	std::future<buffer_snapshot<DataT, Dims>> get_future() { return m_promise.get_future(); }

	void fulfill() override { m_promise.set_value(buffer_snapshot<DataT, Dims>(m_subrange, std::move(m_data))); }

	allocation_id get_user_allocation_id() override { return m_aid; }

  private:
	subrange<Dims> m_subrange;
	std::unique_ptr<DataT[]> m_data;
	allocation_id m_aid;
	std::promise<buffer_snapshot<DataT, Dims>> m_promise;
};

template <typename T>
std::future<T> fence(const experimental::host_object<T>& obj) {
	static_assert(std::is_object_v<T>, "host_object<T&> and host_object<void> are not allowed as parameters to fence()");
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::fence", Green2);

	detail::side_effect_map side_effects;
	side_effects.add_side_effect(detail::get_host_object_id(obj), experimental::side_effect_order::sequential);
	auto promise = std::make_unique<detail::host_object_fence_promise<T>>(detail::get_host_object_instance(obj));
	auto future = promise->get_future();
	[[maybe_unused]] const auto tid = detail::runtime::get_instance().fence({}, std::move(side_effects), std::move(promise));

	CELERITY_DETAIL_TRACY_ZONE_NAME("T{} fence", tid);
	return future;
}

template <typename DataT, int Dims>
std::future<buffer_snapshot<DataT, Dims>> fence(const buffer<DataT, Dims>& buf, const subrange<Dims>& sr) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("queue::fence", Green2);

	std::vector<detail::buffer_access> accesses;
	accesses.push_back(detail::buffer_access{detail::get_buffer_id(buf), access_mode::read,
	    std::make_unique<detail::range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), buf.get_range())});
	auto promise = std::make_unique<detail::buffer_fence_promise<DataT, Dims>>(sr);
	auto future = promise->get_future();
	[[maybe_unused]] const auto tid =
	    detail::runtime::get_instance().fence(detail::buffer_access_map(std::move(accesses), detail::task_geometry{}), {}, std::move(promise));

	CELERITY_DETAIL_TRACY_ZONE_NAME("T{} fence", tid);
	return future;
}

} // namespace celerity::detail

namespace celerity {
class distr_queue;
}

namespace celerity::experimental {

template <typename T, int Dims>
using buffer_snapshot [[deprecated("buffer_snapshot is no longer experimental, use celerity::buffer_snapshot")]] = celerity::buffer_snapshot<T, Dims>;

template <typename T, int Dims>
[[deprecated("fence is no longer experimental, use celerity::queue::fence")]] [[nodiscard]] auto fence(
    celerity::distr_queue& /* q */, const buffer<T, Dims>& buf, const subrange<Dims>& sr) {
	return detail::fence(buf, sr);
}

template <typename T, int Dims>
[[deprecated("fence is no longer experimental, use celerity::queue::fence")]] [[nodiscard]] auto fence(
    celerity::distr_queue& /* q */, const buffer<T, Dims>& buf) {
	return detail::fence(buf, {{}, buf.get_range()});
}

template <typename T>
[[deprecated("fence is no longer experimental, use celerity::queue::fence")]] [[nodiscard]] auto fence(
    celerity::distr_queue& /* q */, const host_object<T>& obj) {
	return detail::fence(obj);
}

} // namespace celerity::experimental
