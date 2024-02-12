#pragma once

#include <memory>
#include <type_traits>

#include "buffer_storage.h"
#include "distr_queue.h"
#include "host_object.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity::detail {

template <typename DataT, int Dims>
class buffer_fence_promise;

} // namespace celerity::detail

namespace celerity {

/**
 * Owned representation of buffer contents as captured by celerity::distr_queue::fence.
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
class host_object_fence_promise final : public detail::fence_promise {
  public:
	explicit host_object_fence_promise(const experimental::host_object<T>& obj) : m_host_object(obj) {}

	std::future<T> get_future() { return m_promise.get_future(); }

	void fulfill() override { m_promise.set_value(std::as_const(detail::get_host_object_instance(m_host_object))); }

	allocation_id get_user_allocation_id() override { utils::panic("host_object_fence_promise::get_user_allocation_id"); }

  private:
	experimental::host_object<T> m_host_object;
	std::promise<T> m_promise;
};

template <typename DataT, int Dims>
class buffer_fence_promise final : public detail::fence_promise {
  public:
	explicit buffer_fence_promise(const buffer<DataT, Dims>& buf, const subrange<Dims>& sr)
	    : m_buffer(buf), m_subrange(sr), m_aid(null_allocation_id /* [IDAG placeholder] */) {}

	std::future<buffer_snapshot<DataT, Dims>> get_future() { return m_promise.get_future(); }

	void fulfill() override {
		const auto access_info =
		    runtime::get_instance().get_buffer_manager().access_host_buffer<DataT, Dims>(get_buffer_id(m_buffer), access_mode::read, m_subrange);
		assert(all_true(id_cast<Dims>(access_info.backing_buffer_offset) <= m_subrange.offset));
		auto data = std::make_unique<DataT[]>(m_subrange.range.size());
		memcpy_strided_host(access_info.ptr, data.get(), sizeof(DataT), range_cast<Dims>(access_info.backing_buffer_range),
		    m_subrange.offset - id_cast<Dims>(access_info.backing_buffer_offset), m_subrange.range, {}, m_subrange.range);
		m_promise.set_value(buffer_snapshot<DataT, Dims>(m_subrange, std::move(data)));
	}

	allocation_id get_user_allocation_id() override { return m_aid; }

  private:
	buffer<DataT, Dims> m_buffer;
	subrange<Dims> m_subrange;
	allocation_id m_aid;
	std::promise<buffer_snapshot<DataT, Dims>> m_promise;
};

} // namespace celerity::detail

namespace celerity {

template <typename T>
std::future<T> distr_queue::fence(const experimental::host_object<T>& obj) {
	static_assert(std::is_object_v<T>, "host_object<T&> and host_object<void> are not allowed as parameters to fence()");

	detail::side_effect_map side_effects;
	side_effects.add_side_effect(detail::get_host_object_id(obj), experimental::side_effect_order::sequential);
	auto promise = std::make_unique<detail::host_object_fence_promise<T>>(obj);
	auto future = promise->get_future();
	detail::runtime::get_instance().get_task_manager().generate_fence_task({}, std::move(side_effects), std::move(promise));
	return future;
}

template <typename DataT, int Dims>
std::future<buffer_snapshot<DataT, Dims>> distr_queue::fence(const buffer<DataT, Dims>& buf, const subrange<Dims>& sr) {
	detail::buffer_access_map access_map;
	access_map.add_access(detail::get_buffer_id(buf),
	    std::make_unique<detail::range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), access_mode::read, buf.get_range()));
	auto promise = std::make_unique<detail::buffer_fence_promise<DataT, Dims>>(buf, sr);
	auto future = promise->get_future();
	detail::runtime::get_instance().get_task_manager().generate_fence_task(std::move(access_map), {}, std::move(promise));
	return future;
}

} // namespace celerity

namespace celerity::experimental {

template <typename T, int Dims>
using buffer_snapshot [[deprecated("buffer_snapshot is no longer experimental, use celerity::buffer_snapshot")]] = celerity::buffer_snapshot<T, Dims>;

template <typename... Params>
[[deprecated("fence is no longer experimental, use celerity::distr_queue::fence")]] [[nodiscard]] auto fence(celerity::distr_queue& q, const Params&... args) {
	return q.fence(args...);
}

} // namespace celerity::experimental
