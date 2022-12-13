#pragma once

#include <memory>
#include <type_traits>

#include "accessor.h"
#include "buffer_manager.h"
#include "device_queue.h"
#include "host_object.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity {
class distr_queue;
}

namespace celerity::detail {
template <typename DataT, int Dims>
class buffer_fence_promise;
}

namespace celerity::experimental {

template <typename T, int Dims>
class buffer_snapshot {
  public:
	buffer_snapshot() : m_sr({}, detail::zero_range) {}

	explicit operator bool() const { return m_data != nullptr; }

	range<Dims> get_offset() const { return m_sr.offset; }

	range<Dims> get_range() const { return m_sr.range; }

	subrange<Dims> get_subrange() const { return m_sr; }

	const T* get_data() const { return m_data.get(); }

	std::unique_ptr<T[]> into_data() && { return std::move(m_data); }

	inline const T& operator[](id<Dims> index) const { return m_data[detail::get_linear_index(m_sr.range, index)]; }

	inline detail::subscript_result_t<Dims, const buffer_snapshot> operator[](size_t index) const { return detail::subscript<Dims>(*this, index); }

	friend bool operator==(const buffer_snapshot& lhs, const buffer_snapshot& rhs) { return lhs.m_sr == rhs.m_sr && lhs.m_data == rhs.m_data; }

	friend bool operator!=(const buffer_snapshot& lhs, const buffer_snapshot& rhs) { return !operator==(lhs, rhs); }

  private:
	template <typename U, int Dims2>
	friend class detail::buffer_fence_promise;

	subrange<Dims> m_sr;
	std::unique_ptr<T[]> m_data; // cannot use std::vector here because of vector<bool> m(

	explicit buffer_snapshot(subrange<Dims> sr, std::unique_ptr<T[]> data) : m_sr(sr), m_data(std::move(data)) {}
};

} // namespace celerity::experimental

namespace celerity::detail {

template <typename T>
class host_object_fence_promise : public detail::fence_promise {
  public:
	explicit host_object_fence_promise(const experimental::host_object<T>& obj) : m_host_object(obj) {}

	std::future<T> get_future() { return m_promise.get_future(); }

	void fulfill() override { m_promise.set_value(std::as_const(detail::get_host_object_instance(m_host_object))); }

  private:
	experimental::host_object<T> m_host_object;
	std::promise<T> m_promise;
};

template <typename DataT, int Dims>
class buffer_fence_promise : public detail::fence_promise {
  public:
	explicit buffer_fence_promise(const buffer<DataT, Dims>& buf, const subrange<Dims>& sr) : m_buffer(buf), m_subrange(sr) {}

	std::future<experimental::buffer_snapshot<DataT, Dims>> get_future() { return m_promise.get_future(); }

	void fulfill() override {
		const auto access_info = runtime::get_instance().get_buffer_manager().get_host_buffer<DataT, Dims>(
		    get_buffer_id(m_buffer), access_mode::read, range_cast<3>(m_subrange.range), id_cast<3>(m_subrange.offset));
		assert((access_info.offset <= m_subrange.offset) == id_cast<Dims>(id<3>(true, true, true)));
		auto data = std::make_unique<DataT[]>(m_subrange.range.size());
		memcpy_strided(access_info.buffer.get_pointer(), data.get(), sizeof(DataT), access_info.buffer.get_range(), m_subrange.offset - access_info.offset,
		    m_subrange.range, {}, m_subrange.range);
		m_promise.set_value(experimental::buffer_snapshot<DataT, Dims>(m_subrange, std::move(data)));
	}

  private:
	buffer<DataT, Dims> m_buffer;
	subrange<Dims> m_subrange;
	std::promise<experimental::buffer_snapshot<DataT, Dims>> m_promise;
};

class distr_queue_tracker {
  public:
	~distr_queue_tracker() { runtime::get_instance().shutdown(); }
};

template <typename CGF>
constexpr bool is_safe_cgf = std::is_standard_layout<CGF>::value;

} // namespace celerity::detail

namespace celerity {

struct allow_by_ref_t {};

inline constexpr allow_by_ref_t allow_by_ref{};

class distr_queue {
  public:
	distr_queue() { init(detail::auto_select_device{}); }

	[[deprecated("Use the overload with device selector instead, this will be removed in future release")]] distr_queue(cl::sycl::device& device) {
		if(detail::runtime::is_initialized()) { throw std::runtime_error("Passing explicit device not possible, runtime has already been initialized."); }
		init(device);
	}

	template <typename DeviceSelector>
	distr_queue(const DeviceSelector& device_selector) {
		if(detail::runtime::is_initialized()) {
			throw std::runtime_error("Passing explicit device selector not possible, runtime has already been initialized.");
		}
		init(device_selector);
	}

	distr_queue(const distr_queue&) = default;
	distr_queue(distr_queue&&) = default;

	distr_queue& operator=(const distr_queue&) = delete;
	distr_queue& operator=(distr_queue&&) = delete;

	/**
	 * Submits a command group to the queue.
	 *
	 * Invoke via `q.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {...})`.
	 *
	 * With this overload, CGF may capture by-reference. This may lead to lifetime issues with asynchronous execution, so using the `submit(cgf)` overload is
	 * preferred in the common case.
	 */
	template <typename CGF>
	void submit(allow_by_ref_t, CGF cgf) { // NOLINT(readability-convert-member-functions-to-static)
		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		detail::runtime::get_instance().get_task_manager().submit_command_group(std::move(cgf));
	}

	/**
	 * Submits a command group to the queue.
	 *
	 * CGF must not capture by reference. This is a conservative safety check to avoid lifetime issues when command groups are executed asynchronously.
	 *
	 * If you know what you are doing, you can use the `allow_by_ref` overload of `submit` to bypass this check.
	 */
	template <typename CGF>
	void submit(CGF cgf) {
		static_assert(detail::is_safe_cgf<CGF>, "The provided command group function is not multi-pass execution safe. Please make sure to only capture by "
		                                        "value. If you know what you're doing, use submit(celerity::allow_by_ref, ...).");
		submit(allow_by_ref, std::move(cgf));
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes accross the entire cluster. }
	 */
	void slow_full_sync() { detail::runtime::get_instance().sync(); } // NOLINT(readability-convert-member-functions-to-static)

  private:
	std::shared_ptr<detail::distr_queue_tracker> m_tracker;

	void init(detail::device_or_selector device_or_selector) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr, device_or_selector); }
		try {
			detail::runtime::get_instance().startup();
		} catch(detail::runtime_already_started_error&) {
			throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)");
		}
		m_tracker = std::make_shared<detail::distr_queue_tracker>();
	}
};

} // namespace celerity

namespace celerity::experimental {

template <typename T>
std::future<T> fence(celerity::distr_queue& q, const experimental::host_object<T>& obj) {
	detail::side_effect_map side_effects;
	side_effects.add_side_effect(detail::get_host_object_id(obj), experimental::side_effect_order::sequential);
	auto promise = std::make_unique<detail::host_object_fence_promise<T>>(obj);
	auto future = promise->get_future();
	detail::runtime::get_instance().get_task_manager().generate_fence_task({}, std::move(side_effects), std::move(promise));
	return future;
}

template <typename DataT, int Dims>
std::future<buffer_snapshot<DataT, Dims>> fence(celerity::distr_queue& q, const buffer<DataT, Dims>& buf, const subrange<Dims>& sr) {
	detail::buffer_access_map access_map;
	access_map.add_access(detail::get_buffer_id(buf),
	    std::make_unique<detail::range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), access_mode::read, buf.get_range()));
	auto promise = std::make_unique<detail::buffer_fence_promise<DataT, Dims>>(buf, sr);
	auto future = promise->get_future();
	detail::runtime::get_instance().get_task_manager().generate_fence_task(std::move(access_map), {}, std::move(promise));
	return future;
}

template <typename DataT, int Dims>
std::future<buffer_snapshot<DataT, Dims>> fence(celerity::distr_queue& q, const buffer<DataT, Dims>& buf) {
	return fence(q, buf, {{}, buf.get_range()});
}

} // namespace celerity::experimental
