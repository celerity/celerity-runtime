#pragma once

#include "accessor.h"
#include "host_object.h"
#include "task.h"


namespace celerity::detail {

struct capture_inspector {
  public:
	template <typename... Captures>
	static std::tuple<buffer_capture_map, side_effect_map> collect_requirements(const std::tuple<Captures...>& caps) {
		return collect_requirements(caps, std::make_index_sequence<sizeof...(Captures)>{});
	}

	template <typename... Captures>
	static auto exfiltrate_by_copy(const std::tuple<Captures...>& caps) {
		return exfiltrate_by_copy(caps, std::make_index_sequence<sizeof...(Captures)>{});
	}

	template <typename... Captures>
	static auto exfiltrate_by_move(const std::tuple<Captures...>& caps) {
		return exfiltrate_by_move(caps, std::make_index_sequence<sizeof...(Captures)>{});
	}

  private:
	template <typename... Captures, size_t... Is>
	static std::tuple<buffer_capture_map, side_effect_map> collect_requirements(const std::tuple<Captures...>& caps, std::index_sequence<Is...>) {
		buffer_capture_map bcm;
		side_effect_map sem;
		(std::get<Is>(caps).record_requirements(bcm, sem), ...);
		return {std::move(bcm), std::move(sem)};
	}

	template <typename... Captures, size_t... Is>
	static auto exfiltrate_by_copy(const std::tuple<Captures...>& caps, std::index_sequence<Is...>) {
		return std::tuple{std::get<Is>(caps).exfiltrate_by_copy()...};
	}

	template <typename... Captures, size_t... Is>
	static auto exfiltrate_by_move(const std::tuple<Captures...>& caps, std::index_sequence<Is...>) {
		return std::tuple{std::get<Is>(caps).exfiltrate_by_move()...};
	}
};

} // namespace celerity::detail

namespace celerity::experimental {

template <typename>
class capture;

template <typename T, int Dims>
class buffer_snapshot {
  public:
	buffer_snapshot() : m_sr({}, detail::zero_range) {}

	explicit operator bool() const { return !m_data.empty(); }

	range<Dims> get_offset() const { return m_sr.offset; }

	range<Dims> get_range() const { return m_sr.range; }

	subrange<Dims> get_subrange() const { return m_sr; }

	const std::vector<T>& get_data() const { return m_data; }

	std::vector<T> into_data() && { return std::move(m_data); }

	inline const T& operator[](id<Dims> index) const { return m_data[detail::get_linear_index(m_sr.range, index)]; }

	inline detail::subscript_result_t<Dims, const buffer_snapshot> operator[](size_t index) const { return detail::subscript<Dims>(*this, index); }

	friend bool operator==(const buffer_snapshot& lhs, const buffer_snapshot& rhs) { return lhs.m_sr == rhs.m_sr && lhs.m_data == rhs.m_data; }

	friend bool operator!=(const buffer_snapshot& lhs, const buffer_snapshot& rhs) { return !operator==(lhs, rhs); }

  private:
	friend class capture<buffer<T, Dims>>;

	subrange<Dims> m_sr;
	std::vector<T> m_data;

	explicit buffer_snapshot(subrange<Dims> sr, std::vector<T> data) : m_sr(sr), m_data(std::move(data)) { assert(m_data.size() == m_sr.range.size()); }
};

template <typename T, int Dims>
class capture<buffer<T, Dims>> {
  public:
	using value_type = buffer_snapshot<T, Dims>;

	explicit capture(buffer<T, Dims> buf) : m_buffer(std::move(buf)), m_sr({}, m_buffer.get_range()) {}
	explicit capture(buffer<T, Dims> buf, const subrange<Dims>& sr) : m_buffer(std::move(buf)), m_sr(sr) {}

  private:
	friend struct detail::capture_inspector;

	buffer<T, Dims> m_buffer;
	subrange<Dims> m_sr;

	void record_requirements(detail::buffer_capture_map& captures, detail::side_effect_map&) const {
		captures.add_read_access(detail::get_buffer_id(m_buffer), detail::subrange_cast<3>(m_sr));
	}

	value_type exfiltrate_by_copy() const {
		auto& bm = detail::runtime::get_instance().get_buffer_manager();
		const auto access_info =
		    bm.get_host_buffer<T, Dims>(detail::get_buffer_id(m_buffer), access_mode::read, detail::range_cast<3>(m_sr.range), detail::id_cast<3>(m_sr.offset));

		// TODO this should be able to use host_buffer_storage::get_data
		const auto allocation_window = buffer_allocation_window<T, Dims>{
		    access_info.buffer.get_pointer(),
		    m_buffer.get_range(),
		    access_info.buffer.get_range(),
		    m_sr.range,
		    access_info.offset,
		    m_sr.offset,
		};
		const auto allocation_range_3 = detail::range_cast<3>(allocation_window.get_allocation_range());
		const auto window_range_3 = detail::range_cast<3>(allocation_window.get_window_range());
		const auto read_offset_3 = detail::id_cast<3>(allocation_window.get_window_offset_in_allocation());
		std::vector<T> data(allocation_window.get_window_range().size());
		for(id<3> item{0, 0, 0}; item[0] < window_range_3[0]; ++item[0]) {
			for(item[1] = 0; item[1] < window_range_3[1]; ++item[1]) {
				for(item[2] = 0; item[2] < window_range_3[2]; ++item[2]) {
					data[detail::get_linear_index(window_range_3, item)] =
					    allocation_window.get_allocation()[detail::get_linear_index(allocation_range_3, item + read_offset_3)];
				}
			}
		}

		return value_type{m_sr, std::move(data)};
	}

	value_type exfiltrate_by_move() const { return exfiltrate_by_copy(); }
};

template <typename T, int Dims>
capture(buffer<T, Dims>) -> capture<buffer<T, Dims>>;

template <typename T, int Dims>
capture(buffer<T, Dims>, subrange<Dims>) -> capture<buffer<T, Dims>>;

template <typename T>
class capture<host_object<T>> {
  public:
	static_assert(std::is_object_v<T>);

	using value_type = T;

	explicit capture(host_object<T> ho) : m_ho(std::move(ho)) {}

  private:
	friend struct detail::capture_inspector;

	host_object<T> m_ho;

	void record_requirements(detail::buffer_capture_map&, detail::side_effect_map& side_effects) const {
		side_effects.add_side_effect(m_ho.get_id(), side_effect_order::sequential);
	}

	value_type exfiltrate_by_copy() const { return value_type{std::as_const(*m_ho.get_object())}; }

	value_type exfiltrate_by_move() const { return value_type{std::move(*m_ho.get_object())}; }
};

template <typename T>
capture(host_object<T>) -> capture<host_object<T>>;

} // namespace celerity::experimental
