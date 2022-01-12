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
class buffer_data {
  public:
	buffer_data() : range{detail::zero_range} {}

	explicit operator bool() const { return !data.empty(); }

	celerity::range<Dims> get_range() const { return range; }
	const T* get_pointer() const { return data.data(); }
	T* get_pointer() { return data.data(); }

	// TODO accessor semantics with operator[]; into_vector()

  private:
	friend class capture<buffer<T, Dims>>;

	celerity::range<Dims> range;
	std::vector<T> data;

	explicit buffer_data(celerity::range<Dims> range, std::vector<T> data) : range{range}, data{std::move(data)} {
		assert(this->data.size() == this->range.size());
	}
};

template <typename T, int Dims>
class capture<buffer<T, Dims>> {
  public:
	using value_type = buffer_data<T, Dims>;

	explicit capture(buffer<T, Dims> buf) : buffer{std::move(buf)}, sr{{}, buffer.get_range()} {}
	explicit capture(buffer<T, Dims> buf, const subrange<Dims>& sr) : buffer{std::move(buf)}, sr{sr} {}

  private:
	friend struct detail::capture_inspector;

	buffer<T, Dims> buffer;
	subrange<Dims> sr;

	void record_requirements(detail::buffer_capture_map& accesses, detail::side_effect_map&) const {
		accesses.add_read_access(detail::get_buffer_id(buffer), sr);
	}

	value_type exfiltrate_by_copy() const {
		auto& bm = detail::runtime::get_instance().get_buffer_manager();
		const auto access_info =
		    bm.get_host_buffer<T, Dims>(detail::get_buffer_id(buffer), access_mode::read, detail::range_cast<3>(sr.range), detail::id_cast<3>(sr.offset));

		// TODO this should be able to use host_buffer_storage::get_data
		const auto allocation_window = buffer_allocation_window<T, Dims>{
		    access_info.buffer.get_pointer(),
		    buffer.get_range(),
		    access_info.buffer.get_range(),
		    sr.range,
		    access_info.offset,
		    sr.offset,
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

		return value_type{allocation_window.get_window_range(), std::move(data)};
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

	explicit capture(host_object<T> ho) : ho{std::move(ho)} {}

  private:
	friend struct detail::capture_inspector;

	host_object<T> ho;

	void record_requirements(detail::buffer_capture_map&, detail::side_effect_map& side_effects) const {
		side_effects.add_side_effect(ho.get_id(), side_effect_order::sequential);
	}

	value_type exfiltrate_by_copy() const { return value_type{std::as_const(*ho.get_object())}; }

	value_type exfiltrate_by_move() const { return value_type{std::move(*ho.get_object())}; }
};

template <typename T>
capture(host_object<T>) -> capture<host_object<T>>;

} // namespace celerity::experimental
