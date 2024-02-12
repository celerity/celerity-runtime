#pragma once

#include <type_traits>

#include <CL/sycl.hpp>

#include "access_modes.h"
#include "buffer.h"
#include "buffer_storage.h"
#include "cgf_diagnostics.h"
#include "closure_hydrator.h"
#include "handler.h"
#include "sycl_wrappers.h"

namespace celerity {

template <int Dims>
class partition;

template <typename DataT, int Dims, access_mode Mode, target Target>
class accessor;

namespace detail {

	template <typename DataT, int Dims, access_mode Mode, target Target>
	class accessor_base {
	  public:
		static_assert(Dims <= 3, "accessors can only have 3 dimensions or less");
		using value_type = DataT;
		using reference = DataT&;
		using const_reference = const DataT&;
	};

	struct accessor_testspy;

} // namespace detail
} // namespace celerity

namespace celerity {

/**
 * In addition to the usual per-item access through the subscript operator, accessors in distributed and collective host tasks can access the underlying memory
 * of the node-local copy of a buffer directly through `accessor::get_allocation_window()`. Celerity does not replicate buffers fully on all nodes unless
 * necessary, instead keeping an allocation of a subset that is resized as needed.
 *
 * buffer_allocation_window denotes how indices in the subrange assigned to one node (the _window_) map to the underlying buffer storage (the
 * _allocation_). The structure threrefore describes three subranges: The buffer, the allocation, and the window, with the latter being fully contained in the
 * former. Popular third-party APIs, such as HDF5 hyperslabs, can accept parameters from such an explicit description in one or the other form.
 */
template <typename T, int Dims>
class buffer_allocation_window {
  public:
	T* get_allocation() const { return m_allocation; }

	range<Dims> get_buffer_range() const { return m_buffer_range; }

	range<Dims> get_allocation_range() const { return m_allocation_range; }

	range<Dims> get_window_range() const { return m_window_range; }

	id<Dims> get_allocation_offset_in_buffer() const { return m_allocation_offset_in_buffer; }

	id<Dims> get_window_offset_in_buffer() const { return m_window_offset_in_buffer; }

	id<Dims> get_window_offset_in_allocation() const { return m_window_offset_in_buffer - m_allocation_offset_in_buffer; }

  private:
	T* m_allocation;
	range<Dims> m_buffer_range;
	range<Dims> m_allocation_range;
	range<Dims> m_window_range;
	id<Dims> m_allocation_offset_in_buffer;
	id<Dims> m_window_offset_in_buffer;

  public:
	buffer_allocation_window(T* allocation, const range<Dims>& buffer_range, const range<Dims>& allocation_range, const range<Dims>& window_range,
	    const id<Dims>& allocation_offset_in_buffer, const id<Dims>& window_offset_in_buffer)
	    : m_allocation(allocation), m_buffer_range(buffer_range), m_allocation_range(allocation_range), m_window_range(window_range),
	      m_allocation_offset_in_buffer(allocation_offset_in_buffer), m_window_offset_in_buffer(window_offset_in_buffer) {}

	template <typename, int, access_mode, target>
	friend class accessor;
};

#define CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR [[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]]

/**
 * Celerity wrapper around SYCL accessors.
 *
 * @note The Celerity accessor currently does not support get_size, get_count, get_range, get_offset and get_pointer,
 * as their semantics in a distributed context are unclear.
 */
template <typename DataT, int Dims, access_mode Mode>
class accessor<DataT, Dims, Mode, target::device> : public detail::accessor_base<DataT, Dims, Mode, target::device> {
	friend struct detail::accessor_testspy;

	struct ctor_internal_tag {};

  public:
	static_assert(Mode != access_mode::atomic, "access_mode::atomic is not supported. Please use atomic_ref instead.");

	accessor() noexcept = default;

	template <typename Functor>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, TagModeNoInit, target::device> /* tag */)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::device> /* tag */,
	    const property::no_init& /* no_init */)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh) : accessor(buff, cgh, access::all()) {}

	template <access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<Mode, TagModeNoInit, target::device> tag)
	    : accessor(buff, cgh, access::all(), tag) {}

	template <access_mode TagMode, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& no_init)
	    : accessor(buff, cgh, access::all(), tag, no_init) {}

	template <access_mode TagMode, access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : accessor(buff, cgh, access::all(), tag, prop_list) {}

	template <typename Functor>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagModeNoInit>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, TagModeNoInit, target::device> /* tag */)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::device> /* tag */, const property::no_init& /* no_init */)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode, access_mode TagModeNoInit>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, TagModeNoInit, target::device> /* tag */, const property_list& /* prop_list */) {
		static_assert(detail::constexpr_false<Functor>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property celerity::no_init "
		    "as a last argument in the constructor");
	}

	// explicitly defaulted because we define operator=(value_type) for Dims == 0
	accessor(accessor&&) noexcept = default;
	accessor& operator=(accessor&&) noexcept = default;

#if !defined(__SYCL_DEVICE_ONLY__)
	accessor(const accessor& other) { copy_and_hydrate(other); }

	accessor& operator=(const accessor& other) {
		if(this != &other) { copy_and_hydrate(other); }
		return *this;
	}
#else
	accessor(const accessor&) = default;
	accessor& operator=(const accessor&) = default;
#endif

	// SYCL allows assigning values to accessors directly in the 0-dimensional case

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	const accessor& operator=(const DataT& other) const {
		*m_device_ptr = other;
		return *this;
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	const accessor& operator=(DataT&& other) const {
		*m_device_ptr = std::move(other);
		return *this;
	}

	template <access_mode M = Mode>
	inline std::conditional_t<detail::access::mode_traits::is_producer(M), DataT&, const DataT&> operator[](const id<Dims>& index) const {
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		// We currently don't support boundary checking for accessors created using accessor_testspy::make_device_accessor,
		// which does not set m_oob_indices.
		if(m_oob_indices != nullptr) {
			const bool is_within_bounds_lo = all_true(index >= m_accessed_virtual_subrange.offset);
			const bool is_within_bounds_hi = all_true(index < (m_accessed_virtual_subrange.offset + m_accessed_virtual_subrange.range));
			if((!is_within_bounds_lo || !is_within_bounds_hi)) {
				for(int d = 0; d < Dims; ++d) {
					sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>{m_oob_indices[0][d]}.fetch_min(index[d]);
					sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device>{m_oob_indices[1][d]}.fetch_max(index[d] + 1);
				}
				return m_oob_fallback_value;
			}
		}
#endif
		return m_device_ptr[get_linear_offset(index)];
	}

	template <int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t index) const {
		return detail::subscript<D>(*this, index);
	}

	template <access_mode M = Mode, std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), int> = 0>
	inline operator DataT&() const {
		return *m_device_ptr;
	}

	template <access_mode M = Mode, std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), int> = 0>
	inline operator const DataT&() const {
		return *m_device_ptr;
	}

	// we provide operator* and operator-> in addition to SYCL's operator reference() as we feel it better represents the pointer semantics of accessors

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), DataT&> operator*() const {
		return *m_device_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator*() const {
		return *m_device_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), DataT*> operator->() const {
		return m_device_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), const DataT*> operator->() const {
		return m_device_ptr;
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.m_device_ptr == rhs.m_device_ptr && lhs.m_backing_buffer_range == rhs.m_backing_buffer_range
		       && lhs.m_backing_buffer_offset == rhs.m_backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

  private:
	DataT* m_device_ptr = nullptr;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> m_backing_buffer_offset;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_backing_buffer_range = detail::zeros;
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	id<3>* m_oob_indices = nullptr;
	subrange<Dims> m_accessed_virtual_subrange = {};
	// This value (or a reference to it) is returned for all out-of-bounds accesses.
	mutable DataT m_oob_fallback_value = DataT{};
#endif

	template <typename Functor>
	accessor(const ctor_internal_tag /* tag */, const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) {
		using range_mapper = detail::range_mapper<Dims, std::decay_t<Functor>>; // decay function type to function pointer
		const auto hid = detail::add_requirement(cgh, detail::get_buffer_id(buff), std::make_unique<range_mapper>(rmfn, Mode, buff.get_range()));
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(buff));
		m_device_ptr = detail::embed_hydration_id<DataT*>(hid);
	}

	// Constructor for tests, called through accessor_testspy.
	accessor(DataT* const ptr, const id<Dims>& backing_buffer_offset, const range<Dims>& backing_buffer_range)
	    : m_device_ptr(ptr), m_backing_buffer_offset(backing_buffer_offset), m_backing_buffer_range(backing_buffer_range) {
#if defined(__SYCL_DEVICE_ONLY__)
#if CELERITY_WORKAROUND_HIPSYCL // hipSYCL does not yet implement is_device_copyable_v
		static_assert(std::is_trivially_copyable_v<accessor>);
#else
		static_assert(sycl::is_device_copyable_v<accessor>);
#endif
#endif
	}

	// Constructor for tests, called through accessor_testspy.
	accessor(const detail::hydration_id hid, const id<Dims>& backing_buffer_offset, const range<Dims>& backing_buffer_range)
	    : accessor(detail::embed_hydration_id<DataT*>(hid), backing_buffer_offset, backing_buffer_range) {}

	void copy_and_hydrate(const accessor& other) {
		m_device_ptr = other.m_device_ptr;
		m_backing_buffer_offset = other.m_backing_buffer_offset;
		m_backing_buffer_range = other.m_backing_buffer_range;
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		m_oob_indices = other.m_oob_indices;
		m_accessed_virtual_subrange = other.m_accessed_virtual_subrange;
#endif

#if !defined(__SYCL_DEVICE_ONLY__)
		if(detail::is_embedded_hydration_id(m_device_ptr)) {
			if(detail::cgf_diagnostics::is_available() && detail::cgf_diagnostics::get_instance().is_checking()) {
				detail::cgf_diagnostics::get_instance().register_accessor(detail::extract_hydration_id(m_device_ptr), target::device);
			}

			if(detail::closure_hydrator::is_available() && detail::closure_hydrator::get_instance().is_hydrating()) {
				const auto info = detail::closure_hydrator::get_instance().get_accessor_info<target::device>(detail::extract_hydration_id(m_device_ptr));
				m_device_ptr = static_cast<DataT*>(info.ptr);
				m_backing_buffer_offset = detail::id_cast<Dims>(info.backing_buffer_offset);
				m_backing_buffer_range = detail::range_cast<Dims>(info.backing_buffer_range);
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
				m_oob_indices = info.out_of_bounds_indices;
				m_accessed_virtual_subrange = detail::subrange_cast<Dims>(info.accessed_virtual_subrange);
#endif
			}
		}
#endif
	}

	size_t get_linear_offset(const id<Dims>& index) const { return detail::get_linear_index(m_backing_buffer_range, index - m_backing_buffer_offset); }
};

template <typename DataT, int Dims, access_mode Mode>
class accessor<DataT, Dims, Mode, target::host_task> : public detail::accessor_base<DataT, Dims, Mode, target::host_task> {
	friend struct detail::accessor_testspy;

	struct ctor_internal_tag {};

  public:
	static_assert(Mode != access_mode::atomic, "access_mode::atomic is not supported.");

	accessor() noexcept = default;

	template <typename Functor>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, TagModeNoInit, target::host_task> /* tag */)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	/**
	 * TODO: As of ComputeCpp 2.5.0 they do not support no_init prop, hence this constructor is needed along with discard deduction guide.
	 *    but once they do this should be replace for a constructor that takes a prop list as an argument.
	 */
	template <typename Functor, access_mode TagMode, access_mode M = Mode, typename = std::enable_if_t<detail::access::mode_traits::is_producer(M)>>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::host_task> /* tag */,
	    const property::no_init& /* no_init */)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh) : accessor(buff, cgh, access::all()) {}

	template <access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<Mode, TagModeNoInit, target::host_task> tag)
	    : accessor(buff, cgh, access::all(), tag) {}

	template <access_mode TagMode, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, Mode, target::host_task> tag, const property::no_init& no_init)
	    : accessor(buff, cgh, access::all(), tag, no_init) {}

	template <access_mode TagMode, access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag, const property_list& prop_list)
	    : accessor(buff, cgh, access::all(), tag, prop_list) {}

	template <typename Functor>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagModeNoInit>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, TagModeNoInit, target::host_task> /* tag */)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	/**
	 * TODO: As of ComputeCpp 2.5.0 they do not support no_init prop, hence this constructor is needed along with discard deduction guide.
	 *    but once they do this should be replace for a constructor that takes a prop list as an argument.
	 */
	template <typename Functor, access_mode TagMode, access_mode M = Mode, typename = std::enable_if_t<detail::access::mode_traits::is_producer(M)>>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, Mode, target::host_task> /* tag */, const property::no_init& /* no_init */)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode, access_mode TagModeNoInit>
	CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn,
	    const detail::access_tag<TagMode, TagModeNoInit, target::host_task> /* tag */, const property_list& /* prop_list */) {
		static_assert(detail::constexpr_false<Functor>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property celerity::no_init "
		    "as a last argument in the constructor");
	}

	// explicitly defaulted because we define operator=(value_type) for Dims == 0
	accessor(accessor&&) noexcept = default;
	accessor& operator=(accessor&&) noexcept = default;

	accessor(const accessor& other) { copy_and_hydrate(other); }

	accessor& operator=(const accessor& other) {
		if(this != &other) { copy_and_hydrate(other); }
		return *this;
	}

	// SYCL allows assigning values to accessors directly in the 0-dimensional case

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	const accessor& operator=(const DataT& other) const {
		*m_host_ptr = other;
		return *this;
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	const accessor& operator=(DataT&& other) const {
		*m_host_ptr = std::move(other);
		return *this;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<detail::access::mode_traits::is_producer(M), DataT&> operator[](const id<Dims>& index) const {
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		if(m_oob_indices != nullptr) {
			const bool is_within_bounds_lo = all_true(index >= m_accessed_virtual_subrange.offset);
			const bool is_within_bounds_hi = all_true(index < (m_accessed_virtual_subrange.offset + m_accessed_virtual_subrange.range));

			if((!is_within_bounds_lo || !is_within_bounds_hi)) {
				std::lock_guard<std::mutex> guard(m_oob_mutex);
				for(int d = 0; d < Dims; ++d) {
					m_oob_indices[0][d] = std::min(m_oob_indices[0][d], index[d]);
					m_oob_indices[1][d] = std::max(m_oob_indices[1][d], index[d] + 1);
				}
				return m_oob_fallback_value;
			}
		}
#endif

		return m_host_ptr[get_linear_offset(index)];
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator[](const id<Dims>& index) const {
		return m_host_ptr[get_linear_offset(index)];
	}

	template <int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t index) const {
		return detail::subscript<D>(*this, index);
	}

	template <access_mode M = Mode, std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), int> = 0>
	inline operator DataT&() const {
		return *m_host_ptr;
	}

	template <access_mode M = Mode, std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), int> = 0>
	inline operator const DataT&() const {
		return *m_host_ptr;
	}

	// we provide operator* and operator-> in addition to SYCL's operator reference() as we feel it better represents the pointer semantics of accessors

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), DataT&> operator*() const {
		return *m_host_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator*() const {
		return *m_host_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), DataT*> operator->() const {
		return m_host_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), const DataT*> operator->() const {
		return m_host_ptr;
	}

	/**
	 * @brief Returns a pointer to the underlying buffer.
	 *
	 * Since the physical (or "backing") buffer underlying a Celerity buffer might have a different size,
	 * it is not always safe to return a pointer, as the stride might not be what is expected.
	 *
	 * However, it is always possible to get a pointer if the full buffer is being accessed.
	 *
	 * This API will be deprecated and subsequently removed in a future version.
	 *
	 * @throws Throws a std::logic_error if the buffer cannot be accessed with the expected stride.
	 */
	DataT* get_pointer() const {
		bool illegal_access = false;
		if(m_backing_buffer_offset != detail::id_cast<Dims>(id<3>{0, 0, 0})) { illegal_access = true; }
		// We can be a bit more lenient for 1D buffers, in that the backing buffer doesn't have to have the full size.
		// (Dereferencing the pointer outside of the requested range is UB anyways).
		if(Dims > 1 && m_backing_buffer_range != m_virtual_buffer_range) { illegal_access = true; }
		if(illegal_access) { throw std::logic_error("Buffer cannot be accessed with expected stride"); }
		return m_host_ptr;
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.m_host_ptr == rhs.m_host_ptr && lhs.m_accessed_virtual_subrange == rhs.m_accessed_virtual_subrange
		       && lhs.m_backing_buffer_range == rhs.m_backing_buffer_range && lhs.m_virtual_buffer_range == rhs.m_virtual_buffer_range
		       && lhs.m_backing_buffer_offset == rhs.m_backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

	/**
	 * Returns a pointer to the buffer allocation local to this node along with a description how this potentially smaller allocation maps to the underlying
	 * virtual buffer and the partition available in the current host task.
	 *
	 * Accessing the returned allocation outside the window will lead to undefined results.
	 */
	template <int KernelDims>
	buffer_allocation_window<DataT, Dims> get_allocation_window(const partition<KernelDims>& part) const {
		// We already know the range mapper output for "chunk" from the constructor. The parameter is a purely semantic dependency which ensures that
		// this function is not called outside a host task.
		(void)part;

		return {
		    m_host_ptr,
		    m_virtual_buffer_range,
		    m_backing_buffer_range,
		    m_accessed_virtual_subrange.range,
		    m_backing_buffer_offset,
		    m_accessed_virtual_subrange.offset,
		};
	}

  private:
	// Subange of the accessor, as set by the range mapper or requested by the user (master node host tasks only).
	// This does not necessarily correspond to the backing buffer's range.
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS subrange<Dims> m_accessed_virtual_subrange;

	// Offset of the backing buffer relative to the virtual buffer.
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> m_backing_buffer_offset;

	// Range of the backing buffer.
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_backing_buffer_range = detail::zeros;

	// The range of the Celerity buffer as created by the user.
	// We only need this to check whether it is safe to call get_pointer() or not.
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_virtual_buffer_range = detail::zeros;

	// m_host_ptr must be defined *last* for it to overlap with the sequence of range and id members in the 0-dimensional case
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS DataT* m_host_ptr = nullptr;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	id<3>* m_oob_indices = nullptr;
	// This mutex has to be inline static, since accessors are copyable making the mutex otherwise useless.
	// It is a workaround until atomic_ref() can be used on m_oob_indices in c++20.
	inline static std::mutex m_oob_mutex;

	// This value (or a reference to it) is returned for all out-of-bounds accesses.
	mutable DataT m_oob_fallback_value = DataT{};
#endif

	template <target Target = target::host_task, typename Functor>
	accessor(ctor_internal_tag /* tag */, const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) : m_virtual_buffer_range(buff.get_range()) {
		using range_mapper = detail::range_mapper<Dims, std::decay_t<Functor>>; // decay function type to function pointer
		const auto hid = detail::add_requirement(cgh, detail::get_buffer_id(buff), std::make_unique<range_mapper>(rmfn, Mode, buff.get_range()));
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(buff));
		m_host_ptr = detail::embed_hydration_id<DataT*>(hid);
	}

	// Constructor for tests, called through accessor_testspy.
	accessor(const subrange<Dims> accessed_virtual_subrange, DataT* const ptr, const id<Dims>& backing_buffer_offset, const range<Dims>& backing_buffer_range,
	    const range<Dims>& virtual_buffer_range)
	    : m_accessed_virtual_subrange(accessed_virtual_subrange), m_backing_buffer_offset(backing_buffer_offset), m_backing_buffer_range(backing_buffer_range),
	      m_virtual_buffer_range(virtual_buffer_range), m_host_ptr(ptr) {}

	// Constructor for tests, called through accessor_testspy.
	accessor(const subrange<Dims>& accessed_virtual_subrange, const detail::hydration_id hid, const id<Dims>& backing_buffer_offset,
	    const range<Dims>& backing_buffer_range, range<Dims> virtual_buffer_range)
	    : accessor(accessed_virtual_subrange, detail::embed_hydration_id<DataT*>(hid), backing_buffer_offset, backing_buffer_range, virtual_buffer_range) {}

	void copy_and_hydrate(const accessor& other) {
		m_accessed_virtual_subrange = other.m_accessed_virtual_subrange;
		m_host_ptr = other.m_host_ptr;
		m_backing_buffer_offset = other.m_backing_buffer_offset;
		m_backing_buffer_range = other.m_backing_buffer_range;
		m_virtual_buffer_range = other.m_virtual_buffer_range;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		m_oob_indices = other.m_oob_indices;
#endif

		if(detail::is_embedded_hydration_id(m_host_ptr)) {
			if(detail::cgf_diagnostics::is_available() && detail::cgf_diagnostics::get_instance().is_checking()) {
				detail::cgf_diagnostics::get_instance().register_accessor(detail::extract_hydration_id(m_host_ptr), target::host_task);
			}

			if(detail::closure_hydrator::is_available() && detail::closure_hydrator::get_instance().is_hydrating()) {
				const auto info = detail::closure_hydrator::get_instance().get_accessor_info<target::host_task>(detail::extract_hydration_id(m_host_ptr));
				m_host_ptr = static_cast<DataT*>(info.ptr);
				m_backing_buffer_offset = detail::id_cast<Dims>(info.backing_buffer_offset);
				m_backing_buffer_range = detail::range_cast<Dims>(info.backing_buffer_range);
				m_accessed_virtual_subrange = detail::subrange_cast<Dims>(info.accessed_virtual_subrange);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
				m_oob_indices = info.out_of_bounds_indices;
#endif
			}
		}
	}

	size_t get_linear_offset(const id<Dims>& index) const { return detail::get_linear_index(m_backing_buffer_range, index - m_backing_buffer_offset); }
};

#undef CELERITY_DETAIL_ACCESSOR_DEPRECATED_CTOR

// TODO: Make buffer non-const once corresponding (deprecated!) constructor overloads are removed
template <typename T, int D, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<T, D>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, Target> tag) -> accessor<T, D, Mode, Target>;

template <typename T, int D, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<T, D>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, Target> tag, const property::no_init no_init)
    -> accessor<T, D, ModeNoInit, Target>;

template <typename T, int D, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<T, D>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, Target> tag, const property_list& props)
    -> accessor<T, D, Mode, Target>;

template <typename T, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<T, 0>& buff, handler& cgh, const detail::access_tag<Mode, ModeNoInit, Target> tag) -> accessor<T, 0, Mode, Target>;

template <typename T, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<T, 0>& buff, handler& cgh, const detail::access_tag<Mode, ModeNoInit, Target> tag, const property::no_init no_init)
    -> accessor<T, 0, ModeNoInit, Target>;

template <typename T, access_mode Mode, access_mode ModeNoInit, target Target>
accessor(const buffer<T, 0>& buff, handler& cgh, const detail::access_tag<Mode, ModeNoInit, Target> tag, const property_list& props)
    -> accessor<T, 0, Mode, Target>;


template <typename DataT, int Dims = 1>
class local_accessor {
	friend struct detail::accessor_testspy;

	static_assert(Dims <= 3);
	friend struct detail::accessor_testspy;

  private:
	constexpr static int sycl_dims = std::max(1, Dims);

	using sycl_accessor = cl::sycl::local_accessor<DataT, sycl_dims>;

  public:
	using value_type = DataT;
	using reference = DataT&;
	using const_reference = const DataT&;
	using size_type = size_t;

	local_accessor() : m_sycl_acc{}, m_allocation_size(detail::zeros) {}

	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	local_accessor(handler& cgh) : local_accessor(range<0>(), cgh) {}

#if !defined(__SYCL_DEVICE_ONLY__)
	local_accessor(const range<Dims>& allocation_size, handler& cgh) : m_sycl_acc{}, m_allocation_size(allocation_size) {}

	local_accessor(const local_accessor& other)
	    : m_sycl_acc(
	        detail::closure_hydrator::is_available() && detail::closure_hydrator::get_instance().is_hydrating() && other.sycl_allocation_size().size() > 0
	            ? sycl_accessor{other.sycl_allocation_size(), detail::closure_hydrator::get_instance().get_sycl_handler()}
	            : other.m_sycl_acc),
	      m_allocation_size(other.m_allocation_size) {}
#else
	local_accessor(const range<Dims>& allocation_size, handler& cgh);
	local_accessor(const local_accessor&) = default;
#endif

	local_accessor& operator=(const local_accessor&) = default;
	local_accessor& operator=(local_accessor&&) = default;

	// SYCL allows assigning values to accessors directly in the 0-dimensional case

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	const local_accessor& operator=(const DataT& other) const {
		m_sycl_acc[sycl::id<1>(0)] = other;
		return *this;
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	const local_accessor& operator=(DataT&& other) const {
		m_sycl_acc[sycl::id<1>(0)] = std::move(other);
		return *this;
	}

	size_type byte_size() const noexcept { return m_allocation_size.size() * sizeof(value_type); }

	size_type size() const noexcept { return m_allocation_size.size(); }

	size_type max_size() const noexcept { return m_sycl_acc.max_size(); }

	bool empty() const noexcept { return m_sycl_acc.empty(); }

	range<Dims> get_range() const { return m_allocation_size; }

	std::add_pointer_t<value_type> get_pointer() const noexcept { return m_sycl_acc.get_pointer(); }

	inline DataT& operator[](const id<Dims>& index) const {
		if constexpr(Dims == 0) {
			return m_sycl_acc[sycl::id<1>(0)];
		} else {
			return m_sycl_acc[sycl::id<Dims>(index)];
		}
	}

	template <int D = Dims, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t dim0) const {
		return detail::subscript<Dims>(*this, dim0);
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	operator DataT&() const {
		return m_sycl_acc[sycl::id<1>(0)];
	}

	// we provide operator* and operator-> in addition to SYCL's operator reference() as we feel it better represents the pointer semantics of accessors

	template <int D = Dims>
	std::enable_if_t<D == 0, DataT&> operator*() const {
		return m_sycl_acc[sycl::id<1>(0)];
	}

	template <int D = Dims>
	std::enable_if_t<D == 0, DataT*> operator->() const {
		return &m_sycl_acc[sycl::id<1>(0)];
	}

  private:
	sycl_accessor m_sycl_acc;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_allocation_size;

	sycl::range<sycl_dims> sycl_allocation_size() const { return sycl::range<sycl_dims>(detail::range_cast<sycl_dims>(m_allocation_size)); }

	// Constructor for tests, called through accessor_testspy.
	explicit local_accessor(const range<Dims>& allocation_size) : m_sycl_acc{}, m_allocation_size(allocation_size) {}
};

} // namespace celerity
