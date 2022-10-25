#pragma once

#include <type_traits>

#include <CL/sycl.hpp>

#include "access_modes.h"
#include "buffer.h"
#include "buffer_storage.h"
#include "handler.h"
#include "sycl_wrappers.h"
#include "task_hydrator.h"

namespace celerity {

template <int Dims>
class partition;

template <typename DataT, int Dims, access_mode Mode, target Target>
class accessor;

namespace detail {

	template <typename DataT, int Dims, access_mode Mode, target Target>
	class accessor_base {
	  public:
		static_assert(Dims > 0, "0-dimensional accessors NYI");
		static_assert(Dims <= 3, "accessors can only have 3 dimensions or less");
		using value_type = DataT;
		using reference = DataT&;
		using const_reference = const DataT&;
	};

	template <typename DataT, int Dims, access_mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, target::device> make_device_accessor(Args&&...);

	template <typename DataT, int Dims, access_mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, target::host_task> make_host_accessor(Args&&...);

	template <typename TagT>
	constexpr access_mode deduce_access_mode();

	template <typename TagT>
	constexpr access_mode deduce_access_mode_discard();

	template <typename TagT>
	constexpr target deduce_access_target();

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, target Target, int Index>
	class accessor_subscript_proxy;

	struct accessor_testspy;

// Hack: DPC++ and ComputeCpp do not implement the SYCL 2020 sycl::local_accessor default constructor yet and always require a handler for construction.
// Since there is no SYCL handler in the pre-pass, we abuse inheritance and friend declarations to conjure a temporary sycl::handler that can be passed to the
// existing local_accessor constructor. This works because neither accessor implementation keep references to the handler internally or interacts with context
// surrounding the fake handler instance.
#if CELERITY_WORKAROUND(DPCPP)
	// The DPC++ handler declares `template<...> friend class accessor<...>`, so we specialize sycl::accessor with a made-up type and have it inherit from
	// sycl::handler in order to be able to call the private constructor of sycl::handler.
	struct hack_accessor_specialization_type {};
	using hack_null_sycl_handler = cl::sycl::accessor<celerity::detail::hack_accessor_specialization_type, 0, cl::sycl::access::mode::read,
	    cl::sycl::access::target::host_buffer, cl::sycl::access::placeholder::true_t, void>;
#elif CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 9)
	// ComputeCpp's sycl::handler has a protected constructor, so we expose it to the public through inheritance.
	class hack_null_sycl_handler : public sycl::handler {
	  public:
		hack_null_sycl_handler() : sycl::handler(nullptr) {}
	};
#endif

} // namespace detail
} // namespace celerity

#if CELERITY_WORKAROUND(DPCPP)
// See declaration of celerity::detail::hack_accessor_specialization_type
template <>
class cl::sycl::accessor<celerity::detail::hack_accessor_specialization_type, 0, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer,
    cl::sycl::access::placeholder::true_t, void> : public handler {
  public:
	accessor() : handler{nullptr, false} {}
};
#endif

namespace celerity {

/**
 * Maps slices of the accessor backing buffer present on a host to the virtual global range of the Celerity buffer.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class [[deprecated("host_memory_layout will be removed in favor of buffer_allocation_window in a future version of Celerity")]] host_memory_layout {
  public:
	/**
	 * Layout map for a single dimension describing the offset and strides of its hyperplanes.
	 *
	 * - A zero-dimensional layout corresponds to an individual data item and is not explicitly modelled in the dimension vector.
	 * - A one-dimensional layout is an interval of one-dimensional space and is fully described by global and local offsets and a count of data items (aka
	 * 0-dimensional hyperplanes).
	 * - A two-dimensional layout is modelled as an interval of rows, which manifests as an offset (a multiple of the row width) and a stride (the row width
	 * itself). Each row (aka 1-dimensional hyperplane) is modelled by the same one-dimensional layout.
	 * - and so on for arbitrary dimensioned layouts.
	 */
	class [[deprecated("host_memory_layout will be removed in favor of buffer_allocation_window in a future version of Celerity")]] dimension {
	  public:
		dimension() noexcept = default;

		dimension(size_t global_size, size_t global_offset, size_t local_size, size_t local_offset, size_t extent)
		    : m_global_size(global_size), m_global_offset(global_offset), m_local_size(local_size), m_local_offset(local_offset), m_extent(extent) {
			assert(global_offset >= local_offset);
			assert(global_size >= local_size);
		}

		size_t get_global_size() const { return m_global_size; }

		size_t get_local_size() const { return m_local_size; }

		size_t get_global_offset() const { return m_global_offset; }

		size_t get_local_offset() const { return m_local_offset; }

		size_t get_extent() const { return m_extent; }

	  private:
		size_t m_global_size{};
		size_t m_global_offset{};
		size_t m_local_size{};
		size_t m_local_offset{};
		size_t m_extent{};
	};

	class [[deprecated("host_memory_layout will be removed in favor of buffer_allocation_window in a future version of Celerity")]] dimension_vector {
	  public:
		dimension_vector(size_t size) : m_this_size(size) {}

		dimension& operator[](size_t idx) { return m_values[idx]; }
		const dimension& operator[](size_t idx) const { return m_values[idx]; }

		size_t size() const { return m_this_size; }

	  private:
		/**
		 * Since contiguous dimensions can be merged when generating the memory layout, host_memory_layout is not generic over a fixed dimension count
		 */
		constexpr static size_t max_dimensionality = 4;
		std::array<dimension, max_dimensionality> m_values;
		size_t m_this_size;
	};

	explicit host_memory_layout(const dimension_vector& dimensions) : m_dimensions(dimensions) {}

	/** The layout maps per dimension, in descending dimensionality */
	const dimension_vector& get_dimensions() const { return m_dimensions; }

  private:
	dimension_vector m_dimensions;
};
#pragma GCC diagnostic pop

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

	template <target Target = target::device, typename Functor>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, property::no_init no_init) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, property_list prop_list) {
		static_assert(detail::constexpr_false<TagT>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property celerity::no_init "
		    "as a last argument in the constructor");
	}

	template <target Target = target::device, typename Functor>
	[[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]] accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	[[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]] accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	[[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]] accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, property::no_init no_init)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

#if !defined(__SYCL_DEVICE_ONLY__) && !defined(SYCL_DEVICE_ONLY)
	accessor(const accessor& other) { copy_and_hydrate(other); }

	accessor& operator=(const accessor& other) {
		if(this != &other) { copy_and_hydrate(other); }
		return *this;
	}
#endif

	template <access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_producer(M) && M != access_mode::atomic && (D > 0), DataT&> operator[](id<Dims> index) const {
		return m_device_ptr[get_linear_offset(index)];
	}

	template <access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M) && (D > 0), const DataT&> operator[](id<Dims> index) const {
		return m_device_ptr[get_linear_offset(index)];
	}

	template <int D = Dims>
	std::enable_if_t<(D > 1), detail::accessor_subscript_proxy<DataT, D, Mode, target::device, 1>> operator[](const size_t d0) const {
		return {*this, d0};
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.m_device_ptr == rhs.m_device_ptr && lhs.m_buffer_range == rhs.m_buffer_range && lhs.m_index_offset == rhs.m_index_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

  private:
	DataT* m_device_ptr = nullptr;
	sycl::id<Dims> m_index_offset;
	sycl::range<Dims> m_buffer_range = detail::zero_range;

	template <typename Functor>
	accessor(const ctor_internal_tag, const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn) {
		const auto hid =
		    detail::add_requirement(cgh, detail::get_buffer_id(buff), std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, buff.get_range()));
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(buff));
		m_device_ptr = detail::embed_hydration_id<DataT*>(hid);
	}

	// Constructor for tests, called through accessor_testspy.
	accessor(DataT* ptr, id<Dims> index_offset, range<Dims> buffer_range) : m_device_ptr(ptr), m_index_offset(index_offset), m_buffer_range(buffer_range) {
#if defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_DEVICE_ONLY)
#if CELERITY_WORKAROUND_HIPSYCL
		static_assert(std::is_trivially_copyable_v<accessor>);
#else
		static_assert(sycl::is_device_copyable_v<accessor>);
#endif
#endif
	}

	// Constructor for tests, called through accessor_testspy.
	accessor(detail::hydration_id hid, id<Dims> index_offset, range<Dims> buffer_range)
	    : accessor(detail::embed_hydration_id<DataT*>(hid), index_offset, buffer_range) {}

	void copy_and_hydrate(const accessor& other) {
		m_device_ptr = other.m_device_ptr;
		m_index_offset = other.m_index_offset;
		m_buffer_range = other.m_buffer_range;

#if !defined(__SYCL_DEVICE_ONLY__) && !defined(SYCL_DEVICE_ONLY)
		if(detail::is_embedded_hydration_id(m_device_ptr)) {
			if(detail::task_hydrator::is_available() && detail::task_hydrator::get_instance().can_hydrate()) {
				const auto info = detail::task_hydrator::get_instance().hydrate_accessor(detail::extract_hydration_id(m_device_ptr));
				assert(info.tgt == target::device);
				m_device_ptr = static_cast<DataT*>(info.ptr);
				m_index_offset = detail::id_cast<Dims>(info.buffer_offset);
				m_buffer_range = detail::range_cast<Dims>(info.buffer_range);
			}
		}
#endif
	}

	size_t get_linear_offset(const id<Dims>& index) const { return detail::get_linear_index(m_buffer_range, index - m_index_offset); }
};

// Celerity Accessor Deduction Guides
// TODO: Make buffer non-const once corresponding (deprecated!) constructor overloads are removed
template <typename T, int D, typename Functor, typename TagT>
accessor(const buffer<T, D>& buff, handler& cgh, Functor rmfn, TagT tag)
    -> accessor<T, D, detail::deduce_access_mode<TagT>(), detail::deduce_access_target<std::remove_const_t<TagT>>()>;

template <typename T, int D, typename Functor, typename TagT>
accessor(const buffer<T, D>& buff, handler& cgh, Functor rmfn, TagT tag, property::no_init no_init)
    -> accessor<T, D, detail::deduce_access_mode_discard<TagT>(), detail::deduce_access_target<std::remove_const_t<TagT>>()>;

template <typename T, int D, typename Functor, typename TagT>
accessor(const buffer<T, D>& buff, handler& cgh, Functor rmfn, TagT tag, property_list prop_list)
    -> accessor<T, D, detail::deduce_access_mode_discard<TagT>(), detail::deduce_access_target<std::remove_const_t<TagT>>()>;

template <typename DataT, int Dims, access_mode Mode>
class accessor<DataT, Dims, Mode, target::host_task> : public detail::accessor_base<DataT, Dims, Mode, target::host_task> {
	friend struct detail::accessor_testspy;

	struct ctor_internal_tag {};

  public:
	template <target Target = target::host_task, typename Functor>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	/**
	 * TODO: As of ComputeCpp 2.5.0 they do not support no_init prop, hence this constructor is needed along with discard deduction guide.
	 *    but once they do this should be replace for a constructor that takes a prop list as an argument.
	 */
	template <typename Functor, typename TagT>
	accessor(buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, property::no_init no_init) : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, property_list prop_list) {
		static_assert(detail::constexpr_false<TagT>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property celerity::no_init "
		    "as a last argument in the constructor");
	}

	template <target Target = target::host_task, typename Functor>
	[[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]] accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	[[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]] accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	[[deprecated("Creating accessor from const buffer is deprecated, capture buffer by reference instead")]] accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, property::no_init no_init)
	    : accessor(ctor_internal_tag{}, buff, cgh, rmfn) {}

	accessor(const accessor& other) { copy_and_hydrate(other); }

	accessor& operator=(const accessor& other) {
		if(this != &other) { copy_and_hydrate(other); }
		return *this;
	}

	template <access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_producer(M) && (D > 0), DataT&> operator[](id<Dims> index) const {
		return *(m_host_ptr + get_linear_offset(index));
	}

	template <access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M) && (D > 0), const DataT&> operator[](id<Dims> index) const {
		return *(m_host_ptr + get_linear_offset(index));
	}

	template <int D = Dims>
	std::enable_if_t<(D > 1), detail::accessor_subscript_proxy<DataT, D, Mode, target::host_task, 1>> operator[](const size_t d0) const {
		return {*this, d0};
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
		if(m_index_offset != detail::id_cast<Dims>(id<3>{0, 0, 0})) { illegal_access = true; }
		// We can be a bit more lenient for 1D buffers, in that the backing buffer doesn't have to have the full size.
		// (Dereferencing the pointer outside of the requested range is UB anyways).
		if(Dims > 1 && m_buffer_range != m_virtual_buffer_range) { illegal_access = true; }
		if(illegal_access) { throw std::logic_error("Buffer cannot be accessed with expected stride"); }
		return m_host_ptr;
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.m_host_ptr == rhs.m_host_ptr && lhs.m_mapped_subrange == rhs.m_mapped_subrange && lhs.m_buffer_range == rhs.m_buffer_range
		       && lhs.m_virtual_buffer_range == rhs.m_virtual_buffer_range && lhs.m_index_offset == rhs.m_index_offset;
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
		    m_buffer_range,
		    m_mapped_subrange.range,
		    m_index_offset,
		    m_mapped_subrange.offset,
		};
	}

	/**
	 * Returns a pointer to the host-local backing buffer along with a mapping to the global virtual buffer.
	 *
	 * Each host keeps only part of the global (virtual) buffer locally. The layout information can be used, for example, to perform distributed I/O on the
	 * partial buffer present at each host.
	 */
	// TODO remove this together with host_memory_layout after a grace period
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	template <int KernelDims>
	[[deprecated("get_host_memory will be removed in a future version of Celerity. Use get_allocation_window instead")]] std::pair<DataT*, host_memory_layout>
	get_host_memory(const partition<KernelDims>& part) const {
		// We already know the range mapper output for "chunk" from the constructor. The parameter is a purely semantic dependency which ensures that
		// this function is not called outside a host task.
		(void)part;

		host_memory_layout::dimension_vector dimensions(Dims);
		for(int d = 0; d < Dims; ++d) {
			dimensions[d] = {/* global_size */ m_virtual_buffer_range[d],
			    /* global_offset */ m_mapped_subrange.offset[d],
			    /* local_size */ m_buffer_range[d],
			    /* local_offset */ m_mapped_subrange.offset[d] - m_index_offset[d],
			    /* extent */ m_mapped_subrange.range[d]};
		}

		return {m_host_ptr, host_memory_layout{dimensions}};
	}
#pragma GCC diagnostic pop

  private:
	// Subange of the accessor, as set by the range mapper or requested by the user (master node host tasks only).
	// This does not necessarily correspond to the backing buffer's range.
	subrange<Dims> m_mapped_subrange;

	DataT* m_host_ptr = nullptr;

	// Offset of the backing buffer relative to the virtual buffer.
	id<Dims> m_index_offset;

	// Range of the backing buffer.
	range<Dims> m_buffer_range = detail::zero_range;

	// The range of the Celerity buffer as created by the user.
	// We only need this to check whether it is safe to call get_pointer() or not.
	range<Dims> m_virtual_buffer_range = detail::zero_range;

	template <target Target = target::host_task, typename Functor>
	accessor(ctor_internal_tag, const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn) : m_virtual_buffer_range(buff.get_range()) {
		static_assert(!std::is_same_v<Functor, range<Dims>>, "The accessor constructor overload for master-access tasks (now called 'host tasks') has "
		                                                     "been removed with Celerity 0.2.0. Please provide a range mapper instead.");
		const auto hid =
		    detail::add_requirement(cgh, detail::get_buffer_id(buff), std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, buff.get_range()));
		detail::extend_lifetime(cgh, detail::get_lifetime_extending_state(buff));
		m_host_ptr = detail::embed_hydration_id<DataT*>(hid);
	}

	// Constructor for tests, called through accessor_testspy.
	accessor(subrange<Dims> mapped_subrange, DataT* ptr, id<Dims> backing_buffer_offset, range<Dims> backing_buffer_range, range<Dims> virtual_buffer_range)
	    : m_mapped_subrange(mapped_subrange), m_host_ptr(ptr), m_index_offset(backing_buffer_offset), m_buffer_range(backing_buffer_range),
	      m_virtual_buffer_range(virtual_buffer_range) {}

	// Constructor for tests, called through accessor_testspy.
	accessor(subrange<Dims> mapped_subrange, detail::hydration_id hid, id<Dims> backing_buffer_offset, range<Dims> backing_buffer_range,
	    range<Dims> virtual_buffer_range)
	    : accessor(mapped_subrange, detail::embed_hydration_id<DataT*>(hid), backing_buffer_offset, backing_buffer_range, virtual_buffer_range) {}

	void copy_and_hydrate(const accessor& other) {
		m_mapped_subrange = other.m_mapped_subrange;
		m_host_ptr = other.m_host_ptr;
		m_index_offset = other.m_index_offset;
		m_buffer_range = other.m_buffer_range;
		m_virtual_buffer_range = other.m_virtual_buffer_range;

		if(detail::is_embedded_hydration_id(m_host_ptr)) {
			if(detail::task_hydrator::is_available() && detail::task_hydrator::get_instance().can_hydrate()) {
				const auto info = detail::task_hydrator::get_instance().hydrate_accessor(detail::extract_hydration_id(m_host_ptr));
				assert(info.tgt == target::host_task);
				m_host_ptr = static_cast<DataT*>(info.ptr);
				m_index_offset = detail::id_cast<Dims>(info.buffer_offset);
				m_buffer_range = detail::range_cast<Dims>(info.buffer_range);
				m_mapped_subrange = detail::subrange_cast<Dims>(info.accessor_sr);
			}
		}
	}

	size_t get_linear_offset(id<Dims> index) const { return detail::get_linear_index(m_buffer_range, index - m_index_offset); }
};


template <typename DataT, int Dims = 1>
class local_accessor {
	friend struct detail::accessor_testspy;

  private:
#if CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 6)
	using sycl_accessor = cl::sycl::accessor<DataT, Dims, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;
#else
	using sycl_accessor = cl::sycl::local_accessor<DataT, Dims>;
#endif

  public:
	using value_type = DataT;
	using reference = DataT&;
	using const_reference = const DataT&;
	using size_type = size_t;

	local_accessor() : m_sycl_acc{make_placeholder_sycl_accessor()}, m_allocation_size(detail::zero_range) {}

#if !defined(__SYCL_DEVICE_ONLY__) && !defined(SYCL_DEVICE_ONLY)
	local_accessor(const range<Dims>& allocation_size, handler& cgh) : m_sycl_acc{make_placeholder_sycl_accessor()}, m_allocation_size(allocation_size) {}

	local_accessor(const local_accessor& other)
	    : m_sycl_acc(detail::task_hydrator::is_available() && detail::task_hydrator::get_instance().has_sycl_handler() && other.m_allocation_size.size() > 0
	                     ? sycl_accessor{other.m_allocation_size, detail::task_hydrator::get_instance().get_sycl_handler()}
	                     : other.m_sycl_acc),
	      m_allocation_size(other.m_allocation_size) {}
#else
	local_accessor(const range<Dims>& allocation_size, handler& cgh);
	local_accessor(const local_accessor&) = default;
#endif

	local_accessor& operator=(const local_accessor&) = default;

	size_type byte_size() const noexcept { return m_allocation_size.size() * sizeof(value_type); }

	size_type size() const noexcept { return m_allocation_size.size(); }

	size_type max_size() const noexcept { return m_sycl_acc.max_size(); }

	bool empty() const noexcept { return m_sycl_acc.empty(); }

	range<Dims> get_range() const { return m_allocation_size; }

	std::add_pointer_t<value_type> get_pointer() const noexcept { return m_sycl_acc.get_pointer(); }

	// Workaround: ComputeCpp's legacy clang-8 has trouble deducing the return type of operator[] with decltype(auto), so we derive it manually.
	// TODO replace trailing return type with decltype(auto) once we require the new ComputeCpp (experimental) compiler.
	template <typename Index>
	inline auto operator[](const Index& index) const -> decltype(std::declval<const sycl_accessor&>()[index]) {
		return m_sycl_acc[index];
	}

  private:
	sycl_accessor m_sycl_acc;
	range<Dims> m_allocation_size;

	// Constructor for tests, called through accessor_testspy.
	local_accessor(const range<Dims>& allocation_size) : m_sycl_acc{make_placeholder_sycl_accessor()}, m_allocation_size(allocation_size) {}

	static sycl_accessor make_placeholder_sycl_accessor() {
#if CELERITY_WORKAROUND(DPCPP) || CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 9)
		detail::hack_null_sycl_handler null_cgh;
		return sycl_accessor{detail::zero_range, null_cgh};
#else
		return sycl_accessor{};
#endif
	}
};


namespace detail {

	template <typename TagT>
	constexpr access_mode deduce_access_mode() {
		if constexpr(std::is_same_v<const TagT, decltype(celerity::read_only)> || //
		             std::is_same_v<const TagT, decltype(celerity::read_only_host_task)>) {
			return access_mode::read;
		} else if constexpr(std::is_same_v<const TagT, decltype(celerity::read_write)> || //
		                    std::is_same_v<const TagT, decltype(celerity::read_write_host_task)>) {
			return access_mode::read_write;
		} else if constexpr(std::is_same_v<const TagT, decltype(celerity::write_only)> || //
		                    std::is_same_v<const TagT, decltype(celerity::write_only_host_task)>) {
			return access_mode::write;
		} else {
			static_assert(constexpr_false<TagT>, "Invalid access tag, expecting one of celerity::{read_only,read_write,write_only}[_host_task]");
		}
	}

	template <typename TagT>
	constexpr access_mode deduce_access_mode_discard() {
		if constexpr(std::is_same_v<const TagT, decltype(celerity::read_only)> || //
		             std::is_same_v<const TagT, decltype(celerity::read_only_host_task)>) {
			static_assert(constexpr_false<TagT>, "Invalid access mode + no_init");
		} else if constexpr(std::is_same_v<const TagT, decltype(celerity::read_write)> || //
		                    std::is_same_v<const TagT, decltype(celerity::read_write_host_task)>) {
			return access_mode::discard_read_write;
		} else if constexpr(std::is_same_v<const TagT, decltype(celerity::write_only)> || //
		                    std::is_same_v<const TagT, decltype(celerity::write_only_host_task)>) {
			return access_mode::discard_write;
		} else {
			static_assert(constexpr_false<TagT>, "Invalid access tag, expecting one of celerity::{read_only,read_write,write_only}[_host_task]");
		}
	}

	template <typename TagT>
	constexpr target deduce_access_target() {
		if constexpr(std::is_same_v<const TagT, decltype(celerity::read_only)> ||  //
		             std::is_same_v<const TagT, decltype(celerity::read_write)> || //
		             std::is_same_v<const TagT, decltype(celerity::write_only)>) {
			return target::device;
		} else if constexpr(std::is_same_v<const TagT, decltype(celerity::read_only_host_task)> ||  //
		                    std::is_same_v<const TagT, decltype(celerity::read_write_host_task)> || //
		                    std::is_same_v<const TagT, decltype(celerity::write_only_host_task)>) {
			return target::host_task;
		} else {
			static_assert(constexpr_false<TagT>, "Invalid access tag, expecting one of celerity::{read_only,read_write,write_only}[_host_task]");
		}
	}


	template <typename DataT, cl::sycl::access::mode Mode, target Target>
	class accessor_subscript_proxy<DataT, 3, Mode, Target, 2> {
		using AccessorT = celerity::accessor<DataT, 3, Mode, Target>;

	  public:
		accessor_subscript_proxy(const AccessorT& acc, const size_t d0, const size_t d1) : m_acc(acc), m_d0(d0), m_d1(d1) {}

		decltype(std::declval<AccessorT>()[{0, 0, 0}]) operator[](const size_t d2) const { return m_acc[{m_d0, m_d1, d2}]; }

	  private:
		const AccessorT& m_acc;
		size_t m_d0;
		size_t m_d1;
	};

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, target Target>
	class accessor_subscript_proxy<DataT, Dims, Mode, Target, 1> {
		template <int D>
		using AccessorT = celerity::accessor<DataT, D, Mode, Target>;

	  public:
		accessor_subscript_proxy(const AccessorT<Dims>& acc, const size_t d0) : m_acc(acc), m_d0(d0) {}

		// Note that we currently have to use SFINAE over constexpr-if + decltype(auto), as ComputeCpp 2.6.0 has
		// problems inferring the correct type in some cases (e.g. when DataT == sycl::id<>).
		template <int D = Dims>
		std::enable_if_t<D == 2, decltype(std::declval<AccessorT<2>>()[{0, 0}])> operator[](const size_t d1) const {
			return m_acc[{m_d0, d1}];
		}

		template <int D = Dims>
		std::enable_if_t<D == 3, accessor_subscript_proxy<DataT, 3, Mode, Target, 2>> operator[](const size_t d1) const {
			return {m_acc, m_d0, d1};
		}

	  private:
		const AccessorT<Dims>& m_acc;
		size_t m_d0;
	};

} // namespace detail

} // namespace celerity
