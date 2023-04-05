#pragma once

#include <type_traits>

#include <CL/sycl.hpp>

#include "access_modes.h"
#include "buffer.h"
#include "buffer_storage.h"
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

	accessor() noexcept = default;

	template <typename Functor>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagModeNoInit>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, TagModeNoInit, target::device> /* tag */)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::device> /* tag */,
	    const property::no_init& /* no_init */)
	    : accessor(ctor_internal_tag(), buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode, access_mode TagModeNoInit>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, TagModeNoInit, target::device> /* tag */,
	    const property_list& /* prop_list */) {
		static_assert(detail::constexpr_false<Functor>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property celerity::no_init "
		    "as a last argument in the constructor");
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh) : accessor(buff, cgh, access::all()) {}

	template <access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<Mode, TagModeNoInit, target::device> tag)
	    : accessor(buff, cgh, access::all(), tag) {}

	template <access_mode TagMode, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, Mode, target::device> tag, const property::no_init& no_init)
	    : accessor(buff, cgh, access::all(), tag, no_init) {}

	template <access_mode TagMode, access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::device> tag, const property_list& prop_list)
	    : accessor(buff, cgh, access::all(), tag, prop_list) {}

	template <access_mode M = Mode>
	inline std::enable_if_t<detail::access::mode_traits::is_producer(M), DataT&> operator[](const id<Dims>& index) const {
		return m_device_ptr[get_linear_offset(index)];
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator[](const id<Dims>& index) const {
		return m_device_ptr[get_linear_offset(index)];
	}

	template <int D = Dims>
	inline std::enable_if_t<(D > 0), detail::subscript_result_t<D, const accessor>> operator[](const size_t index) const {
		return detail::subscript<D>(*this, index);
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), DataT&> operator*() const {
		return *m_device_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator*() const {
		return *m_device_ptr;
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.m_device_ptr == rhs.m_device_ptr && lhs.m_buffer_range == rhs.m_buffer_range && lhs.m_index_offset == rhs.m_index_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

  private:
	DataT* m_device_ptr = nullptr;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> m_index_offset;
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_buffer_range = detail::zero_range;

	// Constructor for tests, called through accessor_testspy.
	accessor(DataT* ptr, id<Dims> index_offset, range<Dims> buffer_range) : m_device_ptr(ptr), m_index_offset(index_offset), m_buffer_range(buffer_range) {
#if CELERITY_WORKAROUND_HIPSYCL // hipSYCL does not yet implement is_device_copyable_v
		static_assert(std::is_trivially_copyable_v<accessor>);
#else
		static_assert(sycl::is_device_copyable_v<accessor>);
#endif
	}

	template <typename Functor>
	accessor(const ctor_internal_tag /* tag */, const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) {
		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh);
			using range_mapper = detail::range_mapper<Dims, std::decay_t<Functor>>; // decay function type to function pointer
			prepass_cgh.add_requirement(detail::get_buffer_id(buff), std::make_unique<range_mapper>(rmfn, Mode, buff.get_range()));
		} else {
			if(detail::get_handler_execution_target(cgh) != detail::execution_target::device) {
				throw std::runtime_error(
				    "Calling accessor constructor with device target is only allowed in parallel_for tasks."
				    "If you want to access this buffer from within a host task, please specialize the call using one of the *_host_task tags");
			}

			auto& live_cgh = dynamic_cast<detail::live_pass_device_handler&>(cgh);
			// It's difficult to figure out which stored range mapper corresponds to this constructor call, which is why we just call the raw mapper manually.
			const auto mapped_sr = live_cgh.apply_range_mapper<Dims>(rmfn, buff.get_range());
			auto access_info =
			    detail::runtime::get_instance().get_buffer_manager().access_device_buffer<DataT, Dims>(detail::get_buffer_id(buff), Mode, mapped_sr);

			m_device_ptr = static_cast<DataT*>(access_info.ptr);
			m_index_offset = detail::id_cast<Dims>(access_info.backing_buffer_offset);
			m_buffer_range = detail::range_cast<Dims>(access_info.backing_buffer_range);
		}
	}

	size_t get_linear_offset(const id<Dims>& index) const { return detail::get_linear_index(m_buffer_range, index - m_index_offset); }
};

template <typename DataT, int Dims, access_mode Mode>
class accessor<DataT, Dims, Mode, target::host_task> : public detail::accessor_base<DataT, Dims, Mode, target::host_task> {
	friend struct detail::accessor_testspy;

  public:
	static_assert(Mode != access_mode::atomic, "access_mode::atomic is not supported.");

	accessor() noexcept = default;

	template <typename Functor>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn) {
		static_assert(!std::is_same_v<Functor, range<Dims>>, "The accessor constructor overload for master-access tasks (now called 'host tasks') has "
		                                                     "been removed with Celerity 0.2.0. Please provide a range mapper instead.");

		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(detail::get_buffer_id(buff), std::make_unique<detail::range_mapper<Dims, Functor>>(rmfn, Mode, buff.get_range()));
		} else {
			if(detail::get_handler_execution_target(cgh) != detail::execution_target::host) {
				throw std::runtime_error(
				    "Calling accessor constructor with host_buffer target is only allowed in host tasks."
				    "If you want to access this buffer from within a parallel_for task, please specialize the call using one of the non host tags");
			}
			auto& live_cgh = dynamic_cast<detail::live_pass_host_handler&>(cgh);
			// It's difficult to figure out which stored range mapper corresponds to this constructor call, which is why we just call the raw mapper
			// manually.
			const auto sr = live_cgh.apply_range_mapper<Dims>(rmfn, buff.get_range());
			auto access_info = detail::runtime::get_instance().get_buffer_manager().access_host_buffer<DataT, Dims>(detail::get_buffer_id(buff), Mode, sr);

			m_mapped_subrange = sr;
			m_host_ptr = static_cast<DataT*>(access_info.ptr);
			m_index_offset = detail::id_cast<Dims>(access_info.backing_buffer_offset);
			m_buffer_range = detail::range_cast<Dims>(access_info.backing_buffer_range);
			m_virtual_buffer_range = buff.get_range();
		}
	}

	template <typename Functor, access_mode TagModeNoInit>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, TagModeNoInit, target::host_task> /* tag */)
	    : accessor(buff, cgh, rmfn) {}

	/**
	 * TODO: As of ComputeCpp 2.5.0 they do not support no_init prop, hence this constructor is needed along with discard deduction guide.
	 *    but once they do this should be replace for a constructor that takes a prop list as an argument.
	 */
	template <typename Functor, access_mode TagMode, access_mode M = Mode, typename = std::enable_if_t<detail::access::mode_traits::is_producer(M)>>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::host_task> /* tag */,
	    const property::no_init& /* no_init */)
	    : accessor(buff, cgh, rmfn) {}

	template <typename Functor, access_mode TagMode, access_mode TagModeNoInit>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, TagModeNoInit, target::host_task> /* tag */,
	    const property_list& /* prop_list */) {
		static_assert(detail::constexpr_false<Functor>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property celerity::no_init "
		    "as a last argument in the constructor");
	}

	template <int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh) : accessor(buff, cgh, access::all()) {}

	template <access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<Mode, TagModeNoInit, target::host_task> tag)
	    : accessor(buff, cgh, access::all(), tag) {}

	template <access_mode TagMode, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, Mode, target::host_task> tag, const property::no_init& no_init)
	    : accessor(buff, cgh, access::all(), tag, no_init) {}

	template <access_mode TagMode, access_mode TagModeNoInit, int D = Dims, std::enable_if_t<D == 0, int> = 0>
	accessor(
	    const buffer<DataT, Dims>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag, const property_list& prop_list)
	    : accessor(buff, cgh, access::all(), tag, prop_list) {}

	template <access_mode M = Mode>
	inline std::enable_if_t<detail::access::mode_traits::is_producer(M), DataT&> operator[](const id<Dims>& index) const {
		return m_host_ptr[get_linear_offset(index)];
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator[](const id<Dims>& index) const {
		return m_host_ptr[get_linear_offset(index)];
	}

	template <int D = Dims>
	inline std::enable_if_t<(D > 0), detail::subscript_result_t<D, const accessor>> operator[](const size_t index) const {
		return detail::subscript<D>(*this, index);
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_producer(M), DataT&> operator*() const {
		return *m_host_ptr;
	}

	template <access_mode M = Mode>
	inline std::enable_if_t<Dims == 0 && detail::access::mode_traits::is_pure_consumer(M), const DataT&> operator*() const {
		return *m_host_ptr;
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
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS id<Dims> m_index_offset;

	// Range of the backing buffer.
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_buffer_range = detail::zero_range;

	// The range of the Celerity buffer as created by the user.
	// We only need this to check whether it is safe to call get_pointer() or not.
	CELERITY_DETAIL_NO_UNIQUE_ADDRESS range<Dims> m_virtual_buffer_range = detail::zero_range;

	// Constructor for tests, called through accessor_testspy.
	accessor(subrange<Dims> mapped_subrange, DataT* ptr, id<Dims> backing_buffer_offset, range<Dims> backing_buffer_range, range<Dims> virtual_buffer_range)
	    : m_mapped_subrange(mapped_subrange), m_host_ptr(ptr), m_index_offset(backing_buffer_offset), m_buffer_range(backing_buffer_range),
	      m_virtual_buffer_range(virtual_buffer_range) {}

	size_t get_linear_offset(const id<Dims>& index) const { return detail::get_linear_index(m_buffer_range, index - m_index_offset); }
};


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
	static_assert(Dims <= 3);

  private:
	constexpr static int sycl_dims = std::max(1, Dims);

#if CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 6)
	using sycl_accessor = cl::sycl::accessor<DataT, sycl_dims, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;
#else
	using sycl_accessor = cl::sycl::local_accessor<DataT, sycl_dims>;
#endif

  public:
	using value_type = DataT;
	using reference = DataT&;
	using const_reference = const DataT&;
	using size_type = size_t;

	local_accessor() : m_sycl_acc{make_placeholder_sycl_accessor()}, m_allocation_size(detail::zero_range) {}

	template <int D = Dims, typename = std::enable_if_t<D == 0>>
	local_accessor(handler& cgh) : local_accessor(range<0>(), cgh) {}

#if !defined(__SYCL_DEVICE_ONLY__) && !defined(SYCL_DEVICE_ONLY)
	local_accessor(const range<Dims>& allocation_size, handler& cgh) : m_sycl_acc{make_placeholder_sycl_accessor()}, m_allocation_size(allocation_size) {
		if(!detail::is_prepass_handler(cgh)) {
			auto& device_handler = dynamic_cast<detail::live_pass_device_handler&>(cgh);
			m_eventual_sycl_cgh = device_handler.get_eventual_sycl_cgh();
		}
	}

	local_accessor(const local_accessor& other)
	    : m_sycl_acc(other.sycl_cgh() ? sycl_accessor{other.sycl_allocation_size(), *other.sycl_cgh()} : other.m_sycl_acc),
	      m_allocation_size(other.m_allocation_size), m_eventual_sycl_cgh(other.sycl_cgh() ? nullptr : other.m_eventual_sycl_cgh) {}
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

	inline DataT& operator[](const id<Dims>& index) const {
		if constexpr(Dims == 0) {
			return m_sycl_acc[sycl::id<1>(0)];
		} else {
			return m_sycl_acc[sycl::id<Dims>(index)];
		}
	}

	inline detail::subscript_result_t<Dims, const local_accessor> operator[](const size_t dim0) const { return detail::subscript<Dims>(*this, dim0); }

	template <int D = Dims>
	std::enable_if_t<D == 0, DataT&> operator*() const {
		return m_sycl_acc[sycl::id<1>(0)];
	}

  private:
	sycl_accessor m_sycl_acc;
	range<Dims> m_allocation_size;
	cl::sycl::handler* const* m_eventual_sycl_cgh = nullptr;

	static sycl_accessor make_placeholder_sycl_accessor() {
#if CELERITY_WORKAROUND(DPCPP) || CELERITY_WORKAROUND_LESS_OR_EQUAL(COMPUTECPP, 2, 9)
		detail::hack_null_sycl_handler null_cgh;
		return sycl_accessor{sycl::range<sycl_dims>(celerity::range<sycl_dims>(detail::zero_range)), null_cgh};
#else
		return sycl_accessor{};
#endif
	}

	cl::sycl::handler* sycl_cgh() const { return m_eventual_sycl_cgh != nullptr ? *m_eventual_sycl_cgh : nullptr; }

	sycl::range<sycl_dims> sycl_allocation_size() const { return sycl::range<sycl_dims>(detail::range_cast<sycl_dims>(m_allocation_size)); }
};

} // namespace celerity
