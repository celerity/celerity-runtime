#pragma once

#include <type_traits>

#include <CL/sycl.hpp>
#include <boost/container/static_vector.hpp>

#include "access_modes.h"
#include "buffer_storage.h"


namespace celerity {

template <int Dims>
class partition;

template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
class accessor;

namespace detail {

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
	class accessor_base {
	  public:
		static_assert(Dims > 0, "0-dimensional accessors NYI");
		static_assert(Dims <= 3, "accessors can only have 3 dimensions or less");
		using value_type = DataT;
		using reference = DataT&;
		using const_reference = const DataT&;
	};

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> make_device_accessor(Args&&...);

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> make_host_accessor(Args&&...);

} // namespace detail

/**
 * Maps slices of the accessor backing buffer present on a host to the virtual global range of the Celerity buffer.
 */
class host_memory_layout {
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
	class dimension {
	  public:
		dimension() noexcept = default;

		dimension(size_t global_size, size_t global_offset, size_t local_size, size_t local_offset, size_t extent)
		    : global_size(global_size), global_offset(global_offset), local_size(local_size), local_offset(local_offset), extent(extent) {
			assert(global_offset >= local_offset);
			assert(global_size >= local_size);
		}

		size_t get_global_size() const { return global_size; }

		size_t get_local_size() const { return local_size; }

		size_t get_global_offset() const { return global_offset; }

		size_t get_local_offset() const { return local_offset; }

		size_t get_extent() const { return extent; }

	  private:
		size_t global_size{};
		size_t global_offset{};
		size_t local_size{};
		size_t local_offset{};
		size_t extent{};
	};

	/** Since contiguous dimensions can be merged when generating the memory layout, host_memory_layout is not generic over a fixed dimension count */
	constexpr static size_t max_dimensionality = 4;

	using dimension_vector = boost::container::static_vector<dimension, max_dimensionality>;

	explicit host_memory_layout(const dimension_vector& dimensions) : dimensions(dimensions) {}

	/** The layout maps per dimension, in descending dimensionality */
	const dimension_vector& get_dimensions() const { return dimensions; }

  private:
	dimension_vector dimensions;
};

/**
 * Celerity wrapper around SYCL accessors.
 *
 * @note The Celerity accessor currently does not support get_size, get_count, get_range, get_offset and get_pointer,
 * as their semantics in a distributed context are unclear.
 */
template <typename DataT, int Dims, cl::sycl::access::mode Mode>
class accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>
    : public detail::accessor_base<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> {
  public:
	accessor(const accessor& other) : sycl_accessor(other.sycl_accessor) { init_from(other); }

	accessor& operator=(const accessor& other) {
		if(this != &other) {
			sycl_accessor = other.sycl_accessor;
			init_from(other);
		}
		return *this;
	}

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_producer(M) && M != cl::sycl::access::mode::atomic && (D > 0), DataT&> operator[](
	    cl::sycl::id<Dims> index) const {
		return sycl_accessor[index - backing_buffer_offset];
	}

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M) && (D > 0), DataT> operator[](cl::sycl::id<Dims> index) const {
		return sycl_accessor[index - backing_buffer_offset];
	}

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<M == cl::sycl::access::mode::atomic && (D > 0), cl::sycl::atomic<DataT>> operator[](cl::sycl::id<Dims> index) const {
		return sycl_accessor[index - backing_buffer_offset];
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.sycl_accessor == rhs.sycl_accessor && lhs.backing_buffer_offset == rhs.backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

  private:
	using sycl_accessor_t = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>;

	template <typename T, int D, cl::sycl::access::mode M, typename... Args>
	friend accessor<T, D, M, cl::sycl::access::target::global_buffer> detail::make_device_accessor(Args&&...);

	// see init_from
	cl::sycl::handler* const* eventual_sycl_cgh = nullptr;
	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> sycl_accessor;
	cl::sycl::id<Dims> backing_buffer_offset;

	// TODO remove this once we have SYCL 2020 default-constructible accessors
	accessor(cl::sycl::buffer<DataT, Dims>& faux_buffer)
	    : sycl_accessor(cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>(faux_buffer)) {}

	accessor(cl::sycl::handler* const* eventual_sycl_cgh, cl::sycl::buffer<DataT, Dims>& buffer, const cl::sycl::range<Dims>& range,
	    cl::sycl::id<Dims> backing_buffer_offset)
	    : eventual_sycl_cgh(eventual_sycl_cgh),
	      // We pass a range and offset here to avoid interference from SYCL, but the offset must be relative to the *backing buffer*.
	      sycl_accessor(cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>(
	          buffer, range, backing_buffer_offset)),
	      backing_buffer_offset(backing_buffer_offset) {
		// SYCL 1.2.1 dictates that all kernel parameters must have standard layout.
		// However, since we are wrapping a SYCL accessor, this assertion fails for some implementations,
		// as it is currently unclear whether SYCL accessors must also have standard layout.
		// See https://github.com/KhronosGroup/SYCL-Docs/issues/94
		// static_assert(std::is_standard_layout<accessor>::value, "accessor must have standard layout");
	}

	void init_from(const accessor& other) {
		eventual_sycl_cgh = other.eventual_sycl_cgh;
		backing_buffer_offset = other.backing_buffer_offset;

		// The call to sycl::handler::require must happen inside the SYCL CGF, but since get_access within the Celerity CGF is executed before
		// the submission to SYCL, it needs to be deferred. We capture a reference to a SYCL handler pointer owned by the live pass handler that is
		// initialized once the SYCL CGF is entered. We then abuse the copy constructor that is called implicitly when the lambda is copied for the SYCL
		// kernel submission to call require().
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(SYCL_DEVICE_ONLY)
		if(eventual_sycl_cgh != nullptr && *eventual_sycl_cgh != nullptr) {
			(*eventual_sycl_cgh)->require(sycl_accessor);
			eventual_sycl_cgh = nullptr; // only `require` once
		}
#endif
	}
};

template <typename DataT, int Dims, cl::sycl::access::mode Mode>
class accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer>
    : public detail::accessor_base<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> {
  public:
	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_producer(M) && (D > 0), DataT&> operator[](cl::sycl::id<Dims> index) const {
		return *(get_buffer().get_pointer() + get_linear_offset(index));
	}

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M) && (D > 0), DataT> operator[](cl::sycl::id<Dims> index) const {
		return *(get_buffer().get_pointer() + get_linear_offset(index));
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
		if(backing_buffer_offset != detail::id_cast<Dims>(cl::sycl::id<3>{0, 0, 0})) { illegal_access = true; }
		// We can be a bit more lenient for 1D buffers, in that the backing buffer doesn't have to have the full size.
		// (Dereferencing the pointer outside of the requested range is UB anyways).
		if(Dims > 1 && get_buffer().get_range() != virtual_buffer_range) { illegal_access = true; }
		if(illegal_access) { throw std::logic_error("Buffer cannot be accessed with expected stride"); }
		return get_buffer().get_pointer();
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return (lhs.optional_buffer == rhs.optional_buffer || (lhs.optional_buffer && rhs.optional_buffer && *lhs.optional_buffer == *rhs.optional_buffer))
		       && lhs.backing_buffer_offset == rhs.backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

	/**
	 * Returns a pointer to the host-local backing buffer along with a mapping to the global virtual buffer.
	 *
	 * Each host keeps only part of the global (virtual) buffer locally. The layout information can be used, for example, to perform distributed I/O on the
	 * partial buffer present at each host.
	 */
	template <int KernelDims>
	std::pair<DataT*, host_memory_layout> get_host_memory(const partition<KernelDims>& part) const {
		// We already know the range mapper output for "chunk" from the constructor. The parameter is a purely semantic dependency which ensures that this
		// function is not called outside a host task.
		(void)part;

		host_memory_layout::dimension_vector dimensions(Dims);
		for(int d = 0; d < Dims; ++d) {
			dimensions[d] = {/* global_size */ virtual_buffer_range[d],
			    /* global_offset */ mapped_subrange.offset[d],
			    /* local_size */ get_buffer().get_range()[d],
			    /* local_offset */ mapped_subrange.offset[d] - backing_buffer_offset[d],
			    /* extent */ mapped_subrange.range[d]};
		}

		return {get_buffer().get_pointer(), host_memory_layout{dimensions}};
	}

  private:
	template <typename T, int D, cl::sycl::access::mode M, typename... Args>
	friend accessor<T, D, M, cl::sycl::access::target::host_buffer> detail::make_host_accessor(Args&&...);

	// Subange of the accessor, as set by the range mapper or requested by the user (master node host tasks only).
	// This does not necessarily correspond to the backing buffer's range.
	subrange<Dims> mapped_subrange;

	mutable detail::host_buffer<DataT, Dims>* optional_buffer = nullptr;

	// Offset of the backing buffer relative to the virtual buffer.
	cl::sycl::id<Dims> backing_buffer_offset;

	// The range of the Celerity buffer as created by the user.
	// We only need this to check whether it is safe to call get_pointer() or not.
	cl::sycl::range<Dims> virtual_buffer_range;

	/**
	 * Constructor for pre-pass.
	 */
	accessor() = default;

	/**
	 * Constructor for live-pass.
	 */
	accessor(subrange<Dims> mapped_subrange, detail::host_buffer<DataT, Dims>& buffer, cl::sycl::id<Dims> backing_buffer_offset,
	    cl::sycl::range<Dims> virtual_buffer_range)
	    : mapped_subrange(mapped_subrange), optional_buffer(&buffer), backing_buffer_offset(backing_buffer_offset), virtual_buffer_range(virtual_buffer_range) {
	}

	detail::host_buffer<DataT, Dims>& get_buffer() const {
		assert(optional_buffer != nullptr);
		return *optional_buffer;
	}

	size_t get_linear_offset(cl::sycl::id<Dims> index) const { return detail::get_linear_index(get_buffer().get_range(), index - backing_buffer_offset); }
};

namespace detail {

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> make_device_accessor(Args&&... args) {
		return {std::forward<Args>(args)...};
	}

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> make_host_accessor(Args&&... args) {
		return {std::forward<Args>(args)...};
	}

} // namespace detail

} // namespace celerity
