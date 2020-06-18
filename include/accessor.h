#pragma once

#include <type_traits>

#include <CL/sycl.hpp>
#include <boost/optional.hpp>

#include "access_modes.h"
#include "buffer_storage.h"

namespace celerity {

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
 * Celerity wrapper around SYCL accessors.
 *
 * @note The Celerity accessor currently does not support get_size, get_count, get_range, get_offset and get_pointer,
 * as their semantics in a distributed context are unclear.
 */
template <typename DataT, int Dims, cl::sycl::access::mode Mode>
class accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer>
    : public detail::accessor_base<DataT, Dims, Mode, cl::sycl::access::target::global_buffer> {
  public:
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

	cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t> sycl_accessor;

	cl::sycl::id<Dims> backing_buffer_offset;

	accessor(sycl_accessor_t sycl_accessor, cl::sycl::id<Dims> backing_buffer_offset)
	    : sycl_accessor(sycl_accessor), backing_buffer_offset(backing_buffer_offset) {
		// SYCL 1.2.1 dictates that all kernel parameters must have standard layout.
		// However, since we are wrapping a SYCL accessor, this assertion fails for some implementations,
		// as it is currently unclear whether SYCL accessors must also have standard layout.
		// See https://github.com/KhronosGroup/SYCL-Docs/issues/94
		// static_assert(std::is_standard_layout<accessor>::value, "accessor must have standard layout");
	}
};

template <typename DataT, int Dims, cl::sycl::access::mode Mode>
class accessor<DataT, Dims, Mode, cl::sycl::access::target::host_buffer>
    : public detail::accessor_base<DataT, Dims, Mode, cl::sycl::access::target::host_buffer> {
  public:
	size_t get_size() const { return requested_range.size() * sizeof(DataT); }

	size_t get_count() const { return requested_range.size(); }

	cl::sycl::range<Dims> get_range() const { return requested_range; }

	cl::sycl::id<Dims> get_offset() const { return requested_offset; }

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
		return lhs.optional_buffer == rhs.optional_buffer && lhs.backing_buffer_offset == rhs.backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

  private:
	template <typename T, int D, cl::sycl::access::mode M, typename... Args>
	friend accessor<T, D, M, cl::sycl::access::target::host_buffer> detail::make_host_accessor(Args&&...);

	// The range of the accessor, as requested by the user.
	// This does not correspond to the backing buffer's range.
	cl::sycl::range<Dims> requested_range;

	// The offset of the accessor, as requested by the user.
	// This does not correspond to the backing buffer's offset.
	cl::sycl::id<Dims> requested_offset;

	mutable boost::optional<detail::host_buffer<DataT, Dims>&> optional_buffer = boost::none;

	cl::sycl::id<Dims> backing_buffer_offset;

	// The range of the Celerity buffer as created by the user.
	// We only need this to check whether it is safe to call get_pointer() or not.
	cl::sycl::range<Dims> virtual_buffer_range;

	/**
	 * Constructor for pre-pass.
	 */
	accessor(cl::sycl::range<Dims> requested_range, cl::sycl::id<Dims> requested_offset)
	    : requested_range(requested_range), requested_offset(requested_offset) {}

	/**
	 * Constructor for live-pass.
	 */
	accessor(cl::sycl::range<Dims> requested_range, cl::sycl::id<Dims> requested_offset, detail::host_buffer<DataT, Dims>& buffer,
	    cl::sycl::id<Dims> backing_buffer_offset, cl::sycl::range<Dims> virtual_buffer_range)
	    : requested_range(requested_range), requested_offset(requested_offset), optional_buffer(buffer), backing_buffer_offset(backing_buffer_offset),
	      virtual_buffer_range(virtual_buffer_range) {}

	detail::host_buffer<DataT, Dims>& get_buffer() const {
		assert(optional_buffer != boost::none);
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
