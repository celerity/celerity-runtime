#pragma once

#include <type_traits>

#include <SYCL/sycl.hpp>

#include "buffer_storage.h"

namespace celerity {

// TODO: Looks like we will have to provide the full (mocked) accessor API
template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::access::target Target>
class prepass_accessor {
  public:
	DataT& operator[](cl::sycl::id<Dims> index) const { throw std::runtime_error("Accessor used outside kernel / functor"); }

	template <cl::sycl::access::target T = Target>
	std::enable_if_t<T == cl::sycl::access::target::host_buffer, DataT*> get_pointer() const {
		throw std::runtime_error("Accessor used outside kernel / functor");
	}
};

namespace detail {

	template <typename DataT, int Dims, cl::sycl::access::mode Mode>
	class host_accessor_impl {
	  public:
		host_accessor_impl(std::shared_ptr<detail::buffer_storage<DataT, Dims>> buffer_storage, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {})
		    : buffer_storage(buffer_storage), range(range), offset(offset) {
			buffer_range = cl::sycl::range<Dims>(buffer_storage->get_range());
			// TODO: Also for read_write
			if(Mode == cl::sycl::access::mode::read) {
				read_handle = buffer_storage->get_data(cl::sycl::id<3>(offset), cl::sycl::range<3>(range));
				linearized_data_ptr = reinterpret_cast<DataT*>(read_handle->linearized_data_ptr);
			} else {
				write_buffer = std::make_unique<DataT[]>(range[0] * range[1] * range[2]);
				linearized_data_ptr = write_buffer.get();
			}
		}

		~host_accessor_impl() {
			// TODO: Also for read_write
			if(Mode == cl::sycl::access::mode::write) {
				if(linearized_data_ptr == nullptr) return;
				detail::raw_data_handle data_handle;
				data_handle.linearized_data_ptr = linearized_data_ptr;
				data_handle.range = cl::sycl::range<3>(range);
				data_handle.offset = cl::sycl::range<3>(offset);
				buffer_storage->set_data(data_handle);
			}
		}

		// TODO: Where should this point if the accessor has an offset? How does SYCL do it? Investigate
		DataT* get_pointer() { return linearized_data_ptr; }

		/**
		 * A note on indexing:
		 * SYCL accessors always use absolute indexes, regardless of the accessor offset.
		 * Since we store the linearized pointer to the offset range, we have to subtract the offset
		 * before doing the index calculation to end up in the same location as a SYCL accessor.
		 */

		template <cl::sycl::access::mode M = Mode, int D = Dims>
		std::enable_if_t<M == cl::sycl::access::mode::write && D == 1, DataT&> operator[](size_t index) {
			return linearized_data_ptr[index - offset[0]];
		}

		template <cl::sycl::access::mode M = Mode, int D = Dims>
		std::enable_if_t<M == cl::sycl::access::mode::read && D == 1, const DataT&> operator[](size_t index) {
			return linearized_data_ptr[index - offset[0]];
		}

		template <cl::sycl::access::mode M = Mode>
		std::enable_if_t<M == cl::sycl::access::mode::write, DataT&> operator[](cl::sycl::id<Dims> index) {
			return linearized_data_ptr[detail::get_linear_index(range, index - offset)];
		}

		template <cl::sycl::access::mode M = Mode>
		std::enable_if_t<M == cl::sycl::access::mode::read, const DataT&> operator[](cl::sycl::id<Dims> index) {
			return linearized_data_ptr[detail::get_linear_index(range, index - offset)];
		}

	  private:
		cl::sycl::range<Dims> buffer_range;
		std::shared_ptr<detail::buffer_storage<DataT, Dims>> buffer_storage;
		cl::sycl::range<Dims> range;
		cl::sycl::id<Dims> offset;
		std::shared_ptr<detail::raw_data_read_handle> read_handle;
		std::unique_ptr<DataT[]> write_buffer;
		DataT* linearized_data_ptr = nullptr;
	};

} // namespace detail

/**
 * The host_accessor
 *
 * It is implemented using a PIMPL to detail::host_accessor_impl, so that it can be captured
 * by value in host access tasks, similar to how accessors are captured by value into SYCL kernels.
 */
template <typename DataT, int Dims, cl::sycl::access::mode Mode>
class host_accessor {
  public:
	host_accessor(std::shared_ptr<detail::buffer_storage<DataT, Dims>> buffer_storage, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {})
	    : pimpl(new detail::host_accessor_impl<DataT, Dims, Mode>(buffer_storage, range, offset)) {}

	DataT* get_pointer() const { return (*pimpl).get_pointer(); }

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<M == cl::sycl::access::mode::write && D == 1, DataT&> operator[](size_t index) const {
		return (*pimpl)[index];
	}

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<M == cl::sycl::access::mode::read && D == 1, const DataT&> operator[](size_t index) const {
		return (*pimpl)[index];
	}

	template <cl::sycl::access::mode M = Mode>
	std::enable_if_t<M == cl::sycl::access::mode::write, DataT&> operator[](cl::sycl::id<Dims> index) const {
		return (*pimpl)[index];
	}

	template <cl::sycl::access::mode M = Mode>
	std::enable_if_t<M == cl::sycl::access::mode::read, const DataT&> operator[](cl::sycl::id<Dims> index) const {
		return (*pimpl)[index];
	}

  private:
	// Unfortunately we have to make this mutable to get the same behavior as with SYCL kernels,
	// i.e. being able to capture mutable accessors by value within non-mutable lambdas.
	// TODO: Can this cause unexpected problems for celerity users?
	mutable std::shared_ptr<detail::host_accessor_impl<DataT, Dims, Mode>> pimpl;
};


} // namespace celerity
