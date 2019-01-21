#pragma once

#include <type_traits>

#include <CL/sycl.hpp>

#include "access_modes.h"
#include "buffer_storage.h"
#include "runtime.h"

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
		static_assert(Mode != cl::sycl::access::mode::atomic, "Atomic host access is NYI");

	  public:
		host_accessor_impl(std::shared_ptr<buffer_storage<DataT, Dims>> buf_storage, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {})
		    : buf_storage(buf_storage), range(range), offset(offset) {
			buffer_range = cl::sycl::range<Dims>(buf_storage->get_range());
			auto queue = runtime::get_instance().get_device_queue().get_sycl_queue();
			if(access::detail::mode_traits::is_consumer(Mode)) {
				read_handle = buf_storage->get_data(queue, cl::sycl::id<3>(offset), cl::sycl::range<3>(range));
				linearized_data_ptr = reinterpret_cast<DataT*>(read_handle->linearized_data_ptr);
			}

			if(access::detail::mode_traits::is_producer(Mode)) {
				write_buffer = std::make_unique<DataT[]>(range[0] * range[1] * range[2]);
				linearized_data_ptr = write_buffer.get();

				// Retain previous contents
				if(access::detail::mode_traits::is_consumer(Mode)) {
					std::memcpy(linearized_data_ptr, read_handle->linearized_data_ptr, sizeof(DataT) * range[0] * range[1] * range[2]);
					read_handle = nullptr;
				}
			}
		}

		~host_accessor_impl() {
			if(access::detail::mode_traits::is_producer(Mode)) {
				if(linearized_data_ptr == nullptr) return;
				raw_data_handle data_handle;
				data_handle.linearized_data_ptr = linearized_data_ptr;
				data_handle.range = cl::sycl::range<3>(range);
				data_handle.offset = cl::sycl::range<3>(offset);
				auto queue = runtime::get_instance().get_device_queue().get_sycl_queue();
				buf_storage->set_data(queue, data_handle);
			}
		}

		// FIXME: This currently does NOT behave the same way as SYCL if the accessor has an offset
		// (See runtime tests; basically SYCL always points to the first item in the buffer, regardless of offset).
		DataT* get_pointer() { return linearized_data_ptr; }

		/**
		 * A note on indexing:
		 * SYCL accessors always use absolute indexes, regardless of the accessor offset.
		 * Since we store the linearized pointer to the offset range, we have to subtract the offset
		 * before doing the index calculation to end up in the same location as a SYCL accessor.
		 */

		template <cl::sycl::access::mode M = Mode, int D = Dims>
		std::enable_if_t<access::detail::mode_traits::is_producer(M) && D == 1, DataT&> operator[](size_t index) {
			return linearized_data_ptr[index - offset[0]];
		}

		template <cl::sycl::access::mode M = Mode, int D = Dims>
		std::enable_if_t<access::detail::mode_traits::is_pure_consumer(M) && D == 1, const DataT&> operator[](size_t index) const {
			return linearized_data_ptr[index - offset[0]];
		}

		template <cl::sycl::access::mode M = Mode>
		std::enable_if_t<access::detail::mode_traits::is_producer(M), DataT&> operator[](cl::sycl::id<Dims> index) {
			return linearized_data_ptr[detail::get_linear_index(range, index - offset)];
		}

		template <cl::sycl::access::mode M = Mode>
		std::enable_if_t<access::detail::mode_traits::is_pure_consumer(M), const DataT&> operator[](cl::sycl::id<Dims> index) const {
			return linearized_data_ptr[detail::get_linear_index(range, index - offset)];
		}

	  private:
		cl::sycl::range<Dims> buffer_range;
		std::shared_ptr<buffer_storage<DataT, Dims>> buf_storage;
		cl::sycl::range<Dims> range;
		cl::sycl::id<Dims> offset;
		std::shared_ptr<raw_data_read_handle> read_handle;
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
	std::enable_if_t<access::detail::mode_traits::is_producer(M) && D == 1, DataT&> operator[](size_t index) const {
		return (*pimpl)[index];
	}

	template <cl::sycl::access::mode M = Mode, int D = Dims>
	std::enable_if_t<access::detail::mode_traits::is_pure_consumer(M) && D == 1, const DataT&> operator[](size_t index) const {
		return (*pimpl)[index];
	}

	template <cl::sycl::access::mode M = Mode>
	std::enable_if_t<access::detail::mode_traits::is_producer(M), DataT&> operator[](cl::sycl::id<Dims> index) const {
		return (*pimpl)[index];
	}

	template <cl::sycl::access::mode M = Mode>
	std::enable_if_t<access::detail::mode_traits::is_pure_consumer(M), const DataT&> operator[](cl::sycl::id<Dims> index) const {
		return (*pimpl)[index];
	}

  private:
	// Unfortunately we have to make this mutable to get the same behavior as with SYCL kernels,
	// i.e. being able to capture mutable accessors by value within non-mutable lambdas.
	// TODO: Can this cause unexpected problems for celerity users?
	mutable std::shared_ptr<detail::host_accessor_impl<DataT, Dims, Mode>> pimpl;
};


} // namespace celerity
