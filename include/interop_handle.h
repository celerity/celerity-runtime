#pragma once

#include "access_modes.h"
#include "sycl_wrappers.h"

namespace celerity {

template <typename DataT, int Dims, access_mode Mode, target Target>
class accessor;

namespace experimental {

	// TODO: Align with SYCL spec where possible
	class interop_handle {
	  public:
		interop_handle(sycl::interop_handle& sycl_ih) : m_sycl_ih(sycl_ih) {}

		template <sycl::backend Backend, typename DataT, int Dims, access_mode Mode, target Target>
		auto get_native_mem(const accessor<DataT, Dims, Mode, Target>& acc) const {
			// TODO: Also allow host accessors?
			static_assert(Target == target::device);
			static_assert(Backend == sycl::backend::cuda);

			if constexpr(celerity::detail::access::mode_traits::is_pure_consumer(Mode)) {
				return static_cast<std::add_const_t<decltype(acc.m_device_ptr)>>(acc.m_device_ptr);
			} else {
				return acc.m_device_ptr;
			}
		}

		// TODO: This should probably not even exist; unify with host_memory_layout API
		template <typename DataT, int Dims, access_mode Mode, target Target>
		auto get_backing_buffer_range(const accessor<DataT, Dims, Mode, Target>& acc) const {
			return acc.m_buffer_range;
		}

		template <typename DataT, int Dims, access_mode Mode, target Target>
		auto get_backing_buffer_offset(const accessor<DataT, Dims, Mode, Target>& acc) const {
			return acc.m_index_offset;
		}

		template <sycl::backend Backend>
		auto get_native_queue() const noexcept {
			return m_sycl_ih.get_native_queue<Backend>();
		}

	  private:
		sycl::interop_handle& m_sycl_ih;
	};

} // namespace experimental
} // namespace celerity