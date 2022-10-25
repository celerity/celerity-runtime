#pragma once

#include "test_utils.h"

namespace celerity {

namespace detail {
	struct accessor_testspy {
		template <typename DataT, int Dims, access_mode Mode, typename... Args>
		static accessor<DataT, Dims, Mode, target::device> make_device_accessor(Args&&... args) {
			return {std::forward<Args>(args)...};
		}

		template <typename DataT, int Dims, access_mode Mode, typename... Args>
		static accessor<DataT, Dims, Mode, target::host_task> make_host_accessor(Args&&... args) {
			return {std::forward<Args>(args)...};
		}

		template <typename DataT, int Dims, typename... Args>
		static local_accessor<DataT, Dims> make_local_accessor(Args&&... args) {
			return {std::forward<Args>(args)...};
		}
	};
} // namespace detail

namespace test_utils {

	// Convenience function for submitting parallel_for with global offset without having to create a CGF
	template <typename KernelName = detail::unnamed_kernel, int Dims, typename KernelFn>
	void run_parallel_for(sycl::queue& q, const range<Dims>& global_range, const id<Dims>& global_offset, KernelFn fn) {
		q.submit([=](sycl::handler& cgh) {
			cgh.parallel_for<KernelName>(global_range, detail::bind_simple_kernel(fn, global_range, global_offset, global_offset));
		});
		q.wait_and_throw();
	}

	class buffer_manager_fixture : public device_queue_fixture {
	  public:
		enum class access_target { host, device };

		void initialize(detail::buffer_manager::buffer_lifecycle_callback cb = [](detail::buffer_manager::buffer_lifecycle_event, detail::buffer_id) {}) {
			assert(!m_bm);
			m_bm = std::make_unique<detail::buffer_manager>(get_device_queue(), cb);
			m_bm->enable_test_mode();
		}

		detail::buffer_manager& get_buffer_manager() {
			if(!m_bm) initialize();
			return *m_bm;
		}

		static access_target get_other_target(access_target tgt) {
			if(tgt == access_target::host) return access_target::device;
			return access_target::host;
		}

		template <typename DataT, int Dims>
		range<Dims> get_backing_buffer_range(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset) {
			if(tgt == access_target::host) {
				auto info = m_bm->access_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, detail::range_cast<3>(range), detail::id_cast<3>(offset));
				return detail::range_cast<Dims>(info.backing_buffer_range);
			}
			auto info = m_bm->access_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::range_cast<Dims>(info.backing_buffer_range);
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename KernelName = class buffer_for_each, typename Callback>
		void buffer_for_each(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, Callback cb) {
			const auto range3 = detail::range_cast<3>(range);
			const auto offset3 = detail::id_cast<3>(offset);

			if(tgt == access_target::host) {
				auto info = m_bm->access_host_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_range = detail::range_cast<3>(info.backing_buffer_range);
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - info.backing_buffer_offset;
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							cb(detail::id_cast<Dims>(global_idx), static_cast<DataT*>(info.ptr)[linear_idx]);
						}
					}
				}
			}

			if(tgt == access_target::device) {
				auto info = m_bm->access_device_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_offset = detail::id_cast<Dims>(info.backing_buffer_offset);
				const auto buf_range = detail::range_cast<Dims>(info.backing_buffer_range);
				get_device_queue()
				    .submit([&](cl::sycl::handler& cgh) {
					    auto ptr = info.ptr;
					    cgh.parallel_for<detail::bind_kernel_name<KernelName>>(range, [=](cl::sycl::id<Dims> global_idx) {
						    global_idx += offset;
						    const auto local_idx = global_idx - buf_offset;
						    cb(global_idx, static_cast<DataT*>(ptr)[detail::get_linear_index(buf_range, local_idx)]);
					    });
				    })
				    .wait();
			}
		}

		template <typename DataT, int Dims, typename KernelName = class buffer_reduce, typename ReduceT, typename Operation>
		ReduceT buffer_reduce(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, ReduceT init, Operation op) {
			const auto range3 = detail::range_cast<3>(range);
			const auto offset3 = detail::id_cast<3>(offset);

			if(tgt == access_target::host) {
				auto info = m_bm->access_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
				const auto buf_range = info.backing_buffer_range;
				ReduceT result = init;
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - info.backing_buffer_offset;
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							result = op(detail::id_cast<Dims>(global_idx), result, static_cast<DataT*>(info.ptr)[linear_idx]);
						}
					}
				}
				return result;
			}

			auto info = m_bm->access_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
			const auto buf_offset = info.backing_buffer_offset;
			const auto buf_range = info.backing_buffer_range;
			cl::sycl::buffer<ReduceT, 1> result_buf(1); // Use 1-dimensional instead of 0-dimensional since it's NYI in hipSYCL as of 0.8.1
			// Simply do a serial reduction on the device as well
			get_device_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto ptr = static_cast<DataT*>(info.ptr);
				    auto result_acc = result_buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
				    cgh.single_task<detail::bind_kernel_name<KernelName>>([=]() {
					    result_acc[0] = init;
					    for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
						    for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
							    for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
								    const auto global_idx = cl::sycl::id<3>(i, j, k);
								    const cl::sycl::id<3> local_idx = global_idx - buf_offset;
								    result_acc[0] = op(detail::id_cast<Dims>(global_idx), result_acc[0],
								        ptr[detail::get_linear_index(detail::range_cast<Dims>(buf_range), detail::id_cast<Dims>(local_idx))]);
							    }
						    }
					    }
				    });
			    })
			    .wait();

			ReduceT result;
			get_device_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = result_buf.template get_access<cl::sycl::access::mode::read>(cgh);
				    cgh.copy(acc, &result);
			    })
			    .wait();

			return result;
		}

		template <typename DataT, int Dims, access_mode Mode>
		accessor<DataT, Dims, Mode, target::device> get_device_accessor(
		    detail::buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = m_bm->access_device_buffer<DataT, Dims>(bid, Mode, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::accessor_testspy::make_device_accessor<DataT, Dims, Mode>(static_cast<DataT*>(buf_info.ptr),
			    detail::id_cast<Dims>(buf_info.backing_buffer_offset), detail::range_cast<Dims>(buf_info.backing_buffer_range));
		}

		template <typename DataT, int Dims, access_mode Mode>
		accessor<DataT, Dims, Mode, target::host_task> get_host_accessor(
		    detail::buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = m_bm->access_host_buffer<DataT, Dims>(bid, Mode, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::accessor_testspy::make_host_accessor<DataT, Dims, Mode>(subrange<Dims>(offset, range), static_cast<DataT*>(buf_info.ptr),
			    detail::id_cast<Dims>(buf_info.backing_buffer_offset), detail::range_cast<Dims>(buf_info.backing_buffer_range),
			    detail::range_cast<Dims>(m_bm->get_buffer_info(bid).range));
		}

	  private:
		bool m_initialized = false;
		std::unique_ptr<detail::buffer_manager> m_bm;
	};

} // namespace test_utils
} // namespace celerity