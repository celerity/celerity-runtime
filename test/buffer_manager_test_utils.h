#pragma once

#include "test_utils.h"

namespace celerity {
namespace detail {

	struct buffer_manager_testspy {
		template <typename DataT, int Dims>
		static buffer_manager::access_info<DataT, Dims, device_buffer> get_device_buffer(buffer_manager& bm, buffer_id bid) {
			std::unique_lock lock(bm.m_mutex);
			auto& buf = bm.m_buffers.at(bid).device_buf;
			return {dynamic_cast<device_buffer_storage<DataT, Dims>*>(buf.storage.get())->get_device_buffer(), id_cast<Dims>(buf.offset)};
		}
	};
} // namespace detail

namespace test_utils {

	class buffer_manager_fixture : public device_queue_fixture {
	  public:
		enum class access_target { host, device };

		void initialize(detail::buffer_manager::buffer_lifecycle_callback cb = [](detail::buffer_manager::buffer_lifecycle_event, detail::buffer_id) {}) {
			assert(!bm);
			bm = std::make_unique<detail::buffer_manager>(get_device_queue(), cb);
			bm->enable_test_mode();
		}

		detail::buffer_manager& get_buffer_manager() {
			if(!bm) initialize();
			return *bm;
		}

		static access_target get_other_target(access_target tgt) {
			if(tgt == access_target::host) return access_target::device;
			return access_target::host;
		}

		template <typename DataT, int Dims>
		cl::sycl::range<Dims> get_backing_buffer_range(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset) {
			if(tgt == access_target::host) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, detail::range_cast<3>(range), detail::id_cast<3>(offset));
				return info.buffer.get_range();
			}
			auto info = bm->get_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return info.buffer.get_range();
		}

		template <typename DataT, int Dims, cl::sycl::access::mode Mode, typename KernelName = class buffer_for_each, typename Callback>
		void buffer_for_each(detail::buffer_id bid, access_target tgt, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset, Callback cb) {
			const auto range3 = detail::range_cast<3>(range);
			const auto offset3 = detail::id_cast<3>(offset);

			if(tgt == access_target::host) {
				auto info = bm->get_host_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_range = detail::range_cast<3>(info.buffer.get_range());
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - detail::id_cast<3>(info.offset);
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							cb(detail::id_cast<Dims>(global_idx), info.buffer.get_pointer()[linear_idx]);
						}
					}
				}
			}

			if(tgt == access_target::device) {
				auto info = bm->get_device_buffer<DataT, Dims>(bid, Mode, range3, offset3);
				const auto buf_offset = info.offset;
				get_device_queue()
				    .submit([&](cl::sycl::handler& cgh) {
					    auto acc = info.buffer.template get_access<Mode>(cgh);
					    cgh.parallel_for<detail::bind_kernel_name<KernelName>>(range, [=](cl::sycl::id<Dims> global_idx) {
						    global_idx += offset;
						    const auto local_idx = global_idx - buf_offset;
						    cb(global_idx, acc[local_idx]);
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
				auto info = bm->get_host_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
				const auto buf_range = detail::range_cast<3>(info.buffer.get_range());
				ReduceT result = init;
				for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
					for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
						for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
							const auto global_idx = cl::sycl::id<3>(i, j, k);
							const cl::sycl::id<3> local_idx = global_idx - detail::id_cast<3>(info.offset);
							const size_t linear_idx = local_idx[0] * buf_range[1] * buf_range[2] + local_idx[1] * buf_range[2] + local_idx[2];
							result = op(detail::id_cast<Dims>(global_idx), result, info.buffer.get_pointer()[linear_idx]);
						}
					}
				}
				return result;
			}

			auto info = bm->get_device_buffer<DataT, Dims>(bid, cl::sycl::access::mode::read, range3, offset3);
			const auto buf_offset = info.offset;
			cl::sycl::buffer<ReduceT, 1> result_buf(1); // Use 1-dimensional instead of 0-dimensional since it's NYI in hipSYCL as of 0.8.1
			// Simply do a serial reduction on the device as well
			get_device_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    auto acc = info.buffer.template get_access<cl::sycl::access::mode::read>(cgh);
				    auto result_acc = result_buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
				    cgh.single_task<detail::bind_kernel_name<KernelName>>([=]() {
					    result_acc[0] = init;
					    for(size_t i = offset3[0]; i < offset3[0] + range3[0]; ++i) {
						    for(size_t j = offset3[1]; j < offset3[1] + range3[1]; ++j) {
							    for(size_t k = offset3[2]; k < offset3[2] + range3[2]; ++k) {
								    const auto global_idx = cl::sycl::id<3>(i, j, k);
								    const cl::sycl::id<3> local_idx = global_idx - detail::id_cast<3>(buf_offset);
								    result_acc[0] = op(detail::id_cast<Dims>(global_idx), result_acc[0], acc[detail::id_cast<Dims>(local_idx)]);
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
		    detail::live_pass_device_handler& cgh, detail::buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = bm->get_device_buffer<DataT, Dims>(bid, Mode, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::make_device_accessor<DataT, Dims, Mode>(cgh.get_eventual_sycl_cgh(), buf_info.buffer,
			    detail::get_effective_sycl_accessor_subrange(buf_info.offset, subrange<Dims>(offset, range)), offset);
		}

		template <typename DataT, int Dims, access_mode Mode>
		accessor<DataT, Dims, Mode, target::host_task> get_host_accessor(
		    detail::buffer_id bid, const cl::sycl::range<Dims>& range, const cl::sycl::id<Dims>& offset) {
			auto buf_info = bm->get_host_buffer<DataT, Dims>(bid, Mode, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			return detail::make_host_accessor<DataT, Dims, Mode>(
			    subrange<Dims>(offset, range), buf_info.buffer, buf_info.offset, detail::range_cast<Dims>(bm->get_buffer_info(bid).range));
		}

	  private:
		bool initialized = false;
		std::unique_ptr<detail::buffer_manager> bm;
	};

} // namespace test_utils
} // namespace celerity