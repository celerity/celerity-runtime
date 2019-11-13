#pragma once

#include <cassert>
#include <memory>

#include <CL/sycl.hpp>

#include "ranges.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	struct raw_data_handle {
		void* linearized_data_ptr = nullptr;
		cl::sycl::range<3> range;
		cl::sycl::id<3> offset;

		raw_data_handle() = default;
		raw_data_handle(void* linearized_data_ptr, cl::sycl::range<3> range, cl::sycl::id<3> offset)
		    : linearized_data_ptr(linearized_data_ptr), range(range), offset(offset) {}
		virtual ~raw_data_handle() = default;
	};

	struct raw_data_read_handle : raw_data_handle {
		size_t linearized_data_size = 0;

		void allocate(size_t byte_size) {
			linearized_data_ptr = malloc(byte_size);
			own_data = true;
		}

		~raw_data_read_handle() {
			if(linearized_data_ptr != nullptr && own_data) { free(linearized_data_ptr); }
		}

	  private:
		bool own_data = false;
	};

	class buffer_storage_base {
	  public:
		/**
		 * @param range The size of the buffer
		 */
		buffer_storage_base(cl::sycl::range<3> range) : range(range) {}

		cl::sycl::range<3> get_range() const { return range; }

		virtual std::shared_ptr<raw_data_read_handle> get_data(cl::sycl::queue& queue, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) = 0;
		virtual void set_data(cl::sycl::queue& queue, const raw_data_handle& dh) = 0;
		virtual ~buffer_storage_base() = default;

	  private:
		cl::sycl::range<3> range;
	};

	// FIXME: Remove this
	template <typename DataT, int Dims>
	class computecpp_get_data_workaround {};
	template <typename DataT, int Dims>
	class computecpp_set_data_workaround {};

	template <typename DataT, int Dims>
	class buffer_storage : public virtual buffer_storage_base {
	  public:
		buffer_storage(cl::sycl::range<Dims> range) : buffer_storage_base(detail::range_cast<3>(range)) {
			// TODO: Especially on master node it is likely overkill to initialize all buffers eagerly
			sycl_buf = std::make_unique<cl::sycl::buffer<DataT, Dims>>(detail::range_cast<Dims>(get_range()));
		}

		/**
		 * @brief Returns the underlying SYCL buffer.
		 */
		cl::sycl::buffer<DataT, Dims>& get_sycl_buffer() { return *sycl_buf; }

		std::shared_ptr<raw_data_read_handle> get_data(cl::sycl::queue& queue, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) override {
			assert(Dims > 1 || (offset[1] == 0 && range[1] == 1));
			assert(Dims > 2 || (offset[2] == 0 && range[2] == 1));

			auto result = std::make_shared<raw_data_read_handle>();
			result->range = range;
			result->offset = offset;
			result->linearized_data_size = sizeof(DataT) * range[0] * range[1] * range[2];

			result->allocate(result->linearized_data_size);
			// TODO: Ideally we'd not wait here and instead return some sort of async handle that can be waited upon
			auto buf = get_sycl_buffer();
			// Explicit memory operations appear to be broken in ComputeCpp as of version 1.0.5
			// As a workaround we create a temporary buffer and copy the contents manually.
#if WORKAROUND(COMPUTECPP, 1, 0, 5)
			cl::sycl::buffer<DataT, Dims> tmp_dst_buf(reinterpret_cast<DataT*>(result->linearized_data_ptr), cl::sycl::range<Dims>(range));
			const auto dim_offset = cl::sycl::id<Dims>(offset);
			auto event = queue.submit([&](cl::sycl::handler& cgh) {
				auto src_acc = buf.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<Dims>(range), dim_offset);
				auto dst_acc = tmp_dst_buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
				cgh.parallel_for<computecpp_get_data_workaround<DataT, Dims>>(
				    cl::sycl::range<Dims>(range), [=](cl::sycl::item<Dims> item) { dst_acc[item] = src_acc[item.get_id() + dim_offset]; });
			});
#else
			auto event = queue.submit([&](cl::sycl::handler& cgh) {
				auto acc = buf.template get_access<cl::sycl::access::mode::read>(cgh, detail::range_cast<Dims>(range), detail::id_cast<Dims>(offset));
				cgh.copy(acc, reinterpret_cast<DataT*>(result->linearized_data_ptr));
			});
#endif
			event.wait();

			return result;
		}

		void set_data(cl::sycl::queue& queue, const raw_data_handle& dh) override {
			// TODO: Ideally we'd not wait here and instead return some sort of async handle that can be waited upon
			auto buf = get_sycl_buffer();
			// Explicit memory operations appear to be broken in ComputeCpp as of version 1.0.5
			// As a workaround we create a temporary buffer and copy the contents manually.
#if WORKAROUND(COMPUTECPP, 1, 0, 5)
			cl::sycl::buffer<DataT, Dims> tmp_src_buf(reinterpret_cast<DataT*>(dh.linearized_data_ptr), cl::sycl::range<Dims>(dh.range));
			const auto dim_offset = cl::sycl::id<Dims>(dh.offset);
			auto event = queue.submit([&](cl::sycl::handler& cgh) {
				auto src_acc = tmp_src_buf.template get_access<cl::sycl::access::mode::read>(cgh);
				auto dst_acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, cl::sycl::range<Dims>(dh.range), dim_offset);
				cgh.parallel_for<computecpp_set_data_workaround<DataT, Dims>>(
				    cl::sycl::range<Dims>(dh.range), [=](cl::sycl::item<Dims> item) { dst_acc[item.get_id() + dim_offset] = src_acc[item]; });
			});
#else
			auto event = queue.submit([&](cl::sycl::handler& cgh) {
				auto acc =
				    buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, detail::range_cast<Dims>(dh.range), detail::id_cast<Dims>(dh.offset));
				cgh.copy(reinterpret_cast<DataT*>(dh.linearized_data_ptr), acc);
			});
#endif
			event.wait();
		}

	  private:
		std::unique_ptr<cl::sycl::buffer<DataT, Dims>> sycl_buf;
	};

} // namespace detail
} // namespace celerity
