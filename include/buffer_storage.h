#pragma once

#include <array>
#include <cassert>

#include <SYCL/sycl.hpp>

namespace celerity {
namespace detail {

	struct raw_data_range {
		void* base_ptr;
		int dimensions;
		std::array<int, 3> full_size;
		std::array<int, 3> subsize;
		std::array<int, 3> offsets;
	};

	struct raw_data_read_handle : raw_data_range {
		explicit raw_data_read_handle(cl::sycl::accessor_base accessor) : accessor(accessor){};

	  private:
		// We store the related accessor (_base, for convenience) to ensure the memory remains valid
		// FIXME: accessor_base is a non-standard (ComputeCpp internal) object
		// TODO: An alternative approach would be to copy the data somewhere before sending it
		cl::sycl::accessor_base accessor;
	};

	struct buffer_storage_base {
		virtual raw_data_read_handle get_data(const cl::sycl::range<3>& offset, const cl::sycl::range<3>& range) = 0;
		virtual void set_data(const raw_data_range& dr) = 0;
		virtual ~buffer_storage_base() = default;
	};

	// TODO: We store just the SYCL buffer instead of a celerity buffer to avoid a circular dependency issue
	// In case we need the celerity buffer, we'll have to work around that somehow
	template <typename DataT, int Dims>
	struct buffer_storage : buffer_storage_base {};

	template <typename DataT>
	struct buffer_storage<DataT, 1> : buffer_storage_base {
		cl::sycl::buffer<DataT, 1>& buf;

		buffer_storage(cl::sycl::buffer<DataT, 1>& buf) : buf(buf) {}

		raw_data_read_handle get_data(const cl::sycl::range<3>& offset, const cl::sycl::range<3>& range) override {
			assert(offset[1] == 0 && range[1] == 0);
			assert(offset[2] == 0 && range[2] == 0);

			auto acc = buf.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(range[0]), cl::sycl::id<1>(offset[0]));
			auto buf_size = buf.get_range();
			raw_data_read_handle result(acc);
			result.dimensions = 1;
			result.base_ptr = acc.get_pointer();
			// FIXME: Check bounds before casting to int!
			result.full_size = {(int)buf_size[0], 0, 0};
			result.subsize = {(int)range[0], 0, 0};
			result.offsets = {(int)offset[0], 0, 0};
			return result;
		}

		void set_data(const raw_data_range& dr) override {
			assert(dr.dimensions == 1);
			auto acc = buf.template get_access<cl::sycl::access::mode::write>(cl::sycl::range<1>(dr.subsize[0]), cl::sycl::id<1>(dr.offsets[0]));
			const auto dst_ptr = acc.get_pointer();
			std::memcpy(dst_ptr + dr.offsets[0], dr.base_ptr, dr.subsize[0] * sizeof(DataT));
		}
	};

	template <typename DataT>
	struct buffer_storage<DataT, 2> : buffer_storage_base {
		cl::sycl::buffer<DataT, 2>& buf;

		buffer_storage(cl::sycl::buffer<DataT, 2>& buf) : buf(buf) {}

		raw_data_read_handle get_data(const cl::sycl::range<3>& offset, const cl::sycl::range<3>& range) override {
			assert(offset[2] == 0 && range[2] == 0);

			auto acc = buf.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<2>(range[0], range[1]), cl::sycl::id<2>(offset[0], offset[1]));
			auto buf_size = buf.get_range();
			raw_data_read_handle result(acc);
			result.dimensions = 2;
			result.base_ptr = acc.get_pointer();
			// FIXME: Check bounds before casting to int!
			result.full_size = {(int)buf_size[0], (int)buf_size[1], 0};
			result.subsize = {(int)range[0], (int)range[1], 0};
			result.offsets = {(int)offset[0], (int)offset[1], 0};
			return result;
		}

		void set_data(const raw_data_range& dr) override {
			assert(dr.dimensions == 2);
			auto acc = buf.template get_access<cl::sycl::access::mode::write>(
			    cl::sycl::range<2>(dr.subsize[0], dr.subsize[1]), cl::sycl::id<2>(dr.offsets[0], dr.offsets[1]));
			const auto dst_ptr = acc.get_pointer();
			const auto buffer_size = buf.get_range();
			const auto base_row_offset = dr.offsets[0] * buffer_size[0];
			for(auto i = 0; i < dr.subsize[0]; ++i) {
				std::memcpy(dst_ptr + base_row_offset + i * buffer_size[0] + dr.offsets[1], reinterpret_cast<DataT*>(dr.base_ptr) + i * dr.subsize[1],
				    dr.subsize[1] * sizeof(DataT));
			}
		}
	};

	template <typename DataT>
	struct buffer_storage<DataT, 3> : buffer_storage_base {
		cl::sycl::buffer<DataT, 3>& buf;

		buffer_storage(cl::sycl::buffer<DataT, 3>& buf) : buf(buf) {}

		raw_data_read_handle get_data(const cl::sycl::range<3>& offset, const cl::sycl::range<3>& range) override {
			auto acc = buf.template get_access<cl::sycl::access::mode::read>(
			    cl::sycl::range<3>(range[0], range[1], range[2]), cl::sycl::id<3>(offset[0], offset[1], offset[2]));
			auto buf_size = buf.get_range();
			raw_data_read_handle result(acc);
			result.dimensions = 3;
			result.base_ptr = acc.get_pointer();
			// FIXME: Check bounds before casting to int!
			result.full_size = {(int)buf_size[0], (int)buf_size[1], (int)buf_size[2]};
			result.subsize = {(int)range[0], (int)range[1], (int)range[2]};
			result.offsets = {(int)offset[0], (int)offset[1], (int)offset[2]};
			return result;
		}

		void set_data(const raw_data_range& dr) override {
			assert(dr.dimensions == 3);
			throw std::runtime_error("set_data_range for 3 dimensions NYI");
		}
	};

} // namespace detail
} // namespace celerity
