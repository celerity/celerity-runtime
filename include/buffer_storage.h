#pragma once

#include <cassert>
#include <cstring>
#include <memory>

#include <CL/sycl.hpp>

#include "ranges.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	size_t get_linear_index(const cl::sycl::range<1>& buffer_range, const cl::sycl::id<1>& index);

	size_t get_linear_index(const cl::sycl::range<2>& buffer_range, const cl::sycl::id<2>& index);

	size_t get_linear_index(const cl::sycl::range<3>& buffer_range, const cl::sycl::id<3>& index);

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<1>& source_range,
	    const cl::sycl::id<1>& source_offset, const cl::sycl::range<1>& target_range, const cl::sycl::id<1>& target_offset,
	    const cl::sycl::range<1>& copy_range);

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<2>& source_range,
	    const cl::sycl::id<2>& source_offset, const cl::sycl::range<2>& target_range, const cl::sycl::id<2>& target_offset,
	    const cl::sycl::range<2>& copy_range);

	void memcpy_strided(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const cl::sycl::range<3>& source_range,
	    const cl::sycl::id<3>& source_offset, const cl::sycl::range<3>& target_range, const cl::sycl::id<3>& target_offset,
	    const cl::sycl::range<3>& copy_range);

	/**
	 * Dense, linearized host-side storage for buffer data.
	 */
	class raw_buffer_data {
	  public:
		raw_buffer_data() {}

		raw_buffer_data(size_t elem_size, cl::sycl::range<3> range) : elem_size(elem_size), range(range) {
			const size_t size = get_size();
			data = std::make_unique<unsigned char[]>(size);
		}

		raw_buffer_data(const raw_buffer_data&) = delete;

		raw_buffer_data(raw_buffer_data&& other) noexcept { *this = std::move(other); }

		raw_buffer_data& operator=(raw_buffer_data&& other) noexcept {
			elem_size = other.elem_size;
			range = other.range;
			data = std::move(other.data);
			return *this;
		}

		/**
		 * Changes the element size and range of this buffer.
		 * Note that the resulting data size must remain the same as before.
		 */
		void reinterpret(size_t elem_size, cl::sycl::range<3> range) {
			assert(elem_size * range.size() == this->elem_size * this->range.size());
			this->elem_size = elem_size;
			this->range = range;
		}

		/**
		 * Returns the pointer to the dense, linearized data location.
		 */
		void* get_pointer() const { return data.get(); }

		cl::sycl::range<3> get_range() const { return range; }

		/**
		 * Returns the data size, in bytes.
		 */
		size_t get_size() const { return elem_size * range.size(); }

		/**
		 * Copies the specified data subrange into a new (unstrided) raw_buffer_data instance.
		 */
		raw_buffer_data copy(cl::sycl::id<3> offset, cl::sycl::range<3> copy_range);

	  private:
		size_t elem_size = 0;
		cl::sycl::range<3> range = {};
		std::unique_ptr<unsigned char[]> data;
	};

	template <typename DataT, int Dims>
	using device_buffer = cl::sycl::buffer<DataT, Dims>;

	template <typename DataT, int Dims>
	class host_buffer {
	  public:
		host_buffer(cl::sycl::range<Dims> range) : range(range) {
			auto r3 = range_cast<3>(range);
			data = std::make_unique<DataT[]>(r3[0] * r3[1] * r3[2]);
		}

		cl::sycl::range<Dims> get_range() const { return range; };

		DataT* get_pointer() { return data.get(); }

		const DataT* get_pointer() const { return data.get(); }

		bool operator==(const host_buffer& rhs) const { return data.get() == rhs.data.get(); }

	  private:
		cl::sycl::range<Dims> range;
		std::unique_ptr<DataT[]> data;
	};

	enum class buffer_type { DEVICE_BUFFER, HOST_BUFFER };

	class buffer_storage {
	  public:
		/**
		 * @param range The size of the buffer
		 */
		buffer_storage(cl::sycl::range<3> range, buffer_type type) : range(range), type(type) {}

		cl::sycl::range<3> get_range() const { return range; }

		buffer_type get_type() const { return type; }

		/**
		 * Returns the buffer size, in bytes.
		 */
		virtual size_t get_size() const = 0;

		virtual raw_buffer_data get_data(const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) const = 0;

		virtual void set_data(cl::sycl::id<3> offset, raw_buffer_data data) = 0;

		/**
		 * Convenience function to create new buffer_storages of the same (templated) type, useful in contexts where template type information is not available.
		 */
		virtual buffer_storage* make_new_of_same_type(cl::sycl::range<3> range) const = 0;

		/**
		 * Copy data from the given source buffer into this buffer.
		 *
		 * TODO: Consider making this non-blocking, returning an async handle instead.
		 */
		virtual void copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) = 0;

		virtual ~buffer_storage() = default;

	  private:
		cl::sycl::range<3> range;
		buffer_type type;
	};

	// FIXME: Remove this
	template <typename DataT, int Dims>
	class computecpp_get_data_workaround {};
	template <typename DataT, int Dims>
	class computecpp_set_data_workaround {};

	template <typename DataT, int Dims>
	class device_buffer_storage : public buffer_storage {
	  public:
		device_buffer_storage(cl::sycl::range<Dims> range, cl::sycl::queue transfer_queue)
		    : buffer_storage(range_cast<3>(range), buffer_type::DEVICE_BUFFER), transfer_queue(transfer_queue), device_buf(range) {
			// We never want SYCL to do any buffer write-backs. While we don't pass any host pointers to SYCL buffers,
			// meaning there shouldn't be any write-back in the first place, it doesn't hurt to make sure.
			// (This was prompted by a hipSYCL bug that did superfluous write-backs).
			device_buf.set_write_back(false);
		}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		/**
		 * @brief Returns the underlying SYCL buffer.
		 */
		device_buffer<DataT, Dims>& get_device_buffer() { return device_buf; }

		const device_buffer<DataT, Dims>& get_device_buffer() const { return device_buf; }

		raw_buffer_data get_data(const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) const override {
			assert(Dims > 1 || (offset[1] == 0 && range[1] == 1));
			assert(Dims > 2 || (offset[2] == 0 && range[2] == 1));

			auto result = raw_buffer_data{sizeof(DataT), range};
			auto buf = get_device_buffer();

			// ComputeCpp (as of version 2.1.0) expects the target pointer of an explicit copy operation to have the same size as the buffer,
			// even though the SYCL 1.2.1 Rev 7 spec states that "dest must have at least as many bytes as the range accessed by src", which
			// in my (psalz) opinion indicates that this is supposed to result in a contiguous copy of potentially strided source data.
			// As a workaround, we copy the data manually using a kernel.
#if WORKAROUND_COMPUTECPP
			cl::sycl::buffer<DataT, Dims> tmp_dst_buf(reinterpret_cast<DataT*>(result.get_pointer()), range_cast<Dims>(range));
			auto event = transfer_queue.submit([&](cl::sycl::handler& cgh) {
				auto src_acc = buf.template get_access<cl::sycl::access::mode::read>(cgh, range_cast<Dims>(range), id_cast<Dims>(offset));
				auto dst_acc = tmp_dst_buf.template get_access<cl::sycl::access::mode::discard_write>(cgh);
				cgh.parallel_for<computecpp_get_data_workaround<DataT, Dims>>(
				    range_cast<Dims>(range), [=, offset = id_cast<Dims>(offset)](cl::sycl::id<Dims> id) { dst_acc[id] = src_acc[offset + id]; });
			});
#else
			auto event = transfer_queue.submit([&](cl::sycl::handler& cgh) {
				auto acc = buf.template get_access<cl::sycl::access::mode::read>(cgh, range_cast<Dims>(range), id_cast<Dims>(offset));
				cgh.copy(acc, reinterpret_cast<DataT*>(result.get_pointer()));
			});
#endif

			// TODO: Ideally we'd not wait here and instead return some sort of async handle that can be waited upon
			event.wait();

			return result;
		}

		void set_data(cl::sycl::id<3> offset, raw_buffer_data data) override {
			assert(Dims > 1 || (offset[1] == 0 && data.get_range()[1] == 1));
			assert(Dims > 2 || (offset[2] == 0 && data.get_range()[2] == 1));
			assert(data.get_size() == data.get_range().size() * sizeof(DataT));
			assert(data.get_size() <= device_buf.get_range().size() * sizeof(DataT));

			auto buf = get_device_buffer();

			// See above for why this workaround is needed.
#if WORKAROUND_COMPUTECPP
			cl::sycl::buffer<DataT, Dims> tmp_src_buf(reinterpret_cast<DataT*>(data.get_pointer()), range_cast<Dims>(data.get_range()));
			auto event = transfer_queue.submit([&](cl::sycl::handler& cgh) {
				auto src_acc = tmp_src_buf.template get_access<cl::sycl::access::mode::read>(cgh);
				auto dst_acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, range_cast<Dims>(data.get_range()), id_cast<Dims>(offset));
				cgh.parallel_for<computecpp_set_data_workaround<DataT, Dims>>(
				    range_cast<Dims>(data.get_range()), [=, offset = id_cast<Dims>(offset)](cl::sycl::id<Dims> id) { dst_acc[offset + id] = src_acc[id]; });
			});
#else
			auto event = transfer_queue.submit([&](cl::sycl::handler& cgh) {
				auto acc = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, range_cast<Dims>(data.get_range()), id_cast<Dims>(offset));
				cgh.copy(reinterpret_cast<DataT*>(data.get_pointer()), acc);
			});
#endif

			// TODO: Ideally we'd not wait here and instead return some sort of async handle that can be waited upon
			event.wait();
		}

		buffer_storage* make_new_of_same_type(cl::sycl::range<3> range) const override {
			return new device_buffer_storage<DataT, Dims>(range_cast<Dims>(range), transfer_queue);
		}

		void copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) override;

		cl::sycl::queue& get_transfer_queue() { return transfer_queue; }

	  private:
		mutable cl::sycl::queue transfer_queue;
		device_buffer<DataT, Dims> device_buf;
	};

	template <typename DataT, int Dims>
	class host_buffer_storage : public buffer_storage {
	  public:
		host_buffer_storage(cl::sycl::range<Dims> range) : buffer_storage(range_cast<3>(range), buffer_type::HOST_BUFFER), host_buf(range) {}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		raw_buffer_data get_data(const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) const override {
			assert(Dims > 1 || (offset[1] == 0 && range[1] == 1));
			assert(Dims > 2 || (offset[2] == 0 && range[2] == 1));

			auto result = raw_buffer_data{sizeof(DataT), range};
			memcpy_strided(host_buf.get_pointer(), result.get_pointer(), sizeof(DataT), range_cast<Dims>(host_buf.get_range()), id_cast<Dims>(offset),
			    range_cast<Dims>(range), id_cast<Dims>(cl::sycl::id<3>{0, 0, 0}), range_cast<Dims>(range));
			return result;
		}

		void set_data(cl::sycl::id<3> offset, raw_buffer_data data) override {
			assert(Dims > 1 || (offset[1] == 0 && data.get_range()[1] == 1));
			assert(Dims > 2 || (offset[2] == 0 && data.get_range()[2] == 1));
			assert(data.get_size() == data.get_range().size() * sizeof(DataT));
			assert(data.get_size() <= host_buf.get_range().size() * sizeof(DataT));

			memcpy_strided(reinterpret_cast<DataT*>(data.get_pointer()), host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(data.get_range()),
			    id_cast<Dims>(cl::sycl::id<3>(0, 0, 0)), range_cast<Dims>(host_buf.get_range()), id_cast<Dims>(offset), range_cast<Dims>(data.get_range()));
		}

		buffer_storage* make_new_of_same_type(cl::sycl::range<3> range) const override { return new host_buffer_storage<DataT, Dims>(range_cast<Dims>(range)); }

		void copy(const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) override;

		host_buffer<DataT, Dims>& get_host_buffer() { return host_buf; }

		const host_buffer<DataT, Dims>& get_host_buffer() const { return host_buf; }

	  private:
		host_buffer<DataT, Dims> host_buf;
	};

	inline void assert_copy_is_in_range(const cl::sycl::range<3>& source_range, const cl::sycl::range<3>& target_range, const cl::sycl::id<3>& source_offset,
	    const cl::sycl::id<3>& target_offset, const cl::sycl::range<3>& copy_range) {
		assert(max_range(source_range, range_cast<3>(source_offset + copy_range)) == source_range);
		assert(max_range(target_range, range_cast<3>(target_offset + copy_range)) == target_range);
	}

	template <typename DataT, int Dims>
	void device_buffer_storage<DataT, Dims>::copy(
	    const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) {
		assert_copy_is_in_range(source.get_range(), range_cast<3>(device_buf.get_range()), source_offset, target_offset, copy_range);

		if(source.get_type() == buffer_type::DEVICE_BUFFER) {
			auto& device_source = dynamic_cast<const device_buffer_storage<DataT, Dims>&>(source);
			auto event = transfer_queue.submit([&](cl::sycl::handler& cgh) {
				// FIXME: Getting read access is currently not a const operation on SYCL buffers
				// Resolve once https://github.com/KhronosGroup/SYCL-Docs/issues/10 has been clarified
				auto source_acc = const_cast<device_buffer<DataT, Dims>&>(device_source.get_device_buffer())
				                      .template get_access<cl::sycl::access::mode::read>(cgh, range_cast<Dims>(copy_range), id_cast<Dims>(source_offset));
				auto target_acc =
				    device_buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, range_cast<Dims>(copy_range), id_cast<Dims>(target_offset));
				cgh.copy(source_acc, target_acc);
			});
			event.wait();
		}

		// TODO: Optimize for contiguous copies - we could do a single SYCL H->D copy directly.
		else if(source.get_type() == buffer_type::HOST_BUFFER) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			set_data(target_offset, host_source.get_data(source_offset, copy_range));
		}

		else {
			assert(false);
		}
	}

	template <typename DataT, int Dims>
	void host_buffer_storage<DataT, Dims>::copy(
	    const buffer_storage& source, cl::sycl::id<3> source_offset, cl::sycl::id<3> target_offset, cl::sycl::range<3> copy_range) {
		assert_copy_is_in_range(source.get_range(), range_cast<3>(host_buf.get_range()), source_offset, target_offset, copy_range);

		// TODO: Optimize for contiguous copies - we could do a single SYCL D->H copy directly.
		if(source.get_type() == buffer_type::DEVICE_BUFFER) {
			auto data = source.get_data(source_offset, copy_range);
			set_data(target_offset, std::move(data));
		}

		else if(source.get_type() == buffer_type::HOST_BUFFER) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			memcpy_strided(host_source.get_host_buffer().get_pointer(), host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(host_source.get_range()),
			    id_cast<Dims>(source_offset), range_cast<Dims>(host_buf.get_range()), range_cast<Dims>(target_offset), range_cast<Dims>(copy_range));
		}

		else {
			assert(false);
		}
	}

} // namespace detail
} // namespace celerity
