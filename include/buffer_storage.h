#pragma once

#include <cassert>
#include <cstring>
#include <memory>

#include <CL/sycl.hpp>

#include "device_queue.h"
#include "legacy_backend/backend.h"
#include "payload.h"
#include "ranges.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<0>& source_range, const id<0>& source_offset,
	    const range<0>& target_range, const id<0>& target_offset, const range<0>& copy_range);

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<1>& source_range, const id<1>& source_offset,
	    const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range);

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<2>& source_range, const id<2>& source_offset,
	    const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range);

	void memcpy_strided_host(const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<3>& source_range, const id<3>& source_offset,
	    const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range);

	void linearize_subrange(const void* source_base_ptr, void* target_ptr, size_t elem_size, const range<3>& source_range, const subrange<3>& copy_sr);

	template <typename DataT, int Dims>
	class device_buffer {
	  public:
		device_buffer(const range<Dims>& range, device_queue& queue) : m_range(range), m_queue(queue) {
			if(m_range.size() != 0) { m_device_allocation = m_queue.malloc<DataT>(m_range.size()); }
		}

		~device_buffer() { m_queue.free(m_device_allocation); }

		device_buffer(const device_buffer&) = delete;
		device_buffer(device_buffer&&) noexcept = default;
		device_buffer& operator=(const device_buffer&) = delete;
		device_buffer& operator=(device_buffer&&) noexcept = default;

		range<Dims> get_range() const { return m_range; }

		DataT* get_pointer() { return static_cast<DataT*>(m_device_allocation.ptr); }

		const DataT* get_pointer() const { return static_cast<DataT*>(m_device_allocation.ptr); }

	  private:
		range<Dims> m_range;
		device_queue& m_queue;
		device_allocation m_device_allocation;
	};

	template <typename DataT, int Dims>
	class host_buffer {
	  public:
		explicit host_buffer(range<Dims> range) : m_range(range) {
			auto r3 = range_cast<3>(range);
			m_data = std::make_unique<DataT[]>(r3[0] * r3[1] * r3[2]);
		}

		range<Dims> get_range() const { return m_range; };

		DataT* get_pointer() { return m_data.get(); }

		const DataT* get_pointer() const { return m_data.get(); }

		bool operator==(const host_buffer& rhs) const { return m_data.get() == rhs.m_data.get(); }

	  private:
		range<Dims> m_range;
		std::unique_ptr<DataT[]> m_data;
	};

	enum class buffer_type { device_buffer, host_buffer };

	class buffer_storage {
	  public:
		/**
		 * @param range The size of the buffer
		 */
		buffer_storage(range<3> range, buffer_type type) : m_range(range), m_type(type) {}

		range<3> get_range() const { return m_range; }

		buffer_type get_type() const { return m_type; }

		/**
		 * Returns the buffer size, in bytes.
		 */
		virtual size_t get_size() const = 0;

		virtual void* get_pointer() = 0;

		virtual void get_data(const subrange<3>& sr, void* out_linearized) const = 0;

		virtual void set_data(const subrange<3>& sr, const void* in_linearized) = 0;

		/**
		 * Copy data from the given source buffer into this buffer.
		 *
		 * TODO: Consider making this non-blocking, returning an async handle instead.
		 */
		virtual void copy(const buffer_storage& source, id<3> source_offset, id<3> target_offset, range<3> copy_range) = 0;

		virtual ~buffer_storage() = default;

	  private:
		range<3> m_range;
		buffer_type m_type;
	};

	inline void assert_copy_is_in_range(
	    const range<3>& source_range, const range<3>& target_range, const id<3>& source_offset, const id<3>& target_offset, const range<3>& copy_range) {
		assert(range_max(source_range, range_cast<3>(source_offset + copy_range)) == source_range);
		assert(range_max(target_range, range_cast<3>(target_offset + copy_range)) == target_range);
	}

	template <typename DataT, int Dims>
	class device_buffer_storage : public buffer_storage {
	  public:
		device_buffer_storage(range<Dims> range, device_queue& owning_queue)
		    : buffer_storage(range_cast<3>(range), buffer_type::device_buffer), m_owning_queue(owning_queue.get_sycl_queue()),
		      m_device_buf(range, owning_queue) {}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		void* get_pointer() override { return m_device_buf.get_pointer(); }

		device_buffer<DataT, Dims>& get_device_buffer() { return m_device_buf; }

		const device_buffer<DataT, Dims>& get_device_buffer() const { return m_device_buf; }

		void get_data(const subrange<3>& sr, void* out_linearized) const override {
			assert(Dims > 0 || (sr.offset[0] == 0 && sr.range[0] == 1));
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));
			assert_copy_is_in_range(range_cast<3>(m_device_buf.get_range()), sr.range, sr.offset, id<3>{}, sr.range);

			// TODO: Ideally we'd make this non-blocking and return some sort of async handle that can be waited upon
			legacy_backend::memcpy_strided_device(m_owning_queue, m_device_buf.get_pointer(), out_linearized, sizeof(DataT), m_device_buf.get_range(),
			    id_cast<Dims>(sr.offset), range_cast<Dims>(sr.range), id<Dims>{}, range_cast<Dims>(sr.range));
		}

		void set_data(const subrange<3>& sr, const void* in_linearized) override {
			assert(Dims > 0 || (sr.offset[0] == 0 && sr.range[0] == 1));
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));
			assert_copy_is_in_range(sr.range, range_cast<3>(m_device_buf.get_range()), id<3>{}, sr.offset, sr.range);

			// TODO: Ideally we'd make this non-blocking and return some sort of async handle that can be waited upon
			legacy_backend::memcpy_strided_device(m_owning_queue, in_linearized, m_device_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(sr.range),
			    id<Dims>{}, m_device_buf.get_range(), id_cast<Dims>(sr.offset), range_cast<Dims>(sr.range));
		}

		void copy(const buffer_storage& source, id<3> source_offset, id<3> target_offset, range<3> copy_range) override;

	  private:
		mutable sycl::queue m_owning_queue;
		device_buffer<DataT, Dims> m_device_buf;
	};

	template <typename DataT, int Dims>
	class host_buffer_storage : public buffer_storage {
	  public:
		explicit host_buffer_storage(range<Dims> range) : buffer_storage(range_cast<3>(range), buffer_type::host_buffer), m_host_buf(range) {}

		size_t get_size() const override { return get_range().size() * sizeof(DataT); };

		void* get_pointer() override { return m_host_buf.get_pointer(); }

		void get_data(const subrange<3>& sr, void* out_linearized) const override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));
			assert_copy_is_in_range(range_cast<3>(m_host_buf.get_range()), sr.range, sr.offset, id<3>{}, sr.range);

			memcpy_strided_host(m_host_buf.get_pointer(), out_linearized, sizeof(DataT), range_cast<Dims>(m_host_buf.get_range()), id_cast<Dims>(sr.offset),
			    range_cast<Dims>(sr.range), id<Dims>(), range_cast<Dims>(sr.range));
		}

		void set_data(const subrange<3>& sr, const void* in_linearized) override {
			assert(Dims > 1 || (sr.offset[1] == 0 && sr.range[1] == 1));
			assert(Dims > 2 || (sr.offset[2] == 0 && sr.range[2] == 1));
			assert_copy_is_in_range(sr.range, range_cast<3>(m_host_buf.get_range()), id<3>{}, sr.offset, sr.range);

			memcpy_strided_host(in_linearized, m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(sr.range), id<Dims>(),
			    range_cast<Dims>(m_host_buf.get_range()), id_cast<Dims>(sr.offset), range_cast<Dims>(sr.range));
		}

		void copy(const buffer_storage& source, id<3> source_offset, id<3> target_offset, range<3> copy_range) override;

		host_buffer<DataT, Dims>& get_host_buffer() { return m_host_buf; }

		const host_buffer<DataT, Dims>& get_host_buffer() const { return m_host_buf; }

	  private:
		host_buffer<DataT, Dims> m_host_buf;
	};

	template <typename DataT, int Dims>
	void device_buffer_storage<DataT, Dims>::copy(const buffer_storage& source, id<3> source_offset, id<3> target_offset, range<3> copy_range) {
		assert_copy_is_in_range(source.get_range(), range_cast<3>(m_device_buf.get_range()), source_offset, target_offset, copy_range);

		if(source.get_type() == buffer_type::device_buffer) {
			auto& device_source = dynamic_cast<const device_buffer_storage<DataT, Dims>&>(source);
			legacy_backend::memcpy_strided_device(m_owning_queue, device_source.m_device_buf.get_pointer(), m_device_buf.get_pointer(), sizeof(DataT),
			    device_source.m_device_buf.get_range(), id_cast<Dims>(source_offset), m_device_buf.get_range(), id_cast<Dims>(target_offset),
			    range_cast<Dims>(copy_range));
		}

		// TODO: Optimize for contiguous copies - we could do a single SYCL H->D copy directly.
		else if(source.get_type() == buffer_type::host_buffer) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			// TODO: No need for intermediate copy with native backend 2D/3D copy capabilities
			auto tmp = make_uninitialized_payload<DataT>(copy_range.size());
			host_source.get_data(subrange{source_offset, copy_range}, static_cast<DataT*>(tmp.get_pointer()));
			set_data(subrange{target_offset, copy_range}, static_cast<const DataT*>(tmp.get_pointer()));
		}

		else {
			assert(false);
		}
	}

	template <typename DataT, int Dims>
	void host_buffer_storage<DataT, Dims>::copy(const buffer_storage& source, id<3> source_offset, id<3> target_offset, range<3> copy_range) {
		assert_copy_is_in_range(source.get_range(), range_cast<3>(m_host_buf.get_range()), source_offset, target_offset, copy_range);

		// TODO: Optimize for contiguous copies - we could do a single SYCL D->H copy directly.
		if(source.get_type() == buffer_type::device_buffer) {
			// This looks more convoluted than using a vector<DataT>, but that would break if DataT == bool
			// TODO: No need for intermediate copy with native backend 2D/3D copy capabilities
			auto tmp = make_uninitialized_payload<DataT>(copy_range.size());
			source.get_data(subrange{source_offset, copy_range}, static_cast<DataT*>(tmp.get_pointer()));
			set_data(subrange{target_offset, copy_range}, static_cast<const DataT*>(tmp.get_pointer()));
		}

		else if(source.get_type() == buffer_type::host_buffer) {
			auto& host_source = dynamic_cast<const host_buffer_storage<DataT, Dims>&>(source);
			memcpy_strided_host(host_source.get_host_buffer().get_pointer(), m_host_buf.get_pointer(), sizeof(DataT), range_cast<Dims>(host_source.get_range()),
			    id_cast<Dims>(source_offset), range_cast<Dims>(m_host_buf.get_range()), range_cast<Dims>(target_offset), range_cast<Dims>(copy_range));
		}

		else {
			assert(false);
		}
	}

} // namespace detail
} // namespace celerity
