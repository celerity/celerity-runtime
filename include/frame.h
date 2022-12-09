#pragma once

#include <cassert>

#include "host_allocator.h"
#include "payload.h"

namespace celerity::detail {

struct from_payload_count_tag {
} inline constexpr from_payload_count;

struct from_size_bytes_tag {
} inline constexpr from_size_bytes;

/**
 * Owning smart pointer for variable-sized structures with a 0-sized array of type Frame::payload_type as the last member.
 */
template <typename Frame>
class unique_frame_ptr : private std::unique_ptr<Frame, std::function<void(Frame*)>> {
  private:
	using impl = std::unique_ptr<Frame, std::function<void(Frame*)>>;

	friend class unique_payload_ptr;

  public:
	using payload_type = typename Frame::payload_type;

	unique_frame_ptr() = default;

	unique_frame_ptr(from_payload_count_tag, size_t payload_count, size_t packet_size_bytes = 1)
	    : unique_frame_ptr(from_size_bytes, sizeof(Frame) + sizeof(payload_type) * payload_count, packet_size_bytes) {}

	unique_frame_ptr(from_size_bytes_tag, size_t size_bytes, size_t packet_size_bytes = 1)

	    : impl(make_frame(pad_to_packet_size(size_bytes, packet_size_bytes)),
	        [size_bytes, packet_size_bytes](Frame* frame) { delete_frame(frame, pad_to_packet_size(size_bytes, packet_size_bytes)); }),
	      m_size_bytes(pad_to_packet_size(size_bytes, packet_size_bytes)) {}

	unique_frame_ptr(unique_frame_ptr&& other) noexcept : impl(static_cast<impl&&>(other)), m_size_bytes(other.m_size_bytes) { other.m_size_bytes = 0; }

	unique_frame_ptr& operator=(unique_frame_ptr&& other) noexcept {
		if(this == &other) return *this;                        // gracefully handle self-assignment
		static_cast<impl&>(*this) = static_cast<impl&&>(other); // delegate to base class unique_ptr<Frame>::operator=() to delete previously held frame
		m_size_bytes = other.m_size_bytes;
		other.m_size_bytes = 0;
		return *this;
	}

	Frame* get_pointer() { return impl::get(); }
	const Frame* get_pointer() const { return impl::get(); }
	size_t get_size_bytes() const { return m_size_bytes; }

	using impl::operator bool;
	using impl::operator*;
	using impl::operator->;

	unique_payload_ptr into_payload_ptr() && {
		unique_payload_ptr::deleter_type deleter{[size_bytes = m_size_bytes](void* ptr) {
			delete_frame_from_payload(ptr, size_bytes);
		}}; // allocate deleter (aka std::function) first so `result` construction is noexcept
		const auto frame = this->release();
		const auto payload = reinterpret_cast<typename Frame::payload_type*>(frame + 1); // payload is located at +sizeof(Frame) bytes (+1 Frame object)
		return unique_payload_ptr{payload, std::move(deleter)};
	}

  private:
	size_t m_size_bytes = 0;

	static Frame* make_frame(const size_t size_bytes) {
		assert(size_bytes >= sizeof(Frame));
		assert((size_bytes - sizeof(Frame)) % sizeof(payload_type) == 0);
		const auto mem = host_allocator::get_instance().allocate(size_bytes);
		try {
			new(mem) Frame;
		} catch(...) {
			host_allocator::get_instance().free(mem, size_bytes);
			throw;
		}
		return static_cast<Frame*>(mem);
	}

	static size_t pad_to_packet_size(const size_t size_bytes, const size_t packet_size_bytes) {
		return (size_bytes + packet_size_bytes - 1) / packet_size_bytes * packet_size_bytes;
	}

  private:
	static void delete_frame(Frame* frame, const size_t size_bytes) {
		if(frame) {
			assert(size_bytes != 0);
			frame->~Frame();
			host_allocator::get_instance().free(frame, size_bytes);
		}
	}

	static void delete_frame_from_payload(void* const type_erased_payload, const size_t size_bytes) {
		const auto payload = static_cast<typename Frame::payload_type*>(type_erased_payload);
		const auto frame = reinterpret_cast<Frame*>(payload) - 1; // frame header is located at -sizeof(Frame) bytes (-1 Frame object)
		assert(size_bytes != 0);
		host_allocator::get_instance().free(frame, size_bytes);
	}
};

} // namespace celerity::detail
