#pragma once

#include <cassert>
#include <memory>
#include <utility>

namespace celerity::detail {

namespace mpi_support {

	constexpr int TAG_CMD = 0;
	constexpr int TAG_DATA_TRANSFER = 1;
	constexpr int TAG_TELEMETRY = 2;

} // namespace mpi_support

struct from_payload_size_tag {
} inline constexpr from_payload_size;

struct from_frame_bytes_tag {
} inline constexpr from_frame_bytes;

/**
 * Owning smart pointer for variable-sized structures with a 0-sized array of type Frame::payload_type as the last member.
 */
template <typename Frame>
class unique_frame_ptr : private std::unique_ptr<Frame> {
  private:
	using impl = std::unique_ptr<Frame>;

  public:
	using payload_type = typename Frame::payload_type;

	unique_frame_ptr() = default;

	unique_frame_ptr(from_payload_size_tag, size_t payload_size)
	    : impl(static_cast<Frame*>(operator new(frame_bytes_from_payload_size(payload_size)))), payload_size(payload_size) {
		new(impl::get()) Frame; // permits later deletion through std::default_deleter
	}

	unique_frame_ptr(from_frame_bytes_tag, size_t frame_bytes)
	    : impl(static_cast<Frame*>(operator new(frame_bytes))), payload_size(payload_size_from_frame_bytes(frame_bytes)) {
		new(impl::get()) Frame; // permits later deletion through std::default_deleter
	}

	unique_frame_ptr(unique_frame_ptr&& other) noexcept : impl(static_cast<impl&&>(other)), payload_size(other.payload_size) { other.payload_size = 0; }

	unique_frame_ptr& operator=(unique_frame_ptr&& other) noexcept {
		static_cast<impl&>(*this) = static_cast<impl&&>(other);
		payload_size = other.payload_size;
		other.payload_size = 0;
		return *this;
	}

	Frame* get_pointer() { return impl::get(); }
	const Frame* get_pointer() const { return impl::get(); }
	size_t get_payload_size() const { return payload_size; }
	size_t get_frame_size_bytes() const { return frame_bytes_from_payload_size(payload_size); }

	using impl::operator bool;
	using impl::operator*;
	using impl::operator->;

  private:
	size_t payload_size = 0;

	static size_t frame_bytes_from_payload_size(size_t payload_size) { //
		return sizeof(Frame) + sizeof(payload_type) * payload_size;
	}

	static size_t payload_size_from_frame_bytes(size_t frame_bytes) {
		assert(frame_bytes >= sizeof(Frame) && (frame_bytes - sizeof(Frame)) % sizeof(payload_type) == 0);
		return (frame_bytes - sizeof(Frame)) / sizeof(payload_type);
	}
};

} // namespace celerity::detail
