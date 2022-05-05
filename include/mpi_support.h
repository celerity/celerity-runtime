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

struct from_payload_count_tag {
} inline constexpr from_payload_count;

struct from_size_bytes_tag {
} inline constexpr from_size_bytes;

// unique_frame_ptr manually `operator new`s the underlying frame memory, placement-new-constructs the frame and casts it to a frame pointer.
// I'm convinced that I'm actually, technically allowed to use the resulting frame pointer in a delete-expression and therefore keep `std::default_delete` as
// the deleter type for `unique_frame_ptr::impl`: Following the standard, delete-expression requires its operand to originate from a new-expression,
// and placement-new is defined to be a new-expression. The following implicit call to operator delete is also legal, since memory was obtained from
// `operator new`. Despite the beauty of this standards loophole, @BlackMark29A and @PeterTh couldn't be convinced to let me merge it :(   -- @fknorr
template <typename Frame>
struct unique_frame_delete {
	void operator()(Frame* frame) const {
		if(frame) {
			frame->~Frame();
			operator delete(frame);
		}
	}
};

/**
 * Owning smart pointer for variable-sized structures with a 0-sized array of type Frame::payload_type as the last member.
 */
template <typename Frame>
class unique_frame_ptr : private std::unique_ptr<Frame, unique_frame_delete<Frame>> {
  private:
	using impl = std::unique_ptr<Frame, unique_frame_delete<Frame>>;

  public:
	using payload_type = typename Frame::payload_type;

	unique_frame_ptr() = default;

	unique_frame_ptr(from_payload_count_tag, size_t payload_count) : unique_frame_ptr(from_size_bytes, sizeof(Frame) + sizeof(payload_type) * payload_count) {}

	unique_frame_ptr(from_size_bytes_tag, size_t size_bytes) : impl(make_frame(size_bytes)), size_bytes(size_bytes) {}

	unique_frame_ptr(unique_frame_ptr&& other) noexcept : impl(static_cast<impl&&>(other)), size_bytes(other.size_bytes) { other.size_bytes = 0; }

	unique_frame_ptr& operator=(unique_frame_ptr&& other) noexcept {
		if(this == &other) return *this;                        // gracefully handle self-assignment
		static_cast<impl&>(*this) = static_cast<impl&&>(other); // delegate to base class unique_ptr<Frame>::operator=() to delete previously held frame
		size_bytes = other.size_bytes;
		other.size_bytes = 0;
		return *this;
	}

	Frame* get_pointer() { return impl::get(); }
	const Frame* get_pointer() const { return impl::get(); }
	size_t get_size_bytes() const { return size_bytes; }
	size_t get_payload_count() const { return (size_bytes - sizeof(Frame)) / sizeof(payload_type); }

	using impl::operator bool;
	using impl::operator*;
	using impl::operator->;

  private:
	size_t size_bytes = 0;

	static Frame* make_frame(const size_t size_bytes) {
		assert(size_bytes >= sizeof(Frame));
		assert((size_bytes - sizeof(Frame)) % sizeof(payload_type) == 0);
		const auto mem = operator new(size_bytes);
		try {
			new(mem) Frame;
		} catch(...) {
			operator delete(mem);
			throw;
		}
		return static_cast<Frame*>(mem);
	}
};

} // namespace celerity::detail
