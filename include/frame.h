#pragma once

#include <cassert>

#include "payload.h"
#include "utils.h"

namespace celerity::detail {

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

	friend class unique_payload_ptr;

	template <typename>
	friend class shared_frame_ptr;

  public:
	using payload_type = typename Frame::payload_type;

	unique_frame_ptr() noexcept = default;

	unique_frame_ptr(from_payload_count_tag, size_t payload_count) : unique_frame_ptr(from_size_bytes, sizeof(Frame) + sizeof(payload_type) * payload_count) {}

	unique_frame_ptr(from_size_bytes_tag, size_t size_bytes) : impl(make_frame(size_bytes)), m_size_bytes(size_bytes) {}

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
	size_t get_payload_count() const { return (m_size_bytes - sizeof(Frame)) / sizeof(payload_type); }

	using impl::operator bool;
	using impl::operator*;
	using impl::operator->;

	unique_payload_ptr into_payload_ptr() && {
		unique_payload_ptr::deleter_type deleter{delete_frame_from_payload}; // allocate deleter (aka std::function) first so `result` construction is noexcept
		const auto frame = this->release();
		const auto payload = reinterpret_cast<typename Frame::payload_type*>(frame + 1); // payload is located at +sizeof(Frame) bytes (+1 Frame object)
		return unique_payload_ptr{payload, std::move(deleter)};
	}

  private:
	size_t m_size_bytes = 0;

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


  private:
	static void delete_frame_from_payload(void* const type_erased_payload) {
		const auto payload = static_cast<typename Frame::payload_type*>(type_erased_payload);
		const auto frame = reinterpret_cast<Frame*>(payload) - 1; // frame header is located at -sizeof(Frame) bytes (-1 Frame object)
		delete frame;
	}
};

template <typename Frame>
class shared_frame_ptr : private std::shared_ptr<Frame> {
  private:
	using impl = std::shared_ptr<Frame>;

  public:
	using payload_type = typename Frame::payload_type;

	shared_frame_ptr() noexcept = default;

	shared_frame_ptr(unique_frame_ptr<Frame>&& unique)
	    : impl(static_cast<typename unique_frame_ptr<Frame>::impl&&>(unique)), m_size_bytes(unique.m_size_bytes) {}

	template <typename T>
	shared_frame_ptr(const std::shared_ptr<T>& alias, Frame* const frame, const size_t frame_size_bytes) : impl(alias, frame), m_size_bytes(frame_size_bytes) {}

	shared_frame_ptr(const shared_frame_ptr&) = default;

	shared_frame_ptr(shared_frame_ptr&& other) noexcept : impl(static_cast<impl&&>(other)) { std::swap(m_size_bytes, other.m_size_bytes); }

	shared_frame_ptr& operator=(const shared_frame_ptr& other) = default;

	shared_frame_ptr& operator=(shared_frame_ptr&& other) noexcept {
		if(this == &other) return *this;                        // gracefully handle self-assignment
		static_cast<impl&>(*this) = static_cast<impl&&>(other); // delegate to base class unique_ptr<Frame>::operator=() to delete previously held frame
		m_size_bytes = other.m_size_bytes;
		other.m_size_bytes = 0;
		return *this;
	}

	Frame* get_pointer() { return impl::get(); }
	const Frame* get_pointer() const { return impl::get(); }
	size_t get_size_bytes() const { return m_size_bytes; }
	size_t get_payload_count() const { return (m_size_bytes - sizeof(Frame)) / sizeof(payload_type); }

	using impl::operator bool;
	using impl::operator*;
	using impl::operator->;

  private:
	size_t m_size_bytes = 0;
};

template <typename Frame>
class frame_vector_layout {
  public:
	void reserve_back(from_size_bytes_tag, const size_t frame_size_bytes) {
		assert(frame_size_bytes >= sizeof(Frame));
		m_frame_count += 1;
		m_aligned_frames_size = utils::ceil(m_aligned_frames_size, alignof(Frame)) + frame_size_bytes;
	}

	void reserve_back(from_payload_count_tag, const size_t payload_count) {
		reserve_back(from_size_bytes, sizeof(Frame) + sizeof(typename Frame::payload_type) * payload_count);
	}

	size_t get_frame_count() const { return m_frame_count; }

	size_t get_size_bytes() const { return utils::ceil(sizeof(size_t) * (1 + m_frame_count), alignof(Frame)) + m_aligned_frames_size; }

  private:
	size_t m_frame_count = 0;
	size_t m_aligned_frames_size = 0;
};

template <typename Frame>
class frame_vector : public std::enable_shared_from_this<frame_vector<Frame>> {
  public:
	class const_iterator;

  private:
	template <typename Iterator /* CRTP */, typename MaybeConstVector, typename MaybeConstFrame>
	class iterator_impl {
	  public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = MaybeConstFrame;
		using reference = value_type&;
		using pointer = value_type*;
		using difference_type = std::ptrdiff_t;

		explicit iterator_impl(MaybeConstVector& vector, const size_t index, MaybeConstFrame* const frame)
		    : m_vector(&vector), m_index(index), m_frame(frame) {}

		Iterator& operator++() {
			const auto num_frames = m_vector->get_frame_count();
			if(m_index + 1 < num_frames) {
				const auto cur_frame_padded_size = utils::ceil(m_vector->get_frame_size_bytes(m_index), alignof(Frame));
				m_frame = reinterpret_cast<MaybeConstFrame*>(reinterpret_cast<uintptr_t>(m_frame) + cur_frame_padded_size);
				m_index += 1;
			} else if(m_index + 1 == num_frames) {
				m_frame = nullptr;
				m_index = num_frames;
			}
			return *static_cast<Iterator*>(this);
		}

		Iterator operator++(const int) {
			const auto prev = *static_cast<Iterator*>(this);
			++*this;
			return prev;
		}

		friend bool operator==(const Iterator lhs, const Iterator rhs) { return lhs.m_vector == rhs.m_vector && lhs.m_index == rhs.m_index; }

		friend bool operator!=(const Iterator lhs, const Iterator rhs) { return !(lhs == rhs); }

		reference operator*() const {
			assert(m_frame != nullptr);
			return *m_frame;
		}

		pointer operator->() const {
			assert(m_frame != nullptr);
			return m_frame;
		}

		size_t get_size_bytes() const { return m_vector->get_frame_size_bytes(m_index); }

		size_t get_payload_count() const { return m_vector->get_frame_payload_count(m_index); }

		shared_frame_ptr<value_type> get_shared_from_this() const {
			assert(m_frame != nullptr);
			return shared_frame_ptr<MaybeConstFrame>(m_vector->shared_from_this(), m_frame, m_vector->get_frame_size_bytes(m_index));
		}

	  private:
		friend class const_iterator;

		MaybeConstVector* m_vector;
		size_t m_index;
		MaybeConstFrame* m_frame;
	};

  public:
	using value_type = Frame;
	using payload_type = typename Frame::payload_type;

	class iterator final : public iterator_impl<iterator, frame_vector, Frame> {
	  public:
		using iterator_impl<iterator, frame_vector, Frame>::iterator_impl;
	};

	class const_iterator final : public iterator_impl<const_iterator, const frame_vector, const Frame> {
	  public:
		using iterator_impl<const_iterator, const frame_vector, const Frame>::iterator_impl;

		const_iterator(iterator non_const) : const_iterator(*non_const.m_vector, non_const.m_index, non_const.m_frame) {}
	};

	frame_vector() noexcept = default;

	frame_vector(from_size_bytes_tag, size_t size_bytes) : m_alloc(operator new(size_bytes)), m_size_bytes(size_bytes) {}

	frame_vector(frame_vector&& other) noexcept : m_alloc(other.m_alloc), m_size_bytes(other.m_size_bytes) {
		other.m_alloc = nullptr;
		other.m_size_bytes = 0;
	}

	~frame_vector() { reset(); }

	frame_vector& operator=(frame_vector&& other) noexcept {
		if(this == &other) return *this;
		reset();
		std::swap(m_alloc, other.m_alloc);
		std::swap(m_size_bytes, other.m_size_bytes);
		return *this;
	}

	iterator begin() { return iterator{*this, 0, get_first_frame()}; }
	iterator end() { return iterator{*this, get_frame_count(), nullptr}; }
	const_iterator cbegin() const { return const_iterator{*this, 0, get_first_frame()}; }
	const_iterator cend() const { return const_iterator{*this, get_frame_count(), nullptr}; }
	const_iterator begin() const { return cbegin(); }
	const_iterator end() const { return cend(); }

	void* get_pointer() { return m_alloc; }
	const void* get_pointer() const { return m_alloc; }

	size_t get_size_bytes() const { return m_size_bytes; }
	size_t get_frame_count() const { return m_alloc ? get_header()[0] : 0; }

	size_t get_frame_size_bytes(const size_t index) const {
		assert(index < get_frame_count());
		return get_header()[1 + index];
	}

	size_t get_frame_payload_count(const size_t index) const { return (get_frame_size_bytes(index) - sizeof(Frame)) / sizeof(payload_type); }

	explicit operator bool() const { return m_alloc != nullptr; }

  private:
	template <typename F>
	friend class frame_vector_builder;

	void* m_alloc = nullptr;
	size_t m_size_bytes = 0;

	size_t* get_header() const {
		assert(m_alloc != nullptr);
		return static_cast<size_t*>(m_alloc);
	}

	void set_frame_count(const size_t count) { get_header()[0] = count; }

	void set_frame_size_bytes(const size_t index, const size_t size_bytes) { get_header()[1 + index] = size_bytes; }

	Frame* get_first_frame() const {
		if(m_alloc != nullptr) {
			const auto offset = utils::ceil((1 + get_frame_count()) * sizeof(size_t), alignof(Frame));
			return reinterpret_cast<Frame*>(static_cast<std::byte*>(m_alloc) + offset);
		} else {
			return nullptr;
		}
	}

	void reset() {
		// TODO this cannot call Frame destructors because frame_vector_builder will leave the structure partially initialized.
		// At the same time, unique/shared_frame_ptrs will call destructors. This should be made consistent.
		operator delete(m_alloc);
		m_alloc = nullptr;
		m_size_bytes = 0;
	}
};

template <typename Frame>
class frame_vector_builder {
  public:
	explicit frame_vector_builder(frame_vector_layout<Frame> layout) : m_vector(from_size_bytes, layout.get_size_bytes()) {
		m_vector.get_header()[0] = layout.get_frame_count();
		m_next_frame = m_vector.get_first_frame();
	}

	Frame& emplace_back(from_size_bytes_tag, const size_t frame_size_bytes) {
		assert(frame_size_bytes >= sizeof(Frame));
		assert(m_vector);
		assert(m_next_index < m_vector.get_frame_count());
		assert(reinterpret_cast<std::byte*>(m_next_frame) - static_cast<std::byte*>(m_vector.get_pointer()) + frame_size_bytes <= m_vector.get_size_bytes());

		m_vector.set_frame_size_bytes(m_next_index, frame_size_bytes);
		new(m_next_frame) Frame;
		const auto frame = m_next_frame;

		const auto frame_padded_size = utils::ceil(frame_size_bytes, alignof(Frame));
		m_next_frame = reinterpret_cast<Frame*>(reinterpret_cast<uintptr_t>(frame) + frame_padded_size);
		m_next_index += 1;

		return *frame;
	}

	Frame& emplace_back(from_payload_count_tag, const size_t payload_count) {
		return emplace_back(from_size_bytes, sizeof(Frame) + sizeof(typename Frame::payload_type) * payload_count);
	}

	frame_vector<Frame> into_vector() && {
		assert(m_next_index == m_vector.get_frame_count());
		return std::move(m_vector);
	}

  private:
	frame_vector<Frame> m_vector;
	size_t m_next_index = 0;
	Frame* m_next_frame;
};

} // namespace celerity::detail
