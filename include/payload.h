#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "host_allocator.h"

namespace celerity::detail {

/*
 * Owning smart pointer for arbitrary structures with a type-erased deleter.
 */
class unique_payload_ptr : private std::unique_ptr<void, std::function<void(void*)>> {
  private:
	friend class shared_payload_ptr;

	using impl = std::unique_ptr<void, std::function<void(void*)>>;

  public:
	using typename impl::deleter_type;

	unique_payload_ptr() noexcept = default;
	unique_payload_ptr(void* const ptr, deleter_type&& deleter) : impl{ptr, std::move(deleter)} {}

	void* get_pointer() { return impl::get(); }
	const void* get_pointer() const { return impl::get(); }

	using impl::operator bool;
};

/*
 * Owning smart pointer for arbitrary structures with a type-erased deleter.
 */
class shared_payload_ptr : private std::shared_ptr<const void> {
  private:
	using impl = std::shared_ptr<const void>;

  public:
	shared_payload_ptr() noexcept = default;
	shared_payload_ptr(unique_payload_ptr&& owning) : impl(static_cast<unique_payload_ptr::impl&&>(owning)) {}
	shared_payload_ptr(const shared_payload_ptr& base, const size_t offset_bytes)
	    : impl(static_cast<const impl&>(base), static_cast<const std::byte*>(base.get_pointer()) + offset_bytes) {}

	const void* get_pointer() const { return impl::get(); }

	using impl::operator bool;
};

template <typename T>
unique_payload_ptr make_uninitialized_payload(const size_t count) {
	// allocate deleter (aka std::function) first so construction unique_payload_ptr is noexcept
	unique_payload_ptr::deleter_type deleter{[size_bytes = count * sizeof(T)](void* const p) { host_allocator::get_instance().free(p, size_bytes); }};
	const auto payload = host_allocator::get_instance().allocate(count * sizeof(T));
	return unique_payload_ptr{payload, std::move(deleter)};
}

} // namespace celerity::detail
