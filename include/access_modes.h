#pragma once

#include <array>

#include <sycl/sycl.hpp>

namespace celerity::detail::access {

constexpr std::array<sycl::access::mode, 6> all_modes = {sycl::access::mode::atomic, sycl::access::mode::discard_read_write, sycl::access::mode::discard_write,
    sycl::access::mode::read, sycl::access::mode::read_write, sycl::access::mode::write};

constexpr std::array<sycl::access::mode, 4> consumer_modes = {
    sycl::access::mode::atomic, sycl::access::mode::read, sycl::access::mode::read_write, sycl::access::mode::write};

constexpr std::array<sycl::access::mode, 5> producer_modes = {sycl::access::mode::atomic, sycl::access::mode::discard_read_write,
    sycl::access::mode::discard_write, sycl::access::mode::read_write, sycl::access::mode::write};

struct mode_traits {
	static constexpr bool is_producer(sycl::access::mode m) {
		using namespace sycl::access;
		return m != mode::read;
	}

	static constexpr bool is_consumer(sycl::access::mode m) {
		using namespace sycl::access;
		return m != mode::discard_read_write && m != mode::discard_write;
	}

	static constexpr bool is_pure_consumer(sycl::access::mode m) { return is_consumer(m) && !is_producer(m); }

	static constexpr const char* name(sycl::access::mode m) {
		using sycl::access::mode;
		switch(m) {
		case mode::atomic: return "atomic";
		case mode::discard_read_write: return "discard_read_write";
		case mode::discard_write: return "discard_write";
		case mode::read: return "read";
		case mode::read_write: return "read_write";
		case mode::write: return "write";
		default: return nullptr;
		}
	}
};

} // namespace celerity::detail::access
