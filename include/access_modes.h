#pragma once

#include <array>

#include <CL/sycl.hpp>

namespace celerity::detail::access {

constexpr std::array<cl::sycl::access::mode, 6> all_modes = {cl::sycl::access::mode::atomic, cl::sycl::access::mode::discard_read_write,
    cl::sycl::access::mode::discard_write, cl::sycl::access::mode::read, cl::sycl::access::mode::read_write, cl::sycl::access::mode::write};

constexpr std::array<cl::sycl::access::mode, 4> consumer_modes = {
    cl::sycl::access::mode::atomic, cl::sycl::access::mode::read, cl::sycl::access::mode::read_write, cl::sycl::access::mode::write};

constexpr std::array<cl::sycl::access::mode, 5> producer_modes = {cl::sycl::access::mode::atomic, cl::sycl::access::mode::discard_read_write,
    cl::sycl::access::mode::discard_write, cl::sycl::access::mode::read_write, cl::sycl::access::mode::write};

struct mode_traits {
	static constexpr bool is_producer(cl::sycl::access::mode m) {
		using namespace cl::sycl::access;
		return m != mode::read;
	}

	static constexpr bool is_consumer(cl::sycl::access::mode m) {
		using namespace cl::sycl::access;
		return m != mode::discard_read_write && m != mode::discard_write;
	}

	static constexpr bool is_pure_consumer(cl::sycl::access::mode m) { return is_consumer(m) && !is_producer(m); }

	static constexpr const char* name(cl::sycl::access::mode m) {
		using cl::sycl::access::mode;
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
