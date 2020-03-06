#pragma once

#include <array>

#include <CL/sycl.hpp>

namespace celerity {
namespace access {
	namespace detail {

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
				using namespace cl::sycl::access;
				// Unfortunately as of ComputeCpp 1.0.1 we have to compile with MSVC toolset 140 (2015) on Windows,
				// which doesn't support multiple return statements in constexpr functions (C++14).
				// TODO: Get rid of this once ComputeCpp supports toolset 141
				return m == mode::atomic
				           ? "atomic"
				           : (m == mode::discard_read_write
				                   ? "discard_read_write"
				                   : (m == mode::discard_write
				                           ? "discard_write"
				                           : (m == mode::read ? "read" : (m == mode::read_write ? "read_write" : (m == mode::write ? "write" : nullptr)))));
			}
		};

	} // namespace detail
} // namespace access
} // namespace celerity
