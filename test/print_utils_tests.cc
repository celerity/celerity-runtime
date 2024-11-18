#include <catch2/catch_test_macros.hpp>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("sycl_backend_type is correctly stringified", "[print_utils]") {
	CHECK(fmt::format("{}", sycl_backend_type::generic) == "generic");
	CHECK(fmt::format("{}", sycl_backend_type::cuda) == "CUDA");
}

TEST_CASE("durations are correctly stringified as_sub_second()", "[print_utils]") {
	CHECK(fmt::format("{:.2f}", as_sub_second(std::chrono::nanoseconds(123'456))) == "123.46 µs");
	CHECK(fmt::format("{:.2f}", as_sub_second(std::chrono::milliseconds(123'456))) == "123.46 s");
	CHECK(fmt::format("{:.0f}", as_sub_second(std::chrono::milliseconds(100))) == "100 ms");

	// Padding works as expected
	CHECK(fmt::format("{:>10.1f}", as_sub_second(std::chrono::microseconds(100'500))) == "  100.5 ms");
	CHECK(fmt::format("{:<10.1f}", as_sub_second(std::chrono::microseconds(100'500))) == "100.5 ms  ");

	// Custom unit table can be used
	CHECK(fmt::format("{:.2f}", as_sub_second<default_time_units>(std::chrono::milliseconds(123'456))) == "123.46 s");
	CHECK(fmt::format("{:.2f}", as_sub_second<right_padded_time_units>(std::chrono::milliseconds(123'456))) == "123.46 s ");
}

TEST_CASE("sizes are correctly stringified as_decimal_size()", "[print_utils]") {
	CHECK(fmt::format("{:.2f}", as_decimal_size(123'456)) == "123.46 kB");
	CHECK(fmt::format("{:.0f}", as_decimal_size(1e6)) == "1 MB");
	CHECK(fmt::format("{:.0f}", as_decimal_size(1e15)) == "1000 TB");

	// Padding works as expected
	CHECK(fmt::format("{:>10.1f}", as_decimal_size(100'500)) == "  100.5 kB");
	CHECK(fmt::format("{:<10.1f}", as_decimal_size(100'500)) == "100.5 kB  ");

	// Custom unit table can be used
	CHECK(fmt::format("{:.0f}", as_decimal_size<default_byte_size_units>(123)) == "123 bytes");
	CHECK(fmt::format("{:.0f}", as_decimal_size<single_digit_right_padded_byte_size_units>(123)) == "123 B ");
}

TEST_CASE("throughputs are correctly stringified as_decimal_throughput()", "[print_utils]") {
	CHECK(fmt::format("{:.2f}", as_decimal_throughput(123'456, std::chrono::seconds(1))) == "123.46 kB/s");
	CHECK(fmt::format("{:.0f}", as_decimal_throughput(1000, std::chrono::milliseconds(1))) == "1 MB/s");
}
