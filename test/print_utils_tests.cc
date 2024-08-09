#include <catch2/catch_test_macros.hpp>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("command_type is correctly stringified", "[print_utils]") {
	CHECK(fmt::format("{}", command_type::epoch) == "epoch");
	CHECK(fmt::format("{}", command_type::horizon) == "horizon");
	CHECK(fmt::format("{}", command_type::execution) == "execution");
	CHECK(fmt::format("{}", command_type::push) == "push");
	CHECK(fmt::format("{}", command_type::await_push) == "await push");
	CHECK(fmt::format("{}", command_type::reduction) == "reduction");
	CHECK(fmt::format("{}", command_type::fence) == "fence");
}

TEST_CASE("sycl_backend_type is correctly stringified", "[print_utils]") {
	CHECK(fmt::format("{}", sycl_backend_type::generic) == "generic");
	CHECK(fmt::format("{}", sycl_backend_type::cuda) == "CUDA");
}

TEST_CASE("durations are correctly stringified as_sub_second()", "[print_utils]") {
	CHECK(fmt::format("{:.2f}", as_sub_second(std::chrono::nanoseconds(123'456))) == "123.46 µs");
	CHECK(fmt::format("{:.2f}", as_sub_second(std::chrono::milliseconds(123'456))) == "123.46 s");
	CHECK(fmt::format("{:.0f}", as_sub_second(std::chrono::milliseconds(100))) == "100 ms");
}

TEST_CASE("sizes are correctly stringified as_decimal_size()", "[print_utils]") {
	CHECK(fmt::format("{:.2f}", as_decimal_size(123'456)) == "123.46 kB");
	CHECK(fmt::format("{:.0f}", as_decimal_size(1e6)) == "1 MB");
	CHECK(fmt::format("{:.0f}", as_decimal_size(1e15)) == "1000 TB");
}

TEST_CASE("throughputs are correctly stringified as_decimal_throughput()", "[print_utils]") {
	CHECK(fmt::format("{:.2f}", as_decimal_throughput(123'456, std::chrono::seconds(1))) == "123.46 kB/s");
	CHECK(fmt::format("{:.0f}", as_decimal_throughput(1000, std::chrono::milliseconds(1))) == "1 MB/s");
}
