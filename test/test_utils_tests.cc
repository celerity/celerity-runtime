#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

TEST_CASE("tests that log any message in excess of level::info fail by default", "[test_utils][log][!shouldfail]") { CELERITY_WARN("spooky message!"); }

// This is a non-default (i.e. manual) test, because it aborts when passing
TEST_CASE("tests that log messages in excess of level::info from a secondary thread abort", "[test_utils][log][!shouldfail][.]") {
	std::thread([] { CELERITY_WARN("abort() in 3... 2... 1..."); }).join();
}

TEST_CASE("test_utils::set_max_expected_log_level() allows tests with warning / error messages to pass", "[test_utils][log]") {
	test_utils::allow_max_log_level(spdlog::level::err);
	CELERITY_WARN("spooky message!");
	CELERITY_ERROR("spooky message!");
}
