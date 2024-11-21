#include "named_threads.h"
#include "test_utils.h"
#include "thread_queue.h"

#include <catch2/catch_test_macros.hpp>

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("thread_queue forwards job results", "[thread_queue]") {
	thread_queue tq(named_threads::thread_type::first_test);

	auto evt1 = tq.submit([] {});
	auto evt2 = tq.submit([] { return nullptr; });
	auto evt3 = tq.submit([] { return reinterpret_cast<void*>(16); });
	auto evt4 = tq.submit([] { return reinterpret_cast<int*>(64); });

	CHECK(test_utils::await(evt1) == nullptr);
	CHECK(evt1.get_native_execution_time() == std::nullopt);

	CHECK(test_utils::await(evt2) == nullptr);
	CHECK(evt2.get_native_execution_time() == std::nullopt);

	CHECK(test_utils::await(evt3) == reinterpret_cast<void*>(16));
	CHECK(evt3.get_native_execution_time() == std::nullopt);

	CHECK(test_utils::await(evt4) == reinterpret_cast<void*>(64));
	CHECK(evt4.get_native_execution_time() == std::nullopt);
}

TEST_CASE("thread_queue reports execution times when profiling is enabled", "[thread_queue]") {
	thread_queue tq(named_threads::thread_type::first_test, true /* enable profiling */);

	auto evt1 = tq.submit([] {});
	auto evt2 = tq.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(99)); });

	test_utils::await(evt1);
	test_utils::await(evt2);

	CHECK(evt1.get_native_execution_time().has_value());
	CHECK(evt2.get_native_execution_time().value() >= std::chrono::milliseconds(99));
}
