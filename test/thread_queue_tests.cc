#include "named_threads.h"
#include "test_utils.h"
#include "thread_queue.h"

#include <catch2/catch_test_macros.hpp>

using namespace celerity;
using namespace celerity::detail;

namespace celerity::detail {

struct thread_queue_testspy {
	static std::thread& get_thread(thread_queue& tq) { return tq.m_impl->thread; }
};

} // namespace celerity::detail

#if CELERITY_DETAIL_HAS_NAMED_THREADS

TEST_CASE("thread_queue sets its thread name", "[thread_queue]") {
	thread_queue tq("funny-name");
	test_utils::await(tq.submit([] {})); // wait for thread to enter loop
	auto& thread = thread_queue_testspy::get_thread(tq);
	CHECK(get_thread_name(thread.native_handle()) == "funny-name");
}

#endif

TEST_CASE("thread_queue forwards job results", "[thread_queue]") {
	thread_queue tq("cy-test");

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
	thread_queue tq("cy-test", true /* enable profiling */);

	auto evt1 = tq.submit([] {});
	auto evt2 = tq.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(99)); });

	test_utils::await(evt1);
	test_utils::await(evt2);

	CHECK(evt1.get_native_execution_time().has_value());
	CHECK(evt2.get_native_execution_time().value() >= std::chrono::milliseconds(99));
}
