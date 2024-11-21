#include <celerity.h>

#include <catch2/matchers/catch_matchers_string.hpp>

#include "live_executor.h"
#include "named_threads.h"
#include "test_utils.h"
#include "thread_queue.h"

using namespace celerity;
using namespace celerity::detail::named_threads;

namespace celerity::detail {
struct executor_testspy {
	static std::thread& get_thread(live_executor& exec) { return exec.m_thread; }
};
} // namespace celerity::detail

TEST_CASE("semantic thread type enum entries can be constructed and turned into strings", "[named_threads]") {
	using namespace detail::thread_pinning;
	CHECK(thread_type_to_string(thread_type::application) == "cy-application");
	CHECK(thread_type_to_string(thread_type::scheduler) == "cy-scheduler");
	CHECK(thread_type_to_string(thread_type::executor) == "cy-executor");
	CHECK(thread_type_to_string(thread_type::alloc) == "cy-alloc");
	CHECK(thread_type_to_string(thread_type::first_device_submitter) == "cy-dev-sub-0");
	CHECK(thread_type_to_string(task_type_device_submitter(13)) == "cy-dev-sub-13");
	CHECK(thread_type_to_string(thread_type::first_host_queue) == "cy-host-0");
	CHECK(thread_type_to_string(task_type_host_queue(42)) == "cy-host-42");
	CHECK(thread_type_to_string(thread_type::first_test) == "cy-test-0");
	CHECK(thread_type_to_string(task_type_test(3)) == "cy-test-3");
	CHECK(thread_type_to_string(static_cast<thread_type>(3133337)) == "unknown(3133337)"); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}

#if CELERITY_DETAIL_HAS_NAMED_THREADS

namespace celerity::detail::named_threads {
// These functions have a per-platform implementation in the platform-specific files
// They only work if CELERITY_DETAIL_HAS_NAMED_THREADS is defined
std::string get_thread_name(const std::thread::native_handle_type thread_handle);
std::string get_current_thread_name();
} // namespace celerity::detail::named_threads

TEST_CASE_METHOD(test_utils::runtime_fixture, "thread names are set", "[named_threads]") {
	queue q;

	auto& rt = detail::runtime::get_instance();
	auto& schdlr = detail::runtime_testspy::get_schdlr(rt);
	auto& exec = *detail::utils::as<detail::live_executor>(&detail::runtime_testspy::get_exec(rt));

	q.submit([](handler& cgh) {
		cgh.host_task(experimental::collective, [&](experimental::collective_partition) {
			const auto base_name = std::string("cy-host-");
			const auto worker_thread_name = get_current_thread_name();
			CHECK_THAT(worker_thread_name, Catch::Matchers::StartsWith(base_name));
		});
	});
	q.wait(); // make sure that the runtime threads are actually running for the subsequent checks

	const auto application_thread_name = get_current_thread_name();
	CHECK(application_thread_name == thread_type_to_string(thread_type::application));

	const auto scheduler_thread_name = detail::scheduler_testspy::inspect_thread(schdlr, [](const auto&) { return get_current_thread_name(); });
	CHECK(scheduler_thread_name == thread_type_to_string(thread_type::scheduler));

	const auto executor_thread_name = get_thread_name(detail::executor_testspy::get_thread(exec).native_handle());
	CHECK(executor_thread_name == thread_type_to_string(thread_type::executor));
}

TEST_CASE("thread_queue sets its thread name", "[named_threads][thread_queue]") {
	detail::thread_queue tq(thread_type::first_test);
	test_utils::await(tq.submit([] { CHECK(get_current_thread_name() == thread_type_to_string(thread_type::first_test)); }));
}

#endif
