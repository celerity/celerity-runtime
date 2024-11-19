#include <celerity.h>

#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <libenvpp/env.hpp>
#include <sycl/sycl.hpp>

#include "affinity.h"
#include "test_utils.h"

using namespace celerity;

namespace {

using core_set = std::unordered_set<uint32_t>;

// RAII utility for setting and restoring the process affinity mask
class raii_affinity_masking {
  public:
	raii_affinity_masking(const core_set& cores, const bool clean_check = true) : m_clean_check(clean_check) {
		REQUIRE(sched_getaffinity(0, sizeof(cpu_set_t), &m_prior_mask) == 0);
		CPU_ZERO(&m_new_mask);
		for(auto core : cores) {
			CPU_SET(core, &m_new_mask);
		}
		REQUIRE(sched_setaffinity(0, sizeof(cpu_set_t), &m_new_mask) == 0);
	}
	~raii_affinity_masking() {
		if(m_clean_check) {
			cpu_set_t mask_after = {};
			REQUIRE(sched_getaffinity(0, sizeof(cpu_set_t), &mask_after) == 0);
			CHECK(CPU_EQUAL(&mask_after, &m_new_mask));
		}
		REQUIRE(sched_setaffinity(0, sizeof(cpu_set_t), &m_prior_mask) == 0);
	}
	raii_affinity_masking(const raii_affinity_masking&) = delete;
	raii_affinity_masking(raii_affinity_masking&&) = delete;
	raii_affinity_masking& operator=(const raii_affinity_masking&) = delete;
	raii_affinity_masking& operator=(raii_affinity_masking&&) = delete;

  private:
	cpu_set_t m_prior_mask = {};
	cpu_set_t m_new_mask = {};
	bool m_clean_check = true;
};

// RAII utility to start the runtime with exactly `n` devices
class raii_test_runtime {
  public:
	raii_test_runtime(int n) {
		// devices are default constructible, and we don't care if we use the same more than once
		std::vector<sycl::device> devices(n);
		celerity::runtime::init(nullptr, nullptr, devices);
	}
	~raii_test_runtime() { celerity::runtime::shutdown(); }
	raii_test_runtime(const raii_test_runtime&) = delete;
	raii_test_runtime(raii_test_runtime&&) = delete;
	raii_test_runtime& operator=(const raii_test_runtime&) = delete;
	raii_test_runtime& operator=(raii_test_runtime&&) = delete;
};

constexpr void skip_unsupported_configurations() {
#ifdef _WIN32
	SKIP("Affinity is not supported on Windows");
#endif
}

core_set get_current_cores() {
	cpu_set_t mask = {};
	REQUIRE(sched_getaffinity(0, sizeof(cpu_set_t), &mask) == 0);
	core_set cores;
	for(uint32_t i = 0; i < CPU_SETSIZE; ++i) {
		if(CPU_ISSET(i, &mask)) { cores.insert(i); }
	}
	return cores;
}

bool have_cores(const core_set& desired_set) {
	auto current_set = get_current_cores();
	return std::ranges::all_of(desired_set, [&](uint32_t core) { return current_set.contains(core); });
}

} // namespace

TEST_CASE("semantic thread type enum entries can be turned into strings", "[affinity]") {
	using namespace detail::thread_pinning;
	auto to_tt([&](uint32_t t) { return static_cast<thread_type>(t); });
	CHECK(thread_type_to_string(thread_type::application) == "application");
	CHECK(thread_type_to_string(thread_type::scheduler) == "scheduler");
	CHECK(thread_type_to_string(thread_type::executor) == "executor");
	CHECK(thread_type_to_string(thread_type::first_backend_worker) == "backend_worker_0");
	CHECK(thread_type_to_string(to_tt(thread_type::first_backend_worker + 1)) == "backend_worker_1");
	CHECK(thread_type_to_string(thread_type::first_host_queue) == "host_queue_0");
	CHECK(thread_type_to_string(to_tt(thread_type::first_host_queue + 1)) == "host_queue_1");
	CHECK(thread_type_to_string(to_tt(3133337)) == "unknown(3133337)");
}

TEST_CASE("thread pinning environment is correctly parsed", "[affinity][config]") {
	test_utils::allow_max_log_level(detail::log_level::warn);
	using namespace std::string_view_literals;

	auto check_cfg = [](const detail::thread_pinning::environment_configuration& cfg, const bool enabled, const uint32_t starting_from_core,
	                     const std::vector<uint32_t>& hardcoded_core_ids) {
		CHECK(cfg.enabled == enabled);
		CHECK(cfg.starting_from_core == starting_from_core);
		CHECK(cfg.hardcoded_core_ids == hardcoded_core_ids);
	};

	SECTION("when unset") {
		// we expect unset to be auto
		const auto cfg = detail::thread_pinning::parse_validate_env(""sv);
		check_cfg(cfg, true, 1, {});
	}

	SECTION("when auto") {
		const auto cfg = detail::thread_pinning::parse_validate_env("auto"sv);
		check_cfg(cfg, true, 1, {});
	}

	SECTION("when from:3") {
		const auto cfg = detail::thread_pinning::parse_validate_env("from:3"sv);
		check_cfg(cfg, true, 3, {});
	}

	SECTION("when 1,2,3") {
		const auto cfg = detail::thread_pinning::parse_validate_env("1,2,3"sv);
		check_cfg(cfg, true, 0, {1, 2, 3});
	}

	SECTION("when bool") {
		const auto cfg = detail::thread_pinning::parse_validate_env("false"sv);
		check_cfg(cfg, false, 1, {});
	}
}

TEST_CASE("thread pinning environment error handling", "[affinity][config]") {
	test_utils::allow_max_log_level(detail::log_level::warn);
	using namespace std::string_view_literals;

	const auto error_string = "Cannot parse CELERITY_THREAD_PINNING setting";

	SECTION("for the 'from' case") { CHECK_THROWS_WITH(detail::thread_pinning::parse_validate_env("from:a"sv), Catch::Matchers::StartsWith(error_string)); }

	SECTION("for the core list case") { CHECK_THROWS_WITH(detail::thread_pinning::parse_validate_env("1,3#4"sv), Catch::Matchers::StartsWith(error_string)); }

	SECTION("for random strings") { CHECK_THROWS_WITH(detail::thread_pinning::parse_validate_env("foo"sv), Catch::Matchers::StartsWith(error_string)); }
}

TEST_CASE("a warning is emitted if insufficient cores are available", "[affinity]") {
	skip_unsupported_configurations();
	test_utils::allow_max_log_level(detail::log_level::warn);
	raii_affinity_masking mask({0, 1, 2});

	SECTION("if pinning enabled") {
		detail::thread_pinning::thread_pinner pinner({.enabled = true, .num_devices = 3, .num_legacy_processes = 1});
		CHECK(test_utils::log_contains_substring(detail::log_level::warn, "Ran out of available cores for thread pinning"));
	}
	SECTION("if pinning disabled") {
		detail::thread_pinning::thread_pinner pinner({.enabled = false, .num_devices = 3, .num_legacy_processes = 1});
		CHECK(test_utils::log_contains_substring(detail::log_level::warn,
		    fmt::format("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {}", 3, 6)));
	}
}

TEST_CASE("do not plan for device submission threads if they are unused", "[affinity]") {
	skip_unsupported_configurations();
	raii_affinity_masking mask({1, 2, 3, 4});
	const detail::thread_pinning::runtime_configuration cfg = {.enabled = true, .num_devices = 10, .use_backend_device_submission_threads = false};
	detail::thread_pinning::thread_pinner pinner(cfg);
	// no additional check, a warning will make the test fail if we do not handle this case correctly
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime warns on manual core list of wrong size", "[affinity][config]") {
	skip_unsupported_configurations();
	test_utils::allow_max_log_level(detail::log_level::warn);
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "1,2");
	{ raii_test_runtime rt(1); }
	CHECK(test_utils::log_contains_substring(detail::log_level::warn, "Hardcoded core ID count (2) does not match the number of threads to be pinned ("));
}

// SimSYCL has no backend submission thread support
#if !CELERITY_SYCL_IS_SIMSYCL
TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime system claims to pin its threads as desired", "[affinity][runtime]") {
	skip_unsupported_configurations();
	auto test = [](const std::vector<uint32_t>& core_ids) {
		if(!have_cores({core_ids.cbegin(), core_ids.cend()})) {
			SKIP("Skipping test because not all needed cores are available");
			return;
		}
		REQUIRE(core_ids.size() == 5);

		{ raii_test_runtime rt(2); }

		constexpr auto msg_template = "Affinity: pinned thread '{}' to core {}";
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "application", core_ids.at(0))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "scheduler", core_ids.at(1))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "executor", core_ids.at(2))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "backend_worker_0", core_ids.at(3))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "backend_worker_1", core_ids.at(4))));
	};

	SECTION("for auto") {
		env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "auto");
		test({1, 2, 3, 4, 5});
	}
	SECTION("for hardcoded") {
		env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "2,3,7,6,6");
		test({2, 3, 7, 6, 6});
	}
	SECTION("for from") {
		env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "from:3");
		test({3, 4, 5, 6, 7});
	}
}
#endif // !CELERITY_SYCL_IS_SIMSYCL

TEST_CASE_METHOD(test_utils::runtime_fixture, "when pinning disabled: rt warns on insufficient threads and does not pin", "[affinity][runtime]") {
	skip_unsupported_configurations();
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "false");
	raii_affinity_masking mask({0});

	celerity::queue q;

	CHECK(test_utils::log_contains_substring(
	    detail::log_level::warn, fmt::format("only {} logical cores are available to this process. It is recommended to assign at least", 1)));
	CHECK_FALSE(test_utils::log_contains_substring(detail::log_level::debug, "Affinity"));
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "the application thread is actually pinned when pinning is enabled", "[affinity][runtime]") {
	skip_unsupported_configurations();
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "auto");

	const core_set process_mask = {3, 4, 5, 6, 7};
	if(!have_cores(process_mask)) {
		SKIP("Skipping test because not all needed cores are available");
		return;
	}
	raii_affinity_masking mask(process_mask);

	{
		raii_test_runtime rt(1);
		CHECK(get_current_cores() == core_set{3});
	}

	// check that the application thread was unpinned on shutdown
	CHECK(get_current_cores() == process_mask);

	CHECK(test_utils::log_contains_substring(detail::log_level::debug, "Affinity: pinned thread 'application' to core 3"));
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "when pinning is disabled, no threads are pinned", "[affinity][runtime]") {
	skip_unsupported_configurations();
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "false");
	const auto initial_core_set = get_current_cores();

	{
		raii_test_runtime rt(1);
		CHECK(get_current_cores() == initial_core_set);
	}

	CHECK(get_current_cores() == initial_core_set);
	CHECK_FALSE(test_utils::log_contains_substring(detail::log_level::debug, "Affinity"));
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "rt warns when the application thread changes pinning unexpectedly", "[affinity][runtime]") {
	skip_unsupported_configurations();
	test_utils::allow_max_log_level(detail::log_level::warn);
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "auto");

	const core_set process_mask = {3, 4, 5, 6, 7};
	if(!have_cores(process_mask)) {
		SKIP("Skipping test because not all needed cores are available");
		return;
	}
	raii_affinity_masking mask(process_mask, false); // no cleanup check, since we aren't clean on purpose in this test

	{
		raii_test_runtime rt(1);
		CHECK(get_current_cores() == core_set{3});

		// now change the affinity of the application thread
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(4, &cpuset);
		REQUIRE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0);
	}

	CHECK(test_utils::log_contains_substring(
	    detail::log_level::warn, fmt::format("Thread affinity of thread {} was changed unexpectedly, skipping restoration.", pthread_self())));
}

TEST_CASE("multiple subsequent non-overlapping pinning ranges are handled correctly", "[affinity]") {
	skip_unsupported_configurations();

	const core_set process_mask = {3, 4, 5, 6, 7};
	if(!have_cores(process_mask)) {
		SKIP("Skipping test because not all needed cores are available");
		return;
	}

	raii_affinity_masking mask(process_mask);

	REQUIRE(get_current_cores() == process_mask);

	{
		detail::thread_pinning::thread_pinner pinner({.enabled = true, .num_devices = 1, .num_legacy_processes = 1});
		detail::thread_pinning::pin_this_thread(detail::thread_pinning::thread_type::application);
		CHECK(get_current_cores() == core_set{3});
	}

	CHECK(get_current_cores() == process_mask);

	{
		detail::thread_pinning::thread_pinner pinner({.enabled = true, .num_devices = 1, .num_legacy_processes = 1, .standard_core_start_id = 4});
		detail::thread_pinning::pin_this_thread(detail::thread_pinning::thread_type::application);
		CHECK(get_current_cores() == core_set{4});
	}

	CHECK(get_current_cores() == process_mask);
}