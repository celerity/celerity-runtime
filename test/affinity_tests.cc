#include <catch2/catch_test_macros.hpp>

#include <celerity.h>
#include <sycl/sycl.hpp>

#include "affinity.h"
#include "test_utils.h"

#include <libenvpp/env.hpp>

#include <unordered_set>

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

TEST_CASE("a warning is emitted if insufficient cores are available", "[affinity]") {
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

TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime system claims to pin its threads as desired", "[affinity][runtime]") {
	auto test = [](const std::vector<uint32_t>& core_ids) {
		if(!have_cores({core_ids.cbegin(), core_ids.cend()})) {
			SKIP("Skipping test because not all needed cores are available");
			return;
		}
		REQUIRE(core_ids.size() == 5);

		{
			// force exactly 2 devices
			sycl::device dev;
			celerity::runtime::init(nullptr, nullptr, {dev, dev});
			celerity::runtime::shutdown();
		}

		constexpr auto msg_template = "Affinity: pinned thread of type '{}' to core {}";
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "user", core_ids.at(0))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "scheduler", core_ids.at(1))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "executor", core_ids.at(2))));
		// SimSYCL has no backend worker support
#if !CELERITY_SYCL_IS_SIMSYCL
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "backend_worker_0", core_ids.at(3))));
		CHECK(test_utils::log_contains_substring(detail::log_level::debug, fmt::format(msg_template, "backend_worker_1", core_ids.at(4))));
#endif
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

TEST_CASE_METHOD(test_utils::runtime_fixture, "when pinning disabled: rt warns on insufficient threads and does not pin", "[affinity][runtime]") {
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "false");
	raii_affinity_masking mask({0});

	celerity::queue q;

	CHECK(test_utils::log_contains_substring(
	    detail::log_level::warn, fmt::format("only {} logical cores are available to this process. It is recommended to assign at least", 1)));
	CHECK_FALSE(test_utils::log_contains_substring(detail::log_level::debug, "Affinity"));
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "the user thread is actually pinned when pinning is enabled", "[affinity][runtime]") {
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "auto");

	const core_set process_mask = {3, 4, 5, 6, 7};
	if(!have_cores(process_mask)) {
		SKIP("Skipping test because not all needed cores are available");
		return;
	}
	raii_affinity_masking mask(process_mask);

	{
		// just 1 device
		celerity::runtime::init(nullptr, nullptr, {sycl::device{}});
		CHECK(get_current_cores() == core_set{3});
		celerity::runtime::shutdown();
	}

	// check that the user thread was unpinned on shutdown
	CHECK(get_current_cores() == process_mask);

	CHECK(test_utils::log_contains_substring(detail::log_level::debug, "Affinity: pinned thread of type 'user' to core 3"));
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "when pinning is disabled, no threads are pinned", "[affinity][runtime]") {
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "false");
	const auto initial_core_set = get_current_cores();

	{
		// just 1 device
		celerity::runtime::init(nullptr, nullptr, {sycl::device{}});
		CHECK(get_current_cores() == initial_core_set);
		celerity::runtime::shutdown();
	}

	CHECK(get_current_cores() == initial_core_set);
	CHECK_FALSE(test_utils::log_contains_substring(detail::log_level::debug, "Affinity"));
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "rt warns when the user thread changes pinning unexpectedly", "[affinity][runtime]") {
	test_utils::allow_max_log_level(detail::log_level::warn);
	env::scoped_test_environment ste("CELERITY_THREAD_PINNING", "auto");

	const core_set process_mask = {3, 4, 5, 6, 7};
	if(!have_cores(process_mask)) {
		SKIP("Skipping test because not all needed cores are available");
		return;
	}
	raii_affinity_masking mask(process_mask, false); // no cleanup check, since we aren't clean on purpose in this test

	{
		// just 1 device
		celerity::runtime::init(nullptr, nullptr, {sycl::device{}});
		CHECK(get_current_cores() == core_set{3});

		// now change the affinity of the user thread
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(4, &cpuset);
		REQUIRE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0);
		celerity::runtime::shutdown();
	}

	CHECK(test_utils::log_contains_substring(
	    detail::log_level::warn, fmt::format("Thread affinity of thread {} was changed unexpectedly, skipping restoration.", pthread_self())));
}
