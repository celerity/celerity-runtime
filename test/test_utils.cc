#include "test_utils.h"
#include "catch2/internal/catch_context.hpp"

#include <regex>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <spdlog/details/os.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/sinks/sink.h>

namespace celerity::test_utils_detail {

struct log_message {
	detail::log_level level = spdlog::level::trace;
	std::string text;
};

/// Thread-safe global capture for logs encountered in Catch2 test cases.
class test_log_capture {
  public:
	test_log_capture(std::thread::id catch2_safe_thread) : m_catch2_safe_thread(catch2_safe_thread) {
		// Re-create the color codes used by spdlog for printing the captured log. If spdlog does not detect a terminal, m_ccodes will contain empty strings.
		if(spdlog::details::os::in_terminal(stdout) && spdlog::details::os::is_color_terminal()) {
			const spdlog::sinks::ansicolor_stdout_sink_st dummy_color_sink;
			// See spdlog::sinks::ansicolor_sink::ansicolor_sink()
			m_ccodes[spdlog::level::trace] = fmt::to_string(dummy_color_sink.white);
			m_ccodes[spdlog::level::debug] = fmt::to_string(dummy_color_sink.cyan);
			m_ccodes[spdlog::level::info] = fmt::to_string(dummy_color_sink.green);
			m_ccodes[spdlog::level::warn] = fmt::to_string(dummy_color_sink.yellow_bold);
			m_ccodes[spdlog::level::err] = fmt::to_string(dummy_color_sink.red_bold);
			m_ccodes[spdlog::level::critical] = fmt::to_string(dummy_color_sink.bold_on_red);
			m_ccodes[spdlog::level::off] = fmt::to_string(dummy_color_sink.reset);
		}
	}

	void clear() {
		std::lock_guard lock(m_mutex);
		m_messages.clear();
		m_max_expected_level = spdlog::level::info;
		m_max_level_state = max_level_state::not_exceeded;
		m_expected_log_messages.clear();
	}

	void allow_max_log_level(const spdlog::level::level_enum level) {
		std::lock_guard lock(m_mutex);
		m_max_expected_level = level;
	}

	void allow_higher_level_log_message(detail::log_level level, const std::string& text_regex) {
		std::lock_guard lock(m_mutex);
		m_expected_log_messages.push_back({level, std::regex(text_regex)});
	}

	void log(const spdlog::details::log_msg& msg) {
		std::lock_guard lock(m_mutex);
		m_messages.push_back({msg.level, fmt::to_string(msg.payload)});

		if(msg.level > m_max_expected_level && m_max_level_state != max_level_state::exceeded_and_failed) {
			if(!std::any_of(m_expected_log_messages.begin(), m_expected_log_messages.end(),
			       [&](const message_template& tmpl) { return tmpl.level == msg.level && std::regex_match(m_messages.back().text, tmpl.text_regex); })) {
				m_max_level_state = max_level_state::exceeded_and_fail_pending;
			}
		}

		// If the log level exceeds the allowed maximum, either FAIL_CHECK() if we're in Catch2's thread (its macros are not thread safe), or remember that we
		// need to do so either when we get the next call to `log` or after the test run ends.
		if(m_max_level_state == max_level_state::exceeded_and_fail_pending && std::this_thread::get_id() == m_catch2_safe_thread) {
			// If this reports an earlier error it would be possible to create a weird error message by changing m_max_expected_level in-between, but there is a
			// limit to how robust debugging code needs to be.
			FAIL_CHECK(format_error_message_for_exceeding_max_level());
			m_max_level_state = max_level_state::exceeded_and_failed;
		}
	}

	bool print_log_if_nonempty() const {
		std::lock_guard lock(m_mutex);
		return print_log_if_nonempty_internal();
	}

	bool print_if_max_level_was_exceeded_but_not_reported(const Catch::TestCaseInfo& info) const {
		std::lock_guard lock(m_mutex);
		if(m_max_level_state != max_level_state::exceeded_and_fail_pending) return false;

		// we're outside of Catch2's "test guard" where we can FAIL() or FAIL_CHECK() or where an abort() would be verbosely reported, so we just imitate that
		fmt::print("-------------------------------------------------------------------------------\n");
		fmt::print("{}\n", info.name);
		fmt::print("-------------------------------------------------------------------------------\n");
		fmt::print("{}:{}\n", info.lineInfo.file, info.lineInfo.line);
		fmt::print("...............................................................................\n");
		fmt::print("\n{}Irrecoverable error{} from secondary thread:\n  {}\n\n", m_ccodes[spdlog::level::err], m_ccodes[spdlog::level::off],
		    format_error_message_for_exceeding_max_level());
		print_log_if_nonempty_internal();
		return true;
	}

	template <typename Predicate>
	bool log_contains_if(const Predicate& p) {
		std::lock_guard lock(m_mutex);
		return std::find_if(m_messages.begin(), m_messages.end(), p) != m_messages.end();
	}

  private:
	enum class max_level_state {
		not_exceeded,              ///< all good.
		exceeded_and_failed,       ///< the log level was exceeded, and FAIL_CHECK() was called because we were in m_catch2_safe_thread at the time.
		exceeded_and_fail_pending, ///< the log level was exceeded, but not reported yet.
	};

	struct message_template {
		detail::log_level level;
		std::regex text_regex;
	};

	// immutable after construction
	std::array<std::string, spdlog::level::n_levels> m_ccodes;
	std::thread::id m_catch2_safe_thread;

	mutable std::mutex m_mutex;
	std::vector<log_message> m_messages;
	spdlog::level::level_enum m_max_expected_level = spdlog::level::info;
	std::vector<message_template> m_expected_log_messages;
	max_level_state m_max_level_state = max_level_state::not_exceeded;

	std::string format_error_message_for_exceeding_max_level() const {
		return fmt::format("Observed a log message exceeding the expected maximum of \"{}\". If this is correct, increase the expected log level through "
		                   "test_utils::expect_max_log_level().",
		    spdlog::level::to_string_view(m_max_expected_level));
	}

	bool print_log_if_nonempty_internal() const {
		if(m_messages.empty()) return false;

		fmt::print("captured log:\n");
		for(const auto& msg : m_messages) {
			// ccodes are empty strings if we are not writing to a color terminal
			fmt::print("  [{}{}{}] {}\n", m_ccodes[msg.level], spdlog::level::to_string_view(msg.level), m_ccodes[spdlog::level::off], msg.text);
		}
		fmt::print("\n");
		return true;
	}
};

/// A spdlog sink forwarding all log messages to the test_log_capture.
class test_capture_sink final : public spdlog::sinks::sink {
  public:
	explicit test_capture_sink(test_log_capture& capture) : m_capture(&capture) {}

	void log(const spdlog::details::log_msg& msg) override { m_capture->log(msg); }
	void flush() override {}
	void set_pattern(const std::string& pattern) override {}
	void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) override {}

  private:
	test_log_capture* m_capture;
};

std::unique_ptr<test_log_capture> g_test_log_capture;

struct global_setup_and_teardown : Catch::EventListenerBase {
	using EventListenerBase::EventListenerBase;

	void testRunStarting(const Catch::TestRunInfo& /* info */) override {
		celerity::detail::closure_hydrator::make_available();

		g_test_log_capture = std::make_unique<test_log_capture>(std::this_thread::get_id());

		auto capture_sink = std::make_shared<test_capture_sink>(*g_test_log_capture);
		auto capture_logger = std::make_shared<spdlog::logger>("test", std::move(capture_sink));
		capture_logger->set_level(spdlog::level::trace);
		capture_logger->flush_on(spdlog::level::trace);
		spdlog::set_default_logger(std::move(capture_logger));
	}

	void testCasePartialStarting(const Catch::TestCaseInfo& info, uint64_t /* partNumber */) override {
		// Logs are always fully captured, independent of previous log_level settings or environment. A test can still influence the log level while it's
		// running (but very likely should not!)
		spdlog::set_level(spdlog::level::trace);
	}

	void testCasePartialEnded(const Catch::TestCaseStats& stats, uint64_t /* partNumber */) override {
		// Reset REQUIRE_LOOP after each test case, section or generator value.
		celerity::test_utils::require_loop_assertion_registry::get_instance().reset();

		if(g_test_log_capture->print_if_max_level_was_exceeded_but_not_reported(*stats.testInfo)) { abort(); }
		if(stats.totals.testCases.failed + stats.totals.testCases.failedButOk != 0) { g_test_log_capture->print_log_if_nonempty(); }
		g_test_log_capture->clear();
	}

	void testRunEnded(const Catch::TestRunStats& stats) override {
		g_test_log_capture->print_log_if_nonempty(); // if so, this is likely due to a fatal signal
		g_test_log_capture->clear();
	}

	void benchmarkPreparing(Catch::StringRef name) override {
		// Do not include log-capturing in benchmark times (log level would remain at `trace` otherwise).
		log_level_before_benchmark = spdlog::get_level();
		spdlog::set_level(spdlog::level::off);
	}

	void benchmarkEnded(const Catch::BenchmarkStats<>& /* benchmarkStats */) override { spdlog::set_level(log_level_before_benchmark); }
	void benchmarkFailed(Catch::StringRef error) override { spdlog::set_level(log_level_before_benchmark); }

	spdlog::level::level_enum log_level_before_benchmark = spdlog::level::trace;
};

} // namespace celerity::test_utils_detail

namespace celerity::test_utils {

void allow_max_log_level(const spdlog::level::level_enum level) { test_utils_detail::g_test_log_capture->allow_max_log_level(level); }

void allow_higher_level_log_messages(const detail::log_level level, const std::string& text_regex) {
	test_utils_detail::g_test_log_capture->allow_higher_level_log_message(level, text_regex);
}

bool log_contains_exact(const detail::log_level level, const std::string& text) {
	return test_utils_detail::g_test_log_capture->log_contains_if(
	    [&](const test_utils_detail::log_message& msg) { return msg.level == level && msg.text == text; });
}

bool log_contains_substring(const detail::log_level level, const std::string& substring) {
	return test_utils_detail::g_test_log_capture->log_contains_if(
	    [&](const test_utils_detail::log_message& msg) { return msg.level == level && msg.text.find(substring) != std::string::npos; });
}

} // namespace celerity::test_utils

namespace celerity::test_utils_detail {

// These error and warning messages will appear depending on the system the (runtime) tests are executed on, so we must not fail tests because of them.

const char* const expected_runtime_init_warnings_regex = "Celerity has detected that only .* logical cores are available to this process.*";

const char* const expected_device_enumeration_warnings_regex =
    "Selected devices are of different type and/or do not belong to the same platform.*|No suitable platform found that can provide.*|No backend "
    "specialization available for selected platform.*|Selected platform .* is compatible with specialized .* backend, but it has not been compiled.*";

} // namespace celerity::test_utils_detail

namespace celerity::test_utils {

detail::instruction_graph_generator::system_info make_system_info(const size_t num_devices, const bool supports_d2d_copies) {
	using namespace detail;
	instruction_graph_generator::system_info info;
	info.devices.resize(num_devices);
	info.memories.resize(first_device_memory_id + num_devices);
	info.memories[host_memory_id].copy_peers.set(host_memory_id);
	info.memories[user_memory_id].copy_peers.set(user_memory_id);
	info.memories[host_memory_id].copy_peers.set(user_memory_id);
	info.memories[user_memory_id].copy_peers.set(host_memory_id);
	for(device_id did = 0; did < num_devices; ++did) {
		info.devices[did].native_memory = first_device_memory_id + did;
	}
	for(memory_id mid = first_device_memory_id; mid < info.memories.size(); ++mid) {
		info.memories[mid].copy_peers.set(mid);
		info.memories[mid].copy_peers.set(host_memory_id);
		info.memories[host_memory_id].copy_peers.set(mid);
		if(supports_d2d_copies) {
			for(memory_id peer = first_device_memory_id; peer < info.memories.size(); ++peer) {
				info.memories[mid].copy_peers.set(peer);
			}
		}
	}
	return info;
}

runtime_fixture::runtime_fixture() {
	detail::runtime::test_case_enter();
	allow_higher_level_log_messages(spdlog::level::warn, test_utils_detail::expected_runtime_init_warnings_regex);
	allow_higher_level_log_messages(spdlog::level::warn, test_utils_detail::expected_device_enumeration_warnings_regex);
}

runtime_fixture::~runtime_fixture() {
	if(!detail::runtime::test_runtime_was_instantiated()) { WARN("Test specified a runtime_fixture, but did not end up instantiating the runtime"); }
	detail::runtime::test_case_exit();
}

device_queue_fixture::~device_queue_fixture() { get_device_queue().get_sycl_queue().wait_and_throw(); }

detail::device_queue& device_queue_fixture::get_device_queue() {
	if(!m_dq) {
		allow_higher_level_log_messages(spdlog::level::warn, test_utils_detail::expected_device_enumeration_warnings_regex);
		m_cfg = std::make_unique<detail::config>(nullptr, nullptr);
		m_dq = std::make_unique<detail::device_queue>();
		m_dq->init(*m_cfg, detail::auto_select_device{});
	}
	return *m_dq;
}

} // namespace celerity::test_utils

CATCH_REGISTER_LISTENER(celerity::test_utils_detail::global_setup_and_teardown);
