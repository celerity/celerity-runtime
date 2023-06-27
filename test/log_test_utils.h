#pragma once

#include "log.h" // Need to include before spdlog
#include <catch2/catch_test_macros.hpp>
#include <spdlog/sinks/ostream_sink.h>

namespace celerity::test_utils {

class log_capture {
  public:
	explicit log_capture(spdlog::level::level_enum level = spdlog::level::trace) {
		const auto logger = spdlog::default_logger_raw();
		m_ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_st>(m_oss);
		m_ostream_sink->set_level(level);
		logger->sinks().push_back(m_ostream_sink);
		m_level_before = spdlog::get_level();
		m_tmp_level = std::min(m_level_before, level);
		spdlog::set_level(m_tmp_level);
		m_flush_before = logger->flush_level();
		m_tmp_flush = std::min(m_flush_before, level);
		logger->flush_on(m_tmp_flush);
	}

	~log_capture() {
		spdlog::set_level(m_level_before);
		const auto logger = spdlog::default_logger_raw();
		logger->flush_on(m_flush_before);
		assert(*logger->sinks().rbegin() == m_ostream_sink);
		logger->sinks().pop_back();
	}

	std::string get_log() {
		assert_log_level_unchanged();
		return m_oss.str();
	}

  private:
	spdlog::level::level_enum m_tmp_level, m_level_before, m_tmp_flush, m_flush_before;
	std::ostringstream m_oss;
	std::shared_ptr<spdlog::sinks::ostream_sink_st> m_ostream_sink;

	void assert_log_level_unchanged() {
		if(spdlog::get_level() != m_tmp_level) {
			FAIL_CHECK("global spdlog level has changed since log_capture was instantiated - captured logs may depend on CELERITY_LOG_LEVEL!");
			m_tmp_level = spdlog::get_level();
		}
		if(spdlog::default_logger_raw()->flush_level() != m_tmp_flush) {
			FAIL_CHECK("spdlog flush level has changed since log_capture was instantiated!");
			m_tmp_flush = spdlog::default_logger_raw()->flush_level();
		}
	}
};

} // namespace celerity::test_utils
