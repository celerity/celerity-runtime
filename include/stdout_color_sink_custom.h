//
// Copyright(c) 2018 spdlog
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#pragma once

#ifndef SPDLOG_H
#include <spdlog/spdlog.h>
#endif

#ifdef _WIN32
#include <spdlog/sinks/wincolor_sink.h>
#else
#include "ansicolor_sink_custom.h"
#endif

namespace spdlog {
namespace sinks {
// for windows there is no checking for terminal, don't need to force color sequence
#ifdef _WIN32

	template <typename Factory = default_factory>
	inline std::shared_ptr<logger> stdout_color_mt(const std::string& logger_name, bool force_color_seq = false) {
		return Factory::template create<sinks::wincolor_stdout_sink_mt>(logger_name);
	}

#else

	template <typename Factory = default_factory>
	inline std::shared_ptr<logger> stdout_color_mt(const std::string& logger_name, bool force_color_seq = false) {
		return Factory::template create<sinks::ansicolor_stdout_custom_sink_mt>(logger_name, force_color_seq);
	}

#endif
} // namespace sinks
} // namespace spdlog