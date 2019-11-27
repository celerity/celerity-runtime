#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "print_utils.h"

namespace spd = spdlog;

namespace celerity {
namespace detail {

	using logger_map = std::unordered_map<std::string, std::string>;

	class map_serializer {
	  public:
		map_serializer(const logger_map& map, bool log_machine_readable) : map(map), log_machine_readable(log_machine_readable) {}

		template <typename OStream>
		friend OStream& operator<<(OStream& os, const map_serializer& serializer) {
			int i = 0;
			for(auto& kv : serializer.map) {
				if(i++ > 0) { os << ", "; }
				if(serializer.log_machine_readable) {
					os << fmt::format(R"("{}": "{}")", kv.first, kv.second);
				} else {
					os << fmt::format("{} = {}", kv.first, kv.second);
				}
			}
			return os;
		}

	  private:
		const logger_map& map;
		bool log_machine_readable;
	};

	enum class log_level {
		trace = SPDLOG_LEVEL_TRACE,
		debug = SPDLOG_LEVEL_DEBUG,
		info = SPDLOG_LEVEL_INFO,
		warn = SPDLOG_LEVEL_WARN,
		err = SPDLOG_LEVEL_ERROR,
		critical = SPDLOG_LEVEL_CRITICAL,
		off = SPDLOG_LEVEL_OFF,
	};

	enum class log_color_mode {
		always = std::underlying_type<spdlog::color_mode>::type(spdlog::color_mode::always),
		automatic = std::underlying_type<spdlog::color_mode>::type(spdlog::color_mode::automatic),
		never = std::underlying_type<spdlog::color_mode>::type(spdlog::color_mode::never),
	};

	class logger {
	  public:
		logger(const std::string& channel, log_level level = log_level::info, log_color_mode mode = log_color_mode::automatic) {
			this->channel = channel;
			spd_logger = spd::stderr_color_mt(channel, static_cast<spdlog::color_mode>(mode));
			set_level(level);
		}

		~logger() {
			if(channel != "") { spd::drop(channel); }
		}

		std::shared_ptr<logger> create_context(logger_map context_values) const {
			assert(!context_values.empty());
			logger_map combined = this->context_values;
			std::copy(context_values.cbegin(), context_values.cend(), std::inserter(combined, combined.end()));
			return std::shared_ptr<logger>(new logger(spd_logger, combined));
		}

		void set_level(log_level level) {
			spd_logger->set_level(static_cast<spdlog::level::level_enum>(level));
			update_format();
		}

		log_level get_level() const { return static_cast<log_level>(spd_logger->level()); }

		template <typename Arg1, typename... Args>
		void trace(const char* fmt, const Arg1& arg1, const Args&... args) {
			log(spd::level::trace, fmt, arg1, args...);
		}

		template <typename Arg1, typename... Args>
		void debug(const char* fmt, const Arg1& arg1, const Args&... args) {
			log(spd::level::debug, fmt, arg1, args...);
		}

		template <typename Arg1, typename... Args>
		void info(const char* fmt, const Arg1& arg1, const Args&... args) {
			log(spd::level::info, fmt, arg1, args...);
		}

		template <typename Arg1, typename... Args>
		void warn(const char* fmt, const Arg1& arg1, const Args&... args) {
			log(spd::level::warn, fmt, arg1, args...);
		}

		template <typename Arg1, typename... Args>
		void error(const char* fmt, const Arg1& arg1, const Args&... args) {
			log(spd::level::err, fmt, arg1, args...);
		}

		template <typename Arg1, typename... Args>
		void critical(const char* fmt, const Arg1& arg1, const Args&... args) {
			log(spd::level::critical, fmt, arg1, args...);
		}

		template <typename T>
		void trace(const T& msg) {
			log(spd::level::trace, msg);
		}

		template <typename T>
		void debug(const T& msg) {
			log(spd::level::debug, msg);
		}

		template <typename T>
		void info(const T& msg) {
			log(spd::level::info, msg);
		}

		template <typename T>
		void warn(const T& msg) {
			log(spd::level::warn, msg);
		}

		template <typename T>
		void error(const T& msg) {
			log(spd::level::err, msg);
		}

		template <typename T>
		void critical(const T& msg) {
			log(spd::level::critical, msg);
		}

	  private:
		bool log_machine_readable = false;
		std::string channel;
		std::shared_ptr<spd::logger> spd_logger;

		logger_map context_values;
		std::string context;

		logger(std::shared_ptr<spd::logger> spd_logger, logger_map context_values)
		    : spd_logger(std::move(spd_logger)), context_values(std::move(context_values)) {
			update_format();
		}

		void update_format() {
			context = "";
			log_machine_readable = get_level() == log_level::trace; // TODO: Make this configurable as well?
			if(log_machine_readable) {
				// FIXME: We need proper JSON serialization solution
				spd_logger->set_pattern(R"({"at": "%E%f", "channel": "%n", "level": "%l"%v})");
				context = ", ";
			}

			std::ostringstream oss;
			oss << map_serializer(context_values, log_machine_readable);
			if(!context_values.empty() && log_machine_readable) oss << ", ";
			context = context + oss.str();
		}

		template <typename... Args>
		void log(spd::level::level_enum lvl, const char* fmt, const Args&... args) {
			spd_logger->log(lvl, add_context(fmt).c_str(), args...);
		}

		template <typename T>
		void log(spd::level::level_enum lvl, const T& msg) {
			spd_logger->log(lvl, add_context(msg).c_str());
		}

		template <typename T>
		std::string add_context(const T& msg) const {
			if(log_machine_readable) { return fmt::format(R"({}"message": "{}")", context, msg); }
			return fmt::format("[{}] {}", context, msg);
		}

		std::string add_context(const logger_map& log_values) const {
			if(log_machine_readable) { return fmt::format("{}{}", context, map_serializer(log_values, log_machine_readable)); }
			return fmt::format("[{}] {}", context, map_serializer(log_values, log_machine_readable));
		}
	};

} // namespace detail
} // namespace celerity
