#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include "config.h"
#include "stdout_color_sink_custom.h"

namespace spd = spdlog;

namespace celerity {
namespace detail {

	using logger_map = std::unordered_map<std::string, std::string>;

	class map_serializer {
	  public:
		const logger_map& map;

		map_serializer(const logger_map& map) : map(map) {}

		template <typename OStream>
		friend OStream& operator<<(OStream& os, const map_serializer& serializer) {
			int i = 0;
			for(auto& kv : serializer.map) {
				if(i++ > 0) { os << ", "; }
				os << fmt::format(R"("{}": "{}")", kv.first, kv.second);
			}
			return os;
		}
	};


	class logger {
	  public:
		logger(const std::string& channel, bool force_color_seq = true) {
			this->channel = channel;
			spd_logger = spd::sinks::stdout_color_mt(channel, force_color_seq);
			spd_logger->set_pattern(R"({"at": "%Y-%m-%d %T.%e", "channel": "%n", "level": "%^%l%$"%v})");
			size_t log_level = config::get_log_level();
			spd_logger->set_level(spd::level::level_enum(log_level));
			spd_logger->flush_on(spd::level::level_enum(log_level));
		}

		~logger() {
			if(channel != "") { spd::drop(channel); }
		}

		std::shared_ptr<logger> create_context(logger_map context_values) const {
			assert(!context_values.empty());
			std::ostringstream oss;
			oss << context;
			oss << map_serializer(context_values);
			oss << ", ";
			return std::shared_ptr<logger>(new logger(spd_logger, oss.str()));
		}

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
		std::string channel;
		std::shared_ptr<spd::logger> spd_logger;
		std::string context = ", ";

		logger(std::shared_ptr<spd::logger> spd_logger, std::string context) : spd_logger(std::move(spd_logger)), context(std::move(context)) {}

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
			return fmt::format(R"({}"message": "{}")", context, msg);
		}

		std::string add_context(const logger_map& log_values) const { return fmt::format(R"({}{})", context, map_serializer(log_values)); }
	};

} // namespace detail
} // namespace celerity
