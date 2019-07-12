#pragma once

#include <chrono>
#include <memory>
#include <stack>

#include "logger.h"
#include "types.h"

namespace celerity {
namespace detail {
	class config;
}

namespace experimental {
	namespace bench {
		namespace detail {
			using logger = celerity::detail::logger;
			using node_id = celerity::detail::node_id;
			using logger_map = celerity::detail::logger_map;
			using config = celerity::detail::config;

			class user_benchmarker {
			  public:
				static void initialize(config& cfg, node_id this_nid);
				static void destroy();

				user_benchmarker(const user_benchmarker&) = delete;
				user_benchmarker(user_benchmarker&&) = delete;
				~user_benchmarker();

				static user_benchmarker& get_instance();

				void log_user_config(logger_map lm) const;

				template <typename... Args>
				void begin(const char* fmt, Args... args) {
					begin_section(fmt::format(fmt, std::forward<Args>(args)...));
				}

				template <typename... Args>
				void end(const char* fmt, Args... args) {
					end_section(fmt::format(fmt, std::forward<Args>(args)...));
				}

				template <typename... Args>
				void event(const char* fmt, Args... args) {
					log_event(fmt::format(fmt, std::forward<Args>(args)...));
				}

				void event(const logger_map& lm) const { log_event(lm); }

			  private:
				using bench_clock = std::chrono::steady_clock;
				using section_id = size_t;
				struct section {
					section_id id;
					std::string name;
					bench_clock::time_point start;
				};

				static std::unique_ptr<user_benchmarker> instance;
				std::shared_ptr<logger> bench_logger;
				node_id this_nid;
				section_id next_section_id = 0;
				std::stack<section> sections;

				user_benchmarker(config& cfg, node_id this_nid);

				void begin_section(std::string name);
				void end_section(std::string name);
				void log_event(const std::string& message) const;
				void log_event(logger_map lm) const;
			};
		} // namespace detail

		/**
		 * @brief Logs structured user configuration data. Only logged once (on the master node).
		 */
		inline void log_user_config(const detail::logger_map& lm) { detail::user_benchmarker::get_instance().log_user_config(lm); }

		/**
		 * @brief Begins a new benchmarking section.
		 *
		 * Sections can be nested to any depth.
		 */
		template <typename... Args>
		void begin(const char* bench_section_fmt, Args... args) {
			detail::user_benchmarker::get_instance().begin(bench_section_fmt, std::forward<Args>(args)...);
		}

		/**
		 * @brief Ends an existing benchmarking section.
		 */
		template <typename... Args>
		void end(const char* bench_section_fmt, Args... args) {
			detail::user_benchmarker::get_instance().end(bench_section_fmt, std::forward<Args>(args)...);
		}

		/**
		 * @brief Logs a benchmarking event.
		 */
		template <typename... Args>
		void event(const char* event_fmt, Args... args) {
			detail::user_benchmarker::get_instance().event(event_fmt, std::forward<Args>(args)...);
		}

		inline void event(const detail::logger_map& lm) { detail::user_benchmarker::get_instance().event(lm); }

	} // namespace bench
} // namespace experimental
} // namespace celerity
