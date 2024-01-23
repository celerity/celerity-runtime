#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <string_view>

#include "log.h"
#include "types.h"

namespace celerity {
namespace detail {
	class config;
}

namespace experimental {
	namespace bench {
		namespace detail {
			using node_id = celerity::detail::node_id;
			using config = celerity::detail::config;

			class user_benchmarker {
			  public:
				user_benchmarker(config& cfg, node_id this_nid);
				user_benchmarker(const user_benchmarker&) = delete;
				user_benchmarker(user_benchmarker&&) = delete;
				~user_benchmarker();

				template <typename... Args>
				void log_once(const std::string& format_string, Args&&... args) const {
					if(m_this_nid == 0) { log(format_string, std::forward<Args>(args)...); }
				}

				template <typename... Args>
				void begin(std::string_view format_string, Args&&... args) {
					begin_section(fmt::format(format_string, std::forward<Args>(args)...));
				}

				template <typename... Args>
				void end(std::string_view format_string, Args&&... args) {
					end_section(fmt::format(format_string, std::forward<Args>(args)...));
				}

				template <typename... Args>
				void event(const std::string& format_string, Args&&... args) {
					std::lock_guard lk{m_mutex};
					const auto now = bench_clock::now();
					const auto dt = (m_last_event_tp != bench_clock::time_point{})
					                    ? std::chrono::duration_cast<std::chrono::microseconds>(now - m_last_event_tp).count()
					                    : 0;
					log(format_string + " (+{}us)", std::forward<Args>(args)..., dt);
					m_last_event_tp = now;
				}

			  private:
				using bench_clock = std::chrono::steady_clock;
				using section_id = size_t;
				struct section {
					section_id id;
					std::string name;
					bench_clock::time_point start;
				};

				mutable std::mutex m_mutex;

				node_id m_this_nid;
				section_id m_next_section_id = 0;
				std::stack<section> m_sections;
				bench_clock::time_point m_last_event_tp = {};

				void begin_section(std::string name);
				void end_section(const std::string& name);

				template <typename... Args>
				void log(fmt::format_string<Args...> fmt_string, Args&&... args) const {
					CELERITY_DEBUG("[user] {}", fmt::format(fmt_string, std::forward<Args>(args)...));
				}
			};

			user_benchmarker& get_user_benchmarker();
		} // namespace detail

		/**
		 * @brief Logs a message only once (on the master node).
		 */
		template <typename... Args>
		void log_once(const std::string& format_string, Args&&... args) {
			detail::get_user_benchmarker().log_once(format_string, std::forward<Args>(args)...);
		}

		/**
		 * @brief Begins a new benchmarking section.
		 *
		 * Sections can be nested to any depth.
		 */
		template <typename... Args>
		void begin(std::string_view format_string, Args&&... args) {
			detail::get_user_benchmarker().begin(format_string, std::forward<Args>(args)...);
		}

		/**
		 * @brief Ends an existing benchmarking section.
		 */
		template <typename... Args>
		void end(std::string_view format_string, Args&&... args) {
			detail::get_user_benchmarker().end(format_string, std::forward<Args>(args)...);
		}

		/**
		 * @brief Logs a benchmarking event.
		 */
		template <typename... Args>
		void event(const std::string& format_string, Args&&... args) {
			detail::get_user_benchmarker().event(format_string, std::forward<Args>(args)...);
		}

	} // namespace bench
} // namespace experimental
} // namespace celerity
