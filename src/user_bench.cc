#include "user_bench.h"

#include <ratio>

#include "config.h"
#include "runtime.h"

namespace celerity {
namespace experimental {
	namespace bench {
		namespace detail {
			user_benchmarker::~user_benchmarker() {
				while(!sections.empty()) {
					const auto sec = sections.top();
					end_section(sec.name);
				}
			}

			void user_benchmarker::log_user_config(logger_map lm) const {
				assert(lm.count("event") == 0 && "Key 'event' not allowed in user config");
				if(this_nid == 0) {
					lm["event"] = "userConfig";
					bench_logger->info(lm);
				}
			}

			user_benchmarker::user_benchmarker(config& cfg, node_id this_nid) : this_nid(this_nid) {
				bench_logger = logger("bench", cfg.get_log_level()).create_context({{"rank", std::to_string(this_nid)}});
				if(static_cast<double>(bench_clock::period::num) / bench_clock::period::den > static_cast<double>(std::micro::num) / std::micro::den) {
					bench_logger->warn("Available clock does not have sufficient precision");
				}
			}

			user_benchmarker& get_user_benchmarker() { return celerity::detail::runtime::get_instance().get_user_benchmarker(); }

			void user_benchmarker::begin_section(std::string name) {
				const section sec = {next_section_id++, name, bench_clock::now()};
				sections.push(sec);
				bench_logger->info(logger_map{{"event", "beginSection"}, {"id", std::to_string(sec.id)}, {"name", name}});
			}

			void user_benchmarker::end_section(std::string name) {
				const auto sec = sections.top();
				sections.pop();
				assert(sec.name == name && "Section name does not equal last section");
				const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(bench_clock::now() - sec.start);
				bench_logger->info(
				    logger_map{{"event", "endSection"}, {"id", std::to_string(sec.id)}, {"name", name}, {"duration", std::to_string(duration.count())}});
			}

			void user_benchmarker::log_event(const std::string& message) const { log_event(logger_map{{"message", message}}); }

			void user_benchmarker::log_event(logger_map lm) const {
				assert(lm.count("event") == 0 && "Key 'event' not allowed in bench event");
				lm["event"] = "userEvent";
				bench_logger->info(lm);
			}
		} // namespace detail
	}     // namespace bench
} // namespace experimental
} // namespace celerity