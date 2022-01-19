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

			void user_benchmarker::log_user_config(log_map lm) const {
				if(this_nid == 0) { CELERITY_INFO("User config: {}", lm); }
			}

			user_benchmarker::user_benchmarker(config& cfg, node_id this_nid) : this_nid(this_nid) {
				if(static_cast<double>(bench_clock::period::num) / bench_clock::period::den > static_cast<double>(std::micro::num) / std::micro::den) {
					CELERITY_WARN("Available clock does not have sufficient precision");
				}
			}

			user_benchmarker& get_user_benchmarker() { return celerity::detail::runtime::get_instance().get_user_benchmarker(); }

			void user_benchmarker::begin_section(std::string name) {
				const section sec = {next_section_id++, name, bench_clock::now()};
				sections.push(sec);
				CELERITY_INFO("Begin section {}: \"{}\"", sec.id, name);
			}

			void user_benchmarker::end_section(std::string name) {
				const auto sec = sections.top();
				sections.pop();
				assert(sec.name == name && "Section name does not equal last section");
				const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(bench_clock::now() - sec.start);
				CELERITY_INFO("End section {}: \"{}\". Duration {}us", sec.id, name, duration.count());
			}

			void user_benchmarker::log_event(const std::string& message) const { CELERITY_INFO("User event: \"{}\"", message); }

			void user_benchmarker::log_event(log_map lm) const { CELERITY_INFO("User event: {}", lm); }
		} // namespace detail
	}     // namespace bench
} // namespace experimental
} // namespace celerity