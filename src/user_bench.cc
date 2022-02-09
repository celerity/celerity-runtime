#include "user_bench.h"

#include <ratio>
#include <stdexcept>

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

			user_benchmarker::user_benchmarker(config& cfg, node_id this_nid) : this_nid(this_nid) {
				if(static_cast<double>(bench_clock::period::num) / bench_clock::period::den > static_cast<double>(std::micro::num) / std::micro::den) {
					CELERITY_WARN("Available clock does not have sufficient precision");
				}
			}

			user_benchmarker& get_user_benchmarker() { return celerity::detail::runtime::get_instance().get_user_benchmarker(); }

			void user_benchmarker::begin_section(std::string name) {
				std::lock_guard lk{mutex};
				section sec = {next_section_id++, std::move(name), bench_clock::now()};
				log("Begin section {} \"{}\"", sec.id, sec.name);
				sections.push(std::move(sec));
			}

			void user_benchmarker::end_section(const std::string& name) {
				std::lock_guard lk{mutex};
				const auto sec = sections.top();
				sections.pop();
				if(sec.name != name) { throw std::runtime_error(fmt::format("Section name '{}' does not match last section '{}'", name, sec.name)); }
				const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(bench_clock::now() - sec.start);
				log("End section {} \"{}\" after {}us", sec.id, name, duration.count());
			}

		} // namespace detail
	}     // namespace bench
} // namespace experimental
} // namespace celerity