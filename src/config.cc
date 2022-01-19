#include "config.h"

#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string_view>
#include <thread>
#include <vector>

#include <mpi.h>

#include "log.h"
#include "workaround.h"

std::pair<bool, std::string> get_env(const char* key) {
	bool exists = false;
	std::string str;
#ifdef _MSC_VER
	char* buf;
	_dupenv_s(&buf, nullptr, key);
	if(buf != nullptr) {
		exists = true;
		str = buf;
		delete buf;
	}
#else
	const auto value = std::getenv(key);
	if(value != nullptr) {
		exists = true;
		str = value;
	}
#endif

	std::string_view sv = str;
	sv.remove_prefix(std::min(sv.find_first_not_of(" "), sv.size()));
	sv.remove_suffix(std::min(sv.size() - sv.find_last_not_of(" ") - 1, sv.size()));

	return {exists, std::string{sv}};
}

std::pair<bool, size_t> parse_uint(const char* str) {
	errno = 0;
	char* eptr = nullptr;
	const auto value = std::strtoul(str, &eptr, 10);
	if(errno == 0 && eptr != str) { return {true, value}; }
	return {false, 0};
}

namespace celerity {
namespace detail {
	config::config(int* argc, char** argv[]) {
		// TODO: At some point we might want to parse arguments from argv as well

		{
			// Determine the "host config", i.e., how many nodes are spawned on this host,
			// and what this node's local rank is. We do this by finding all world-ranks
			// that can use a shared-memory transport (if running on OpenMPI, use the
			// per-host split instead).
#ifdef OPEN_MPI
#define SPLIT_TYPE OMPI_COMM_TYPE_HOST
#else
			// TODO: Assert that shared memory is available (i.e. not explicitly disabled)
#define SPLIT_TYPE MPI_COMM_TYPE_SHARED
#endif
			MPI_Comm host_comm;
			MPI_Comm_split_type(MPI_COMM_WORLD, SPLIT_TYPE, 0, MPI_INFO_NULL, &host_comm);

			int local_rank = 0;
			MPI_Comm_rank(host_comm, &local_rank);

			int node_count = 0;
			MPI_Comm_size(host_comm, &node_count);

			host_cfg.local_rank = local_rank;
			host_cfg.node_count = node_count;

			MPI_Comm_free(&host_comm);
		}

		// ------------------------------- CELERITY_LOG_LEVEL ---------------------------------

		{
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			log_lvl = log_level::debug;
#else
			log_lvl = log_level::info;
#endif
			const std::vector<std::pair<log_level, std::string>> possible_values = {
			    {log_level::trace, "trace"},
			    {log_level::debug, "debug"},
			    {log_level::info, "info"},
			    {log_level::warn, "warn"},
			    {log_level::err, "err"},
			    {log_level::critical, "critical"},
			    {log_level::off, "off"},
			};

			const auto result = get_env("CELERITY_LOG_LEVEL");
			if(result.first) {
				bool valid = false;
				for(auto& pv : possible_values) {
					if(result.second == pv.second) {
						log_lvl = pv.first;
						valid = true;
						break;
					}
				}
				if(!valid) {
					std::ostringstream oss;
					oss << "Invalid value \"" << result.second << "\" provided for CELERITY_LOG_LEVEL. ";
					oss << "Possible values are: ";
					for(size_t i = 0; i < possible_values.size(); ++i) {
						oss << possible_values[i].second << (i < possible_values.size() - 1 ? ", " : ".");
					}
					CELERITY_WARN(oss.str());
				}
			}
			spdlog::set_level(log_lvl);
		}

		// --------------------------------- CELERITY_DEVICES ---------------------------------

		{
			const auto [is_set, value] = get_env("CELERITY_DEVICES");
			if(is_set) {
				if(value.empty()) {
					CELERITY_WARN("CELERITY_DEVICES is set but empty - ignoring");
				} else {
					std::istringstream ss{value};
					std::vector<std::string> values{std::istream_iterator<std::string>{ss}, std::istream_iterator<std::string>{}};
					if(static_cast<long>(host_cfg.local_rank) > static_cast<long>(values.size()) - 2) {
						throw std::runtime_error(fmt::format("Process has local rank {}, but CELERITY_DEVICES only includes {} device(s)", host_cfg.local_rank,
						    values.empty() ? 0 : values.size() - 1));
					}

					if(static_cast<long>(values.size()) - 1 > static_cast<long>(host_cfg.node_count)) {
						CELERITY_WARN("CELERITY_DEVICES contains {} device indices, but only {} worker processes were spawned on this host", values.size() - 1,
						    host_cfg.node_count);
					}

					const auto pid_parsed = parse_uint(values[0].c_str());
					const auto did_parsed = parse_uint(values[host_cfg.local_rank + 1].c_str());
					if(!pid_parsed.first || !did_parsed.first) {
						CELERITY_WARN("CELERITY_DEVICES contains invalid value(s) - will be ignored");
					} else {
						device_cfg = device_config{pid_parsed.second, did_parsed.second};
					}
				}
			}
		}

		// ----------------------------- CELERITY_PROFILE_KERNEL ------------------------------
		{
			const auto result = get_env("CELERITY_PROFILE_OCL");
			if(result.first) {
				CELERITY_WARN("CELERITY_PROFILE_OCL has been renamed to CELERITY_PROFILE_KERNEL with Celerity 0.3.0.");
				enable_device_profiling = result.second == "1";
			}
		}

		{
			const auto result = get_env("CELERITY_PROFILE_KERNEL");
			if(result.first) { enable_device_profiling = result.second == "1"; }
		}

		// -------------------------------- CELERITY_FORCE_WG ---------------------------------

		{
			const auto result = get_env("CELERITY_FORCE_WG");
			if(result.first) { CELERITY_WARN("Support for CELERITY_FORCE_WG has been removed with Celerity 0.3.0."); }
		}

		// -------------------------------- CELERITY_HOST_CPUS --------------------------------

		{
			host_cfg.local_num_cpus = std::thread::hardware_concurrency();
			const auto result = get_env("CELERITY_HOST_CPUS");
			if(result.first) {
				const auto parsed = parse_uint(result.second.c_str());
				if(parsed.first) { host_cfg.local_num_cpus = parsed.second; }
			}
		}
	}

} // namespace detail
} // namespace celerity
