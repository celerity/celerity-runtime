#include "config.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <libenvpp/env.hpp>
#include <mpi.h>

static std::vector<std::string> split(const std::string_view str, const char delimiter) {
	auto result = std::vector<std::string>{};
	auto sstream = std::istringstream(std::string(str));
	auto item = std::string{};
	while(std::getline(sstream, item, delimiter)) {
		result.push_back(std::move(item));
	}
	return result;
}

namespace {

bool parse_validate_profile_kernel(const std::string_view str) {
	const auto pk = env::default_parser<bool>{}(str);
	CELERITY_DEBUG("CELERITY_PROFILE_KERNEL={}.", pk ? "on" : "off");
	return pk;
}

size_t parse_validate_dry_run_nodes(const std::string_view str) {
	const size_t drn = env::default_parser<size_t>{}(str);
	int world_size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	if(world_size != 1) throw std::runtime_error("In order to run with CELERITY_DRY_RUN_NODES a single MPI process/rank must be used.");
	CELERITY_WARN("Performing a dry run with {} simulated nodes", drn);
	return drn;
}

std::vector<size_t> parse_validate_devices(const std::string_view str, const celerity::detail::host_config host_cfg) {
	std::vector<size_t> devices;
	const auto split_str = split(str, ' ');
	// Delegate parsing of primitive types to the default_parser
	for(size_t i = 0; i < split_str.size(); ++i) {
		devices.push_back(env::default_parser<size_t>{}(split_str[i]));
	}
	if(devices.size() < 2) {
		throw env::validation_error{fmt::format(
		    "Found {} IDs.\nExpected the following format: CELERITY_DEVICES=\"<platform_id> <first device_id> <second device_id> ... <nth device_id>\"",
		    devices.size())};
	}

	if(static_cast<long>(host_cfg.local_rank) > static_cast<long>(devices.size()) - 2) {
		throw env::validation_error{fmt::format(
		    "Process has local rank {}, but CELERITY_DEVICES only includes {} device(s)", host_cfg.local_rank, devices.empty() ? 0 : devices.size() - 1)};
	}
	if(static_cast<long>(devices.size()) - 1 > static_cast<long>(host_cfg.node_count)) {
		throw env::validation_error{fmt::format(
		    "CELERITY_DEVICES contains {} device indices, but only {} worker processes were spawned on this host", devices.size() - 1, host_cfg.node_count)};
	}

	return devices;
}

} // namespace

namespace celerity {
namespace detail {

	config::config(int* argc, char** argv[]) {
		// TODO: At some point we might want to parse arguments from argv as well

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
		MPI_Comm host_comm = nullptr;
		MPI_Comm_split_type(MPI_COMM_WORLD, SPLIT_TYPE, 0, MPI_INFO_NULL, &host_comm);

		int local_rank = 0;
		MPI_Comm_rank(host_comm, &local_rank);

		int node_count = 0;
		MPI_Comm_size(host_comm, &node_count);

		m_host_cfg.local_rank = local_rank;
		m_host_cfg.node_count = node_count;

		MPI_Comm_free(&host_comm);

		auto pref = env::prefix("CELERITY");
		const auto env_log_level = pref.register_option<log_level>("LOG_LEVEL", //
		    {{"trace", log_level::trace}, {"debug", log_level::debug}, {"info", log_level::info}, {"warn", log_level::warn}, {"err", log_level::err},
		        {"critical", log_level::critical}, {"off", log_level::off}});
		const auto env_devs =
		    pref.register_variable<std::vector<size_t>>("DEVICES", [this](const std::string_view str) { return parse_validate_devices(str, m_host_cfg); });
		const auto env_profile_kernel = pref.register_variable<bool>("PROFILE_KERNEL", parse_validate_profile_kernel);
		const auto env_print_graphs = pref.register_variable<bool>("PRINT_GRAPHS");
		const auto env_dry_run_nodes = pref.register_variable<size_t>("DRY_RUN_NODES", parse_validate_dry_run_nodes);
		constexpr int horizon_max = 1024 * 64;
		const auto env_horizon_step = pref.register_range<int>("HORIZON_STEP", 1, horizon_max);
		const auto env_horizon_max_para = pref.register_range<int>("HORIZON_MAX_PARALLELISM", 1, horizon_max);

		pref.register_deprecated("FORCE_WG", "Support for CELERITY_FORCE_WG has been removed with Celerity 0.3.0.");
		pref.register_deprecated("PROFILE_OCL", "CELERITY_PROFILE_OCL has been renamed to CELERITY_PROFILE_KERNEL with Celerity 0.3.0.");
		pref.register_deprecated("GRAPH_PRINT_MAX_VERTS", "Support for CELERITY_GRAPH_PRINT_MAX_VERTS has been removed with Celerity 0.5.0.\n"
		                                                  "Opt into graph printing by setting CELERITY_PRINT_GRAPHS=1.");

		const auto parsed_and_validated_envs = pref.parse_and_validate();
		if(parsed_and_validated_envs.ok()) {
			// ------------------------------- CELERITY_LOG_LEVEL ---------------------------------

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			m_log_lvl = parsed_and_validated_envs.get_or(env_log_level, log_level::debug);
#else
			m_log_lvl = parsed_and_validated_envs.get_or(env_log_level, log_level::info);
#endif

			// --------------------------------- CELERITY_DEVICES ---------------------------------

			const auto has_devs = parsed_and_validated_envs.get(env_devs);
			if(has_devs) {
				const auto pid_parsed = (*has_devs)[0];
				const auto did_parsed = (*has_devs)[m_host_cfg.local_rank + 1];
				m_device_cfg = device_config{pid_parsed, did_parsed};
			}

			// ----------------------------- CELERITY_PROFILE_KERNEL ------------------------------

			const auto has_profile_kernel = parsed_and_validated_envs.get(env_profile_kernel);
			if(has_profile_kernel) { m_enable_device_profiling = *has_profile_kernel; }

			// -------------------------------- CELERITY_DRY_RUN_NODES ---------------------------------

			const auto has_dry_run_nodes = parsed_and_validated_envs.get(env_dry_run_nodes);
			if(has_dry_run_nodes) { m_dry_run_nodes = *has_dry_run_nodes; }

			m_should_print_graphs = parsed_and_validated_envs.get_or(env_print_graphs, false);
			m_horizon_step = parsed_and_validated_envs.get(env_horizon_step);
			m_horizon_max_parallelism = parsed_and_validated_envs.get(env_horizon_max_para);

		} else {
			for(const auto& warn : parsed_and_validated_envs.warnings()) {
				CELERITY_ERROR("{}", warn.what());
			}
			for(const auto& err : parsed_and_validated_envs.errors()) {
				CELERITY_ERROR("{}", err.what());
			}
			throw std::runtime_error("Failed to parse/validate environment variables.");
		}
	}
} // namespace detail
} // namespace celerity
