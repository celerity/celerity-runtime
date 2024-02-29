#include "config.h"

#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string_view>
#include <thread>

#include <mpi.h>

#include "log.h"

#include <spdlog/sinks/sink.h>

#include <libenvpp/env.hpp>

namespace env {
template <>
struct default_parser<celerity::detail::log_level> {
	celerity::detail::log_level operator()(const std::string_view str) const {
		const std::vector<std::pair<celerity::detail::log_level, std::string>> possible_values = {
		    {celerity::detail::log_level::trace, "trace"},
		    {celerity::detail::log_level::debug, "debug"},
		    {celerity::detail::log_level::info, "info"},
		    {celerity::detail::log_level::warn, "warn"},
		    {celerity::detail::log_level::err, "err"},
		    {celerity::detail::log_level::critical, "critical"},
		    {celerity::detail::log_level::off, "off"},
		};

		auto lvl = celerity::detail::log_level::info;
		bool valid = false;
		for(const auto& pv : possible_values) {
			if(str == pv.second) {
				lvl = pv.first;
				valid = true;
				break;
			}
		}
		auto err_msg = fmt::format("Unable to parse '{}'. Possible values are:", str);
		for(size_t i = 0; i < possible_values.size(); ++i) {
			err_msg += fmt::format(" {}{}", possible_values[i].second, (i < possible_values.size() - 1 ? ", " : "."));
		}
		if(!valid) throw parser_error{err_msg};

		return lvl;
	}
};
} // namespace env

namespace {

size_t parse_validate_graph_print_max_verts(const std::string_view str) {
	throw env::validation_error{"Support for CELERITY_GRAPH_PRINT_MAX_VERTS has been removed with Celerity 0.5.0.\n"
	                            "Opt into graph printing by setting CELERITY_PRINT_GRAPHS=1."};
	return 0;
}

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
	throw env::validation_error{
	    "Support for CELERITY_DEVICES has been removed with Celerity 0.5.0. Please use SYCL or vendor specific means to limit device visibility."};
	return {};
}

bool parse_validate_force_wg(const std::string_view str) {
	throw env::validation_error{"Support for CELERITY_FORCE_WG has been removed with Celerity 0.3.0."};
	return false;
}

bool parse_validate_profile_ocl(const std::string_view str) {
	throw env::validation_error{"CELERITY_PROFILE_OCL has been renamed to CELERITY_PROFILE_KERNEL with Celerity 0.3.0."};
	return false;
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
		const auto env_log_level = pref.register_option<log_level>(
		    "LOG_LEVEL", {log_level::trace, log_level::debug, log_level::info, log_level::warn, log_level::err, log_level::critical, log_level::off});
		[[maybe_unused]] const auto env_devs =
		    pref.register_variable<std::vector<size_t>>("DEVICES", [this](const std::string_view str) { return parse_validate_devices(str, m_host_cfg); });
		const auto env_profile_kernel = pref.register_variable<bool>("PROFILE_KERNEL", parse_validate_profile_kernel);
		const auto env_dry_run_nodes = pref.register_variable<size_t>("DRY_RUN_NODES", parse_validate_dry_run_nodes);
		const auto env_print_graphs = pref.register_variable<bool>("PRINT_GRAPHS");
		constexpr int horizon_max = 1024 * 64;
		const auto env_horizon_step = pref.register_range<int>("HORIZON_STEP", 1, horizon_max);
		const auto env_horizon_max_para = pref.register_range<int>("HORIZON_MAX_PARALLELISM", 1, horizon_max);
		[[maybe_unused]] const auto env_gpmv = pref.register_variable<size_t>("GRAPH_PRINT_MAX_VERTS", parse_validate_graph_print_max_verts);
		[[maybe_unused]] const auto env_force_wg =
		    pref.register_variable<bool>("FORCE_WG", [](const std::string_view str) { return parse_validate_force_wg(str); });
		[[maybe_unused]] const auto env_profile_ocl =
		    pref.register_variable<bool>("PROFILE_OCL", [](const std::string_view str) { return parse_validate_profile_ocl(str); });
		const auto env_disable_d2d_copy = pref.register_variable<bool>("DISABLE_D2D_COPY");

		const auto parsed_and_validated_envs = pref.parse_and_validate();
		if(parsed_and_validated_envs.ok()) {
			// ------------------------------- CELERITY_LOG_LEVEL ---------------------------------

#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
			m_log_lvl = parsed_and_validated_envs.get_or(env_log_level, log_level::debug);
#else
			m_log_lvl = parsed_and_validated_envs.get_or(env_log_level, log_level::info);
#endif

			// ----------------------------- CELERITY_PROFILE_KERNEL ------------------------------

			const auto has_profile_kernel = parsed_and_validated_envs.get(env_profile_kernel);
			if(has_profile_kernel) { m_enable_device_profiling = *has_profile_kernel; }

			// -------------------------------- CELERITY_DRY_RUN_NODES ---------------------------------

			const auto has_dry_run_nodes = parsed_and_validated_envs.get(env_dry_run_nodes);
			if(has_dry_run_nodes) { m_dry_run_nodes = *has_dry_run_nodes; }

			m_should_print_graphs = parsed_and_validated_envs.get_or(env_print_graphs, false);
			m_horizon_step = parsed_and_validated_envs.get(env_horizon_step);
			m_horizon_max_parallelism = parsed_and_validated_envs.get(env_horizon_max_para);
			m_disable_d2d_copy = parsed_and_validated_envs.get_or(env_disable_d2d_copy, false);

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
