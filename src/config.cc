#include "config.h"

#include "affinity.h"
#include "log.h"
#include "types.h"
#include "workaround.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>
#include <libenvpp/env.hpp>

namespace {

bool parse_validate_profile_kernel(const std::string_view str) {
	const auto pk = env::default_parser<bool>{}(str);
	CELERITY_DEBUG("CELERITY_PROFILE_KERNEL={}.", pk ? "on" : "off");
	return pk;
}

size_t parse_validate_dry_run_nodes(const std::string_view str) {
	const size_t drn = env::default_parser<size_t>{}(str);
	CELERITY_WARN("Performing a dry run with {} simulated nodes", drn);
	return drn;
}

} // namespace

namespace celerity {
namespace detail {

	config::config(int* argc, char** argv[]) {
		// TODO: At some point we might want to parse arguments from argv as well

		auto pref = env::prefix("CELERITY");
		const auto env_log_level = pref.register_option<log_level>("LOG_LEVEL", //
		    {{"trace", log_level::trace}, {"debug", log_level::debug}, {"info", log_level::info}, {"warn", log_level::warn}, {"err", log_level::err},
		        {"critical", log_level::critical}, {"off", log_level::off}});
		const auto env_profile_kernel = pref.register_variable<bool>("PROFILE_KERNEL", parse_validate_profile_kernel);
		const auto env_backend_device_submission_threads = pref.register_variable<bool>("BACKEND_DEVICE_SUBMISSION_THREADS");
		const auto env_thread_pinning = pref.register_variable<thread_pinning::environment_configuration>("THREAD_PINNING", thread_pinning::parse_validate_env);
		const auto env_print_graphs = pref.register_variable<bool>("PRINT_GRAPHS");
		const auto env_dry_run_num_nodes = pref.register_variable<size_t>("DRY_RUN_NODES", parse_validate_dry_run_nodes);
		constexpr int horizon_max = 1024 * 64;
		const auto env_horizon_step = pref.register_range<int>("HORIZON_STEP", 1, horizon_max);
		const auto env_horizon_max_para = pref.register_range<int>("HORIZON_MAX_PARALLELISM", 1, horizon_max);
		const auto env_lookahead = pref.register_option<experimental::lookahead>("LOOKAHEAD", //
		    {{"none", experimental::lookahead::none}, {"auto", experimental::lookahead::automatic}, {"infinite", experimental::lookahead::infinite}});
		const auto env_tracy_mode = pref.register_option<tracy_mode>("TRACY", //
		    {{"off", tracy_mode::off}, {"fast", tracy_mode::fast}, {"full", tracy_mode::full}});

		pref.register_deprecated("FORCE_WG", "Support for CELERITY_FORCE_WG has been removed with Celerity 0.3.0.");
		pref.register_deprecated("PROFILE_OCL", "CELERITY_PROFILE_OCL has been renamed to CELERITY_PROFILE_KERNEL with Celerity 0.3.0.");
		pref.register_deprecated("GRAPH_PRINT_MAX_VERTS", "Support for CELERITY_GRAPH_PRINT_MAX_VERTS has been removed with Celerity 0.5.0.\n"
		                                                  "Opt into graph printing by setting CELERITY_PRINT_GRAPHS=1.");
		pref.register_deprecated("DEVICES", "Support for CELERITY_DEVICES has been removed with Celerity 0.6.0.\n"
		                                    "Please use SYCL or vendor specific means to limit device visibility.");

		const auto parsed_and_validated_envs = pref.parse_and_validate();
		if(parsed_and_validated_envs.ok()) {
#if CELERITY_DETAIL_ENABLE_DEBUG
			m_log_lvl = parsed_and_validated_envs.get_or(env_log_level, log_level::debug);
#else
			m_log_lvl = parsed_and_validated_envs.get_or(env_log_level, log_level::info);
#endif

			const auto has_profile_kernel = parsed_and_validated_envs.get(env_profile_kernel);
			if(has_profile_kernel) { m_enable_device_profiling = *has_profile_kernel; }

			m_enable_backend_device_submission_threads = parsed_and_validated_envs.get_or(env_backend_device_submission_threads, true);
			m_thread_pinning_config = parsed_and_validated_envs.get_or(env_thread_pinning, {});

			const auto dry_run_num_nodes = parsed_and_validated_envs.get(env_dry_run_num_nodes);
			if(dry_run_num_nodes) { m_dry_run_num_nodes = static_cast<int>(*dry_run_num_nodes); }

			m_should_print_graphs = parsed_and_validated_envs.get_or(env_print_graphs, false);

			m_horizon_step = parsed_and_validated_envs.get(env_horizon_step);
			m_horizon_max_parallelism = parsed_and_validated_envs.get(env_horizon_max_para);
			m_lookahead = parsed_and_validated_envs.get_or(env_lookahead, experimental::lookahead::automatic);

			m_tracy_mode = parsed_and_validated_envs.get_or(env_tracy_mode, tracy_mode::off);
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
