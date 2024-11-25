#pragma once

#include "affinity.h"
#include "log.h"
#include "types.h"

#include <cstddef>
#include <optional>


namespace celerity {
namespace detail {

	struct host_config {
		size_t node_count = 0;
		size_t local_rank = -1;
	};

	class config {
	  public:
		/**
		 * Initializes the @p config by parsing environment variables and passed arguments.
		 */
		config(int* argc, char** argv[]);

		log_level get_log_level() const { return m_log_lvl; }

		bool should_enable_device_profiling() const { return m_enable_device_profiling.value_or(m_tracy_mode == tracy_mode::full); }
		bool should_use_backend_device_submission_threads() const { return m_enable_backend_device_submission_threads; }
		const thread_pinning::environment_configuration& get_thread_pinning_config() const& { return m_thread_pinning_config; }
		bool should_print_graphs() const { return m_should_print_graphs; }
		bool should_record() const {
			// Currently only graph printing requires recording, but this might change in the future.
			return m_should_print_graphs;
		}
		bool is_dry_run() const { return m_dry_run_num_nodes > 0; }
		int get_dry_run_nodes() const { return m_dry_run_num_nodes; }
		std::optional<int> get_horizon_step() const { return m_horizon_step; }
		std::optional<int> get_horizon_max_parallelism() const { return m_horizon_max_parallelism; }
		experimental::lookahead get_lookahead() { return m_lookahead; }
		tracy_mode get_tracy_mode() const { return m_tracy_mode; }

	  private:
		log_level m_log_lvl;
		std::optional<bool> m_enable_device_profiling;
		bool m_enable_backend_device_submission_threads = true;
		thread_pinning::environment_configuration m_thread_pinning_config;
		int m_dry_run_num_nodes = 0;
		bool m_should_print_graphs = false;
		std::optional<int> m_horizon_step;
		std::optional<int> m_horizon_max_parallelism;
		experimental::lookahead m_lookahead = experimental::lookahead::automatic;
		tracy_mode m_tracy_mode = tracy_mode::off;
	};

} // namespace detail
} // namespace celerity
