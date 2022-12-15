#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace celerity {
namespace detail {

	struct host_config {
		size_t node_count;
		size_t local_rank;
	};

	struct device_config {
		size_t platform_id;
		size_t device_id;
	};

	class config {
		friend struct config_testspy;

	  public:
		/**
		 * Initializes the @p config by parsing environment variables and passed arguments.
		 */
		config(int* argc, char** argv[]);

		const host_config& get_host_config() const { return m_host_cfg; }

		/**
		 * Returns the platform and device id as set by the CELERITY_DEVICES environment variable.
		 * The variable has the form "P D0 [D1 ...]", where P is the platform index, followed by any number
		 * of device indices. Each device is assigned to a different node on the same host, according
		 * to their host-local node id.
		 *
		 * TODO: Should we support multiple platforms on the same host as well?
		 */
		const std::optional<device_config>& get_device_config() const { return m_device_cfg; };
		std::optional<bool> get_enable_device_profiling() const { return m_enable_device_profiling; };
		bool is_dry_run() const { return m_dry_run_nodes > 0; };
		int get_dry_run_nodes() const { return m_dry_run_nodes; }

		size_t get_graph_print_max_verts() const { return m_graph_print_max_verts; };

	  private:
		host_config m_host_cfg;
		std::optional<device_config> m_device_cfg;
		std::optional<bool> m_enable_device_profiling;
		size_t m_graph_print_max_verts = 200;
		size_t m_dry_run_nodes = 0;
	};

} // namespace detail
} // namespace celerity
