#pragma once

#include <optional>

#include "logger.h"

namespace celerity {
namespace detail {

	struct host_config {
		size_t node_count;
		size_t local_rank;
		size_t local_num_cpus;
	};

	struct device_config {
		size_t platform_id;
		size_t device_id;
	};

	class config {
	  public:
		/**
		 * Initializes the @p config by parsing environment variables and passed arguments.
		 *
		 * @param logger The logger is used to print warnings about invalid configuration options.
		 * 				 Additionally, the logger's level is set to the same level as is
		 * 				 returned by ::get_log_level().
		 */
		config(int* argc, char** argv[], logger& logger);

		log_level get_log_level() const { return log_lvl; }

		const host_config& get_host_config() const { return host_cfg; }

		/**
		 * Returns the platform and device id as set by the CELERITY_DEVICES environment variable.
		 * The variable has the form "P D0 [D1 ...]", where P is the platform index, followed by any number
		 * of device indices. Each device is assigned to a different node on the same host, according
		 * to their host-local node id.
		 *
		 * TODO: Should we support multiple platforms on the same host as well?
		 */
		const std::optional<device_config>& get_device_config() const { return device_cfg; };
		std::optional<bool> get_enable_device_profiling() const { return enable_device_profiling; };

	  private:
		log_level log_lvl;
		host_config host_cfg;
		std::optional<device_config> device_cfg;
		std::optional<bool> enable_device_profiling;
	};

} // namespace detail
} // namespace celerity
