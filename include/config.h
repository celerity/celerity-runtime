#pragma once

#include <boost/optional.hpp>
#include <string>

namespace celerity {
namespace detail {

	class logger;

	struct device_config {
		size_t platform_id;
		size_t device_id;
	};

	class config {
	  public:
		config(int* argc, char** argv[], logger& logger);

		/**
		 * Returns the platform and device id as set by the CELERITY_DEVICES environment variable.
		 * The variable has the form "P D0 [D1 ...]", where P is the platform index, followed by any number
		 * of device indices. Each device is assigned to a different node on the same host, according
		 * to their host-local node id.
		 *
		 * TODO: Should we support multiple platforms on the same host as well?
		 */
		boost::optional<device_config> get_device_config() const { return device_cfg; };
		boost::optional<bool> get_enable_device_profiling() const { return enable_device_profiling; };
		boost::optional<size_t> get_forced_work_group_size() const { return forced_work_group_size; };
		static size_t get_log_level();

	  private:
		boost::optional<device_config> device_cfg;
		boost::optional<bool> enable_device_profiling;
		boost::optional<size_t> forced_work_group_size;
	};

} // namespace detail
} // namespace celerity
