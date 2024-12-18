#pragma once

#include "runtime.h"

#include "accessor.h"
#include "buffer.h"
#include "debug.h"
#include "distr_queue.h"
#include "host_utils.h"
#include "log.h"
#include "queue.h"
#include "side_effect.h"
#include "version.h"


namespace celerity {
namespace runtime {

	/**
	 * @brief Initializes the Celerity runtime.
	 */
	inline void init(int* argc, char** argv[]) { detail::runtime::init(argc, argv, detail::auto_select_devices{}); }

	/**
	 * @brief Initializes the Celerity runtime and instructs it to use a particular set of devices.
	 *
	 * @param devices The devices to be used on the current node. This can vary between nodes.
	 *                If there are multiple nodes running on the same host, the list of devices must be the same across nodes on the same host.
	 */
	inline void init(int* argc, char** argv[], const std::vector<sycl::device>& devices) { detail::runtime::init(argc, argv, devices); }

	/**
	 * @brief Initializes the Celerity runtime and instructs it to use a particular set of devices.
	 *
	 * @param device_selector The device selector to be used on the current node. This can vary between nodes.
	 *                        If there are multiple nodes running on the same host, the selector must be the same across nodes on the same host.
	 */
	inline void init(int* argc, char** argv[], const detail::device_selector& device_selector) { detail::runtime::init(argc, argv, device_selector); }

	/**
	 * @brief Manually shuts down the Celerity runtime if it has previously been initialized.
	 *
	 * No Celerity object (buffer, queue or host_object) must be live when calling this function, and the runtime cannot be re-initialized afterwards.
	 *
	 * Shutdown is also performed automatically on application exit.
	 */
	inline void shutdown() { detail::runtime::shutdown(); }

} // namespace runtime
} // namespace celerity
