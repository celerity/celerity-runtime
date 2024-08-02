#pragma once

#include "runtime.h"

#include "accessor.h"
#include "buffer.h"
#include "debug.h"
#include "distr_queue.h"
#include "fence.h"
#include "host_utils.h"
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
} // namespace runtime
} // namespace celerity

// Celerity includes <CL/sycl.hpp> internally, but we want to expose the SYCL 2020 ::sycl namespace to Celerity users.
// TODO: Remove this once Celerity includes <sycl/sycl.hpp> internally.
#include <sycl/sycl.hpp>
