#ifndef RUNTIME_INCLUDE_ENTRY_CELERITY
#define RUNTIME_INCLUDE_ENTRY_CELERITY

#include "device_queue.h"
#include "runtime.h"

#include "accessor.h"
#include "buffer.h"
#include "debug.h"
#include "distr_queue.h"
#include "fence.h"
#include "side_effect.h"
#include "user_bench.h"
#include "version.h"

namespace celerity {
namespace runtime {
	/**
	 * @brief Initializes the Celerity runtime.
	 */
	inline void init(int* argc, char** argv[]) { detail::runtime::init(argc, argv, detail::auto_select_device{}); }

	/**
	 * @brief Initializes the Celerity runtime and instructs it to use a particular device.
	 *
	 * @param device The device to be used on the current node. This can vary between nodes.
	 */
	[[deprecated("Use the overload with device selector instead, this will be removed in future release")]] inline void init(
	    int* argc, char** argv[], sycl::device& device) {
		detail::runtime::init(argc, argv, device);
	}

	/**
	 * @brief Initializes the Celerity runtime and instructs it to use a particular device.
	 *
	 * @param device_selector The device selector to be used on the current node. This can vary between nodes.
	 */
	inline void init(int* argc, char** argv[], const detail::device_selector& device_selector) { detail::runtime::init(argc, argv, device_selector); }
} // namespace runtime
} // namespace celerity

// Celerity includes <CL/sycl.hpp> internally, but we want to expose the SYCL 2020 ::sycl namespace to Celerity users.
// TODO: Remove this once Celerity includes <sycl/sycl.hpp> internally.
#include <sycl/sycl.hpp>

#endif
