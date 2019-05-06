#ifndef RUNTIME_INCLUDE_ENTRY_CELERITY
#define RUNTIME_INCLUDE_ENTRY_CELERITY

#include "runtime.h"

#include "buffer.h"
#include "distr_queue.h"
#include "user_bench.h"

namespace celerity {
namespace runtime {
	/**
	 * @brief Initializes the Celerity runtime.
	 */
	inline void init(int* argc, char** argv[]) { detail::runtime::init(argc, argv, nullptr); }

	/**
	 * @brief Initializes the Celerity runtime and instructs it to use a particular device.
	 *
	 * @param device The device to be used on the current node. This can vary between nodes.
	 */
	inline void init(int* argc, char** argv[], cl::sycl::device& device) { detail::runtime::init(argc, argv, &device); }
} // namespace runtime
} // namespace celerity

#endif
