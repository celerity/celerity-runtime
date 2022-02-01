#include <cassert>
#include <cstdint>

#include <pthread.h>
#include <sched.h>

#include "affinity.h"

namespace celerity {
namespace detail {

	uint32_t affinity_cores_available() {
		cpu_set_t available_cores;
		const auto affinity_error = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &available_cores);
		(void)affinity_error;
		assert(affinity_error == 0 && "Error retrieving affinity mask.");
		return CPU_COUNT(&available_cores);
	}

} // namespace detail
} // namespace celerity