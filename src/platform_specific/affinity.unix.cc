#include <cassert>
#include <cstdint>

#include <pthread.h>
#include <sched.h>

#include "affinity.h"

namespace celerity {
namespace detail {

	uint32_t affinity_cores_available() {
		cpu_set_t available_cores;
		[[maybe_unused]] const auto ret = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &available_cores);
		assert(ret == 0 && "Error retrieving affinity mask.");
		return CPU_COUNT(&available_cores);
	}

} // namespace detail
} // namespace celerity
