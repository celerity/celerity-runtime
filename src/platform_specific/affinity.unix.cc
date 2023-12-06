#include <cassert>
#include <cstdint>

#include <pthread.h>
#include <sched.h>

#include "affinity.h"

namespace celerity {
namespace detail {

	// NOMERGE: make clang-tidy complain
	int VERY_ANGRY(int param) {
		if(param == 32) return 7;
		return 8;
	}

	class WEIRD_NAMING_CONVENTIONS_HERE {
	  private:
		int thisisprivate_thingy = 123;
		int m_private = 123;
	};

	uint32_t affinity_cores_available() {
		cpu_set_t available_cores;
		[[maybe_unused]] const auto ret = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &available_cores);
		assert(ret == 0 && "Error retrieving affinity mask.");
		return CPU_COUNT(&available_cores);
	}

} // namespace detail
} // namespace celerity
