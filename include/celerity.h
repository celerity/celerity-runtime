#ifndef RUNTIME_INCLUDE_ENTRY_CELERITY
#define RUNTIME_INCLUDE_ENTRY_CELERITY

#include "runtime.h"

#include "buffer.h"
#include "distr_queue.h"
#include "user_bench.h"

namespace celerity {
namespace runtime {
	inline void init(int* argc, char** argv[]) { detail::runtime::init(argc, argv); }
} // namespace runtime
} // namespace celerity

#endif
