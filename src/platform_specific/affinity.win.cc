#include <cassert>

#include "affinity.h"
#include "log.h"

namespace celerity::detail::thread_pinning {
thread_pinner::thread_pinner(const runtime_configuration& cfg) {
	if(cfg.enabled) { CELERITY_WARN("Thread pinning is currently not supported on Windows."); }
}
thread_pinner::~thread_pinner() {}
} // namespace celerity::detail::thread_pinning
