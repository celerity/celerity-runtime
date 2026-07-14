// The Windows thread-pinning logic is identical to the POSIX one in affinity.unix.cc. affinity_win32_adapter.h/.cc
// reimplement the POSIX cpu_set_t / pthread_t / CPU_*() / sched_*affinity() / pthread_*affinity_np() surface on top
// of the Win32 GROUP_AFFINITY API, so rather than duplicating affinity.unix.cc we just pull it in here: its own
// _WIN32 branch picks up the adapter header instead of <pthread.h>/<sched.h>, and the shared logic below that
// compiles unchanged.
#include "platform_specific/affinity_win32_adapter.h"

#include "affinity.unix.cc"
