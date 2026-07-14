#ifdef _WIN32
#include "platform_specific/affinity_win32_adapter.h"

#include <algorithm>
#include <numeric>

#include "log.h"

void CPU_ZERO(cpu_set_t* set) { std::memset(set->bits, 0, sizeof(set->bits)); }

void CPU_SET(unsigned cpu, cpu_set_t* set) {
	if(cpu < cpu_set_t::CPU_SETSIZE) set->bits[cpu / 64] |= (uint64_t(1) << (cpu % 64));
}

void CPU_CLR(unsigned cpu, cpu_set_t* set) {
	if(cpu < cpu_set_t::CPU_SETSIZE) set->bits[cpu / 64] &= ~(uint64_t(1) << (cpu % 64));
}

int CPU_ISSET(unsigned cpu, const cpu_set_t* set) {
	if(cpu >= cpu_set_t::CPU_SETSIZE) return 0;
	return (set->bits[cpu / 64] & (uint64_t(1) << (cpu % 64))) != 0;
}

int CPU_COUNT(const cpu_set_t* set) {
	int n = 0;
	for(unsigned i = 0; i < cpu_set_t::WORDS; ++i)
		n += std::popcount(set->bits[i]);
	return n;
}

int CPU_EQUAL(const cpu_set_t* a, const cpu_set_t* b) { return std::memcmp(a->bits, b->bits, sizeof(a->bits)) == 0; }


namespace win32_pthread_detail {} // namespace win32_pthread_detail

int sched_getaffinity(int, size_t, cpu_set_t* mask) {
	GROUP_AFFINITY ga{};

	if(!GetThreadGroupAffinity(GetCurrentThread(), &ga)) return -1;

	CPU_ZERO(mask);

	const unsigned base = ga.Group * win32_pthread_detail::PROCS_PER_GROUP;

	for(unsigned p = 0; p < win32_pthread_detail::PROCS_PER_GROUP; ++p) {
		if(ga.Mask & (KAFFINITY(1) << p)) CPU_SET(base + p, mask);
	}

	return 0;
}

int sched_setaffinity(int, size_t, const cpu_set_t* mask) {
	GROUP_AFFINITY ga{};

	if(!win32_pthread_detail::cpuset_to_group_affinity(mask, ga)) {
		CELERITY_WARN("sched_setaffinity failed (multi-group mask rejected)");
		return -1;
	}

	return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) ? 0 : -1;
}

pthread_t pthread_self() { return GetCurrentThreadId(); }

int pthread_setaffinity_np(pthread_t thread, size_t, const cpu_set_t* mask) {
	const bool self = (thread == GetCurrentThreadId());

	HANDLE h = self ? GetCurrentThread() : OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, thread);

	if(!h) return -1;

	GROUP_AFFINITY ga{};

	if(!win32_pthread_detail::cpuset_to_group_affinity(mask, ga)) {
		if(!self) CloseHandle(h);
		return -1;
	}

	int ok = SetThreadGroupAffinity(h, &ga, nullptr) ? 0 : -1;

	if(!self) CloseHandle(h);
	return ok;
}

int pthread_getaffinity_np(pthread_t thread, size_t, cpu_set_t* mask) {
	const bool self = (thread == GetCurrentThreadId());

	HANDLE h = self ? GetCurrentThread() : OpenThread(THREAD_QUERY_INFORMATION, FALSE, thread);

	if(!h) return -1;

	GROUP_AFFINITY ga{};

	if(!GetThreadGroupAffinity(h, &ga)) {
		if(!self) CloseHandle(h);
		return -1;
	}

	CPU_ZERO(mask);

	const unsigned base = ga.Group * win32_pthread_detail::PROCS_PER_GROUP;

	for(unsigned p = 0; p < win32_pthread_detail::PROCS_PER_GROUP; ++p) {
		if(ga.Mask & (KAFFINITY(1) << p)) CPU_SET(base + p, mask);
	}

	if(!self) CloseHandle(h);
	return 0;
}
#endif
