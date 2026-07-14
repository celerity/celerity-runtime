#pragma once
// We need this ifdef to avoid false positives from clang-tidy CI, which doesn't know that this file is only included on Windows.
#ifdef _WIN32

// This file provides an adaptation layer that lets the Celerity runtime use the Windows API for thread affinity through the POSIX-like
// cpu_set_t/sched_*affinity/pthread_*affinity_np interface expected by the rest of the runtime. cpu_set_t covers up to 1024 logical processors,
// spanning multiple Windows processor groups (up to 16 groups of 64 processors each).
//
// Note: Windows' GROUP_AFFINITY API can only bind a thread to a single processor group at a time. If a cpu_set_t's bits span more than one
// group, cpuset_to_group_affinity() cannot represent that as one affinity mask; it logs a warning and fails in that case.
// TODO: If we ever want to support pinning to multiple groups, we would need to find a way around this limitation.

#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

// Prevent Windows headers from polluting global namespace with min/max macros.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include "log.h"

using pthread_t = DWORD;

struct cpu_set_t {
	static constexpr unsigned CPU_SETSIZE = 1024;
	static constexpr unsigned WORDS = (CPU_SETSIZE + 63) / 64;

	uint64_t bits[WORDS] = {};
};

inline constexpr unsigned CPU_SETSIZE = cpu_set_t::CPU_SETSIZE;

void CPU_ZERO(cpu_set_t* set);
void CPU_SET(unsigned cpu, cpu_set_t* set);
void CPU_CLR(unsigned cpu, cpu_set_t* set);
int CPU_ISSET(unsigned cpu, const cpu_set_t* set);

int CPU_COUNT(const cpu_set_t* set);
int CPU_EQUAL(const cpu_set_t* a, const cpu_set_t* b);

int sched_getaffinity(int pid, size_t cpusetsize, cpu_set_t* mask);
int sched_setaffinity(int pid, size_t cpusetsize, const cpu_set_t* mask);

pthread_t pthread_self();

int pthread_setaffinity_np(pthread_t thread, size_t cpusetsize, const cpu_set_t* mask);
int pthread_getaffinity_np(pthread_t thread, size_t cpusetsize, cpu_set_t* mask);

namespace win32_pthread_detail {

static constexpr unsigned PROCS_PER_GROUP = 64;

struct cpu_topology_entry {
	WORD group;
	WORD count;
};

using cpu_topology = std::vector<cpu_topology_entry>;

struct windows_topology_policy {
	static WORD get_group_count() { return GetActiveProcessorGroupCount(); }

	static unsigned get_proc_count(WORD g) { return GetActiveProcessorCount(g); }
};

using topology_policy = windows_topology_policy;
template <typename Policy = topology_policy>
cpu_topology get_cpu_topology() {
	cpu_topology topo;

	const WORD groups = Policy::get_group_count();

	for(WORD g = 0; g < groups; ++g) {
		const unsigned count = Policy::get_proc_count(g);
		topo.push_back({g, (WORD)count});
	}

	return topo;
}

template <typename Policy = windows_topology_policy>
bool cpuset_to_group_affinity(const cpu_set_t* set, GROUP_AFFINITY& out) {
	const WORD groups = Policy::get_group_count();

	bool found = false;

	for(WORD g = 0; g < groups; ++g) {
		const unsigned procs = Policy::get_proc_count(g);

		KAFFINITY mask = 0;

		for(unsigned p = 0; p < procs && p < PROCS_PER_GROUP; ++p) {
			const unsigned global_cpu = g * PROCS_PER_GROUP + p;

			if(CPU_ISSET(global_cpu, set)) { mask |= (KAFFINITY(1) << p); }
		}

		if(mask == 0) continue;

		if(found) {
			CELERITY_WARN("Affinity mask spans multiple processor groups (not supported on Windows).");
			return false;
		}

		found = true;
		out.Group = g;
		out.Mask = mask;
	}

	return found;
}

} // namespace win32_pthread_detail
#endif