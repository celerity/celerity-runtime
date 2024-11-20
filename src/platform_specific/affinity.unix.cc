#include <algorithm>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <pthread.h>
#include <sched.h>

#include "affinity.h"
#include "log.h"

namespace {

using namespace celerity::detail::thread_pinning;

std::vector<uint32_t> get_available_sequential_cores(const cpu_set_t& available_cores, const uint32_t count, const uint32_t starting_from_core) {
	std::vector<uint32_t> cores;
	uint32_t current_core = starting_from_core;
	for(uint32_t i = 0; i < count; ++i) {
		// find the next sequential core we may use
		while(CPU_ISSET(current_core, &available_cores) == 0 && current_core < CPU_SETSIZE) {
			current_core++;
		}
		if(current_core >= CPU_SETSIZE) { return {}; }
		cores.push_back(current_core++);
	}
	return cores;
}

struct pinned_thread_state {
	cpu_set_t previous_cpuset = {};
	cpu_set_t pinned_to_cpuset = {};
};

struct thread_pinner_state {
	std::mutex mutex;
	bool initialized = false;
	runtime_configuration config;
	cpu_set_t available_cores = {};
	std::unordered_map<thread_type, uint32_t> thread_pinning_plan;
	std::unordered_map<pthread_t, pinned_thread_state> pinned_threads;
};
thread_pinner_state g_state; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// This helper removes a thread from the pinned_threads map when it ends
struct thread_remover {
	thread_remover() = default;
	thread_remover(const thread_remover&) = delete;
	thread_remover& operator=(const thread_remover&) = delete;
	thread_remover(thread_remover&&) = delete;
	thread_remover& operator=(thread_remover&&) = delete;
	~thread_remover() {
		std::lock_guard lock(g_state.mutex);
		if(g_state.initialized) { g_state.pinned_threads.erase(pthread_self()); }
	}
};
thread_local std::optional<thread_remover> t_remover; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Retrieves the threads that should be pinned according to the given configuration, in order
std::vector<thread_type> get_threads_to_pin(const runtime_configuration& cfg) {
	std::vector<thread_type> threads_to_pin = {thread_type::application, thread_type::scheduler, thread_type::executor};
	if(cfg.use_backend_device_submission_threads) {
		for(uint32_t i = 0; i < cfg.num_devices; ++i) {
			threads_to_pin.push_back(static_cast<thread_type>(thread_type::first_device_submitter + i)); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
		}
	}
	return threads_to_pin;
}

// Gets the list of thread types to be pinned for a given configuration as a string, for debug output
std::string get_threads_to_pin_string(const runtime_configuration& cfg) {
	const auto to_pin = get_threads_to_pin(cfg);
	std::vector<std::string> to_pin_names;
	std::transform(to_pin.begin(), to_pin.end(), std::back_inserter(to_pin_names), thread_type_to_string);
	return fmt::format("[{}]", fmt::join(to_pin_names, ", "));
}

// Initializes the thread pinning machinery
// This captures the current thread's affinity mask and sets the thread pinning machinery up
// Calls to pin_this_thread prior to this call will have no effect
bool initialize(const runtime_configuration& cfg) {
	std::lock_guard lock(g_state.mutex);
	if(g_state.initialized) {
		CELERITY_ERROR("Thread pinning already initialized. Ignoring this initialization attempt.");
		return false;
	}
	assert(g_state.thread_pinning_plan.empty() && "Thread pinning plan not initially empty.");
	assert(g_state.pinned_threads.empty() && "Pinned threads not initially empty.");

	g_state.config = cfg;

	const auto ret = sched_getaffinity(0, sizeof(cpu_set_t), &g_state.available_cores);
	if(ret != 0) {
		CELERITY_WARN("Error retrieving initial process affinity mask. Unable to check whether enough logical cores are available to this process.{}",
		    cfg.enabled ? " Will disable thread pinning." : "");
		g_state.config.enabled = false;
		return true;
	}
	{ // log tracing information about available cores, specifically useful to understand MPI implementation behaviour
		std::string available_cores_str = {};
		for(uint32_t i = 0; i < CPU_SETSIZE; ++i) {
			available_cores_str += CPU_ISSET(i, &g_state.available_cores) ? "1" : "0";
		}
		CELERITY_TRACE("Affinity: Initialized, available cores: {}", available_cores_str);
	}

	// pinned threads per process: application, scheduler, executor, 1 device submitter per device if enabled
	uint32_t pinned_threads_per_process = 3;
	if(g_state.config.use_backend_device_submission_threads) { pinned_threads_per_process += g_state.config.num_devices; }
	// total number of threads to be pinned across processes (legacy mode) - we assume that each process has been assigned the same number of device
	const uint32_t total_threads = pinned_threads_per_process * g_state.config.num_legacy_processes;

	if(g_state.config.enabled) {
		// select the core set to use
		std::vector<uint32_t> selected_core_ids = {};
		if(!cfg.hardcoded_core_ids.empty()) {
			// attempt to use the provided hardcoded IDs if they match the number of threads to be pinned
			if(static_cast<uint32_t>(cfg.hardcoded_core_ids.size()) != total_threads) {
				CELERITY_WARN("Hardcoded core ID count ({}) does not match the number of threads to be pinned ({}), downgrading to auto-pinning.",
				    cfg.hardcoded_core_ids.size(), total_threads);
				CELERITY_WARN("Expected core list for these threads: {}", get_threads_to_pin_string(cfg));
			} else {
				// also check if the provided core IDs are actually available
				if(!std::ranges::all_of(cfg.hardcoded_core_ids, [&](const uint32_t core_id) { return CPU_ISSET(core_id, &g_state.available_cores) != 0; })) {
					CELERITY_WARN("Not all hardcoded core IDs are available, downgrading to auto-pinning.");
				} else {
					// if everything checks out, use the provided core IDs
					selected_core_ids = cfg.hardcoded_core_ids;
				}
			}
		}
		if(selected_core_ids.empty()) {
			// otherwise, sequential core assignments for now; it is most important that each of the threads is "close"
			// to the ones next to it in this sequence, so that communication between them is fast
			selected_core_ids = get_available_sequential_cores(g_state.available_cores, total_threads, cfg.standard_core_start_id);
			if(selected_core_ids.empty()) {
				CELERITY_WARN("Insufficient logical cores available for thread pinning (required {} startig from {}, {} available), disabling pinning."
				              " Performance may be negatively impacted.", //
				    total_threads, cfg.standard_core_start_id, CPU_COUNT(&g_state.available_cores));
			}
		}
		// build our pinning plan based on the selected core list
		if(selected_core_ids.empty()) {
			g_state.config.enabled = false;
		} else {
			uint32_t current_core_id = cfg.legacy_process_index * pinned_threads_per_process;
			const auto threads_to_pin = get_threads_to_pin(cfg);
			for(const auto t_type : threads_to_pin) {
				g_state.thread_pinning_plan.emplace(t_type, selected_core_ids[current_core_id++]);
			}
		}
	} else {
		// when pinning is disabled, still validate that we have enough threads, warn otherwise
		const auto cores_available = CPU_COUNT(&g_state.available_cores);
		if(static_cast<uint32_t>(cores_available) < total_threads) {
			CELERITY_WARN("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {} logical "
			              "cores. Performance may be negatively impacted.",
			    cores_available, total_threads);
		} else {
			CELERITY_DEBUG("Thread pinning is disabled.");
		}
	}

	g_state.initialized = true;
	return true;
}

// Tears down the thread pinning machinery
// This restores the affinity mask of all threads that have been pinned, are still running, and had no other changes to their affinity
void teardown() {
	std::lock_guard lock(g_state.mutex);
	assert(g_state.initialized && "Thread pinning not initialized.");

	if(g_state.config.enabled) {
		for(const auto& [thread, pinned_state] : g_state.pinned_threads) {
			// first check if no one else made changes to the affinity of this thread
			cpu_set_t current_cpuset;
			if(pthread_getaffinity_np(thread, sizeof(cpu_set_t), &current_cpuset) != 0) {
				CELERITY_WARN("Error retrieving thread affinity of thread {} to check for unexpected changes.", thread);
				continue;
			}
			if(CPU_EQUAL(&current_cpuset, &pinned_state.pinned_to_cpuset) == 0) {
				CELERITY_WARN("Thread affinity of thread {} was changed unexpectedly, skipping restoration.", thread);
				continue;
			}
			// now we can safely restore the affinity
			const auto ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &pinned_state.previous_cpuset);
			if(ret != 0) { CELERITY_WARN("Error resetting thread affinity."); }
		}
	}

	g_state.pinned_threads.clear();
	g_state.thread_pinning_plan.clear();
	g_state.config.enabled = false;
	g_state.initialized = false;
}

} // namespace

namespace celerity::detail::thread_pinning {

thread_pinner::thread_pinner(const runtime_configuration& cfg) : m_successfully_initialized(initialize(cfg)) {}
thread_pinner::~thread_pinner() {
	if(m_successfully_initialized) { teardown(); }
}

void pin_this_thread(const thread_type t_type) {
	std::lock_guard lock(g_state.mutex);
	// if thread pinning is not initialized or disabled, do nothing
	if(!g_state.initialized || !g_state.config.enabled) return;

	assert(g_state.thread_pinning_plan.find(t_type) != g_state.thread_pinning_plan.end() && "Trying to pin thread of a type which has no core assigned.");
	assert(g_state.pinned_threads.find(pthread_self()) == g_state.pinned_threads.end() && "Trying to pin a thread which was already pinned.");

	// retrieve current thread affinity for later restoration
	const auto this_thread = pthread_self();
	cpu_set_t previous_cpuset;
	if(pthread_getaffinity_np(this_thread, sizeof(cpu_set_t), &previous_cpuset) != 0) {
		CELERITY_WARN("Error retrieving thread affinity for thread '{}', will not pin it.", thread_type_to_string(t_type));
		return;
	}

	// if the application thread was already pinned (i.e. its affinity mask is different from the process mask), we should not pin it again
	if(t_type == thread_type::application && !CPU_EQUAL(&previous_cpuset, &g_state.available_cores)) {
		CELERITY_WARN("Affinity mask for the application thread was modified, will not pin it.");
		return;
	}

	// set new affinity to designated core
	const auto core = g_state.thread_pinning_plan.at(t_type);
	cpu_set_t new_cpuset;
	CPU_ZERO(&new_cpuset);
	CPU_SET(core, &new_cpuset);
	const auto ret = pthread_setaffinity_np(this_thread, sizeof(cpu_set_t), &new_cpuset);
	if(ret != 0) {
		CELERITY_WARN("Could not set affinity fot thread '{}'.", thread_type_to_string(t_type));
		return;
	}

	// allow the affinity of this thread to be restored on teardown
	{
		const pinned_thread_state ps{
		    .previous_cpuset = previous_cpuset,
		    .pinned_to_cpuset = new_cpuset,
		};
		g_state.pinned_threads.emplace(this_thread, ps);
		if(!t_remover.has_value()) t_remover.emplace();
	}

	CELERITY_DEBUG("Affinity: pinned thread '{}' to core {} (local process #{} thread id {:x})", //
	    thread_type_to_string(t_type), core, g_state.config.legacy_process_index, this_thread);
}

} // namespace celerity::detail::thread_pinning
