#if CELERITY_ENABLE_TRACY

#include "tracy.h"

#include <mutex>
#include <set>

namespace celerity::detail {

struct tracy_async_fiber_index_less {
	bool operator()(const tracy_async_fiber* const lhs, const tracy_async_fiber* const rhs) const { return lhs->index < rhs->index; }
};

// we intentionally leak all heap allocations to avoid problems with static-destruction order

struct tracy_thread {
	// std::set: always acquire the free lane with the lowest index
	std::set<tracy_async_fiber*, tracy_async_fiber_index_less> free_lanes;
	size_t next_lane_index = 0;
};

using tracy_thread_map = std::unordered_map<std::string_view, tracy_thread*>;

tracy_thread* tracy_get_thread(const char* const thread_name) {
	static std::mutex mutex;
	static tracy_thread_map* threads;

	std::lock_guard<std::mutex> lock(mutex);
	if(threads == nullptr) { threads = new tracy_thread_map(); }
	if(const auto it = threads->find(thread_name); it != threads->end()) { return it->second; }
	return threads->emplace(thread_name, new tracy_thread()).first->second;
}

tracy_async_lane tracy_acquire_lane(const char* const thread_name) {
	const auto thread = tracy_get_thread(thread_name);
	if(!thread->free_lanes.empty()) {
		const auto first = thread->free_lanes.begin();
		const auto lane = *first;
		thread->free_lanes.erase(first);
		return lane;
	} else {
		return new tracy_async_fiber(thread_name, thread->next_lane_index++);
	}
}

void tracy_release_lane(const tracy_async_lane lane) {
	const auto thread = tracy_get_thread(lane->thread_name);
	thread->free_lanes.insert(lane);
}

} // namespace celerity::detail

#endif
