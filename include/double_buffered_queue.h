#pragma once

#include <atomic>
#include <mutex>
#include <vector>

namespace celerity::detail {

// std::hardware_destructive_interference_size is not implemented in Clang, and GCC makes this dependent on -mtune - so we just redefine it ourselves.
constexpr size_t hardware_destructive_interference_size = 128;

template <typename T>
class alignas(hardware_destructive_interference_size /* avoid false sharing on atomic flag */) double_buffered_queue {
  public:
	// TODO use this instead of push_back() between iggen and executor
	template <typename ForwardIterator>
	void insert_end(ForwardIterator begin, ForwardIterator end) {
		std::lock_guard lock(m_mutex);
		m_queue.insert(m_queue.end(), begin, end);
		m_queue_nonempty.store(true, std::memory_order_relaxed);
	}

	void push_back(T v) {
		std::lock_guard lock(m_mutex);
		m_queue.push_back(std::move(v));
		m_queue_nonempty.store(true, std::memory_order_relaxed);
	}

	// TODO consider having the swap-target as a member of this class
	bool swap_if_nonempty(std::vector<T>& other) {
		if(m_queue_nonempty.load(std::memory_order_relaxed) /* opportunistic, might race */) {
			std::lock_guard lock(m_mutex);
			if(m_queue_nonempty.load(std::memory_order_relaxed) /* synchronized by m_mutex */) {
				swap(m_queue, other);
				m_queue_nonempty.store(false, std::memory_order_relaxed);
				return true;
			}
		}
		return false;
	}

  private:
	std::mutex m_mutex;
	std::vector<T> m_queue;
	std::atomic<bool> m_queue_nonempty{false};
};

} // namespace celerity::detail