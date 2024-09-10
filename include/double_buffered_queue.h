#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "system_info.h"

namespace celerity::detail {

/// (Thread-safe) multi-producer single-consumer queue that uses double-buffering to avoid lock contention and keep dequeueing latency as low as possible.
template <typename T>
class double_buffered_queue {
  public:
	/// Push a single element to the queue. Instead of frequently pushing multiple elements, consider using a vector<T> as the element type.
	void push(T v) {
		{
			std::lock_guard lock(m_write.mutex);
			// This push might allocate, which is the reason why double_buffered_queue retains ownership of both queues in order to re-use allocated memory and
			// keep the lock duration as short as possible.
			m_write.queue.push_back(std::move(v));
			// Notify the reader that it is worth taking the lock
			m_write.queue_nonempty.store(true, std::memory_order_relaxed);
		}
		m_write.resume.notify_one();
	}

	/// Returns all elements pushed to the queue since the last `pop_all`. The returned reference is valid until the next call to `pop_all`.
	[[nodiscard]] std::vector<T>& pop_all() {
		m_read.queue.clear();
		if(m_write.queue_nonempty.load(std::memory_order_relaxed) /* opportunistic */) {
			std::lock_guard lock(m_write.mutex);
			swap(m_read.queue, m_write.queue);
			// m_read.queue was cleared before the swap, so m_write.queue is empty now
			m_write.queue_nonempty.store(false, std::memory_order_relaxed);
		}
		return m_read.queue;
	}

	/// After this function returns, the result of `pop_all` is non-empty as long as there is only a single reader thread.
	void wait_while_empty() {
		if(!m_write.queue_nonempty.load(std::memory_order_relaxed) /* opportunistic */) {
			std::unique_lock lock(m_write.mutex);
			m_write.resume.wait(lock, [&] { return m_write.queue_nonempty.load(std::memory_order_relaxed); });
		}
	}

  private:
	/// Aligned group 1: The write-queue and its associated synchronization primitives will move between threads on push, pop, and wait.
	struct alignas(hardware_destructive_interference_size) write_end {
		std::mutex mutex;
		std::condition_variable resume;
		std::vector<T> queue;
		std::atomic<bool> queue_nonempty{false};
	} m_write;

	/// Aligned group 2: Reader thread can continue to concurrently access read_queue even while a writer thread pushes to write_queue.
	struct alignas(hardware_destructive_interference_size) read_end {
		std::vector<T> queue;
	} m_read;
};

} // namespace celerity::detail
