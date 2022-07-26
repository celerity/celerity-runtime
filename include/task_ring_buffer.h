#pragma once

#include <array>
#include <atomic>
#include <memory>

#include "log.h"
#include "task.h"
#include "types.h"

namespace celerity::detail {

constexpr unsigned long task_ringbuffer_size = 1024;

class task_ring_buffer {
	friend struct task_ring_buffer_testspy;

  public:
	// This is an RAII type for ensuring correct handling of task id reservations
	// in the presence of exceptions (i.e. revoking the reservation on stack unwinding)
	class reservation {
		friend class task_ring_buffer;

	  public:
		reservation(task_id tid, task_ring_buffer& buffer) : m_tid(tid), m_buffer(buffer) {}
		~reservation() {
			if(!m_consumed) {
				CELERITY_WARN("Consumed reservation for tid {} in destructor", m_tid);
				m_buffer.revoke_reservation(std::move(*this));
			}
		}
		reservation(const reservation&) = delete;            // non copyable
		reservation& operator=(const reservation&) = delete; // non assignable
		reservation(reservation&&) = default;                // movable

		task_id get_tid() const { return m_tid; }

	  private:
		void consume() {
			assert(m_consumed == false);
			m_consumed = true;
		}

		bool m_consumed = false;
		task_id m_tid;
		task_ring_buffer& m_buffer;
	};

	bool has_task(task_id tid) const {
		return tid >= m_number_of_deleted_tasks.load(std::memory_order_relaxed) // best effort, only reliable from application thread
		       && tid < m_next_active_tid.load(std::memory_order_acquire);      // synchronizes access to data with put(...)
	}

	size_t get_total_task_count() const { return m_next_active_tid.load(std::memory_order_relaxed); }

	task* find_task(task_id tid) const { return has_task(tid) ? m_data[tid % task_ringbuffer_size].get() : nullptr; }

	task* get_task(task_id tid) const {
		assert(has_task(tid));
		return m_data[tid % task_ringbuffer_size].get();
	}

	// all member functions beyond this point may *only* be called by the main application thread

	size_t get_current_task_count() const { //
		return m_next_active_tid.load(std::memory_order_relaxed) - m_number_of_deleted_tasks.load(std::memory_order_relaxed);
	}

	// the task id passed to the wait callback identifies the lowest in-use TID that the ring buffer is aware of
	using wait_callback = std::function<void(task_id)>;

	reservation reserve_task_entry(const wait_callback& wc) {
		wait_for_available_slot(wc);
		reservation ret(m_next_task_id, *this);
		m_next_task_id++;
		return ret;
	}

	void revoke_reservation(reservation&& reserve) {
		reserve.consume();
		assert(reserve.m_tid == m_next_task_id - 1); // this is the only allowed (and extant) pattern
		m_next_task_id--;
	}

	void put(reservation&& reserve, std::unique_ptr<task> task) {
		reserve.consume();
		assert(m_next_active_tid.load(std::memory_order_relaxed) == reserve.m_tid);
		m_data[reserve.m_tid % task_ringbuffer_size] = std::move(task);
		m_next_active_tid.store(reserve.m_tid + 1, std::memory_order_release);
	}

	void delete_up_to(task_id target_tid) {
		assert(target_tid >= m_number_of_deleted_tasks.load(std::memory_order_relaxed));
		for(task_id tid = m_number_of_deleted_tasks.load(std::memory_order_relaxed); tid < target_tid; ++tid) {
			m_data[tid % task_ringbuffer_size].reset();
		}
		m_number_of_deleted_tasks.store(target_tid, std::memory_order_relaxed);
	}

	void clear() {
		for(auto&& d : m_data) {
			d.reset();
		}
		m_number_of_deleted_tasks.store(m_next_task_id, std::memory_order_relaxed);
	}

	class task_buffer_iterator {
		unsigned long m_id;
		const task_ring_buffer& m_buffer;

	  public:
		task_buffer_iterator(unsigned long id, const task_ring_buffer& buffer) : m_id(id), m_buffer(buffer) {}
		task* operator*() { return m_buffer.get_task(m_id); }
		void operator++() { m_id++; }
		bool operator<(task_buffer_iterator other) { return m_id < other.m_id; }
		bool operator!=(task_buffer_iterator other) { return &m_buffer != &other.m_buffer || m_id != other.m_id; }
	};

	task_buffer_iterator begin() const { //
		return task_buffer_iterator(m_number_of_deleted_tasks.load(std::memory_order_relaxed), *this);
	}
	task_buffer_iterator end() const { return task_buffer_iterator(m_next_task_id, *this); }

  private:
	// the id of the next task that will be reserved
	task_id m_next_task_id = 0;
	// the next task id that will actually be emplaced
	std::atomic<task_id> m_next_active_tid = task_id(0);
	// the number of deleted tasks (which is implicitly the start of the active range of the ringbuffer)
	std::atomic<size_t> m_number_of_deleted_tasks = 0;
	std::array<std::unique_ptr<task>, task_ringbuffer_size> m_data;

	void wait_for_available_slot(const wait_callback& wc) const {
		if(m_next_task_id - m_number_of_deleted_tasks.load(std::memory_order_relaxed) >= task_ringbuffer_size) {
			wc(static_cast<task_id>(m_number_of_deleted_tasks.load(std::memory_order_relaxed)));
		}
	}
};

} // namespace celerity::detail
