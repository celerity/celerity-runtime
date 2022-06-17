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
		reservation(task_id tid, task_ring_buffer& buffer) : tid(tid), buffer(buffer) {}
		~reservation() {
			if(!consumed) {
				CELERITY_WARN("Consumed reservation for tid {} in destructor", tid);
				buffer.revoke_reservation(std::move(*this));
			}
		}
		reservation(const reservation&) = delete;            // non copyable
		reservation& operator=(const reservation&) = delete; // non assignable
		reservation(reservation&&) = default;                // movable

		task_id get_tid() const { return tid; }

	  private:
		void consume() {
			assert(consumed == false);
			consumed = true;
		}

		bool consumed = false;
		task_id tid;
		task_ring_buffer& buffer;
	};

	bool has_task(task_id tid) const {
		return tid >= number_of_deleted_tasks.load(std::memory_order_relaxed) // best effort, only reliable from application thread
		       && tid < next_active_tid.load(std::memory_order_acquire);      // synchronizes access to data with put(...)
	}

	size_t get_total_task_count() const { return next_active_tid.load(std::memory_order_relaxed); }

	task* find_task(task_id tid) const { return has_task(tid) ? data[tid % task_ringbuffer_size].get() : nullptr; }

	task* get_task(task_id tid) const {
		assert(has_task(tid));
		return data[tid % task_ringbuffer_size].get();
	}

	// all member functions beyond this point may *only* be called by the main application thread

	size_t get_current_task_count() const { //
		return next_active_tid.load(std::memory_order_relaxed) - number_of_deleted_tasks.load(std::memory_order_relaxed);
	}

	// the task id passed to the wait callback identifies the lowest in-use TID that the ring buffer is aware of
	using wait_callback = std::function<void(task_id)>;

	reservation reserve_task_entry(const wait_callback& wc) {
		wait_for_available_slot(wc);
		reservation ret(next_task_id, *this);
		next_task_id++;
		return ret;
	}

	void revoke_reservation(reservation&& reserve) {
		reserve.consume();
		assert(reserve.tid == next_task_id - 1); // this is the only allowed (and extant) pattern
		next_task_id--;
	}

	void put(reservation&& reserve, std::unique_ptr<task> task) {
		reserve.consume();
		assert(next_active_tid.load(std::memory_order_relaxed) == reserve.tid);
		data[reserve.tid % task_ringbuffer_size] = std::move(task);
		next_active_tid.store(reserve.tid + 1, std::memory_order_release);
	}

	void delete_up_to(task_id target_tid) {
		assert(target_tid >= number_of_deleted_tasks.load(std::memory_order_relaxed));
		for(task_id tid = number_of_deleted_tasks.load(std::memory_order_relaxed); tid < target_tid; ++tid) {
			data[tid % task_ringbuffer_size].reset();
		}
		number_of_deleted_tasks.store(target_tid, std::memory_order_relaxed);
	}

	void clear() {
		for(auto&& d : data) {
			d.reset();
		}
		number_of_deleted_tasks.store(next_task_id, std::memory_order_relaxed);
	}

	class task_buffer_iterator {
		unsigned long id;
		const task_ring_buffer& buffer;

	  public:
		task_buffer_iterator(unsigned long id, const task_ring_buffer& buffer) : id(id), buffer(buffer) {}
		task* operator*() { return buffer.get_task(id); }
		void operator++() { id++; }
		bool operator<(task_buffer_iterator other) { return id < other.id; }
		bool operator!=(task_buffer_iterator other) { return &buffer != &other.buffer || id != other.id; }
	};

	task_buffer_iterator begin() const { //
		return task_buffer_iterator(number_of_deleted_tasks.load(std::memory_order_relaxed), *this);
	}
	task_buffer_iterator end() const { return task_buffer_iterator(next_task_id, *this); }

  private:
	// the id of the next task that will be reserved
	task_id next_task_id = 0;
	// the next task id that will actually be emplaced
	std::atomic<task_id> next_active_tid = task_id(0);
	// the number of deleted tasks (which is implicitly the start of the active range of the ringbuffer)
	std::atomic<size_t> number_of_deleted_tasks = 0;
	std::array<std::unique_ptr<task>, task_ringbuffer_size> data;

	void wait_for_available_slot(const wait_callback& wc) const {
		if(next_task_id - number_of_deleted_tasks.load(std::memory_order_relaxed) >= task_ringbuffer_size) {
			wc(static_cast<task_id>(number_of_deleted_tasks.load(std::memory_order_relaxed)));
		}
	}
};

} // namespace celerity::detail
