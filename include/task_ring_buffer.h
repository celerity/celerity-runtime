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
				buffer.revoke_reservation(*this);
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

	size_t get_total_task_count() const { return next_active_tid.load(); }
	size_t get_current_task_count() const { return next_active_tid.load() - number_of_deleted_tasks; }

	bool has_task(task_id tid) const {
		return tid >= number_of_deleted_tasks && tid < next_active_tid.load(); //
	}

	task* find_task(task_id tid) const { return has_task(tid) ? data[tid % task_ringbuffer_size].get() : nullptr; }

	task* get_task(task_id tid) const {
		assert(has_task(tid));
		return data[tid % task_ringbuffer_size].get();
	}

	reservation reserve_task_entry() {
		wait_for_available_slot();
		reservation ret(next_task_id, *this);
		next_task_id++;
		return ret;
	}

	void revoke_reservation(reservation& reserve) {
		reserve.consume();
		assert(reserve.tid == next_task_id - 1); // this is the only allowed (and extant) pattern
		next_task_id--;
	}

	void put(reservation& reserve, std::unique_ptr<task> task) {
		reserve.consume();
		task_id expected_tid = reserve.tid;
		[[maybe_unused]] bool successfully_updated = next_active_tid.compare_exchange_strong(expected_tid, next_active_tid.load() + 1);
		assert(successfully_updated); // this is the only allowed (and extant) pattern
		data[reserve.tid % task_ringbuffer_size] = std::move(task);
	}

	// may only be called by one thread
	void delete_up_to(task_id target_tid) {
		for(task_id tid = number_of_deleted_tasks.load(); tid < target_tid; ++tid) {
			data[tid % task_ringbuffer_size].reset();
		}
		number_of_deleted_tasks += target_tid - number_of_deleted_tasks.load();
	}

	void clear() {
		for(auto&& d : data) {
			d.reset();
		}
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

	task_buffer_iterator begin() const { return task_buffer_iterator(number_of_deleted_tasks, *this); }
	task_buffer_iterator end() const { return task_buffer_iterator(next_task_id, *this); }

  private:
	// the id of the next task that will be reserved
	task_id next_task_id = 0;
	// the next task id that will actually be emplaced
	std::atomic<task_id> next_active_tid = task_id(0);
	// the number of deleted tasks (which is implicitly the start of the active range of the ringbuffer)
	std::atomic<unsigned long> number_of_deleted_tasks = 0;
	std::array<std::unique_ptr<task>, task_ringbuffer_size> data;

	void wait_for_available_slot() const {
		while(next_task_id - number_of_deleted_tasks >= task_ringbuffer_size)
			std::this_thread::yield(); // busy wait until we have available slots
	}
};

} // namespace celerity::detail
