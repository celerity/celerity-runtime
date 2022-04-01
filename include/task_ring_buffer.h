#pragma once

#include <array>
#include <atomic>
#include <memory>

#include "task.h"
#include "types.h"

namespace celerity::detail {

constexpr unsigned long task_ringbuffer_size = 1024;

template <unsigned long N>
class task_ring_buffer {
  public:
	size_t get_total_task_count() const { return next_active_tid.load(); }
	size_t get_current_task_count() const { return next_active_tid.load() - number_of_deleted_tasks; }

	bool has_task(task_id tid) const {
		return tid >= number_of_deleted_tasks && tid < next_active_tid.load(); //
	}

	task* find_task(task_id tid) const { return has_task(tid) ? data[tid % N].get() : nullptr; }

	task* get_task(task_id tid) const {
		assert(has_task(tid));
		return data[tid % N].get();
	}

	task_id reserve_new_tid() {
		wait_for_available_slot();
		auto ret = next_task_id;
		next_task_id++;
		return ret;
	}

	// task_id must have been reserved previously
	void emplace(task_id tid, std::unique_ptr<task> task) {
		task_id expected_tid = tid;
		bool successfully_updated = next_active_tid.compare_exchange_strong(expected_tid, next_active_tid.load() + 1);
		assert(successfully_updated); // this is the only allowed (and extant) pattern
		data[tid % N] = std::move(task);
	}

	// may only be called by one thread
	void delete_up_to(task_id target_tid) {
		for(task_id tid = number_of_deleted_tasks.load(); tid < target_tid; ++tid) {
			data[tid % N].reset();
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
	std::array<std::unique_ptr<task>, N> data;

	void wait_for_available_slot() const {
		while(next_task_id - number_of_deleted_tasks >= N)
			; // busy wait until we have available slots
	}
};

} // namespace celerity::detail
