#pragma once

#include <atomic>
#include <list>
#include <mutex>
#include <string>
#include <ostream>

#include "allscale/utils/printer/arrays.h"
#include "allscale/api/core/impl/reference/lock.h"

namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace reference {


	template<typename T, size_t Capacity>
	class BoundQueue {

	public:

		static const size_t capacity = Capacity;

	private:

		using guard = std::lock_guard<SpinLock>;

		static const size_t buffer_size = capacity + 1;

		mutable SpinLock lock;

		std::array<T,buffer_size> data;

		size_t front;
		size_t back;

	public:

		BoundQueue() : lock(), front(0), back(0) {
			for(auto& cur : data) cur = T();
		}

		bool empty() const {
			return front == back;
		}
		bool full() const {
			return ((back + 1) % buffer_size) == front;
		}

		bool push_front(const T& t) {
			guard g(lock);
			if (full()) {
				return false;
			}
			front = (front - 1 + buffer_size) % buffer_size;
			data[front] = t;
			return true;
		}

		bool push_back(const T& t) {
			guard g(lock);
			if (full()) {
				return false;
			}
			data[back] = t;
			back = (back + 1) % buffer_size;
			return true;
		}

	private:

		T pop_front_internal() {
			if (empty()) {
				return T();
			}
			T res(std::move(data[front]));
			front = (front + 1) % buffer_size;
			return res;
		}

		T pop_back_internal() {
			if (empty()) {
				return T();
			}
			back = (back - 1 + buffer_size) % buffer_size;
			T res(std::move(data[back]));
			return res;
		}

	public:

		T pop_front() {
			guard g(lock);
			return pop_front_internal();
		}

		T try_pop_front() {
			if (!lock.try_lock()) {
				return {};
			}
			const T& res = pop_front_internal();
			lock.unlock();
			return res;
		}

		T pop_back() {
			guard g(lock);
			return pop_back_internal();
		}

		T try_pop_back() {
			if (!lock.try_lock()) {
				return {};
			}
			const T& res = pop_back_internal();
			lock.unlock();
			return res;
		}

		size_t size() const {
			guard g(lock);
			return (back >= front) ? (back - front) : (buffer_size - (front - back));
		}

		std::vector<T> getSnapshot() const {
			std::vector<T> res;
			guard g(lock);
			size_t i = front;
			while(i != back) {
				res.push_back(data[i]);
				i += (i + 1) % buffer_size;
			}
			return res;
		}

		friend std::ostream& operator<<(std::ostream& out, const BoundQueue& queue) {
			guard g(queue.lock);
			return out << "[" << queue.data << "," << queue.front << " - " << queue.back << "]";
		}

	};



	template<typename T>
	class UnboundQueue {

		using guard = std::lock_guard<SpinLock>;

		mutable SpinLock lock;

		std::list<T> data;

		std::atomic<std::size_t> num_entries;

	public:

		UnboundQueue() : lock(), num_entries(0) {}

		void push_front(const T& t) {
			guard g(lock);
			data.push_front(t);
			++num_entries;
		}

		void push_back(const T& t) {
			guard g(lock);
			data.push_back(t);
			++num_entries;
		}

	private:

		T pop_front_internal() {
			if (data.empty()) {
				return T();
			}
			T res(std::move(data.front()));
			data.pop_front();
			--num_entries;
			return res;
		}

		T pop_back_internal() {
			if (data.empty()) {
				return T();
			}
			T res(std::move(data.back()));
			data.pop_back();
			--num_entries;
			return res;
		}

	public:

		T pop_front() {
			guard g(lock);
			return pop_front_internal();
		}

		T try_pop_front() {
			if (!lock.try_lock()) {
				return {};
			}
			const T& res = pop_front_internal();
			lock.unlock();
			return res;
		}

		T pop_back() {
			guard g(lock);
			return pop_back_internal();
		}

		T try_pop_back() {
			if (!lock.try_lock()) {
				return {};
			}
			const T& res = pop_back_internal();
			lock.unlock();
			return res;
		}

		bool empty() const {
			return num_entries == 0;
		}

		size_t size() const {
			return num_entries;
		}

		std::vector<T> getSnapshot() const {
			guard g(lock);
			return std::vector<T>(data.begin(),data.end());
		}

	};


	template<typename T>
	class OptimisticUnboundQueue {

		mutable OptimisticReadWriteLock lock;

		std::list<T> data;

		std::atomic<std::size_t> num_entries;

	public:

		OptimisticUnboundQueue() : lock(), num_entries(0) {}

		void push_front(const T& t) {
			lock.start_write();
			data.push_front(t);
			++num_entries;
			lock.end_write();
		}

		void push_back(const T& t) {
			lock.start_write();
			data.push_back(t);
			++num_entries;
			lock.end_write();
		}

	private:

		template<bool tryOnlyOnce>
		T pop_front_internal() {
			// manual tail-recursion optimization since
			// debug builds may fail to do so
			while(true) {

				// start with a read permit
				auto lease = lock.start_read();

				// check whether it is empty
				if (data.empty()) {
					return T();
				}

				// to retrieve data, upgrade to a write
				if (!lock.try_upgrade_to_write(lease)) {
					// if upgrade failed, restart procedure if requested
					if (tryOnlyOnce) return T();
					continue;	// start over again
				}

				// now this one has write access (exclusive)
				T res(std::move(data.front()));
				data.pop_front();
				--num_entries;

				// write is complete
				lock.end_write();

				// done
				return res;

			}
		}

		template<bool tryOnlyOnce>
		T pop_back_internal() {
			// manual tail-recursion optimization since
			// debug builds may fail to do so
			while(true) {

				// start with a read permit
				auto lease = lock.start_read();

				// check whether it is empty
				if (data.empty()) {
					return T();
				}

				// to retrieve data, upgrade to a write
				if (!lock.try_upgrade_to_write(lease)) {
					// if upgrade failed, restart procedure if requested
					if (tryOnlyOnce) return T();
					continue;	// start over again
				}

				// now this one has write access (exclusive)
				T res(std::move(data.back()));
				data.pop_back();
				--num_entries;

				// write is complete
				lock.end_write();

				// done
				return res;
			}
		}

	public:

		T pop_front() {
			return pop_front_internal<false>();
		}

		T try_pop_front() {
			return pop_front_internal<true>();
		}

		T pop_back() {
			return pop_back_internal<false>();
		}

		T try_pop_back() {
			return pop_back_internal<true>();
		}

		bool empty() const {
			return num_entries == 0;
		}

		size_t size() const {
			return num_entries;
		}

		std::vector<T> getSnapshot() const {
			lock.start_write();
			std::vector<T> res(data.begin(),data.end());
			lock.end_write();
			return res;
		}

	};


} // end namespace reference
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale
