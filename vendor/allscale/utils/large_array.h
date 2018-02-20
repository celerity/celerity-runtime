#pragma once

#ifndef _MSC_VER
	#include <sys/mman.h>
	#include <unistd.h>
#else
	#include<stdlib.h>
	#include<malloc.h>

#endif

#include <cstddef>

#include <algorithm>
#include <type_traits>

#include "allscale/utils/assert.h"

#include "allscale/utils/printer/vectors.h"

namespace allscale {
namespace utils {


	namespace detail {

		/**
		 * Intervals are utilized by the LargeArray class to manage active intervals -- those intervals
		 * for which the stored values need to be preserved.
		 */
		class Intervals {

			/**
			 * A list of start/end values of the covered intervals.
			 * For instance, the values [10,15,18,35] correspond to the
			 * intervals [10,..,15) and [18,..,35). The intervals are sorted.
			 * The lower boundary is included, the upper boundary not.
			 */
			std::vector<std::size_t> data;

		public:

			/**
			 * A factory function creating a list of intervals consisting of a single,
			 * closed range [begin,end).
			 */
			static Intervals fromRange(std::size_t begin, std::size_t end) {
				Intervals res;
				res.add(begin,end);
				return res;
			}

			/**
			 * Compares this and the given intervals for equality.
			 */
			bool operator==(const Intervals& other) const {
				return data == other.data;
			}

			/**
			 * Compares this and the given intervals for inequality.
			 */
			bool operator!=(const Intervals& other) const {
				return data != other.data;
			}

			/**
			 * Checks whether this is interval is empty.
			 */
			bool empty() const {
				return data.empty();
			}

			/**
			 * Adds a new interval to the covered intervals.
			 * @param from the start (inclusive) of the interval to be added
			 * @param to the end (exclusive) of the interval to be added
			 */
			void add(std::size_t from, std::size_t to) {

				// skip empty ranges
				if (from >= to) return;

				// insert first element
				if (data.empty()) {
					data.push_back(from);
					data.push_back(to);
				}

				// find positions for from and to
				auto it_begin = data.begin();
				auto it_end = data.end();

				auto it_from = std::upper_bound(it_begin, it_end, from);
				auto it_to = std::upper_bound(it_begin, it_end, to-1);

				std::size_t idx_from = std::distance(it_begin,it_from);
				std::size_t idx_to = std::distance(it_begin,it_to);

				// whether insertion is at a common place
				if (it_from == it_to) {

					// if it is between ranges ...
					if (idx_to % 2 == 0) {

						// check whether it is a gap closing a range
						if (idx_to > 1 && idx_to < data.size() && data[idx_to-1] == from && data[idx_to] == to) {
							data.erase(it_from-1,it_to+1);
							return;
						}

						// check whether it is connecting to the one on the left
						if (idx_to > 1 && data[idx_to-1] == from) {
							data[idx_to-1] = to;
							return;
						}

						// check whether it is connecting to the one on the right
						if (idx_to < data.size() && data[idx_to] == to) {
							data[idx_to] = from;
							return;
						}
					}

					// check whether it is the end
					if (it_from == it_end) {
						data.push_back(from);
						data.push_back(to);
						return;
					}

					// check whether it is within an interval
					if ((idx_from % 2) == 1) {
						return;		// nothing to add
					}

					// insert new pair at insertion position
					data.insert(it_from,2,from);
					data[idx_from+1] = to;

					return;
				}

				// if from references an existing start value => correct it
				if (idx_from % 2 == 0) {
					data[idx_from] = from;
					++it_from;
				} else {
					// all fine
				}

				// correct end of last closed interval
				if (idx_to % 2 == 0) {
					data[idx_to-1] = to;
					it_to -= 1;
				} else {
					// nothing to do here
				}

				if (it_from < it_to) data.erase(it_from,it_to);

			}

			/**
			 * Removes the given interval from the covered range.
			 * @param from the start (inclusive) of the interval to be removed
			 * @param to the end (exclusive) of the interval to be removed
			 */
			void remove(std::size_t from, std::size_t to) {

				// quick exits
				if (from >= to) return;
				if (data.empty()) return;

				// find positions for from and to
				auto it_begin = data.begin();
				auto it_end = data.end();

				auto it_from = std::upper_bound(it_begin, it_end, from);
				auto it_to = std::upper_bound(it_begin, it_end, to-1);

				std::size_t idx_from = std::distance(it_begin,it_from);
				std::size_t idx_to = std::distance(it_begin,it_to);

				// in case they are both at the same spot
				if (idx_from == idx_to) {

					// if it is between two intervals ..
					if (idx_from % 2 == 0) return;		// .. there is nothing to delete

					// it is within a single interval
					assert_eq(1, idx_from % 2);

					// check whether full interval is covered
					if (data[idx_from-1] == from && data[idx_to] == to) {
						data.erase(it_from-1,it_to+1);
						return;
					}

					// check if lower boundary matches
					if (data[idx_from-1] == from) {
						data[idx_from-1] = to;
						return;
					}

					// check if lower boundary matches
					if (data[idx_to] == to) {
						data[idx_to] = from;
						return;
					}

					data.insert(it_from,2,from);
					data[idx_from+1] = to;
					return;

				}

				if (idx_from % 2 == 1) {
					data[idx_from] = from;
					it_from++;
				}

				if (idx_to % 2 == 1) {
					data[idx_to-1] = to;
					it_to--;
				}

				// delete nodes in-between
				data.erase(it_from,it_to);
				return;

			}

			/**
			 * Removes the given intervals from the covered range.
			 * @param other the intervals to be removed
			 */
			void remove(const Intervals& other) {
				// iteratively remove the elements of the given interval
				for(std::size_t i =0; i<other.data.size(); i+=2) {
					remove(other.data[i],other.data[i+1]);
				}
			}

			/**
			 * Inverts the range covered by this instance.
			 */
			void invert() {

				// add new start and end value
				data.insert(data.begin(), std::numeric_limits<std::size_t>::min());
				data.insert(data.end(), std::numeric_limits<std::size_t>::max());

				// remove first pair if it is empty
				if (data[0] == data[1]) {
					for(std::size_t i = 0; i<data.size()-2; ++i) {
						data[i] = data[i+2];
					}
					data.pop_back();
					data.pop_back();
				}

				// if the list is empty now, we are done
				if (data.empty()) return;

				// remove the last pair if it is empty
				if (data[data.size()-2] == data[data.size()-1]) {
					data.pop_back();
					data.pop_back();
				}

			}


			/**
			 * Removes all elements of this range that are not covered by the given range.
			 * @param other the range of entries to be retained
			 */
			void retain(const Intervals& other) {
				invert();
				auto tmp = other;
				tmp.remove(*this);
				swap(tmp);
			}

			/**
			 * Tests whether the given point is covered by this intervals.
			 */
			bool covers(std::size_t idx) const {
				auto begin = data.begin();
				auto end = data.end();
				auto pos = std::upper_bound(begin, end, idx);
				return pos != end && ((std::distance(begin,pos) % 2) == 1);
			}

			/**
			 * Tests whether all the points within the range [from,...,to) are covered by this intervals.
			 */
			bool coversAll(std::size_t from, std::size_t to) const {
				if (from >= to) return true;
				auto begin = data.begin();
				auto end = data.end();
				auto a = std::upper_bound(begin, end, from);
				auto b = std::upper_bound(begin, end, to-1);
				return a == b && a != end && ((std::distance(begin,a) % 2) == 1);
			}

			/**
			 * Tests whether any the points within the range [from,...,to) are covered by this intervals.
			 */
			bool coversAny(std::size_t from, std::size_t to) const {
				if (from >= to) return false;
				auto begin = data.begin();
				auto end = data.end();
				auto a = std::upper_bound(begin, end, from);
				auto b = std::upper_bound(begin, end, to-1);
				return a < b || (a == b && a != end && ((std::distance(begin,a) % 2) == 1));
			}

			/**
			 * Swaps the content of this interval with the given one.
			 */
			void swap(Intervals& other) {
				data.swap(other.data);
			}

			/**
			 * Invokes the given function for each index in the covered intervals.
			 */
			template<typename Fun>
			void forEach(const Fun& fun) const {
				// iterate through the individual intervals
				for(std::size_t i =0; i<data.size(); i+=2) {
					// iterate through the elements of this interval
					auto begin = data[i];
					auto end = data[i+1];
					for(std::size_t j = begin; j < end; ++j) {
						fun(j);
					}
				}
			}

			/**
			 * Enables the printing of the list of intervals.
			 */
			friend std::ostream& operator<<(std::ostream& out, const Intervals& cur) {
				out << "{";
				for(unsigned i=0; i<cur.data.size(); i+=2) {
					if (i != 0) out << ",";
					out << "[" << cur.data[i] << "-" << cur.data[i+1] << "]";
				}
				return out << "}";
			}

		};

	} // end namespace detail


	/**
	 * A large array is an array of objects of type T which can be manually allocated or discarded. The memory
	 * requirements of the array only covers those elements which have been marked active and have actually been used.
	 */
	template<typename T>
	class LargeArray {

		/**
		 * A pointer to the first element of the array.
		 */
		T* data;

		/**
		 * The size of this large array.
		 */
		std::size_t size;

		/**
		 * The list of active ranges in this large array (for which the memory is kept alive).
		 */
		detail::Intervals active_ranges;

	public:

		/**
		 * Creates a new large array of the given size.
		 */
		LargeArray(std::size_t size) : data(nullptr), size(size) {

			// check whether there is something to allocate
			if (size == 0) return;

			// allocate the address space
			#ifdef _MSC_VER
				data = (T*)malloc(sizeof(T)*size);
				assert_true(data != nullptr) << "Failed to allocate memory of size" << sizeof(T)*size;
			#else
				data = (T*)mmap(nullptr,sizeof(T)*size,
						PROT_READ | PROT_WRITE,
						MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE,
						-1,0
					);
			#endif
			assert_ne((void*)-1,(void*)data);
		}

		/**
		 * Explicitly deleted copy constructor.
		 */
		LargeArray(const LargeArray&) = delete;

		/**
		 * A move constructor for large arrays.
		 */
		LargeArray(LargeArray&& other)
			: data(other.data), size(other.size), active_ranges(std::move(other.active_ranges)) {
			assert_true(other.active_ranges.empty());
			other.data = nullptr;
		}

		/**
		 * Destroys this array.
		 */
		~LargeArray() {

			// if there is no data, nothing to do
			if (data == nullptr) return;

			// call the destructor for the remaining objects (if required)
			if (!std::is_trivially_destructible<T>::value) {
				active_ranges.forEach([this](std::size_t i){
					data[i].~T();
				});
			}

			// free the data
			#ifdef _MSC_VER
				::free(data);
			#else
				munmap(data,sizeof(T)*size);
			#endif
		}

		/**
		 * Explicitly deleted copy-assignment operator.
		 */
		LargeArray& operator=(const LargeArray&) = delete;

		/**
		 * Implementation of move assignment operator.
		 */
		LargeArray& operator=(LargeArray&& other) {
			assert_ne(data,other.data);
			if (data) {
			#ifdef _MSC_VER
				::free(data);
			#else
				munmap(data, sizeof(T)*size);
			#endif
			}
			std::swap(data,other.data);
			size = other.size;
			active_ranges.swap(other.active_ranges);
			return *this;
		}

		bool operator==(const LargeArray& other) const {
			// quick check
			if (this == &other) return true;

			// check the same size
			if (size != other.size) return false;

			// make sure both have allocated all the space
			assert_eq(active_ranges, other.active_ranges);

			// compare active ranges
			bool res = true;
			active_ranges.forEach([&](std::size_t pos){
				res = res && (data[pos] == other.data[pos]);
			});
			return res;
		}

		/**
		 * Allocates the given range within this large array.
		 * After this call, the corresponding sub-range can be accessed.
		 */
		void allocate(std::size_t start, std::size_t end) {
			// check for emptiness
			if (start >= end) return;
			assert_le(end, size) << "Invalid range " << start << " - " << end << " for array of size " << size;


			// invoke the constructor for the released objects (if required)
			if (!std::is_trivially_constructible<T>::value) {

				// compute the ranges of new elements
				auto newElements = detail::Intervals::fromRange(start,end);
				newElements.remove(active_ranges);


				// initialize the newly allocated elements
				newElements.forEach([this](std::size_t i){
					new (&data[i]) T();
				});
			}

			// add to active range
			active_ranges.add(start,end);
		}

		/**
		 * Frees the given range, thereby deleting the content and freeing the
		 * associated memory pages.
		 */
		void free(std::size_t start, std::size_t end) {

			// check for emptiness
			if (start >= end) return;
			assert_le(end, size) << "Invalid range " << start << " - " << end << " for array of size " << size;

			// invoke the destructor for the released objects (if required)
			if (!std::is_trivially_destructible<T>::value) {

				// compute the elements to be removed
				auto removedElements = detail::Intervals::fromRange(start,end);
				removedElements.retain(active_ranges);

				// delete elements to be removed
				removedElements.forEach([this](std::size_t i){
					data[i].~T(); // explicit destructor call
				});

			}

			// remove range from active ranges
			active_ranges.remove(start,end);

			#ifdef _MSC_VER
				// do nothing
			#else
				// get address of lower boundary
				uintptr_t ptr_start = (uintptr_t)(data + start);
				uintptr_t ptr_end = (uintptr_t)(data + end);

				auto page_size = getPageSize();
				uintptr_t pg_start = ptr_start - (ptr_start % page_size);
				uintptr_t pg_end = ptr_end - (ptr_end % page_size) + page_size;

				std::size_t idx_start = (pg_start - (uintptr_t)(data)) / sizeof(T);
				std::size_t idx_end   = (pg_end - (uintptr_t)(data)) / sizeof(T);

				assert_le(idx_start,start);
				assert_le(end,idx_end);

				if (active_ranges.coversAny(idx_start,start)) pg_start += page_size;
				if (active_ranges.coversAny(end,idx_end))     pg_end -= page_size;
				pg_end = std::min(pg_end,ptr_end);

				if (pg_start >= pg_end) return;


					void* section_start = (void*)pg_start;
					std::size_t length = pg_end - pg_start;
					munmap(section_start, length);
					auto res = mmap(section_start, length,
							PROT_READ | PROT_WRITE,
							MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE | MAP_FIXED,
							-1,0
						);
					if ((void*)-1 == (void*)res) {
						assert_ne((void*)-1,(void*)res);
					}
			#endif
		}

		/**
		 * Provides mutable access to the element at the given position.
		 */
		T& operator[](std::size_t pos) {
			return data[pos];
		}

		/**
		 * Provides read-only access to the element at the given position.
		 */
		const T& operator[](std::size_t pos) const {
			return data[pos];
		}

	private:

		/**
		 * Determines the memory page size of the system.
		 */
		static long getPageSize() {
			#ifndef _MSC_VER
				static const long PAGE_SIZE = sysconf(_SC_PAGESIZE);
			#else
				static const long PAGE_SIZE = 0;
			#endif
			return PAGE_SIZE;
		}

	};


} // end namespace utils
} // end namespace allscale
