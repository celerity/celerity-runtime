#pragma once

#include <iterator>
#include <vector>

#include "allscale/utils/printer/join.h"

namespace allscale {
namespace utils {

	/**
	 * A data structure for maintaining a collection of
	 * objects with duplicates.
	 */
	template<typename T>
	class Bag {

		// the element type maintained in this bag
		using element_type = T;

		// internally, the data is maintained in a simple list
		std::vector<T> data;

	public:

		/**
		 * Tests whether this bag is empty or not.
		 */
		bool empty() const {
			return data.empty();
		}

		/**
		 * Determines the number of elements in this bag.
		 */
		std::size_t size() const {
			return data.size();
		}

		/**
		 * Inserts a new element in this bag.
		 */
		void insert(const T& element) {
			data.push_back(element);
		}

		/**
		 * Removes an element from this bag.
		 */
		void remove(const T& element) {
			auto pos = std::find(data.begin(),data.end(),element);
			if (pos == data.end()) return;
			data.erase(pos);
		}

		/**
		 * Tests whether the given element is contained within this bag.
		 */
		bool contains(const T& element) {
			auto pos = std::find(data.begin(),data.end(),element);
			return pos != data.end();
		}

		// add support for scans

		/**
		 * Obtains an iterator pointing to the start of the range of
		 * elements contained in this bag.
		 */
		auto begin() const {
			return data.begin();
		}

		/**
		 * Obtains an iterator pointing to the end of the range of
		 * elements contained in this bag.
		 */
		auto end() const {
			return data.end();
		}

		/**
		 * Runs a combined update and filter operation on the elements
		 * in this bag. The elements are passed by reference to the given
		 * body -- which may return false if elements shell be removed, tue
		 * otherwise.
		 */
		template<typename Body>
		void updateFilter(const Body& body) {
			// remove all elements where the predicate is violated
			auto newEnd = std::remove_if(data.begin(), data.end(), [&](T& i) { return !body(i); });
			data.erase(newEnd,data.end());
		}

		/**
		 * Removes all elements from this bag which do not satisfy the
		 * given predicates.
		 */
		template<typename Predicate>
		void filter(const Predicate& pred) {
			updateFilter([&](const T& i) {
				return pred(i);
			});
		}

		/**
		 * Adds printer support to this bag.
		 */
		friend std::ostream& operator<<(std::ostream& out, const Bag& bag) {
			return out << "{" << utils::join(",",bag.data) << "}";
		}

	};


} // end namespace utils
} // end namespace allscale
