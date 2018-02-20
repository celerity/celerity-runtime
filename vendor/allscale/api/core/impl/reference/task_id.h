#pragma once

#include <cstdint>
#include <ostream>
#include <iterator>

#include "allscale/utils/assert.h"

namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace reference {

	/**
	 * The path part of a task ID. The path is the part of and ID addressing
	 * a certain sub-task of a decomposable task.
	 */
	class TaskPath {

		using path_t = std::uint64_t;
		using length_t = std::uint8_t;

		path_t path;
		length_t length;

		TaskPath(path_t path, length_t length) : path(path), length(length) {}

	public:

		TaskPath() = default;

		static TaskPath root() {
			return TaskPath{0,0};
		}

		bool isRoot() const {
			return length == 0;
		}

		path_t getPath() const {
			return path;
		}

		length_t getLength() const {
			return length;
		}

		bool operator==(const TaskPath& other) const {
			return path == other.path && length == other.length;
		}

		bool operator!=(const TaskPath& other) const {
			return !(*this == other);
		}

		bool operator<(const TaskPath& other) const {
			// get common prefix length
			auto min_len = std::min(length, other.length);

			auto pA = path >> (length - min_len);
			auto pB = other.path >> (other.length - min_len);

			// lexicographical compare
			if (pA == pB) {
				return length < other.length;
			}

			// compare prefix comparison
			return pA < pB;
		}

		bool isPrefixOf(const TaskPath& other) const {
			return length < other.length && (path == other.path >> (other.length - length));
		}

		TaskPath getLeftChildPath() const {
			assert_lt((std::size_t)length,sizeof(path)*8);
			auto res = *this;
			res.path = res.path << 1;
			++res.length;
			return res;
		}

		TaskPath getRightChildPath() const {
			auto res = getLeftChildPath();
			res.path = res.path + 1;
			return res;
		}

		TaskPath& descentLeft() {
			path = path << 1;
			return *this;
		}

		TaskPath& descentRight() {
			descentLeft();
			path += 1;
			return *this;
		}

		// --- path iterator support ---

		enum Direction {
			Left = 0, Right = 1
		};


		class path_iterator : public std::iterator<std::forward_iterator_tag,Direction> {

			path_t path;
			length_t pos;
			Direction cur;

			path_iterator(path_t path, length_t pos, Direction cur)
				: path(path), pos(pos), cur(cur) {}

		public:

			static path_iterator begin(path_t path, length_t length) {
				if (length == 0) return end(path);
				return path_iterator( path, length, Direction((path >> (length-1)) % 2) );
			}

			static path_iterator end(path_t path) {
				return path_iterator( path, 0, Left );
			}

			bool operator==(const path_iterator& other) const {
				return pos == other.pos && path == other.path;
			}

			bool operator!=(const path_iterator& other) const {
				return !(*this == other);
			}

			const Direction& operator*() const {
				return cur;
			}

			path_iterator& operator++() {
				--pos;
				if (pos==0) return *this;		// we have reached the end
				cur = Direction((path >> (pos-1)) % 2);
				return *this;
			}

		};

		path_iterator begin() const {
			return path_iterator::begin(path,length);
		}

		path_iterator end() const {
			return path_iterator::end(path);
		}


		// --- print support ---

		friend std::ostream& operator<<(std::ostream& out, const TaskPath& path) {
			for(const auto& cur : path) {
				out << "." << cur;
			}
			return out;
		}

	};

	/**
	 * An identifier of work items. Each work item is either a root-work-item,
	 * created by an initial prec call, or a child work item created through the
	 * splitting of a parent work item. The identifier is tracing this parent-child
	 * relationship.
	 *
	 * E.g. parent work item ID:
	 *
	 * 			T-12.0.1.0.1
	 *
	 * 		child work items:
	 *
	 * 			T-12.0.1.0.1.0 and WI-12.0.1.0.1.1
	 *
	 */
	class TaskID {

		std::uint64_t id;
		TaskPath path;

	public:

		TaskID() = default;

		TaskID(std::uint64_t id) : id(id), path(TaskPath::root()) {}

		TaskID(std::uint64_t id, const TaskPath& path)
			: id(id), path(path) {}


		// -- observers --

		std::uint64_t getRootID() const {
			return id;
		}

		const TaskPath& getPath() const {
			return path;
		}

		auto getDepth() const {
			return path.getLength();
		}

		// -- utility functions --

		bool operator==(const TaskID& other) const {
			return id == other.id && path == other.path;
		}

		bool operator!=(const TaskID& other) const {
			return !(*this == other);
		}

		bool operator<(const TaskID& other) const {
			// check id
			if (id < other.id) return true;
			if (id > other.id) return false;

			// compare the paths
			return path < other.path;
		}

		bool isParentOf(const TaskID& child) const {
			return id == child.id && path.isPrefixOf(child.path);
		}

		TaskID getLeftChild() const {
			return TaskID{ id, path.getLeftChildPath() };
		}

		TaskID getRightChild() const {
			return TaskID{ id, path.getRightChildPath() };
		}


		friend std::ostream& operator<<(std::ostream& out, const TaskID& id) {
			return out << "T-" << id.id << id.path;
		}

	};


} // end namespace reference
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale
