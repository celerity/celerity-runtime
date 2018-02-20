#pragma once

#include <utility>

#include "allscale/utils/assert.h"

#include "allscale/api/core/prec.h"

#include "allscale/utils/vector.h"

namespace allscale {
namespace api {
namespace user {
namespace algorithm {

	// ----- forward declarations ------

	namespace detail {

		/**
		 * The object representing the iterator range of a (parallel) loop.
		 */
		template<typename Iter>
		class range;


		// -- Adaptive Loop Dependencies --

		/**
		 * The token produced by the pfor operator to reference the execution
		 * of a parallel loop.
		 */
		template<typename Iter>
		class loop_reference;

		/**
		 * A marker type for loop dependencies.
		 */
		struct loop_dependency {};

		/**
		 * A test for loop dependencies.
		 */
		template<typename T>
		struct is_loop_dependency : public std::is_base_of<loop_dependency,T> {};

		/**
		 * A small container for splitting dependencies.
		 */
		template<typename Dependency>
		struct SubDependencies {
			Dependency left;
			Dependency right;
		};

	} // end namespace detail

	/**
	 * The dependency to be used if no dependencies are required.
	 */
	struct no_dependencies : public detail::loop_dependency {

		detail::SubDependencies<no_dependencies> split() const {
			return detail::SubDependencies<no_dependencies>();
		}

	};

	// ---------------------------------------------------------------------------------------------
	//									Basic Generic pfor Operators
	// ---------------------------------------------------------------------------------------------

	/**
	 * The generic version of all parallel loops with synchronization dependencies.
	 *
	 * @tparam Iter the type of the iterator to pass over
	 * @tparam Body the type of the body operation, thus the operation to be applied on each element in the given range
	 * @tparam Dependency the type of the dependencies to be enforced
	 *
	 * @param r the range to iterate over
	 * @param body the operation to be applied on each element of the given range
	 * @param dependency the dependencies to be obeyed when scheduling the iterations of this parallel loop
	 *
	 * @return a reference to the iterations of the processed parallel loop to be utilized for forming dependencies
	 */
	template<typename Iter, typename Body, typename Dependency>
	detail::loop_reference<Iter> pfor(const detail::range<Iter>& r, const Body& body, const Dependency& dependency);

	/**
	 * The generic version of all parallel loops without synchronization dependencies.
	 *
	 * @tparam Iter the type of the iterator to pass over
	 * @tparam Body the type of the body operation, thus the operation to be applied on each element in the given range
	 *
	 * @param r the range to iterate over
	 * @param body the operation to be applied on each element of the given range
	 *
	 * @return a reference to the iterations of the processed parallel loop to be utilized for forming dependencies
	 */
	template<typename Iter, typename Body>
	detail::loop_reference<Iter> pfor(const detail::range<Iter>& r, const Body& body, const no_dependencies& = no_dependencies());


	// ---------------------------------------------------------------------------------------------
	//									pfor Operators with Boundaries
	// ---------------------------------------------------------------------------------------------

	/**
	 * The generic version of all parallel loops with synchronization dependencies.
	 *
	 * @tparam Iter the type of the iterator to pass over
	 * @tparam InnerBody the type of the inner body operation, thus the operation to be applied on each element in the given range that is not on the surface
	 * @tparam BoundaryBody the type of the boundary body operation, thus the operation to be applied on each element in the given range that is on the surface
	 * @tparam Dependency the type of the dependencies to be enforced
	 *
	 * @param r the range to iterate over
	 * @param innerBody the operation to be applied on each element of the given range that is not on the surface
	 * @param boundaryBody the operation to be applied on each element of the given range that is on the surface
	 * @param dependency the dependencies to be obeyed when scheduling the iterations of this parallel loop
	 *
	 * @return a reference to the iterations of the processed parallel loop to be utilized for forming dependencies
	 */
	template<typename Iter, typename InnerBody, typename BoundaryBody, typename Dependency>
	detail::loop_reference<Iter> pforWithBoundary(const detail::range<Iter>& r, const InnerBody& innerBody, const BoundaryBody& boundaryBody, const Dependency& dependency);

	/**
	 * The generic version of all parallel loops without synchronization dependencies.
	 *
	 * @tparam Iter the type of the iterator to pass over
	 * @tparam InnerBody the type of the inner body operation, thus the operation to be applied on each element in the given range that is not on the surface
	 * @tparam BoundaryBody the type of the boundary body operation, thus the operation to be applied on each element in the given range that is on the surface
	 *
	 * @param r the range to iterate over
	 * @param innerBody the operation to be applied on each element of the given range that is not on the surface
	 * @param boundaryBody the operation to be applied on each element of the given range that is on the surface
	 *
	 * @return a reference to the iterations of the processed parallel loop to be utilized for forming dependencies
	 */
	template<typename Iter, typename InnerBody, typename BoundaryBody>
	detail::loop_reference<Iter> pforWithBoundary(const detail::range<Iter>& r, const InnerBody& innerBody, const BoundaryBody& boundaryBody, const no_dependencies& = no_dependencies());


	// ---------------------------------------------------------------------------------------------
	//									The after Utility
	// ---------------------------------------------------------------------------------------------

	/**
	 * A generic utility for inserting a single action into a single a chain of dependencies. The given action will be triggered
	 * once the corresponding iteration in the given loop reference has been completed. The resulting loop reference can be utilized
	 * by consecutive operations to synchronize on the completion of the concatenation of the given loop reference and inserted action.
	 *
	 * @tparam Iter the type of iterator the preceding loop operated on
	 * @tparam Point the iterator value of the point this action shell be associated to
	 * @tparam Action the type of action to be performed
	 *
	 * @param loop preceding loop
	 * @param point the point to which this event shell be associated to
	 * @param action the action to be performed
	 * @return a customized loop reference to sync upon the concatenation of this
	 */
	template<typename Iter, typename Point, typename Action>
	detail::loop_reference<Iter> after(const detail::loop_reference<Iter>& loop, const Point& point, const Action& action);


	// ---------------------------------------------------------------------------------------------
	//									adapters for the pfor operator
	// ---------------------------------------------------------------------------------------------

	template<typename Iter, size_t dims, typename Body>
	detail::loop_reference<std::array<Iter,dims>> pfor(const std::array<Iter,dims>& a, const std::array<Iter,dims>& b, const Body& body) {
		return pfor(detail::range<std::array<Iter,dims>>(a,b),body);
	}

	template<typename Iter, size_t dims, typename Body, typename Dependency>
	detail::loop_reference<std::array<Iter,dims>> pfor(const std::array<Iter,dims>& a, const std::array<Iter,dims>& b, const Body& body, const Dependency& dependency) {
		return pfor(detail::range<std::array<Iter,dims>>(a,b),body,dependency);
	}

	template<typename Iter, size_t dims, typename InnerBody, typename BoundaryBody>
	detail::loop_reference<std::array<Iter,dims>> pforWithBoundary(const std::array<Iter,dims>& a, const std::array<Iter,dims>& b, const InnerBody& innerBody, const BoundaryBody& boundaryBody) {
		return pforWithBoundary(detail::range<std::array<Iter,dims>>(a,b),innerBody,boundaryBody);
	}

	template<typename Iter, size_t dims, typename InnerBody, typename BoundaryBody, typename Dependency>
	detail::loop_reference<std::array<Iter,dims>> pforWithBoundary(const std::array<Iter,dims>& a, const std::array<Iter,dims>& b, const InnerBody& innerBody, const BoundaryBody& boundaryBody, const Dependency& dependency) {
		return pforWithBoundary(detail::range<std::array<Iter,dims>>(a,b),innerBody,boundaryBody,dependency);
	}

	/**
	 * A parallel for-each implementation iterating over the given range of elements.
	 */
	template<typename Iter, typename Body, typename Dependency>
	detail::loop_reference<Iter> pfor(const Iter& a, const Iter& b, const Body& body, const Dependency& dependency) {
		return pfor(detail::range<Iter>(a,b),body,dependency);
	}

	template<typename Iter, typename Body>
	detail::loop_reference<Iter> pfor(const Iter& a, const Iter& b, const Body& body) {
		return pfor(a,b,body,no_dependencies());
	}

	template<typename Iter, typename InnerBody, typename BoundaryBody>
	detail::loop_reference<Iter> pforWithBoundary(const Iter& a, const Iter& b, const InnerBody& innerBody, const BoundaryBody& boundaryBody) {
		return pforWithBoundary(detail::range<Iter>(a,b),innerBody,boundaryBody);
	}

	template<typename Iter, typename InnerBody, typename BoundaryBody, typename Dependency>
	detail::loop_reference<Iter> pforWithBoundary(const Iter& a, const Iter& b, const InnerBody& innerBody, const BoundaryBody& boundaryBody, const Dependency& dependency) {
		return pforWithBoundary(detail::range<Iter>(a,b),innerBody,boundaryBody,dependency);
	}

	// ---- container support ----

	/**
	 * A parallel for-each implementation iterating over the elements of the given, mutable container.
	 */
	template<typename Container, typename Op>
	detail::loop_reference<typename Container::iterator>
	pfor(Container& c, const Op& op) {
		return pfor(c.begin(), c.end(), op);
	}

	/**
	 * A parallel for-each implementation iterating over the elements of the given, mutable container.
	 */
	template<typename Container, typename Op, typename Dependency>
	std::enable_if_t<detail::is_loop_dependency<Dependency>::value,detail::loop_reference<typename Container::iterator>>
	pfor(Container& c, const Op& op, const Dependency& dependency) {
		return pfor(c.begin(), c.end(), op, dependency);
	}


	/**
	 * A parallel for-each implementation iterating over the elements of the given container.
	 */
	template<typename Container, typename Op>
	detail::loop_reference<typename Container::const_iterator>
	pfor(const Container& c, const Op& op) {
		return pfor(c.begin(), c.end(), op);
	}

	/**
	 * A parallel for-each implementation iterating over the elements of the given container.
	 */
	template<typename Container, typename Op, typename Dependency>
	detail::loop_reference<typename Container::const_iterator>
	pfor(const Container& c, const Op& op, const Dependency& dependency) {
		return pfor(c.begin(), c.end(), op, dependency);
	}


	// ---- Vector support ----

	/**
	 * A parallel for-each implementation iterating over the elements of the points covered by
	 * the hyper-box limited by the given vectors.
	 */
	template<typename Elem, size_t dims, typename Body>
	detail::loop_reference<utils::Vector<Elem,dims>> pfor(const utils::Vector<Elem,dims>& a, const utils::Vector<Elem,dims>& b, const Body& body) {
		return pfor(detail::range<utils::Vector<Elem,dims>>(a,b),body);
	}

	/**
	 * A parallel for-each implementation iterating over the elements of the points covered by
	 * the hyper-box limited by the given vectors. Optional dependencies may be passed.
	 */
	template<typename Elem, size_t dims, typename Body, typename Dependencies>
	detail::loop_reference<utils::Vector<Elem,dims>> pfor(const utils::Vector<Elem,dims>& a, const utils::Vector<Elem,dims>& b, const Body& body, const Dependencies& dependencies) {
		return pfor(detail::range<utils::Vector<Elem,dims>>(a,b),body,dependencies);
	}

	/**
	 * A parallel for-each implementation iterating over the elements of the points covered by
	 * the hyper-box limited by the given vector.
	 */
	template<typename Elem, size_t Dims, typename Body>
	auto pfor(const utils::Vector<Elem,Dims>& a, const Body& body) {
		return pfor(utils::Vector<Elem,Dims>(0),a,body);
	}

	/**
	 * A parallel for-each implementation iterating over the elements of the points covered by
	 * the hyper-box limited by the given vector. Optional dependencies may be passed.
	 */
	template<typename Elem, size_t Dims, typename Body, typename Dependencies>
	auto pfor(const utils::Vector<Elem,Dims>& a, const Body& body, const Dependencies& dependencies) {
		return pfor(utils::Vector<Elem,Dims>(0),a,body,dependencies);
	}

	// -------------------------------------------------------------------------------------------
	//								Adaptive Synchronization
	// -------------------------------------------------------------------------------------------


	/**
	 * A dependency forming the conjunction of a list of given dependencies.
	 */
	template<typename ... Dependencies>
	class conjunction_sync_dependency;

	/**
	 * A factory for a conjunction of dependencies.
	 */
	template<typename ... Dependencies>
	conjunction_sync_dependency<Dependencies...> sync_all(const Dependencies& ... dependencies) {
		return conjunction_sync_dependency<Dependencies...>(dependencies...);
	}

	/**
	 * A dependency actually representing no dependency. Could be used as a place-holder.
	 */
	class no_dependency;

	/**
	 * A factory for no synchronization dependencies. Could be used as a place-holder.
	 */
	no_dependency no_sync();

	/**
	 * A dependency between loop iterations where iteration i of a new parallel loop may be executed
	 * as soon as iteration i of a given parallel loop has been completed.
	 *
	 * @param Iter the iterator type utilized to address iterations
	 */
	template<typename Iter>
	class one_on_one_dependency;

	/**
	 * A factory for one_on_one dependencies.
	 */
	template<typename Iter>
	one_on_one_dependency<Iter> one_on_one(const detail::loop_reference<Iter>& dep) {
		return one_on_one_dependency<Iter>(dep);
	}

	/**
	 * A dependency between loop iterations where iteration i of a new parallel loop may be executed
	 * as soon as iterations { i + c | c \in {-1,0,1}^n && |c| <= 1 } of a given parallel loop has been completed.
	 *
	 * @param Iter the iterator type utilized to address iterations
	 */
	template<typename Iter,std::size_t radius>
	class small_neighborhood_sync_dependency;

	/**
	 * A factory for small neighborhood sync dependencies.
	 */
	template<std::size_t radius = 1, typename Iter>
	small_neighborhood_sync_dependency<Iter, radius> small_neighborhood_sync(const detail::loop_reference<Iter>& dep) {
		return small_neighborhood_sync_dependency<Iter, radius>(dep);
	}

	/**
	 * A dependency between loop iterations where iteration i of a new parallel loop may be executed
	 * as soon as iterations { i + c | c \in {-1,0,1}^n } of a given parallel loop has been completed.
	 *
	 * @param Iter the iterator type utilized to address iterations
	 */
	template<typename Iter,std::size_t radius>
	class full_neighborhood_sync_dependency;

	/**
	 * A factory for full neighborhood sync dependencies.
	 */
	template<std::size_t radius = 1, typename Iter>
	full_neighborhood_sync_dependency<Iter,radius> full_neighborhood_sync(const detail::loop_reference<Iter>& dep) {
		return full_neighborhood_sync_dependency<Iter,radius>(dep);
	}

	/**
	 * A dependency between loop iterations where iteration i of a new parallel loop may be executed
	 * as soon the entire range of a given loop has been executed.
	 *
	 * @param Iter the iterator type utilized to address iterations
	 */
	template<typename Iter>
	class after_all_sync_dependency;

	/**
	 * A factory for after-all sync dependencies.
	 */
	template<typename Iter>
	after_all_sync_dependency<Iter> after_all_sync(const detail::loop_reference<Iter>& dep) {
		return after_all_sync_dependency<Iter>(dep);
	}


	// -------------------------------------------------------------------------------------------
	//									Range Utils
	// -------------------------------------------------------------------------------------------


	namespace detail {

		// -- obtain number of dimensions of an iterator --

		template<typename Iter>
		struct dimensions {
			enum { value = 1 };
		};

		template<typename Iter, std::size_t D>
		struct dimensions<std::array<Iter,D>> {
			enum { value = D };
		};

		template<typename Iter, std::size_t D>
		struct dimensions<utils::Vector<Iter,D>> {
			enum { value = D };
		};

		// -- distances between begin and end of iterators --

		template<typename Iter,typename filter = bool>
		struct volume {
			size_t operator()(const Iter& a, const Iter& b) const {
				return std::distance(a,b);
			}
		};

		template<typename Int>
		struct volume<Int,std::enable_if_t<std::template is_integral<Int>::value,bool>> {
			size_t operator()(Int a, Int b) const {
				return (a < b) ? b-a : 0;
			}
		};

		template<typename Iter,size_t dims>
		struct volume<std::array<Iter,dims>> {
			size_t operator()(const std::array<Iter,dims>& a, const std::array<Iter,dims>& b) const {
				volume<Iter> inner;
				size_t res = 1;
				for(size_t i = 0; i<dims; i++) {
					res *= inner(a[i],b[i]);
				}
				return res;
			}
		};

		template<typename Iter,size_t dims>
		struct volume<utils::Vector<Iter,dims>> {
			size_t operator()(const utils::Vector<Iter,dims>& a, const utils::Vector<Iter,dims>& b) const {
				return volume<std::array<Iter,dims>>()(a,b);
			}
		};

		// -- minimum distance between elements along individual dimensions --

		template<typename Iter,typename filter = bool>
		struct min_dimension_length {
			size_t operator()(const Iter& a, const Iter& b) const {
				return std::distance(a,b);
			}
		};

		template<typename Int>
		struct min_dimension_length<Int,std::enable_if_t<std::template is_integral<Int>::value,bool>> {
			size_t operator()(Int a, Int b) const {
				return (a < b) ? b-a : 0;
			}
		};

		template<typename Iter,size_t dims>
		struct min_dimension_length<std::array<Iter,dims>> {
			size_t operator()(const std::array<Iter,dims>& a, const std::array<Iter,dims>& b) const {
				min_dimension_length<Iter> inner;
				size_t res = std::numeric_limits<size_t>::max();
				for(size_t i = 0; i<dims; i++) {
					res = std::min(res,inner(a[i],b[i]));
				}
				return res;
			}
		};

		template<typename Iter,size_t dims>
		struct min_dimension_length<utils::Vector<Iter,dims>> {
			size_t operator()(const utils::Vector<Iter,dims>& a, const utils::Vector<Iter,dims>& b) const {
				return min_dimension_length<std::array<Iter,dims>>()(a,b);
			}
		};

		template<typename Iter>
		size_t getMinimumDimensionLength(const range<Iter>& r) {
			return min_dimension_length<Iter>()(r.begin(),r.end());
		}

		// -- coverage --

		template<typename Iter>
		bool covers(const Iter& a_begin, const Iter& a_end, const Iter& b_begin, const Iter& b_end) {
			return b_begin >= b_end || (a_begin <= b_begin && b_end <= a_end);
		}

		template<typename Iter, size_t dims>
		bool covers(const utils::Vector<Iter,dims>& a_begin, const utils::Vector<Iter,dims>& a_end, const utils::Vector<Iter,dims>& b_begin, const utils::Vector<Iter,dims>& b_end) {
			// if the second is empty, it is covered
			for(size_t i=0; i<dims; ++i) {
				if (b_begin[i] >= b_end[i]) return true;
			}
			// check that a non-empty range is covered
			for(size_t i=0; i<dims; ++i) {
				if (!(a_begin[i] <= b_begin[i] && b_end[i] <= a_end[i])) return false;
			}
			return true;
		}


		template<typename Iter, typename Point>
		bool covers(const Iter& begin, const Iter& end, const Point& p) {
			return begin <= p && p < end;
		}

		template<typename Iter, typename Point,size_t dims>
		bool covers(const utils::Vector<Iter,dims>& begin, const utils::Vector<Iter,dims>& end, const utils::Vector<Point,dims>& point) {
			for(size_t i=0; i<dims; ++i) {
				if (point[i] < begin[i] || end[i] <= point[i]) return false;
			}
			return true;
		}

		// -- iterator access utility --

		template<typename Iter>
		auto access(const Iter& iter) -> decltype(*iter) {
			return *iter;
		}

		template<typename T>
		typename std::enable_if<std::is_integral<T>::value,T>::type access(T a) {
			return a;
		}


		// -- scan utility --

		template<typename Iter, typename InnerOp, typename BoundaryOp>
		void forEach(const Iter& fullBegin, const Iter& fullEnd, const Iter& a, const Iter& b, const InnerOp& inner, const BoundaryOp& boundary) {

			// cut off empty loop
			if (a == b) return;

			// get inner range
			Iter innerBegin = a;
			Iter innerEnd = b;

			// check for boundaries
			if (fullBegin == a) {
				boundary(access(a));
				innerBegin++;
			}

			// reduce inner range if b is the end
			if (fullEnd == b) {
				innerEnd--;
			}

			// process inner part
			for(auto it = innerBegin; it != innerEnd; ++it) {
				inner(access(it));
			}

			// process left boundary
			if(fullEnd == b) {
				boundary(access(b-1));
			}
		}


		template<typename Iter, typename InnerOp, typename BoundaryOp>
		void forEach(const Iter& a, const Iter& b, const InnerOp& inner, const BoundaryOp& boundary) {

			// cut off empty loop
			if (a == b) return;

			// process left boundary
			boundary(access(a));
			if (a + 1 == b) return;

			// process inner part
			for(auto it = a+1; it != b-1; ++it) {
				inner(access(it));
			}

			// process left boundary
			boundary(access(b-1));
		}

		template<typename Iter, typename Op>
		void forEach(const Iter& a, const Iter& b, const Op& op) {
			for(auto it = a; it != b; ++it) {
				op(access(it));
			}
		}

		template<typename Point>
		struct point_factory;

		template<typename Iter, size_t dims>
		struct point_factory<std::array<Iter,dims>> {
			template<typename ... Coordinates>
			std::array<Iter,dims> operator()(Coordinates ... coordinates) {
				return { { coordinates ... } };
			}
		};

		template<typename Iter, size_t dims>
		struct point_factory<utils::Vector<Iter,dims>> {
			template<typename ... Coordinates>
			utils::Vector<Iter,dims> operator()(Coordinates ... coordinates) {
				return utils::Vector<Iter,dims>(coordinates...);
			}
		};


		template<size_t idx>
		struct scanner {
			scanner<idx-1> nested;
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Op, typename ... Coordinates>
			void operator()(const Compound<Iter,dims>& begin, const Compound<Iter,dims>& end, const Op& op, Coordinates ... coordinates) {
				auto a = begin[dims-idx];
				auto b = end[dims-idx];
				for(Iter i = a; i != b ; ++i) {
					nested(begin,end,op,coordinates...,i);
				}
			}
		};

		template<>
		struct scanner<0> {
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Op, typename ... Coordinates>
			void operator()(const Compound<Iter,dims>&, const Compound<Iter,dims>&, const Op& op, Coordinates ... coordinates) {
				point_factory<Compound<Iter,dims>> factory;
				op(factory(coordinates...));
			}
		};

		template<size_t idx>
		struct scanner_with_boundary {
			scanner_with_boundary<idx-1> nested;
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Op>
			void operator()(const Compound<Iter,dims>& begin, const Compound<Iter,dims>& end, Compound<Iter,dims>& cur, const Op& op) {
				auto& i = cur[dims-idx];
				for(i = begin[dims-idx]; i != end[dims-idx]; ++i ) {
					nested(begin, end, cur, op);
				}
			}
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Inner, typename Boundary>
			void operator()(const Compound<Iter,dims>& begin, const Compound<Iter,dims>& end, Compound<Iter,dims>& cur, const Inner& inner, const Boundary& boundary) {
				auto& i = cur[dims-idx];

				// extract range
				const auto& a = begin[dims-idx];
				const auto& b = end[dims-idx];

				// check empty range
				if (a==b) return;

				// handle left boundary
				i = a; nested(begin,end,cur,boundary);

				// check whether this has been all
				if (a + 1 == b) return;

				// process inner part
				for(i = a+1; i!=b-1; ++i) {
					nested(begin,end,cur,inner,boundary);
				}

				// handle right boundary
				i = b-1;
				nested(begin,end,cur,boundary);
			}

			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Inner, typename Boundary>
			void operator()(const Compound<Iter,dims>& fullBegin, const Compound<Iter,dims>& fullEnd, const Compound<Iter,dims>& begin, const Compound<Iter,dims>& end, Compound<Iter,dims>& cur, const Inner& inner, const Boundary& boundary) {
				auto& i = cur[dims-idx];

				// extract range
				const auto& fa = fullBegin[dims-idx];
				const auto& fb = fullEnd[dims-idx];

				const auto& a = begin[dims-idx];
				const auto& b = end[dims-idx];

				// check empty range
				if (a==b) return;

				// get inner range
				auto ia = a;
				auto ib = b;

				// handle left boundary
				if (fa == ia) {
					i = ia;
					nested(begin,end,cur,boundary);
					ia++;
				}

				if (fb == b) {
					ib--;
				}

				// process inner part
				for(i = ia; i!=ib; ++i) {
					nested(fullBegin,fullEnd,begin,end,cur,inner,boundary);
				}

				// handle right boundary
				if (fb == b) {
					i = b-1;
					nested(begin,end,cur,boundary);
				}
			}
		};

		template<>
		struct scanner_with_boundary<0> {
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Op>
			void operator()(const Compound<Iter,dims>&, const Compound<Iter,dims>&, Compound<Iter,dims>& cur, const Op& op) {
				op(cur);
			}
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Inner, typename Boundary>
			void operator()(const Compound<Iter,dims>&, const Compound<Iter,dims>&, Compound<Iter,dims>& cur, const Inner& inner, const Boundary&) {
				inner(cur);
			}
			template<template<typename T, size_t d> class Compound, typename Iter, size_t dims, typename Inner, typename Boundary>
			void operator()(const Compound<Iter,dims>&, const Compound<Iter,dims>&, const Compound<Iter,dims>&, const Compound<Iter,dims>&, Compound<Iter,dims>& cur, const Inner& inner, const Boundary&) {
				inner(cur);
			}
		};

		template<typename Iter, size_t dims, typename InnerOp, typename BoundaryOp>
		void forEach(const std::array<Iter,dims>& fullBegin, const std::array<Iter,dims>& fullEnd, const std::array<Iter,dims>& begin, const std::array<Iter,dims>& end, const InnerOp& inner, const BoundaryOp& boundary) {

			// the current position
			std::array<Iter,dims> cur;

			// scan range
			detail::scanner_with_boundary<dims>()(fullBegin, fullEnd, begin, end, cur, inner, boundary);
		}

		template<typename Iter, size_t dims, typename InnerOp, typename BoundaryOp>
		void forEach(const std::array<Iter,dims>& begin, const std::array<Iter,dims>& end, const InnerOp& inner, const BoundaryOp& boundary) {

			// the current position
			std::array<Iter,dims> cur;

			// scan range
			detail::scanner_with_boundary<dims>()(begin, end, cur, inner, boundary);
		}

		template<typename Iter, size_t dims, typename Op>
		void forEach(const std::array<Iter,dims>& begin, const std::array<Iter,dims>& end, const Op& op) {
			// scan range
			detail::scanner<dims>()(begin, end, op);
		}

		template<typename Elem, size_t dims, typename InnerOp, typename BoundaryOp>
		void forEach(const utils::Vector<Elem,dims>& fullBegin, const utils::Vector<Elem,dims>& fullEnd, const utils::Vector<Elem,dims>& begin, const utils::Vector<Elem,dims>& end, const InnerOp& inner, const BoundaryOp& boundary) {

			// the current position
			utils::Vector<Elem,dims> cur;

			// scan range
			detail::scanner_with_boundary<dims>()(fullBegin, fullEnd, begin, end, cur, inner, boundary);
		}

		template<typename Elem, size_t dims, typename InnerOp, typename BoundaryOp>
		void forEach(const utils::Vector<Elem,dims>& begin, const utils::Vector<Elem,dims>& end, const InnerOp& inner, const BoundaryOp& boundary) {

			// the current position
			utils::Vector<Elem,dims> cur;

			// scan range
			detail::scanner_with_boundary<dims>()(begin, end, cur, inner, boundary);
		}


		template<typename Elem, size_t dims, typename Op>
		void forEach(const utils::Vector<Elem,dims>& begin, const utils::Vector<Elem,dims>& end, const Op& op) {
			// scan range
			detail::scanner<dims>()(begin, end, op);
		}


		template<typename Iter>
		Iter grow(const Iter& value, const Iter& limit, int steps) {
			return std::min(limit, value+steps);
		}

		template<typename Iter, size_t dims>
		std::array<Iter,dims> grow(const std::array<Iter,dims>& value, const std::array<Iter,dims>& limit, int steps) {
			std::array<Iter,dims> res;
			for(unsigned i=0; i<dims; i++) {
				res[i] = grow(value[i],limit[i],steps);
			}
			return res;
		}

		template<typename Iter, size_t dims>
		utils::Vector<Iter,dims> grow(const utils::Vector<Iter,dims>& value, const utils::Vector<Iter,dims>& limit, int steps) {
			utils::Vector<Iter,dims> res;
			for(unsigned i=0; i<dims; i++) {
				res[i] = grow(value[i],limit[i],steps);
			}
			return res;
		}


		template<typename Iter>
		Iter shrink(const Iter& value, const Iter& limit, int steps) {
			return std::max(limit, value-steps);
		}

		template<typename Iter, size_t dims>
		std::array<Iter,dims> shrink(const std::array<Iter,dims>& value, const std::array<Iter,dims>& limit, int steps) {
			std::array<Iter,dims> res;
			for(unsigned i=0; i<dims; i++) {
				res[i] = shrink(value[i],limit[i],steps);
			}
			return res;
		}

		template<typename Iter, size_t dims>
		utils::Vector<Iter,dims> shrink(const utils::Vector<Iter,dims>& value, const utils::Vector<Iter,dims>& limit, int steps) {
			utils::Vector<Iter,dims> res;
			for(unsigned i=0; i<dims; i++) {
				res[i] = shrink(value[i],limit[i],steps);
			}
			return res;
		}

		template<typename Iter>
		struct fragments {
			range<Iter> left;
			range<Iter> right;
		};

		template<typename Iter>
		fragments<Iter> make_fragments(const range<Iter>& left, const range<Iter>& right) {
			return fragments<Iter>{ left, right };
		}

		template<typename Iter>
		struct range_spliter;

		/**
		 * The object representing the iterator range of a (parallel) loop.
		 */
		template<typename Iter>
		class range {

			/**
			 * The begin of this range (inclusive).
			 */
			Iter _begin;

			/**
			 * The end of this range (exclusive).
			 */
			Iter _end;

		public:

			range() : _begin(), _end() {}

			range(const Iter& begin, const Iter& end)
				: _begin(begin), _end(end) {
				if (empty()) { _end = _begin; }
			}

			size_t size() const {
				return detail::volume<Iter>()(_begin,_end);
			}

			bool empty() const {
				return size() == 0;
			}

			const Iter& begin() const {
				return _begin;
			}

			const Iter& end() const {
				return _end;
			}

			bool covers(const range<Iter>& r) const {
				return detail::covers(_begin,_end,r._begin,r._end);
			}

			template<typename Point>
			bool covers(const Point& p) const {
				return detail::covers(_begin,_end,p);
			}

			range grow(const range<Iter>& limit, int steps = 1) const {
				return range(
						detail::shrink(_begin,limit.begin(),steps),
						detail::grow(_end,limit.end(),steps)
				);
			}

			range shrink(int steps = 1) const {
				return grow(*this, -steps);
			}

			fragments<Iter> split(std::size_t depth) const {
				return range_spliter<Iter>::split(depth,*this);
			}

			template<typename Op>
			void forEach(const Op& op) const {
				detail::forEach(_begin,_end,op);
			}

			template<typename InnerOp, typename BoundaryOp>
			void forEachWithBoundary(const range& full, const InnerOp& inner, const BoundaryOp& boundary) const {
				detail::forEach(full._begin,full._end,_begin,_end,inner,boundary);
			}

			friend std::ostream& operator<<(std::ostream& out, const range& r) {
				return out << "[" << r.begin() << "," << r.end() << ")";
			}

		};

		template<typename Iter>
		struct range_spliter {

			using rng = range<Iter>;

			static fragments<Iter> split(std::size_t, const rng& r) {
				const auto& a = r.begin();
				const auto& b = r.end();
				auto m = a + (b - a)/2;
				return make_fragments(rng(a,m),rng(m,b));
			}

			static std::size_t getSplitDimension(std::size_t) {
				return 0;
			}
		};

		template<
			template<typename I, size_t d> class Container,
			typename Iter, size_t dims
		>
		struct range_spliter<Container<Iter,dims>> {

			using rng = range<Container<Iter,dims>>;

			static fragments<Container<Iter,dims>> split(std::size_t depth, const rng& r) {

				__allscale_unused const auto volume = detail::volume<Container<Iter,dims>>();

				// get split dimension
				auto splitDim = getSplitDimension(depth);

				// compute range fragments
				const auto& begin = r.begin();
				const auto& end = r.end();

				// split the longest dimension, keep the others as they are
				auto midA = end;
				auto midB = begin;
				midA[splitDim] = midB[splitDim] = range_spliter<Iter>::split(depth,range<Iter>(begin[splitDim],end[splitDim])).left.end();

				// make sure no points got lost
				assert_eq(volume(begin,end), volume(begin,midA) + volume(midB,end));

				// create result
				return make_fragments(rng(begin,midA),rng(midB,end));
			}

			static std::size_t getSplitDimension(std::size_t depth) {
				return depth % dims;
			}

		};

	} // end namespace detail



	// -------------------------------------------------------------------------------------------
	//								 Synchronization Definitions
	// -------------------------------------------------------------------------------------------

	namespace detail {

		/**
		 * An entity to reference ranges of iterations of a loop.
		 */
		template<typename Iter>
		class iteration_reference {

			/**
			 * The range covered by the iterations referenced by this object.
			 */
			range<Iter> _range;

			/**
			 * The reference to the task processing the covered range.
			 */
			core::task_reference handle;

			/**
			 * The recursive depth of the referenced iteration range.
			 */
			std::size_t depth;

		public:

			iteration_reference(const range<Iter>& range, const core::task_reference& handle, std::size_t depth)
				: _range(range), handle(handle), depth(depth) {}

			iteration_reference(const range<Iter>& _range = range<Iter>()) : _range(_range), depth(0) {}

			iteration_reference(const iteration_reference&) = default;
			iteration_reference(iteration_reference&&) = default;

			iteration_reference& operator=(const iteration_reference&) = default;
			iteration_reference& operator=(iteration_reference&&) = default;

			void wait() const {
				if (handle.valid()) handle.wait();
			}

			iteration_reference<Iter> getLeft() const {
				return { range_spliter<Iter>::split(depth,_range).left, handle.getLeft(), depth+1 };
			}

			iteration_reference<Iter> getRight() const {
				return { range_spliter<Iter>::split(depth,_range).right, handle.getRight(), depth+1 };
			}

			operator core::task_reference() const {
				return handle;
			}

			const range<Iter>& getRange() const {
				return _range;
			}

			const core::task_reference& getHandle() const {
				return handle;
			}

			std::size_t getDepth() const {
				return depth;
			}
		};


		/**
		 * An entity to reference the full range of iterations of a loop. This token
		 * can not be copied and will wait for the completion of the loop upon destruction.
		 */
		template<typename Iter>
		class loop_reference : public iteration_reference<Iter> {

		public:

			loop_reference(const range<Iter>& range, core::treeture<void>&& handle)
				: iteration_reference<Iter>(range, std::move(handle), 0) {}

			loop_reference() {};
			loop_reference(const loop_reference&) = delete;
			loop_reference(loop_reference&&) = default;

			loop_reference& operator=(const loop_reference&) = delete;
			loop_reference& operator=(loop_reference&&) = default;

			~loop_reference() { this->wait(); }

		};

	} // end namespace detail



	// ---------------------------------------------------------------------------------------------
	//									Definitions
	// ---------------------------------------------------------------------------------------------


	template<typename Iter, typename Body, typename Dependency>
	detail::loop_reference<Iter> pfor(const detail::range<Iter>& r, const Body& body, const Dependency& dependency) {

		struct RecArgs {
			std::size_t depth;
			detail::range<Iter> range;
			Dependency dependencies;
		};

		// trigger parallel processing
		return { r, core::prec(
			[](const RecArgs& rg) {
				// if there is only one element left, we reached the base case
				return rg.range.size() <= 1;
			},
			[body](const RecArgs& rg) {
				// apply the body operation to every element in the remaining range
				rg.range.forEach(body);
			},
			core::pick(
				[](const RecArgs& rg, const auto& nested) {
					// in the step case we split the range and process sub-ranges recursively
					auto fragments = rg.range.split(rg.depth);
					auto& left = fragments.left;
					auto& right = fragments.right;
					auto dep = rg.dependencies.split(left,right);
					return core::parallel(
						nested(dep.left.toCoreDependencies(), RecArgs{rg.depth+1, left, dep.left} ),
						nested(dep.right.toCoreDependencies(), RecArgs{rg.depth+1, right,dep.right})
					);
				},
				[body](const RecArgs& rg, const auto&) {
					// the alternative is processing the step sequentially
					rg.range.forEach(body);
				}
			)
		)(dependency.toCoreDependencies(),RecArgs{0,r,dependency}) };
	}

	template<typename Iter, typename Body>
	detail::loop_reference<Iter> pfor(const detail::range<Iter>& r, const Body& body, const no_dependencies&) {

		struct RecArgs {
			std::size_t depth;
			detail::range<Iter> range;
		};

		// trigger parallel processing
		return { r, core::prec(
			[](const RecArgs& r) {
				// if there is only one element left, we reached the base case
				return r.range.size() <= 1;
			},
			[body](const RecArgs& r) {
				// apply the body operation to every element in the remaining range
				r.range.forEach(body);
			},
			core::pick(
				[](const RecArgs& r, const auto& nested) {
					// in the step case we split the range and process sub-ranges recursively
					auto fragments = r.range.split(r.depth);
					return core::parallel(
						nested(RecArgs{r.depth+1,fragments.left}),
						nested(RecArgs{r.depth+1,fragments.right})
					);
				},
				[body](const RecArgs& r, const auto&) {
					// the alternative is processing the step sequentially
					r.range.forEach(body);
				}
			)
		)(RecArgs{0,r}) };
	}

	class no_dependency : public detail::loop_dependency {

	public:

		auto toCoreDependencies() const {
			return core::after();
		}

		template<typename Range>
		detail::SubDependencies<no_dependency> split(const Range&, const Range&) const {
			// split dependencies, which is actually nothing to do ...
			return { no_dependency(), no_dependency() };

		}

		friend std::ostream& operator<< (std::ostream& out, const no_dependency&) {
			return out << "none";
		}

	};

	inline no_dependency no_sync() {
		return no_dependency();
	}

	// --------------------------------------------------------------------------------------------------------

	template<typename First, typename ... Rest>
	class conjunction_sync_dependency<First,Rest...> : public detail::loop_dependency {

		using nested_type = conjunction_sync_dependency<Rest...>;

		First first;

		nested_type nested;

		conjunction_sync_dependency(const First& first, const nested_type& nested)
			: first(first), nested(nested) {}

	public:

		conjunction_sync_dependency(const First& first, const Rest& ... rest)
			: first(first), nested(rest...) {}

		auto toCoreDependencies() const {
			return concat(first.toCoreDependencies(),nested.toCoreDependencies());
		}

		template<typename Iter>
		detail::SubDependencies<conjunction_sync_dependency> split(const detail::range<Iter>& left, const detail::range<Iter>& right) const {

			// get fragments
			auto firstFragments = first.split(left,right);
			auto nestedFragments = nested.split(left,right);

			// create resulting dependencies
			return {
				{ firstFragments.left, nestedFragments.left },
				{ firstFragments.right, nestedFragments.right }
			};

		}

		friend std::ostream& operator<< (std::ostream& out, const conjunction_sync_dependency& dep) {
			return out << dep.first << " && " << dep.nested;
		}

	};

	// special case for a conjunction of a single dependency - this is just that dependency
	template<typename Dependency>
	class conjunction_sync_dependency<Dependency> : public Dependency {
	public:
		conjunction_sync_dependency(const Dependency& dep) : Dependency(dep) {}
	};

	// special case for an empty conjunction - this is no dependency
	template<>
	class conjunction_sync_dependency<> : public no_dependency {
	public:
		conjunction_sync_dependency() : no_dependency() {}
		conjunction_sync_dependency(const no_dependency& dep) : no_dependency(dep) {}
	};

	// --------------------------------------------------------------------------------------------------------

	template<typename Iter>
	class one_on_one_dependency : public detail::loop_dependency {

		detail::iteration_reference<Iter> loop;

	public:

		one_on_one_dependency(const detail::iteration_reference<Iter>& loop)
			: loop(loop) {}

		auto getCenterRange() const {
			return loop.getRange();
		}

		core::impl::reference::dependencies<core::impl::reference::fixed_sized<1>> toCoreDependencies() const {
			return core::after(loop.getHandle());
		}

		detail::SubDependencies<one_on_one_dependency<Iter>> split(const detail::range<Iter>& left, const detail::range<Iter>& right) const {

			// get left and right loop fragments
			auto loopLeft = loop.getLeft();
			auto loopRight = loop.getRight();

			// split dependencies, thereby checking range coverage
			return {
				// we take the sub-task if it covers the targeted range, otherwise we stick to the current range
				loopLeft.getRange().covers(left)   ? one_on_one_dependency<Iter>{loopLeft}  : *this,
				loopRight.getRange().covers(right) ? one_on_one_dependency<Iter>{loopRight} : *this
			};

		}

		friend std::ostream& operator<< (std::ostream& out, const one_on_one_dependency& dep) {
			return out << dep.loop.getRange();
		}

	};


	template<typename Iter, std::size_t radius>
	class small_neighborhood_sync_dependency : public detail::loop_dependency {

		// determine the number of dimensions
		enum { num_dimensions = detail::dimensions<Iter>::value };

		// the type of iteration dependency
		using iteration_reference = detail::iteration_reference<Iter>;

		// on each dimension, two dependencies are stored in each direction
		struct deps_pair {
			iteration_reference left;
			iteration_reference right;
		};

		// save two dependencies for each dimension
		using deps_list = std::array<deps_pair,num_dimensions>;

		// on dependency covering the central area
		iteration_reference center;

		// the neighboring dependencies
		deps_list neighborhood;

		// and internal constructor required by the split operation
		small_neighborhood_sync_dependency() {}

	public:

		small_neighborhood_sync_dependency(const iteration_reference& loop)
			: center(loop), neighborhood() {}

		const detail::range<Iter>& getCenterRange() const {
			return center.getRange();
		}

		std::vector<detail::range<Iter>> getRanges() const {
			std::vector<detail::range<Iter>> res;
			res.push_back(center.getRange());
			for(std::size_t i=0; i<num_dimensions; i++) {
				if (!neighborhood[i].left.getRange().empty())  res.push_back(neighborhood[i].left.getRange());
				if (!neighborhood[i].right.getRange().empty()) res.push_back(neighborhood[i].right.getRange());
			}
			return res;
		}

	private:

		template<std::size_t ... Dims>
		core::impl::reference::dependencies<core::impl::reference::fixed_sized<2*num_dimensions+1>> toCoreDependencies(const std::index_sequence<Dims...>&) const {
			return core::after(
					center,
					neighborhood[Dims].left ...,
					neighborhood[Dims].right ...
			);
		}

	public:

		core::impl::reference::dependencies<core::impl::reference::fixed_sized<2*num_dimensions+1>> toCoreDependencies() const {
			return toCoreDependencies(std::make_index_sequence<num_dimensions>());
		}

		detail::SubDependencies<small_neighborhood_sync_dependency<Iter,radius>> split(const detail::range<Iter>& left, const detail::range<Iter>& right) const {

			using splitter = detail::range_spliter<Iter>;

			// create new left and right dependencies
			small_neighborhood_sync_dependency res_left;
			small_neighborhood_sync_dependency res_right;

			// update center
			res_left.center = center.getLeft();
			res_right.center = center.getRight();

			// update neighbors except split dimension
			bool save_left = true;
			bool save_right = true;
			auto splitDim = splitter::getSplitDimension(center.getDepth());
			for(std::size_t i =0; i<num_dimensions; i++) {
				if (i != splitDim) {
					// narrow down dependencies in each dimension
					res_left.neighborhood[i].left  = neighborhood[i].left.getLeft();
					res_left.neighborhood[i].right = neighborhood[i].right.getLeft();
					res_right.neighborhood[i].left  = neighborhood[i].left.getRight();
					res_right.neighborhood[i].right = neighborhood[i].right.getRight();
				} else {
					// for the split dimension, apply special treatment
					res_left.neighborhood[i].left = neighborhood[i].left.getRight();
					res_left.neighborhood[i].right = center.getRight();
					res_right.neighborhood[i].left = center.getLeft();
					res_right.neighborhood[i].right = neighborhood[i].right.getLeft();

				}

				// check that there is still something remaining in left and right
				if (save_left && !neighborhood[i].left.getRange().empty() && getMinimumDimensionLength(res_left.neighborhood[i].left.getRange()) < radius) save_left = false;
				if (save_left && !neighborhood[i].right.getRange().empty() && getMinimumDimensionLength(res_left.neighborhood[i].right.getRange()) < radius) save_left = false;

				if (save_right && !neighborhood[i].left.getRange().empty() && getMinimumDimensionLength(res_right.neighborhood[i].left.getRange()) < radius) save_right = false;
				if (save_right && !neighborhood[i].right.getRange().empty() && getMinimumDimensionLength(res_right.neighborhood[i].right.getRange()) < radius) save_right = false;

			}

			// check coverage and build up result
			return {
				save_left && res_left.center.getRange().covers(left) ? res_left : *this,
				save_right && res_right.center.getRange().covers(right) ? res_right : *this
			};
		}

		friend std::ostream& operator<< (std::ostream& out, const small_neighborhood_sync_dependency& dep) {
			out << "[";
			out << dep.center;
			out << ",";
			for(const auto& cur : dep.neighborhood) {
				out << cur.left << "," << cur.right;
			}
			return out << "]";
		}

	};

	namespace detail {

		template<typename Iter, std::size_t Dims>
		struct full_dependency_block {

			using iteration_reference = detail::iteration_reference<Iter>;

			using nested = full_dependency_block<Iter,Dims-1>;

			enum { num_dependencies = nested::num_dependencies * 3 };

			std::array<nested,3> dependencies;

			void setCenter(const iteration_reference& ref) {
				dependencies[1].setCenter(ref);
			}

			const iteration_reference& getCenter() const {
				return dependencies[1].getCenter();
			}

			template<typename Op>
			void forEach(const Op& op) const {
				dependencies[0].forEach(op);
				dependencies[1].forEach(op);
				dependencies[2].forEach(op);
			}

			core::impl::reference::dependencies<core::impl::reference::fixed_sized<num_dependencies>> toCoreDependencies() const {
				return produceCoreDependencies(*this);
			}

			template<typename ... Blocks>
			static auto produceCoreDependencies(const Blocks& ... blocks) {
				return nested::template produceCoreDependencies(blocks.dependencies[0]...,blocks.dependencies[1]...,blocks.dependencies[2]...);
			}

			full_dependency_block narrowLeft(bool& save, std::size_t splitDimension, std::size_t radius) const {
				full_dependency_block res;
				if (Dims - 1 == splitDimension) {
					res.dependencies[0] = dependencies[0].narrowRight(save,splitDimension, radius);
					res.dependencies[1] = dependencies[1].narrowLeft(save,splitDimension, radius);
					res.dependencies[2] = dependencies[1].narrowRight(save,splitDimension, radius);
				} else {
					res.dependencies[0] = dependencies[0].narrowLeft(save,splitDimension, radius);
					res.dependencies[1] = dependencies[1].narrowLeft(save,splitDimension, radius);
					res.dependencies[2] = dependencies[2].narrowLeft(save,splitDimension, radius);
				}
				return res;
			}

			full_dependency_block narrowRight(bool& save, std::size_t splitDimension, std::size_t radius) const {
				full_dependency_block res;
				if (Dims - 1 == splitDimension) {
					res.dependencies[0] = dependencies[1].narrowLeft(save,splitDimension, radius);
					res.dependencies[1] = dependencies[1].narrowRight(save,splitDimension, radius);
					res.dependencies[2] = dependencies[2].narrowLeft(save,splitDimension, radius);
				} else {
					res.dependencies[0] = dependencies[0].narrowRight(save,splitDimension, radius);
					res.dependencies[1] = dependencies[1].narrowRight(save,splitDimension, radius);
					res.dependencies[2] = dependencies[2].narrowRight(save,splitDimension, radius);
				}
				return res;
			}
		};

		template<typename Iter>
		struct full_dependency_block<Iter,0> {

			using iteration_reference = detail::iteration_reference<Iter>;

			enum { num_dependencies = 1 };

			iteration_reference dependency;

			void setCenter(const iteration_reference& ref) {
				dependency = ref;
			}

			const iteration_reference& getCenter() const {
				return dependency;
			}

			template<typename Op>
			void forEach(const Op& op) const {
				op(dependency);
			}

			core::impl::reference::dependencies<core::impl::reference::fixed_sized<num_dependencies>> toCoreDependencies() const {
				return core::after(dependency);
			}

			template<typename ... Blocks>
			static auto produceCoreDependencies(const Blocks& ... blocks) {
				return core::after(blocks.dependency...);
			}

			full_dependency_block narrowLeft(bool& save, std::size_t, std::size_t radius) const {
				full_dependency_block res;
				res.dependency = dependency.getLeft();
				if (!dependency.getRange().empty() && getMinimumDimensionLength(res.dependency.getRange()) < radius) save = false;
				return res;
			}

			full_dependency_block narrowRight(bool& save, std::size_t, std::size_t radius) const {
				full_dependency_block res;
				res.dependency = dependency.getRight();
				if (!dependency.getRange().empty() && getMinimumDimensionLength(res.dependency.getRange()) < radius) save = false;
				return res;
			}
		};

	}

	template<typename Iter, std::size_t radius>
	class full_neighborhood_sync_dependency : public detail::loop_dependency {

		enum { num_dimensions = detail::dimensions<Iter>::value };

		using deps_block = detail::full_dependency_block<Iter,num_dimensions>;

		deps_block deps;

		full_neighborhood_sync_dependency(const deps_block& deps) : deps(deps) {}

	public:

		full_neighborhood_sync_dependency(const detail::iteration_reference<Iter>& loop) : deps() {
			deps.setCenter(loop);
		}

		const detail::range<Iter>& getCenterRange() const {
			return deps.getCenter().getRange();
		}

		std::vector<detail::range<Iter>> getRanges() const {
			std::vector<detail::range<Iter>> res;
			deps.forEach([&](const auto& dep) {
				if (!dep.getRange().empty()) res.push_back(dep.getRange());
			});
			return res;
		}

		auto toCoreDependencies() const {
			return deps.toCoreDependencies();
		}

		detail::SubDependencies<full_neighborhood_sync_dependency<Iter,radius>> split(const detail::range<Iter>& left, const detail::range<Iter>& right) const {
			using splitter = detail::range_spliter<Iter>;

			auto splitDim = splitter::getSplitDimension(deps.getCenter().getDepth());

			// prepare safety flag
			bool save_left = true;
			bool save_right = true;

			// compute left and right sub-dependencies
			full_neighborhood_sync_dependency res_left(deps.narrowLeft(save_left,splitDim,radius));
			full_neighborhood_sync_dependency res_right(deps.narrowRight(save_right,splitDim,radius));

			// check coverage and build up result
			return {
				save_left && res_left.getCenterRange().covers(left) ? res_left : *this,
				save_right && res_right.getCenterRange().covers(right) ? res_right : *this
			};
		}

		friend std::ostream& operator<< (std::ostream& out, const full_neighborhood_sync_dependency& dep) {
			return out << "[" << utils::join(",", dep.getRanges()) << "]";
		}

	};


	template<typename Iter>
	class after_all_sync_dependency : public detail::loop_dependency {

		// the type of iteration dependency
		using iteration_reference = detail::iteration_reference<Iter>;

		iteration_reference dependency;

	public:

		after_all_sync_dependency(const detail::iteration_reference<Iter>& loop)
			: dependency(loop) {}

		const detail::range<Iter>& getCenterRange() const {
			return dependency.getRange();
		}

		std::vector<detail::range<Iter>> getRanges() const {
			std::vector<detail::range<Iter>> res;
			res.push_back(dependency.getRange());
			return res;
		}

		auto toCoreDependencies() const {
			return core::after(dependency);
		}

		detail::SubDependencies<after_all_sync_dependency<Iter>> split(const detail::range<Iter>&, const detail::range<Iter>&) const {
			// this dependency never changes
			return { *this, *this };
		}

		friend std::ostream& operator<< (std::ostream& out, const after_all_sync_dependency& dep) {
			return out << "[" << dep.getCenterRange() << "]";
		}

	};


	template<typename Iter, typename InnerBody, typename BoundaryBody, typename Dependency>
	detail::loop_reference<Iter> pforWithBoundary(const detail::range<Iter>& r, const InnerBody& innerBody, const BoundaryBody& boundaryBody, const Dependency& dependency) {

		struct RecArgs {
			std::size_t depth;
			detail::range<Iter> range;
			Dependency dependencies;
		};

		// keep a copy of the full range
		auto full = r;

		// trigger parallel processing
		return { r, core::prec(
			[](const RecArgs& rg) {
				// if there is only one element left, we reached the base case
				return rg.range.size() <= 1;
			},
			[innerBody,boundaryBody,full](const RecArgs& rg) {
				// apply the body operation to every element in the remaining range
				rg.range.forEachWithBoundary(full,innerBody,boundaryBody);
			},
			core::pick(
				[](const RecArgs& rg, const auto& nested) {
					// in the step case we split the range and process sub-ranges recursively
					auto fragments = rg.range.split(rg.depth);
					auto& left = fragments.left;
					auto& right = fragments.right;
					auto dep = rg.dependencies.split(left,right);
					return core::parallel(
						nested(dep.left.toCoreDependencies(), RecArgs{rg.depth+1,left, dep.left} ),
						nested(dep.right.toCoreDependencies(), RecArgs{rg.depth+1,right,dep.right})
					);
				},
				[innerBody,boundaryBody,full](const RecArgs& rg, const auto&) {
					// the alternative is processing the step sequentially
					rg.range.forEachWithBoundary(full,innerBody,boundaryBody);
				}
			)
		)(dependency.toCoreDependencies(),RecArgs{0,r,dependency}) };
	}

	template<typename Iter, typename InnerBody, typename BoundaryBody>
	detail::loop_reference<Iter> pforWithBoundary(const detail::range<Iter>& r, const InnerBody& innerBody, const BoundaryBody& boundaryBody, const no_dependencies&) {

		struct RecArgs {
			std::size_t depth;
			detail::range<Iter> range;
		};

		// keep a copy of the full range
		auto full = r;

		// trigger parallel processing
		return { r, core::prec(
			[](const RecArgs& r) {
				// if there is only one element left, we reached the base case
				return r.range.size() <= 1;
			},
			[innerBody,boundaryBody,full](const RecArgs& r) {
				// apply the body operation to every element in the remaining range
				r.range.forEachWithBoundary(full,innerBody,boundaryBody);
			},
			core::pick(
				[](const RecArgs& r, const auto& nested) {
					// in the step case we split the range and process sub-ranges recursively
					auto fragments = r.range.split(r.depth);
					auto& left = fragments.left;
					auto& right = fragments.right;
					return core::parallel(
						nested(RecArgs{ r.depth+1, left }),
						nested(RecArgs{ r.depth+1, right })
					);
				},
				[innerBody,boundaryBody,full](const RecArgs& r, const auto&) {
					// the alternative is processing the step sequentially
					r.range.forEachWithBoundary(full,innerBody,boundaryBody);
				}
			)
		)(RecArgs{ 0 , r }) };
	}




	template<typename Iter, typename Point, typename Action>
	detail::loop_reference<Iter> after(const detail::loop_reference<Iter>& loop, const Point& point, const Action& action) {

		// get the full range
		auto r = loop.getRange();

		struct RecArgs {
			std::size_t depth;
			detail::range<Iter> range;
			one_on_one_dependency<Iter> dependencies;
		};

		// get the initial dependency
		auto dependency = one_on_one(loop);

		// trigger parallel processing
		return { r, core::prec(
			[point](const RecArgs& rg) {
				// check whether the point of action is covered by the current range
				return !rg.range.covers(point);
			},
			[action,point](const RecArgs& rg) {
				// trigger the action if the current range covers the point
				if (rg.range.covers(point)) action();

			},
			core::pick(
				[](const RecArgs& rg, const auto& nested) {
					// in the step case we split the range and process sub-ranges recursively
					auto fragments = rg.range.split(rg.depth);
					auto& left = fragments.left;
					auto& right = fragments.right;
					auto dep = rg.dependencies.split(left,right);
					return core::parallel(
						nested(dep.left.toCoreDependencies(), RecArgs{rg.depth+1, left, dep.left} ),
						nested(dep.right.toCoreDependencies(), RecArgs{rg.depth+1, right,dep.right})
					);
				},
				[action,point](const RecArgs& rg, const auto&) {
					// trigger the action if the current range covers the point
					if (rg.range.covers(point)) action();
				}
			)
		)(dependency.toCoreDependencies(),RecArgs{0,r,dependency}) };
	}

} // end namespace algorithm
} // end namespace user
} // end namespace api
} // end namespace allscale
