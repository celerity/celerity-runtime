#pragma once

#include <array>
#include <utility>
#include <tuple>
#include <vector>

#include "allscale/api/user/data/grid.h"
#include "allscale/api/user/data/static_grid.h"

#include "allscale/api/user/algorithm/pfor.h"
#include "allscale/api/user/algorithm/async.h"
#include "allscale/api/user/algorithm/internal/operation_reference.h"

#include "allscale/utils/bitmanipulation.h"
#include "allscale/utils/unused.h"
#include "allscale/utils/vector.h"

namespace allscale {
namespace api {
namespace user {
namespace algorithm {


	// ---------------------------------------------------------------------------------------------
	//									    Declarations
	// ---------------------------------------------------------------------------------------------


	template<std::size_t dims>
	using Coordinate = utils::Vector<std::int64_t,dims>;

	template<std::size_t dims>
	using Size = Coordinate<dims>;

	namespace implementation {

		struct sequential_iterative;

		struct coarse_grained_iterative;

		struct fine_grained_iterative;

		struct sequential_recursive;

		struct parallel_recursive;

	}

	template<typename TimeStampFilter, typename LocationFilter, typename Action>
	class Observer;

	template<typename T>
	struct is_observer;

	template<typename Impl>
	class stencil_reference;

	template<
		typename Impl = implementation::fine_grained_iterative, typename Container, typename InnerUpdate, typename BoundaryUpdate,
		typename ... ObserverTimeFilters, typename ... ObserverLocationFilters, typename ... ObserverActions
	>
	std::enable_if_t<!is_observer<BoundaryUpdate>::value,stencil_reference<Impl>> stencil(
		Container& res, std::size_t steps, const InnerUpdate& innerUpdate, const BoundaryUpdate& boundaryUpdate,
		const Observer<ObserverTimeFilters,ObserverLocationFilters,ObserverActions>& ... observers
	);

	template<
		typename Impl = implementation::fine_grained_iterative, typename Container, typename Update,
		typename ... ObserverTimeFilters, typename ... ObserverLocationFilters, typename ... ObserverActions
	>
	stencil_reference<Impl> stencil(
		Container& res, std::size_t steps, const Update& update,
		const Observer<ObserverTimeFilters,ObserverLocationFilters,ObserverActions>& ... observers
	);

	template<typename TimeStampFilter, typename LocationFilter, typename Action>
	Observer<TimeStampFilter,LocationFilter,Action> observer(const TimeStampFilter& timeFilter, const LocationFilter& locationFilter, const Action& action);

	// ---------------------------------------------------------------------------------------------
	//									    Definitions
	// ---------------------------------------------------------------------------------------------


	template<typename Impl>
	class stencil_reference : public internal::operation_reference {

	public:

		// inherit all constructors
		using operation_reference::operation_reference;

	};

	template<
		typename Impl, typename Container, typename InnerUpdate, typename BoundaryUpdate,
		typename ... ObserverTimeFilters, typename ... ObserverLocationFilters, typename ... ObserverActions
	>
	std::enable_if_t<!is_observer<BoundaryUpdate>::value,stencil_reference<Impl>> stencil(
			Container& a, std::size_t steps, const InnerUpdate& innerUpdate, const BoundaryUpdate& boundaryUpdate,
			const Observer<ObserverTimeFilters,ObserverLocationFilters,ObserverActions>& ... observers
		) {

		// forward everything to the implementation
		return Impl().process(a,steps,innerUpdate,boundaryUpdate,observers...);

	}

	template<
		typename Impl, typename Container, typename Update,
		typename ... ObserverTimeFilters, typename ... ObserverLocationFilters, typename ... ObserverActions
	>
	stencil_reference<Impl> stencil(Container& a, std::size_t steps, const Update& update,const Observer<ObserverTimeFilters,ObserverLocationFilters,ObserverActions>& ... observers) {

		// use the same update for inner and boundary updates
		return stencil<Impl>(a,steps,update,update,observers...);

	}

	template<typename TimeStampFilter, typename LocationFilter, typename Action>
	class Observer {
	public:
		TimeStampFilter isInterestedInTime;
		LocationFilter isInterestedInLocation;
		Action trigger;

		Observer(const TimeStampFilter& timeFilter, const LocationFilter& locationFilter, const Action& action)
			: isInterestedInTime(timeFilter), isInterestedInLocation(locationFilter), trigger(action) {}
	};

	template<typename T>
	struct is_observer : public std::false_type {};

	template<typename T, typename L, typename A>
	struct is_observer<Observer<T,L,A>> : public std::true_type {};

	template<typename TimeStampFilter, typename LocationFilter, typename Action>
	Observer<TimeStampFilter,LocationFilter,Action> observer(const TimeStampFilter& timeFilter, const LocationFilter& locationFilter, const Action& action) {
		return Observer<TimeStampFilter,LocationFilter,Action>(timeFilter,locationFilter,action);
	}

	namespace implementation {

		namespace detail {

			template<typename Op>
			void staticForEach(const Op&) {
				// nothing to do
			}

			template<typename Op, typename First, typename ... Rest>
			void staticForEach(const Op& op, const First& first, const Rest& ... rest) {
				op(first);
				staticForEach(op,rest...);
			}

		}


		// -- Iterative Stencil Implementation ---------------------------------------------------------

		struct sequential_iterative {

			template<typename Container, typename InnerUpdate, typename BoundaryUpdate, typename ... Observers>
			stencil_reference<sequential_iterative> process(Container& a, std::size_t steps, const InnerUpdate& innerUpdate, const BoundaryUpdate& boundaryUpdate, const Observers& ... observers) {

				// return handle to asynchronous execution
				return async([&a,steps,innerUpdate,boundaryUpdate,observers...]{

					// iterative implementation
					Container b(a.size());

					Container* x = &a;
					Container* y = &b;

					using iter_type = decltype(a.size());

					for(std::size_t t=0; t<steps; t++) {

						// loop based sequential implementation
						user::algorithm::detail::forEach(iter_type(0),a.size(),
							// inner update operation
							[x,y,t,innerUpdate](const iter_type& i){
								(*y)[i] = innerUpdate(t,i,*x);
							},
							// boundary update operation
							[x,y,t,boundaryUpdate](const iter_type& i){
								(*y)[i] = boundaryUpdate(t,i,*x);
							}
						);

						// check observers
						detail::staticForEach(
							[&](const auto& observer){
								// check whether this time step is of interest
								if(!observer.isInterestedInTime(t)) return;
								// walk through space
								user::algorithm::detail::forEach(iter_type(0),a.size(),
									[&](const iter_type& i) {
										if (observer.isInterestedInLocation(i)) {
											observer.trigger(t,i,(*y)[i]);
										}
									}
								);
							},
							observers...
						);

						// swap buffers
						std::swap(x,y);
					}

					// make sure result is in a
					if (x != &a) {
						// move final data to the original container
						std::swap(a,b);
					}

				});

			}
		};


		struct coarse_grained_iterative {

			template<typename Container, typename InnerUpdate, typename BoundaryUpdate, typename ... Observers>
			stencil_reference<coarse_grained_iterative> process(Container& a, std::size_t steps, const InnerUpdate& inner, const BoundaryUpdate& boundary, const Observers& ... observers) {

				// return handle to asynchronous execution
				return async([&a,steps,inner,boundary,observers...]{

					// iterative implementation
					Container b(a.size());

					Container* x = &a;
					Container* y = &b;

					using iter_type = decltype(a.size());

					for(std::size_t t=0; t<steps; t++) {

						// loop based parallel implementation with blocking synchronization
						pforWithBoundary(iter_type(0),a.size(),
							[x,y,t,inner](const iter_type& i){
								(*y)[i] = inner(t,i,*x);
							},
							[x,y,t,boundary](const iter_type& i){
								(*y)[i] = boundary(t,i,*x);
							}
						);

						// check observers
						detail::staticForEach(
							[&](const auto& observer){
								// check whether this time step is of interest
								if(!observer.isInterestedInTime(t)) return;
								// walk through space
								algorithm::pfor(iter_type(0),a.size(),
									[&](const iter_type& i) {
										if (observer.isInterestedInLocation(i)) {
											observer.trigger(t,i,(*y)[i]);
										}
									}
								);
							},
							observers...
						);

						// swap buffers
						std::swap(x,y);
					}

					// make sure result is in a
					if (x != &a) {
						// move final data to the original container
						std::swap(a,b);
					}

				});

			}
		};


		struct fine_grained_iterative {

			template<typename Container, typename InnerUpdate, typename BoundaryUpdate, typename ... Observers>
			stencil_reference<fine_grained_iterative> process(Container& a, std::size_t steps, const InnerUpdate& inner, const BoundaryUpdate& boundary, const Observers& ... observers) {

				// return handle to asynchronous execution
				return async([&a,steps,inner,boundary,observers...]{

					// iterative implementation
					Container b(a.size());

					Container* x = &a;
					Container* y = &b;

					using iter_type = decltype(a.size());

					user::algorithm::detail::loop_reference<iter_type> ref;

					for(std::size_t t=0; t<steps; t++) {

						// loop based parallel implementation with fine grained dependencies
						ref = algorithm::pforWithBoundary(iter_type(0),a.size(),
							[x,y,t,inner](const iter_type& i){
								(*y)[i] = inner(t,i,*x);
							},
							[x,y,t,boundary](const iter_type& i){
								(*y)[i] = boundary(t,i,*x);
							},
							full_neighborhood_sync(ref)
						);

						// check observers
						detail::staticForEach(
							[t,y,&ref,&a](const auto& observer){
								// check whether this time step is of interest
								if(!observer.isInterestedInTime(t)) return;
								// walk through space
								ref = algorithm::pfor(iter_type(0),a.size(),
									[t,y,observer](const iter_type& i) {
										if (observer.isInterestedInLocation(i)) {
											observer.trigger(t,i,(*y)[i]);
										}
									},
									one_on_one(ref)
								);
							},
							observers...
						);

						// swap buffers
						std::swap(x,y);
					}

					// wait for the task completion
					ref.wait();

					// make sure result is in a
					if (x != &a) {
						// move final data to the original container
						std::swap(a,b);
					}

				});

			}
		};



		// -- Recursive Stencil Implementation ---------------------------------------------------------

		namespace detail {

			using index_type = Coordinate<0>::element_type;
			using time_type = std::size_t;


			template<std::size_t dims>
			using Slopes = utils::Vector<index_type,dims>;

			template<std::size_t dims>
			class Base {
			public:

				struct range {
					index_type begin;
					index_type end;
				};

				std::array<range,dims> boundaries;

				static Base zero() {
					return full(0);
				}

				static Base full(std::size_t size) {
					static_assert(dims == 1, "This constructor only supports 1-d bases.");
					Base res;
					res.boundaries[0] = { 0, (index_type)size };
					return res;
				}

				template<typename T>
				static Base full(const utils::Vector<T,dims>& size) {
					Base res;
					for(std::size_t i=0; i<dims; i++) {
						res.boundaries[i] = { 0, size[i] };
					}
					return res;
				}

				bool empty() const {
					return size() == 0;
				}

				std::size_t size() const {
					std::size_t res = 1;
					for(const auto& cur : boundaries) {
						if (cur.begin >= cur.end) return 0;
						res *= (cur.end - cur.begin);
					}
					return res;
				}

				Coordinate<dims> extend() const {
					Coordinate<dims> res;
					for(std::size_t i = 0; i<dims; i++) {
						res[i] = getWidth(i);
					}
					return res;
				}

				index_type getWidth(std::size_t dim) const {
					return boundaries[dim].end - boundaries[dim].begin;
				}

				index_type getMinimumWidth() const {
					index_type res = getWidth(0);
					for(std::size_t i = 1; i<dims; i++) {
						res = std::min(res,getWidth(i));
					}
					return res;
				}

				index_type getMaximumWidth() const {
					index_type res = getWidth(0);
					for(std::size_t i = 1; i<dims; i++) {
						res = std::max(res,getWidth(i));
					}
					return res;
				}

				const range& operator[](std::size_t i) const {
					return boundaries[i];
				}

				range& operator[](std::size_t i) {
					return boundaries[i];
				}

				Base operator+(const Coordinate<dims>& other) const {
					Base res;
					for(std::size_t i=0; i<dims; i++) {
						res.boundaries[i] = { boundaries[i].begin + other[i], boundaries[i].end + other[i] };
					}
					return res;
				}

				friend std::ostream& operator<<(std::ostream& out, const Base& base) {
					if (dims == 0) return out << "[]";
					out << "[";
					for(std::size_t i=0; i<dims-1; i++) {
						out << base.boundaries[i].begin << "-" << base.boundaries[i].end << ",";
					}
					out  << base.boundaries[dims-1].begin << "-" << base.boundaries[dims-1].end;
					return out << "]";
				}
			};


			template<std::size_t dim>
			struct plain_scanner {

				plain_scanner<dim-1> nested;

				template<std::size_t full_dim, typename InnerBody, typename BoundaryBody, typename ObserverBody>
				void operator()(const Base<full_dim>& base, const InnerBody& inner, const BoundaryBody& boundary, const ObserverBody& observer, Coordinate<full_dim>& pos, std::size_t t, const Coordinate<full_dim>& size) const {
					constexpr const auto idx = full_dim - dim - 1;

					// compute boundaries
					auto from = base[idx].begin;
					auto to = base[idx].end;
					auto length = size[idx];

					// shift range to size window
					if (from > length) {
						from -= length;
						to -= length;
					}

					// process range from start to limit
					auto limit = std::min(to,length);
					processRange(base,inner,boundary,observer,pos,t,size,from,limit);

					// and if necessary the elements beyond, after a wrap-around
					if (to <= length) return;

					to -= length;
					processRange(base,inner,boundary,observer,pos,t,size,0,to);
				}

				template<std::size_t full_dim, typename InnerBody, typename BoundaryBody, typename ObserverBody>
				void processRange(const Base<full_dim>& base, const InnerBody& inner, const BoundaryBody& boundary, const ObserverBody& observer, Coordinate<full_dim>& pos, std::size_t t, const Coordinate<full_dim>& size, std::int64_t from, std::int64_t to) const {
					constexpr const auto idx = full_dim - dim - 1;

					// skip an empty range
					if (from >= to) return;

					// get inner range
					auto innerFrom = from;
					auto innerTo = to;

					// check left boundary
					if (innerFrom == 0) {

						// process left as a boundary
						pos[idx] = 0;
						nested(base,boundary,boundary,observer,pos,t,size);

						// skip this one from the inner part
						innerFrom++;
					}

					// check right boundary
					if (innerTo == size[idx]) {
						innerTo--;
					}

					// process inner part
					for(pos[idx]=innerFrom; pos[idx]<innerTo; pos[idx]++) {
						nested(base, inner, boundary, observer, pos, t, size);
					}

					// process right boundary (if necessary)
					if (innerTo != to) {
						pos[idx] = innerTo;
						nested(base,boundary,boundary,observer,pos,t,size);
					}

				}

			};

			template<>
			struct plain_scanner<0> {

				template<std::size_t full_dim, typename InnerBody, typename BoundaryBody, typename ObserverBody>
				void operator()(const Base<full_dim>& base, const InnerBody& inner, const BoundaryBody& boundary, const ObserverBody& observer, Coordinate<full_dim>& pos, std::size_t t, const Coordinate<full_dim>& size) const {
					constexpr const auto idx = full_dim - 1;

					// compute boundaries
					auto from = base[idx].begin;
					auto to = base[idx].end;
					auto length = size[idx];

					// shift range to size window
					if (from > length) {
						from -= length;
						to -= length;
					}

					// process range from start to limit
					auto limit = std::min(to,length);
					processRange(inner,boundary,observer,pos,t,size,from,limit);

					// and if necessary the elements beyond, after a wrap-around
					if (to <= length) return;

					to -= length;
					processRange(inner,boundary,observer,pos,t,size,0,to);
				}

				template<std::size_t full_dim, typename InnerBody, typename BoundaryBody, typename ObserverBody>
				void processRange(const InnerBody& inner, const BoundaryBody& boundary, const ObserverBody& observer, Coordinate<full_dim>& pos, std::size_t t, const Coordinate<full_dim>& size, std::int64_t from, std::int64_t to) const {
					constexpr const auto idx = full_dim - 1;

					// skip an empty range
					if (from >= to) return;

					// get inner range
					auto innerFrom = from;
					auto innerTo = to;

					// check left boundary
					if (innerFrom == 0) {

						// process left as a boundary
						pos[idx] = 0;
						boundary(pos,t);

						// skip this one from the inner part
						innerFrom++;
					}

					// check right boundary
					if (innerTo == size[idx]) {
						innerTo--;
					}

					// process inner part
					for(pos[idx]=innerFrom; pos[idx]<innerTo; pos[idx]++) {
						inner(pos, t);
					}

					// process right boundary (if necessary)
					if (innerTo != to) {
						pos[idx] = innerTo;
						boundary(pos,t);
					}

					// compute start/end for observers
					auto fromCoordinates = pos;
					fromCoordinates[idx] = from;

					auto toCoordinates = pos;
					for(std::size_t i=0; i<idx; i++) {
						toCoordinates[i]++;
					}
					toCoordinates[idx] = to;

					// run observer
					observer(fromCoordinates,toCoordinates,t);

				}

			};

			template<std::size_t length>
			class TaskDependencyList {

				core::task_reference dep;

				TaskDependencyList<length-1> nested;

			public:

				TaskDependencyList() {}

				template<typename ... Rest>
				TaskDependencyList(const core::task_reference& first, const Rest& ... rest)
					: dep(first), nested(rest...) {}


				// support conversion into core dependencies
				auto toCoreDependencies() const {
					return nested.toCoreDependencies(dep);
				}

				template<typename ... Deps>
				auto toCoreDependencies(const Deps& ... deps) const {
					return nested.toCoreDependencies(dep,deps...);
				}

			};

			template<>
			class TaskDependencyList<0> {

			public:

				TaskDependencyList() {}

				// support conversion into core dependencies
				auto toCoreDependencies() const {
					return core::after();
				}

				template<typename ... Deps>
				auto toCoreDependencies(const Deps& ... deps) const {
					return core::after(deps...);
				}

			};


			template<std::size_t dims>
			class ZoidDependencies : public TaskDependencyList<3*dims> {

				using super = TaskDependencyList<3*dims>;

			public:

				// TODO: support dependency refinement

				// inherit constructors
				using super::super;

			};


			template<std::size_t dims>
			class Zoid {

				Base<dims> base;			// the projection of the zoid to the space dimensions

				Slopes<dims> slopes;		// the direction of the slopes

				time_type t_begin;			// the start time
				time_type t_end;			// the end time

			public:

				Zoid() {}

				Zoid(const Base<dims>& base, const Slopes<dims>& slopes, std::size_t t_begin, std::size_t t_end)
					: base(base), slopes(slopes), t_begin(t_begin), t_end(t_end) {}


				template<typename EvenOp, typename OddOp, typename EvenBoundaryOp, typename OddBoundaryOp, typename EvenObserverOp, typename OddObserverOp>
				void forEach(const EvenOp& even, const OddOp& odd, const EvenBoundaryOp& evenBoundary, const OddBoundaryOp& oddBoundary, const EvenObserverOp& evenObserver, const OddObserverOp& oddObserver, const Size<dims>& limits) const {

					// TODO: make this one cache oblivious

					// create the plain scanner
					plain_scanner<dims-1> scanner;

					Coordinate<dims> x;
					auto plainBase = base;

					// over the time
					for(std::size_t t = t_begin; t < t_end; ++t) {

						// process this plain
						if ( t & 0x1 ) {
							scanner(plainBase, odd, oddBoundary, oddObserver, x, t, limits);
						} else {
							scanner(plainBase, even, evenBoundary, evenObserver, x, t, limits);
						}

						// update the plain for the next level
						for(std::size_t i=0; i<dims; ++i) {
							plainBase[i].begin  += slopes[i];
							plainBase[i].end    -= slopes[i];
						}
					}

				}

				template<typename EvenOd, typename OddOp, typename EvenBoundaryOp, typename OddBoundaryOp, typename EvenObserverOp, typename OddObserverOp>
				core::treeture<void> pforEach(const ZoidDependencies<dims>& deps, const EvenOd& even, const OddOp& odd, const EvenBoundaryOp& evenBoundary, const OddBoundaryOp& oddBoundary, const EvenObserverOp& evenObserver, const OddObserverOp& oddObserver, const Size<dims>& limits) const {

					struct Params {
						Zoid zoid;
						ZoidDependencies<dims> deps;
					};

					// recursively decompose the covered space-time volume
					return core::prec(
						[](const Params& params) {
							// check whether this zoid can no longer be divided
							return params.zoid.isTerminal();
						},
						[&](const Params& params) {
							// process final steps sequentially
							params.zoid.forEach(even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits);
						},
						core::pick(
							[](const Params& params, const auto& rec) {
								// unpack parameters
								const auto& zoid = params.zoid;
								const auto& deps = params.deps;

								// make sure the zoid is not terminal
								assert_false(zoid.isTerminal());

								// check whether it can be split in space
								if (!zoid.isSpaceSplitable()) {
									// we need a time split
									auto parts = zoid.splitTime();
									return core::sequential(
										rec(deps.toCoreDependencies(),Params{parts.bottom,deps}),
										rec(deps.toCoreDependencies(),Params{parts.top,deps})
									);
								}

								// let's do a space split
								auto parts = zoid.splitSpace();

								// schedule depending on the orientation
								return (parts.opening)
										? core::sequential(
											rec(deps.toCoreDependencies(),Params{parts.c,deps}),
											core::parallel(
												rec(deps.toCoreDependencies(),Params{parts.l,deps}),
												rec(deps.toCoreDependencies(),Params{parts.r,deps})
											)
										)
										: core::sequential(
											core::parallel(
												rec(deps.toCoreDependencies(),Params{parts.l,deps}),
												rec(deps.toCoreDependencies(),Params{parts.r,deps})
											),
											rec(deps.toCoreDependencies(),Params{parts.c,deps})
										);


							},
							[&](const Params& params, const auto&) {
								// provide sequential alternative
								params.zoid.forEach(even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits);
							}
						)
					)(deps.toCoreDependencies(),Params{*this,deps});

				}

				template<typename EvenOd, typename OddOp, typename EvenBoundaryOp, typename OddBoundaryOp, typename EvenObserverOp, typename OddObserverOp>
				core::treeture<void> pforEach(const EvenOd& even, const OddOp& odd, const EvenBoundaryOp& evenBoundary, const OddBoundaryOp& oddBoundary, const EvenObserverOp& evenObserver, const OddObserverOp& oddObserver, const Size<dims>& limits) const {
					// run the pforEach with no initial dependencies
					return pforEach(ZoidDependencies<dims>(),even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits);
				}


				/**
				 * The height of this zoid in temporal direction.
				 */
				std::size_t getHeight() const {
					return std::size_t(t_end-t_begin);
				}

				/**
				 * Compute the number of elements this volume is covering
				 * when being projected to the space domain.
				 */
				int getFootprint() const {
					int size = 1;
					int dt = getHeight();
					for(std::size_t i=0; i<dims; i++) {
						auto curWidth = base.width(i);
						if (slopes[i] < 0) {
							curWidth += 2*dt;
						}
						size *= curWidth;
					}
					return size;
				}

				friend std::ostream& operator<<(std::ostream& out, const Zoid& zoid) {
					return out << "Zoid(" << zoid.base << "," << zoid.slopes << "," << zoid.t_begin << "-" << zoid.t_end << ")";
				}

			private:

				/**
				 * Tests whether this zoid can not be further divided.
				 */
				bool isTerminal() const {
					// reached when the height is 1 and width is 3
					return getHeight() <= 1 && base.getMaximumWidth() < 3;
				}


				/**
				 * Computes the width of the shadow projected of this zoid on
				 * the given space dimension.
				 */
				std::size_t getWidth(std::size_t dim) const {
					std::size_t res = base.getWidth(dim);
					if (slopes[dim] < 0) res += 2*getHeight();
					return res;
				}


				/**
				 * Determines whether this zoid is splitable in space.
				 */
				bool isSpaceSplitable() const {
					for(std::size_t i=0; i<dims; i++) {
						if (isSplitable(i)) return true;
					}
					return false;
				}

				/**
				 * Tests whether it can be split along the given space dimension.
				 */
				bool isSplitable(std::size_t dim) const {
					return getWidth(dim) > 4*getHeight();
				}

				// the result of a time split
				struct TimeDecomposition {
					Zoid top;
					Zoid bottom;
				};

				/**
				 * Splits this zoid in two sub-zoids along the time dimension. The
				 * First component will be the bottom, the second the top.
				 */
				TimeDecomposition splitTime() const {
					auto split = getHeight() / 2;

					Base<dims> mid = base;

					for(std::size_t i=0; i<dims; i++) {
						auto diff  = slopes[i] * split;
						mid[i].begin += diff;
						mid[i].end   -= diff;
					}

					return {
						Zoid(mid,   slopes, t_begin+split, t_end),		// top
						Zoid(base,  slopes, t_begin, t_begin+split)		// bottom
					};
				}


				// the result of a space split
				struct SpaceDecomposition {
					Zoid l;			// < the left fragment
					Zoid c;			// < the middle fragment
					Zoid r;			// < the right fragment
					bool opening;	// < determines whether the splitting dimension was opening (negative slope) or closing (positive slope)
				};

				/**
				 * Splits this zoid into three sub-zoids along the space dimension.
				 */
				SpaceDecomposition splitSpace() const {
					assert_true(isSpaceSplitable());

					// find longest dimension
					std::size_t max_dim = 0;
					std::size_t max_width = 0;
					for(std::size_t i=0; i<dims; i++) {
						std::size_t width = getWidth(i);
						if (width>max_width) {
							max_width = width;
							max_dim = i;
						}
					}

					// the max dimension is the split dimensin
					auto split_dim = max_dim;

					// check whether longest dimension can be split
					assert(isSplitable(split_dim));

					// create 3 fragments
					SpaceDecomposition res {
						*this, *this, *this, (slopes[split_dim] < 0)
					};

					// get the split point
					auto center = (base.boundaries[split_dim].begin + base.boundaries[split_dim].end) / 2;
					auto left = center;
					auto right = center;

					if (slopes[split_dim] < 0) {
						auto hight = getHeight();
						left -= hight;
						right += hight;
					}

					res.l.base.boundaries[split_dim].end = left;
					res.c.base.boundaries[split_dim] = { left, right };
					res.r.base.boundaries[split_dim].begin = right;

					// invert direction of center piece
					res.c.slopes[split_dim] *= -1;

					// return decomposition
					return res;
				}

			};

			/**
			 * A utility class for enumerating the dependencies of a task in a
			 * n-dimensional top-level task graph.
			 */
			template<std::size_t taskIdx, std::size_t pos>
			struct task_dependency_extractor {

				template<typename Body,typename ... Args>
				void operator()(const Body& body, const Args& ... args) {
					task_dependency_extractor<taskIdx,pos-1> nested;
					if (taskIdx & (1<<pos)) {
						nested(body,args...,taskIdx & ~(1<<pos));
					} else {
						nested(body,args...);
					}
				}

			};

			template<std::size_t taskIdx>
			struct task_dependency_extractor<taskIdx,0> {

				template<typename Body,typename ... Args>
				void operator()(const Body& body, const Args& ... args) {
					if (taskIdx & 0x1) {
						body(args...,taskIdx & ~0x1);
					} else {
						body(args...);
					}
				}

			};


			/**
			 * A utility class for enumerating the dependencies of a task in a
			 * n-dimensional top-level task graph.
			 */
			template<std::size_t Dims, std::size_t taskIdx>
			struct task_dependency_enumerator {

				template<typename Body>
				void operator()(const Body& body) {
					for(std::size_t i=0;i<=Dims;i++) {
						visit(body,i);
					}
				}

				template<typename Body>
				void visit(const Body& body,std::size_t numBits) {
					task_dependency_enumerator<Dims,taskIdx-1>().visit(body,numBits);
					if ((std::size_t)(utils::countOnes(taskIdx))==numBits) {
						task_dependency_extractor<taskIdx,Dims-1>()(body,taskIdx);
					}
				}

			};

			template<std::size_t Dims>
			struct task_dependency_enumerator<Dims,0> {

				template<typename Body>
				void visit(const Body& body,std::size_t numBits) {
					if (numBits == 0) {
						task_dependency_extractor<0,Dims-1>()(body,0);
					}
				}

			};

			/**
			 * A utility to statically enumerate the tasks and dependencies for
			 * the top-level zoid task decomposition scheme. On the top level,
			 * the set of tasks and its dependencies are isomorph to the vertices
			 * and edges in a n-dimensional hyper cube. This utility is enumerating
			 * those edges, as well as listing its predecessors according to the
			 * sub-set relation.
			 */
			template<std::size_t Dims>
			struct task_graph_enumerator {

				template<typename Body>
				void operator()(const Body& body) {
					task_dependency_enumerator<Dims,(1<<Dims)-1> enumerator;
					enumerator(body);
				}

			};


			template<std::size_t Dims>
			class ExecutionPlan {

				using zoid_type = Zoid<Dims>;

				// the execution plan of one layer -- represented as an embedded hyper-cube
				using layer_plan = std::array<zoid_type,1 << Dims>;

				// the list of execution plans of all layers
				std::vector<layer_plan> layers;

			public:

				template<typename EvenOp, typename OddOp, typename EvenBoundaryOp, typename OddBoundaryOp, typename EvenObserver, typename OddObserver>
				void runSequential(const EvenOp& even, const OddOp& odd, const EvenBoundaryOp& evenBoundary, const OddBoundaryOp& oddBoundary, const EvenObserver& evenObserver, const OddObserver& oddObserver, const Size<Dims>& limits) const {
					const std::size_t num_tasks = 1 << Dims;

					// fill a vector with the indices of the tasks
					std::array<std::size_t,num_tasks> order;
					for(std::size_t i = 0; i<num_tasks;++i) {
						order[i] = i;
					}

					// sort the vector by the number of 1-bits
					std::sort(order.begin(),order.end(),[](std::size_t a, std::size_t b){
						return getNumBitsSet(a) < getNumBitsSet(b);
					});

					// process zoids in obtained order
					for(const auto& cur : layers) {
						for(const auto& idx : order) {
							cur[idx].forEach(even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits);
						}
					}
				}

				template<typename EvenOp, typename OddOp, typename EvenBoundaryOp, typename OddBoundaryOp, typename EvenObserver, typename OddObserver>
				core::treeture<void> runParallel(const EvenOp& even, const OddOp& odd, const EvenBoundaryOp& evenBoundary, const OddBoundaryOp& oddBoundary, const EvenObserver& evenObserver, const OddObserver& oddObserver, const Size<Dims>& limits) const {

					const std::size_t num_tasks = 1 << Dims;

					// start tasks with mutual dependencies
					core::treeture<void> last = core::done();
					for(const auto& cur : layers) {

						std::array<core::treeture<void>,num_tasks> jobs;

						// walk through graph dependency graph
						enumTaskGraph([&](std::size_t idx, const auto& ... deps){

							// special case handling for first task (has to depend on previous task)
							if (idx == 0) {
								// create first task
								jobs[idx] = (last.isDone())
										? cur[idx].pforEach(even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits)
										: cur[idx].pforEach(ZoidDependencies<Dims>(last),even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits);
								return;
							}

							// create this task with corresponding dependencies
							jobs[idx] = cur[idx].pforEach(ZoidDependencies<Dims>(jobs[deps]...),even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,limits);

						});

						// update last
						last = std::move(jobs.back());
					}

					// return handle to last task
					return last;

				}

				static ExecutionPlan create(const Base<Dims>& base, std::size_t steps) {

					// get size of structure
					auto size = base.extend();

					// the the smallest width (this is the limiting factor for the height)
					auto width = base.getMinimumWidth();

					// get the height of the largest zoids, thus the height of each layer
					auto height = width/2;

					// compute base area partitioning
					struct split {
						typename Base<Dims>::range left;
						typename Base<Dims>::range right;
					};
					std::array<split,Dims> splits;
					for(std::size_t i = 0; i<Dims; i++) {
						auto curWidth = size[i];
						auto mid = curWidth - (curWidth - width) / 2;
						splits[i].left.begin  = 0;
						splits[i].left.end    = mid;
						splits[i].right.begin = mid;
						splits[i].right.end   = curWidth;
					}

					// create the list of per-layer plans to be processed
					ExecutionPlan plan;

					// process time layer by layer
					for(std::size_t t0=0; t0<steps; t0+=height) {

						// get the top of the current layer
						auto t1 = std::min<std::size_t>(t0+height,steps);

						// create the list of zoids in this step
						plan.layers.emplace_back();
						layer_plan& zoids = plan.layers.back();

						// generate binary patterns from 0 to 2^dims - 1
						for(size_t i=0; i < (1<<Dims); i++) {

							// get base and slopes of the current zoid
							Base<Dims> curBase = base;
							Slopes<Dims> slopes;

							// move base to center on field, edge, or corner
							for(size_t j=0; j<Dims; j++) {
								if (i & ((size_t)(1)<<j)) {
									slopes[j] = -1;
									curBase.boundaries[j] = splits[j].right;
								} else {
									slopes[j] = 1;
									curBase.boundaries[j] = splits[j].left;
								}
							}

							// count the number of ones -- this determines the execution order
							int num_ones = 0;
							for(size_t j=0; j<Dims; j++) {
								if (i & ((size_t)(1)<<j)) num_ones++;
							}

							// add to execution plan
							zoids[i] = Zoid<Dims>(curBase, slopes, t0, t1);
						}

					}

					// build the final result
					return plan;
				}

				template<typename Body>
				static void enumTaskGraph(const Body& body) {
					task_graph_enumerator<Dims>()(body);
				}

			private:

				static std::size_t getNumBitsSet(std::size_t mask) {
					return utils::countOnes((unsigned)mask);
				}

			};


			template<unsigned Dims>
			struct container_info_base {
				constexpr static const unsigned dimensions = Dims;
				using base_type = Base<Dims>;
			};


			template<typename Container>
			struct container_info : public container_info_base<1> {
				using index_type = detail::index_type;
			};

			template<typename T, std::size_t Dims>
			struct container_info<data::Grid<T,Dims>> : public container_info_base<Dims> {
				using index_type = utils::Vector<detail::index_type,Dims>;
			};

			template<typename T, std::size_t ... Sizes>
			struct container_info<data::StaticGrid<T,Sizes...>> : public container_info_base<sizeof...(Sizes)> {
				using index_type = utils::Vector<detail::index_type,sizeof...(Sizes)>;
			};

			template<typename Container>
			struct coordinate_converter {
				auto& operator()(const Coordinate<1>& pos) {
					return pos[0];
				}
			};

			template<typename T, std::size_t Dims>
			struct coordinate_converter<data::Grid<T,Dims>> {
				auto& operator()(const Coordinate<Dims>& pos) {
					return pos;
				}
			};

			template<typename T, std::size_t ... Sizes>
			struct coordinate_converter<data::StaticGrid<T,Sizes...>> {
				auto& operator()(const Coordinate<sizeof...(Sizes)>& pos) {
					return pos;
				}
			};

		}

		struct sequential_recursive {

			template<typename Container, typename InnerUpdate, typename BoundaryUpdate, typename ... Observers>
			stencil_reference<sequential_recursive> process(Container& a, std::size_t steps, const InnerUpdate& inner, const BoundaryUpdate& boundary, const Observers& ... observers) {

				using namespace detail;

				const unsigned dims = container_info<Container>::dimensions;
				using base_t = typename container_info<Container>::base_type;

				// iterative implementation
				Container b(a.size());

				// TODO:
				//  - switch internally to cache-oblivious access pattern (optional)

				// get size of structure
				base_t base = base_t::full(a.size());
				auto size = base.extend();

				// wrap update function into zoid-interface adapter
				auto even = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					b[p] = inner(t,p,a);
				};

				auto odd = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					a[p] = inner(t,p,b);
				};

				auto evenBoundary = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					b[p] = boundary(t,p,a);
				};

				auto oddBoundary = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					a[p] = boundary(t,p,b);
				};

				auto evenObserver = [&](const Coordinate<dims>& from, const Coordinate<dims>& to, time_t t){

					// create a operation handling one observer
					__allscale_unused auto handler = [&](const auto& observer){
						// check whether this time step is of interest
						if(!observer.isInterestedInTime(t)) return;
						// walk through space
						pfor(from,to,
							[&](const Coordinate<container_info<Container>::dimensions>& i) {
								coordinate_converter<Container> conv;
								if (observer.isInterestedInLocation(i)) {
									observer.trigger(t,i,b[conv(i)]);
								}
							}
						);
					};

					// process all observers
					__allscale_unused auto l = { 0,(handler(observers),0)... };
				};

				auto oddObserver = [&](const Coordinate<dims>& from, const Coordinate<dims>& to, time_t t){

					// create a operation handling one observer
					__allscale_unused auto handler = [&](const auto& observer){
						// check whether this time step is of interest
						if(!observer.isInterestedInTime(t)) return;
						// walk through space
						pfor(from,to,
							[&](const Coordinate<container_info<Container>::dimensions>& i) {
								coordinate_converter<Container> conv;
								if (observer.isInterestedInLocation(i)) {
									observer.trigger(t,i,a[conv(i)]);
								}
							}
						);
					};

					// process all observers
					__allscale_unused auto l = { 0,(handler(observers),0)... };
				};

				// get the execution plan
				auto exec_plan = ExecutionPlan<dims>::create(base,steps);

				// process the execution plan
				exec_plan.runSequential(even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,size);


				// make sure the result is in the a copy
				if (steps % 2) {
					std::swap(a,b);
				}

				// done
				return {};
			}
		};


		struct parallel_recursive {

			template<typename Container, typename InnerUpdate, typename BoundaryUpdate, typename ... Observers>
			stencil_reference<parallel_recursive> process(Container& a, std::size_t steps, const InnerUpdate& inner, const BoundaryUpdate& boundary, const Observers& ... observers) {

				using namespace detail;

				const unsigned dims = container_info<Container>::dimensions;
				using base_t = typename container_info<Container>::base_type;

				// iterative implementation
				Container b(a.size());

				// TODO:
				//  - switch internally to cache-oblivious access pattern (optional)
				//  - make parallel with fine-grained dependencies

				// get size of structure
				base_t base = base_t::full(a.size());
				auto size = base.extend();

				// wrap update function into zoid-interface adapter
				auto even = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					b[p] = inner(t,p,a);
				};

				auto odd = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					a[p] = inner(t,p,b);
				};

				auto evenBoundary = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					b[p] = boundary(t,p,a);
				};

				auto oddBoundary = [&](const Coordinate<dims>& pos, time_t t){
					coordinate_converter<Container> conv;
					auto p = conv(pos);
					a[p] = boundary(t,p,b);
				};

				auto evenObserver = [&](const Coordinate<dims>& from, const Coordinate<dims>& to, time_t t){

					// create a operation handling one observer
					__allscale_unused auto handler = [&](const auto& observer){
						// check whether this time step is of interest
						if(!observer.isInterestedInTime(t)) return;
						// walk through space
						pfor(from,to,
							[&](const Coordinate<container_info<Container>::dimensions>& i) {
								coordinate_converter<Container> conv;
								if (observer.isInterestedInLocation(i)) {
									observer.trigger(t,i,b[conv(i)]);
								}
							}
						);
					};

					// process all observers
					__allscale_unused auto l = { 0,(handler(observers),0)... };
				};

				auto oddObserver = [&](const Coordinate<dims>& from, const Coordinate<dims>& to, time_t t){

					// create a operation handling one observer
					__allscale_unused auto handler = [&](const auto& observer){
						// check whether this time step is of interest
						if(!observer.isInterestedInTime(t)) return;
						// walk through space
						pfor(from,to,
							[&](const Coordinate<container_info<Container>::dimensions>& i) {
								coordinate_converter<Container> conv;
								if (observer.isInterestedInLocation(i)) {
									observer.trigger(t,i,a[conv(i)]);
								}
							}
						);
					};

					// process all observers
					__allscale_unused auto l = { 0,(handler(observers),0)... };
				};

				// get the execution plan
				auto exec_plan = ExecutionPlan<dims>::create(base,steps);

				// process the execution plan
				exec_plan.runParallel(even,odd,evenBoundary,oddBoundary,evenObserver,oddObserver,size).wait();

				// make sure the result is in the a copy
				if (steps % 2) {
					std::swap(a,b);
				}

				// done
				return {};
			}
		};

	} // end namespace implementation


} // end namespace algorithm
} // end namespace user
} // end namespace api
} // end namespace allscale
