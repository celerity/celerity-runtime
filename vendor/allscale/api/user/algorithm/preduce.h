#pragma once

#include <utility>

#include "allscale/api/core/prec.h"

#include "allscale/api/user/algorithm/pfor.h"

#include "allscale/utils/assert.h"
#include "allscale/utils/vector.h"

namespace allscale {
namespace api {
namespace user {
namespace algorithm {


	// ----- fold / reduce ------

	/**
	 * The most generic implementation of the reduction operator. All other
	 * reductions are reduced to this implementation.
	 *
	 * @param a the begin of a range of elements to be reduced
	 * @param b the end (exclusive) of a range of elements to be reduced
	 * @param reduce the operation capable of performing a reduction over a subrange
	 * @param aggregate the operation capable of performing a reduction over a subrange
	 */
	template<
		typename Iter,
		typename RangeReductionOp,
		typename AggregationOp
	>
	core::treeture<typename utils::lambda_traits<AggregationOp>::result_type>
	preduce(
			const Iter& a,
			const Iter& b,
			const RangeReductionOp& reduce,
			const AggregationOp& aggregate
		) {

		using res_type = typename utils::lambda_traits<AggregationOp>::result_type;

		// define the argument struct
		struct RecArgs {
			std::size_t depth;
			algorithm::detail::range<Iter> range;
		};

		return core::prec(
			[](const RecArgs& r) {
				return r.range.size() <= 1;
			},
			[reduce](const RecArgs& r)->res_type {
				return reduce(r.range.begin(),r.range.end());
			},
			core::pick(
				[aggregate](const RecArgs& r, const auto& nested) {
					// here we have the binary splitting
					auto fragments = r.range.split(r.depth);
					auto left = fragments.left;
					auto right = fragments.right;
					return core::combine(nested(RecArgs{ r.depth+1, left }),nested(RecArgs{ r.depth+1, right }),aggregate);
				},
				[reduce](const RecArgs& r, const auto&)->res_type {
					return reduce(r.range.begin(),r.range.end());
				}
			)
		)(RecArgs{ 0, algorithm::detail::range<Iter>(a, b) });
	}



	/**
	 * A variant of the preduce operator where the range based reduction step
	 * is assembled from a set of utilities to create, update, and reduce a local temporary value.
	 */
	template<
		typename Iter,
		typename FoldOp,
		typename ReduceOp,
		typename InitLocalState,
		typename FinishLocalState
	>
	core::treeture<typename utils::lambda_traits<ReduceOp>::result_type>
	preduce(
			const Iter& a,
			const Iter& b,
			const FoldOp& fold,
			const ReduceOp& reduce,
			const InitLocalState& init,
			const FinishLocalState& finish
		) {

		return preduce(
				a, b, [init,fold,finish](const Iter& a, const Iter& b) {
					auto res = init();
					algorithm::detail::range<Iter>(a,b).forEach([&](const auto& cur){
						fold(cur,res);
					});
					return finish(res);
				},
				reduce
		);

	}

	// ----- reduction ------

	template<typename Iter, typename Op>
	core::treeture<typename utils::lambda_traits<Op>::result_type>
	preduce(const Iter& a, const Iter& b, const Op& op) {
		using res_type = typename utils::lambda_traits<Op>::result_type;

		return preduce(
				a,b,
				[op](const res_type& cur, res_type& res) {
					res = op(cur,res);
				},
				op,
				[](){ return res_type(); },
				[](const res_type& r) { return r; }
		);

	}

	/**
	 * A parallel reduce implementation over the elements of the given container.
	 */
	template<typename Container, typename Op>
	core::treeture<typename utils::lambda_traits<Op>::result_type>
	preduce(Container& c, Op& op) {
		return preduce(c.begin(), c.end(), op);
	}

	/**
	 * A parallel reduce implementation over the elements of the given container.
	 */
	template<typename Container, typename Op>
	core::treeture<typename utils::lambda_traits<Op>::result_type>
	preduce(const Container& c, const Op& op) {
		return preduce(c.begin(), c.end(), op);
	}


	template<
		typename Iter,
		typename MapOp,
		typename ReduceOp,
		typename InitLocalState
	>
	core::treeture<typename utils::lambda_traits<ReduceOp>::result_type>
	preduce(
			const Iter& a,
			const Iter& b,
			const MapOp& map,
			const ReduceOp& reduce,
			const InitLocalState& init
		) {

		return preduce(a, b, map, reduce, init, ([](typename utils::lambda_traits<ReduceOp>::result_type r) { return r; } ));
	}

	template<
		typename Container,
		typename MapOp,
		typename ReduceOp,
		typename InitLocalState,
		typename ReduceLocalState
	>
	core::treeture<typename utils::lambda_traits<ReduceOp>::result_type>
	preduce(
			const Container& c,
			const MapOp& map,
			const ReduceOp& reduce,
			const InitLocalState& init,
			const ReduceLocalState& exit
		) {

		return preduce(c.begin(), c.end(), map, reduce, init, exit);

	}

	template<
		typename Container,
		typename MapOp,
		typename ReduceOp,
		typename InitLocalState
	>
	core::treeture<typename utils::lambda_traits<ReduceOp>::result_type>
	preduce(
			const Container& c,
			const MapOp& map,
			const ReduceOp& reduce,
			const InitLocalState& init
		) {

		return preduce(c.begin(), c.end(), map, reduce, init);

	}

} // end namespace algorithm
} // end namespace user
} // end namespace api
} // end namespace allscale
