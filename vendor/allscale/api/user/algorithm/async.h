#pragma once

#include <utility>

#include "allscale/utils/assert.h"

#include "allscale/api/core/prec.h"

namespace allscale {
namespace api {
namespace user {
namespace algorithm {



	// ---------------------------------------------------------------------------------------------
	//									    Declarations
	// ---------------------------------------------------------------------------------------------


	/**
	 * A simple job wrapper processing a given task asynchronously. The task
	 * is wrapped to a simple recursion where there is a single base
	 * case step.
	 *
	 * @tparam Action the type of action
	 * @param action the action to be processed
	 * @return a treeture providing a reference the the result
	 */
	template<typename Action>
	core::treeture<std::result_of_t<Action()>> async(const Action& action);


	/**
	 * A simple job wrapper processing a given task asynchronously after the
	 * given dependencies are satisfied. The task is wrapped to a simple recursion
	 * where there is a single base case step.
	 *
	 * @tparam Dependencies the dependencies to await
	 * @tparam Action the type of action
	 * @param action the action to be processed
	 * @return a treeture providing a reference the the result
	 */
	template<typename Dependencies, typename Action>
	core::treeture<std::result_of_t<Action()>> async(Dependencies&& deps, const Action& action);




	// ---------------------------------------------------------------------------------------------
	//									    Definitions
	// ---------------------------------------------------------------------------------------------


	template<typename Action>
	core::treeture<std::result_of_t<Action()>> async(const Action& action) {
		return async(core::after(), action);
	}


	template<typename Dependencies, typename Action>
	core::treeture<std::result_of_t<Action()>> async(Dependencies&& deps, const Action& action) {
		struct empty {};
		return core::prec(
			[](empty){ return true; },
			[=](empty){
				return action();
			},
			[=](empty,const auto&){
				assert_fail() << "Should not be reached!";
				return action();
			}
		)(std::move(deps), empty());
	}


} // end namespace algorithm
} // end namespace user
} // end namespace api
} // end namespace allscale
