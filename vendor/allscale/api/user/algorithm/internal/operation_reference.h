#include <utility>

#include "allscale/api/core/treeture.h"

#include "allscale/utils/assert.h"

namespace allscale {
namespace api {
namespace user {
namespace algorithm {
namespace internal {


	/**
	 * An operation reference is an (optional) base implementation
	 * of the return values of asynchronous operations. Unlike plain
	 * treetures, operator references are waiting for their tasks
	 * to be completed before destruction.
	 */
	class operation_reference {

		/**
		 * The treeture wrapped by this references, which corresponds
		 * to the root task of the asynchronously started task.
		 */
		core::treeture<void> handle;

	public:

		/**
		 * A simple constructor taking 'ownership' on the given treeture.
		 */
		operation_reference(core::treeture<void>&& handle)
			: handle(std::move(handle)) {}

		/**
		 * A simple constructor taking 'ownership' on the given completed task.
		 */
		operation_reference(core::detail::completed_task<void>&&)
			: handle() {}

		/**
		 * A default constructor, not owning or syncing on anything.
		 */
		operation_reference() {};

		/**
		 * Operation references may not be copied.
		 */
		operation_reference(const operation_reference&) = delete;

		/**
		 * Operation references may be moved.
		 */
		operation_reference(operation_reference&&) = default;

		/**
		 * Operation references may not be copied.
		 */
		operation_reference& operator=(const operation_reference&) = delete;

		/**
		 * Operation references may be moved.
		 */
		operation_reference& operator=(operation_reference&&) = default;

		/**
		 * Upon destruction, the references is waiting on the underlying
		 * task if it is still owned.
		 */
		~operation_reference() {
			// if handle is still valid, wait for its completion
			if (handle.isValid()) handle.wait();
		}

		/**
		 * A non-blocking check whether the referenced operation is done.
		 */
		bool isDone() const {
			return handle.isDone();
		}

		/**
		 * Determines whether a task is attached to this reference.
		 */
		bool isValid() const {
			return handle.isValid();
		}

		/**
		 * Disconnects the referenced task, causing this reference no longer
		 * to wait on the given task upon destruction.
		 *
		 * @return returns the maintained task handle
		 */
		core::treeture<void> detach() {
			return std::move(handle);
		}

		/**
		 * Blocks until the underlying operation has been completed.
		 */
		void wait() {
			handle.wait();
		}

	};


} // end namespace internal
} // end namespace algorithm
} // end namespace user
} // end namespace api
} // end namespace allscale
