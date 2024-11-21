#pragma once

#include "async_event.h"
#include "pilot.h"
#include "ranges.h"
#include "types.h"
#include "utils.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>


namespace celerity::detail {

/// Interface for peer-to-peer and collective communication across nodes to be implemented for MPI or similar system APIs.
///
/// Celerity maintains one root communicator which can be cloned collectively, and the same communicator instance in this "cloning tree" must participate in
/// corresponding operations on each node. Communicator instances themselves are not thread-safe, but if there are multiple (cloned) instances, each may be used
/// from their own thread.
///
/// Peer-to-peer operations (send/receive/poll) can be arbitrarily re-ordered by the communicator, but collectives will
/// always be executed precisely in the order they are submitted.
class communicator {
  public:
	/// Addresses a 1/2/3-dimensional subrange of a type-erased (buffer) allocation to be sent from or received into.
	struct stride {
		range<3> allocation_range;
		subrange<3> transfer;
		size_t element_size = 1;

		friend bool operator==(const stride& lhs, const stride& rhs) {
			return lhs.allocation_range == rhs.allocation_range && lhs.transfer == rhs.transfer && lhs.element_size == rhs.element_size;
		}
		friend bool operator!=(const stride& lhs, const stride& rhs) { return !(lhs == rhs); }
	};

	communicator() = default;
	communicator(const communicator&) = delete;
	communicator(communicator&&) = delete;
	communicator& operator=(const communicator&) = delete;
	communicator& operator=(communicator&&) = delete;

	/// Communicator destruction is a collective operation like `collective_barrier`.
	///
	/// The user must ensure that any asynchronous operation is already complete when the destructor runs.
	virtual ~communicator() = default;

	/// Returns the number of nodes (processes) that are part of this communicator.
	virtual size_t get_num_nodes() const = 0;

	/// Returns the 0-based id of the local node in the communicator.
	virtual node_id get_local_node_id() const = 0;

	/// Asynchronously sends a pilot message, returning without acknowledgement from the receiver. The pilot is copied internally and the reference does not
	/// need to remain live after the function returns.
	virtual void send_outbound_pilot(const outbound_pilot& pilot) = 0;

	/// Returns all inbound pilots received on this communicator since the last invocation of the same function. Never blocks.
	[[nodiscard]] virtual std::vector<inbound_pilot> poll_inbound_pilots() = 0;

	/// Begins sending strided data (that was previously announced using an outbound_pilot) to the specified node. The `base` allocation must remain live until
	/// the returned event completes, and no element inside `stride` must be written to during that time.
	[[nodiscard]] virtual async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) = 0;

	/// Begins receiving strided data (which was previously announced using an inbound_pilot) from the specified node. The `base` allocation must remain live
	/// until the returned event completes, and no element inside `stride` must be written to during that time.
	[[nodiscard]] virtual async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) = 0;

	/// Creates a new communicator that is fully concurrent to this one, and which has its own "namespace" for peer-to-peer and collective operations.
	///
	/// Must be ordered identically to all other collective operations on this communicator across all nodes.
	virtual std::unique_ptr<communicator> collective_clone() = 0;

	/// Blocks until all nodes in this communicator have called `collective_barrier()`.
	///
	/// Must be ordered identically to all other collective operations on this communicator across all nodes.
	virtual void collective_barrier() = 0;
};

} // namespace celerity::detail

/// Required for caching strided datatypes in `mpi_communicator`.
template <>
struct std::hash<celerity::detail::communicator::stride> {
	size_t operator()(const celerity::detail::communicator::stride& stride) const {
		size_t h = 0;
		for(int d = 0; d < 3; ++d) {
			celerity::detail::utils::hash_combine(h, stride.allocation_range[d]);
			celerity::detail::utils::hash_combine(h, stride.transfer.offset[d]);
			celerity::detail::utils::hash_combine(h, stride.transfer.range[d]);
		}
		celerity::detail::utils::hash_combine(h, stride.element_size);
		return h;
	}
};
