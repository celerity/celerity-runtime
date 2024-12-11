#pragma once

#include "async_event.h"
#include "communicator.h"
#include "grid.h"
#include "pilot.h"
#include "types.h"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>


namespace celerity::detail::receive_arbiter_detail {

/// A single box received by the communicator, as described earlier by an inbound pilot.
struct incoming_region_fragment {
	detail::box<3> box;
	async_event communication; ///< async communicator event for receiving this fragment

	bool is_complete() const { return communication.is_complete(); }
};

/// State for a single incomplete `receive` operation or a `begin_split_receive` / `await_split_receive_subregion` tree.
struct region_request {
	void* allocation;
	box<3> allocated_box;
	region<3> incomplete_region;
	std::vector<incoming_region_fragment> incoming_fragments;
	bool may_await_subregion;

	region_request(region<3> requested_region, void* const allocation, const box<3>& allocated_bounding_box, const bool may_await_subregion)
	    : allocation(allocation), allocated_box(allocated_bounding_box), incomplete_region(std::move(requested_region)),
	      may_await_subregion(may_await_subregion) {}
	bool do_complete();
};

/// A single chunk in a `gather_request` that is currently being received by the communicator.
struct incoming_gather_chunk {
	async_event communication; ///< async communicator event for receiving this chunk
};

/// State for a single incomplete `gather_receive` operation.
struct gather_request {
	void* allocation;
	size_t chunk_size;                                  ///< in bytes
	size_t num_incomplete_chunks;                       ///< number of chunks that are currently being received or for which we have not seen a pilot yet
	std::vector<incoming_gather_chunk> incoming_chunks; ///< chunks that are currently being received

	gather_request(void* const allocation, const size_t chunk_size, const size_t num_total_chunks)
	    : allocation(allocation), chunk_size(chunk_size), num_incomplete_chunks(num_total_chunks) {}
	bool do_complete();
};

// shared_ptrs for pointer stability (referenced by receive_arbiter::event)
using stable_region_request = std::shared_ptr<region_request>;
using stable_gather_request = std::shared_ptr<gather_request>;

/// A transfer that is only known through inbound pilots so far, but no `receive` / `begin_split_receive` has been issued so far.
struct unassigned_transfer {
	std::vector<inbound_pilot> pilots;
	bool do_complete();
};

/// A (non-gather) transfer that has been mentioned in one or more calls to `receive` / `begin_split_receive`. Note that there may be multiple disjoint
/// receives mapping to the same `transfer_id` as long as their regions are pairwise disconnected.
struct multi_region_transfer {
	size_t elem_size;                                   ///< in bytes
	std::vector<stable_region_request> active_requests; ///< all `receive`s and `begin_split_receive`s active for this transfer id.
	std::vector<inbound_pilot> unassigned_pilots;       ///< all inbound pilots that do not map to any `active_request`.

	explicit multi_region_transfer(const size_t elem_size) : elem_size(elem_size) {}
	explicit multi_region_transfer(const size_t elem_size, std::vector<inbound_pilot>&& unassigned_pilots)
	    : elem_size(elem_size), unassigned_pilots(std::move(unassigned_pilots)) {}
	bool do_complete();
};

/// A transfer originating through `gather_receive`. It is fully described by a single `gather_request`.
struct gather_transfer {
	stable_gather_request request;
	bool do_complete();
};

/// Depending on the order of inputs, transfers may start out as unassigned and will be replaced by either `multi_region_transfer`s or `gather_transfer`s
/// once explicit calls to the respective receive arbiter functions are made.
using transfer = std::variant<unassigned_transfer, multi_region_transfer, gather_transfer>;

} // namespace celerity::detail::receive_arbiter_detail

namespace celerity::detail {

/// Matches receive instructions to inbound pilots and triggers in-place payload receives on the communicator.
///
/// For scalability reasons, distributed command graph generation only yields exact destinations and buffer sub-ranges for push commands, while await-pushes do
/// not carry such information - they just denote the full region to be received. Sender nodes later communicate the exact ranges to the receiver during
/// execution time via pilot messages that are generated alongside the instruction graph.
///
/// The receive_arbiter's job is to match these inbound pilots to receive instructions generated from await-push commands to issue in-place receives (i.e.
/// `MPI_Recv`) of the data into an appropriate host allocation. Since these inputs may arrive in arbitrary order, it maintains a separate state machine for
/// each `transfer_id` to drive all operations that eventually result in completing an `async_event` for each receive instruction.
class receive_arbiter {
  public:
	/// `receive_arbiter` will use `comm` to poll for inbound pilots and issue payload-receives.
	explicit receive_arbiter(communicator& comm);

	receive_arbiter(const receive_arbiter&) = delete;
	receive_arbiter(receive_arbiter&&) = default;
	receive_arbiter& operator=(const receive_arbiter&) = delete;
	receive_arbiter& operator=(receive_arbiter&&) = default;
	~receive_arbiter();

	/// Receive a buffer region associated with a single transfer id `trid` into an existing `allocation` with size `allocated_box.size() * elem_size`. The
	/// `request` region must be fully contained in `allocated_box`, and the caller must ensure that it the communicator will not receive an inbound pilot that
	/// intersects `request` without being fully contained in it. The returned `async_event` will complete once the receive is complete.
	[[nodiscard]] async_event receive(const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size);

	/// Begin the reception of a buffer region into an existing allocation similar to `receive`, but do not await its completion with a single `async_event`.
	/// Instead, the caller must follow up with calls to `await_split_receive_subregion` to the same `transfer_id` whose request regions do not necessarily have
	/// to be disjoint, but whose union must be equal to the original `request`.
	void begin_split_receive(const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size);

	/// To be called after `begin_split_receive` to await receiving a `subregion` of the original request. Subregions passed to different invocations of this
	/// function may overlap, but must not exceed the original request. If the entire split-receive has finished already, this will return an instantly complete
	/// event.
	[[nodiscard]] async_event await_split_receive_subregion(const transfer_id& trid, const region<3>& subregion);

	/// Receive a contiguous chunk of data from every peer node, placing the results in `allocation[node_chunk_size * node_id]`. The location reserved for the
	/// local node is not written to and may be concurrently accessed while this operation is in progress. If a peer node announces that it will not contribute
	/// to this transfer by sending an empty-box pilot, its location will also remain unmodified.
	///
	/// This feature is a temporary solution until we implement inter-node reductions through inter-node collectives.
	[[nodiscard]] async_event gather_receive(const transfer_id& trid, void* allocation, size_t node_chunk_size);

	/// Polls the communicator for inbound pilots and advances the state of all ongoing receive operations. This is expected to be called in a loop
	/// unconditionally.
	void poll_communicator();

  private:
	communicator* m_comm;
	size_t m_num_nodes;

	/// State machines for all `transfer_id`s that were mentioned in an inbound pilot or call to one of the receive functions. Once a transfer is complete, it
	/// is cleared from `m_transfers`, but `multi_region_transfer`s can be re-created if there later appears another pair of inbound pilots and `receive`s for
	/// the same transfer id that did not temporally overlap with the original ones.
	std::unordered_map<transfer_id, receive_arbiter_detail::transfer> m_transfers;

	/// Cache for all transfer ids in m_transfers that are not unassigned_transfers. Bounds complexity of iterating to poll all transfer events.
	std::vector<transfer_id> m_active_transfers;

	/// Initiates a new `region_request` for which the caller can construct events to await either the entire region or sub-regions (may_await_subregion =
	/// true).
	receive_arbiter_detail::stable_region_request& initiate_region_request(
	    const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size, bool may_await_subregion);

	/// Updates the state of an active `region_request` from receiving an inbound pilot.
	void handle_region_request_pilot(receive_arbiter_detail::region_request& rr, const inbound_pilot& pilot, size_t elem_size);

	/// Updates the state of an active `gather_request` from receiving an inbound pilot.
	void handle_gather_request_pilot(receive_arbiter_detail::gather_request& gr, const inbound_pilot& pilot);
};

} // namespace celerity::detail
