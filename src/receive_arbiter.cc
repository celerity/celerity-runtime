#include "receive_arbiter.h"

#include "async_event.h"
#include "communicator.h"
#include "grid.h"
#include "pilot.h"
#include "ranges.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <memory>
#include <utility>
#include <variant>

#include <matchbox.hh>


namespace celerity::detail::receive_arbiter_detail {

// weak-pointers for referencing stable_region/gather_requests that are held by the receive_arbiter. If they expire, we know the event is complete.
using weak_region_request = std::weak_ptr<const region_request>;
using weak_gather_request = std::weak_ptr<const gather_request>;

/// Event for `receive_arbiter::receive`, which immediately awaits the entire receive-region.
class region_receive_event final : public async_event_impl {
  public:
	explicit region_receive_event(const stable_region_request& rr) : m_request(rr) {}

	bool is_complete() override { return m_request.expired(); }

  private:
	weak_region_request m_request;
};

/// Event for `receive_arbiter::await_split_receive_subregion`, which awaits a specific subregion of a split receive.
class subregion_receive_event final : public async_event_impl {
  public:
	explicit subregion_receive_event(const stable_region_request& rr, const region<3>& awaited_subregion)
	    : m_request(rr), m_awaited_region(awaited_subregion) {}

	bool is_complete() override {
		const auto rr = m_request.lock();
		return rr == nullptr || region_intersection(rr->incomplete_region, m_awaited_region).empty();
	}

  private:
	weak_region_request m_request;
	region<3> m_awaited_region;
};

/// Event for `receive_arbiter::gather_receive`, which waits for incoming messages (or empty-box pilots) from all peers.
class gather_receive_event final : public async_event_impl {
  public:
	explicit gather_receive_event(const stable_gather_request& gr) : m_request(gr) {}

	bool is_complete() override { return m_request.expired(); }

  private:
	weak_gather_request m_request;
};

bool region_request::do_complete() {
	// Fast path: Avoid polling the entire fragment set when we know that neither the request as a whole nor any subregion-request will complete at this time
	if(!may_await_subregion && !incoming_fragments.empty() && !incoming_fragments.front().communication.is_complete()) return false;

	std::erase_if(incoming_fragments, [&](const incoming_region_fragment& fragment) {
		if(!fragment.communication.is_complete()) return false;
		if(may_await_subregion) { incomplete_region = region_difference(incomplete_region, fragment.box); }
		incomplete_area -= fragment.box.get_area();
		return true;
	});
	assert(incomplete_area > 0 || incoming_fragments.empty());
	return incomplete_area == 0;
}

bool multi_region_transfer::do_complete() {
	std::erase_if(active_requests, [](stable_region_request& rr) { return rr->do_complete(); });
	return active_requests.empty() && unassigned_pilots.empty();
}

bool gather_request::do_complete() {
	std::erase_if(incoming_chunks, [&](const incoming_gather_chunk& chunk) {
		if(!chunk.communication.is_complete()) return false;
		assert(num_incomplete_chunks > 0);
		num_incomplete_chunks -= 1;
		return true;
	});
	return num_incomplete_chunks == 0;
}

bool gather_transfer::do_complete() { return request->do_complete(); }

bool unassigned_transfer::do_complete() { // NOLINT(readability-make-member-function-const)
	// an unassigned_transfer inside receive_arbiter::m_transfers is never empty.
	assert(!pilots.empty());
	return false;
}

} // namespace celerity::detail::receive_arbiter_detail

namespace celerity::detail {

using namespace receive_arbiter_detail;

receive_arbiter::receive_arbiter(communicator& comm) : m_comm(&comm), m_num_nodes(comm.get_num_nodes()) { assert(m_num_nodes > 0); }

receive_arbiter::~receive_arbiter() { assert(std::uncaught_exceptions() > 0 || m_transfers.empty()); }

receive_arbiter_detail::stable_region_request& receive_arbiter::initiate_region_request(const transfer_id& trid, const region<3>& request,
    void* const allocation, const box<3>& allocated_box, const size_t elem_size, const bool may_await_subregion) //
{
	assert(allocated_box.covers(bounding_box(request)));

	// Ensure there is a multi_region_transfer present - if there is none, create it by consuming unassigned pilots
	auto& transfer = m_transfers[trid]; // allow default-insert as unassigned_transfer
	auto& mrt = matchbox::match(
	    transfer,
	    [&](unassigned_transfer& ut) -> multi_region_transfer& {
		    auto pilots = std::move(ut.pilots);
		    m_active_transfers.push_back(trid);
		    return transfer.emplace<multi_region_transfer>(elem_size, std::move(pilots));
	    },
	    [&](multi_region_transfer& existing_mrt) -> multi_region_transfer& { //
		    return existing_mrt;
	    },
	    [&](gather_transfer& gt) -> multi_region_transfer& { //
		    utils::panic("calling receive_arbiter::begin_receive on an active gather transfer");
	    });

	// Add a new region_request to the `mrt` (transfers have transfer_id granularity, but there might be multiple receives from independent range mappers
	assert(std::all_of(mrt.active_requests.begin(), mrt.active_requests.end(),
	    [&](const stable_region_request& rr) { return region_intersection(rr->incomplete_region, request).empty(); }));
	auto& rr = mrt.active_requests.emplace_back(std::make_shared<region_request>(request, allocation, allocated_box, may_await_subregion));

	// If the new region_request matches any of the still-unassigned pilots associated with `mrt`, immediately initiate the appropriate payload-receives
	std::erase_if(mrt.unassigned_pilots, [&](const inbound_pilot& pilot) {
		assert((region_intersection(rr->incomplete_region, pilot.message.box) != pilot.message.box)
		       == region_intersection(rr->incomplete_region, pilot.message.box).empty());
		if(region_intersection(rr->incomplete_region, pilot.message.box) == pilot.message.box) {
			handle_region_request_pilot(*rr, pilot, elem_size);
			return true;
		}
		return false;
	});

	return rr;
}

void receive_arbiter::begin_split_receive(
    const transfer_id& trid, const region<3>& request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	initiate_region_request(trid, request, allocation, allocated_box, elem_size, true /* may_await_subregion */);
}

async_event receive_arbiter::await_split_receive_subregion(const transfer_id& trid, const region<3>& subregion) {
	// If there is no known associated `transfer`, we must have erased it previously due to the the entire `begin_split_receive` being completed. Any (partial)
	// await thus immediately completes as well.
	const auto transfer_it = m_transfers.find(trid);
	if(transfer_it == m_transfers.end()) { return make_complete_event(); }

	auto& mrt = std::get<multi_region_transfer>(transfer_it->second);

#ifndef NDEBUG
	// all boxes from the awaited region must be contained in a single allocation
	const auto awaited_bounds = bounding_box(subregion);
	assert(std::all_of(mrt.active_requests.begin(), mrt.active_requests.end(), [&](const stable_region_request& rr) {
		const auto overlap = box_intersection(rr->allocated_box, awaited_bounds);
		return overlap.empty() || overlap == awaited_bounds;
	}));
#endif

	// If the transfer (by transfer_id) as a whole has not completed yet but the subregion is, this "await" also completes immediately.
	const auto rr_it = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(),
	    [&](const stable_region_request& rr) { return !region_intersection(rr->incomplete_region, subregion).empty(); });
	if(rr_it == mrt.active_requests.end()) { return make_complete_event(); }

	auto& rr = *rr_it;
	assert(rr->may_await_subregion && "attempting await_split_receive_subregion() on region that was not initiated with begin_split_receive()");
	return make_async_event<subregion_receive_event>(rr, subregion);
}

async_event receive_arbiter::receive(
    const transfer_id& trid, const region<3>& request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) //
{
	auto& rr = initiate_region_request(trid, request, allocation, allocated_box, elem_size, false /* may_await_subregion */);
	return make_async_event<region_receive_event>(rr);
}

async_event receive_arbiter::gather_receive(const transfer_id& trid, void* const allocation, const size_t node_chunk_size) {
	auto gr = std::make_shared<gather_request>(allocation, node_chunk_size, m_num_nodes - 1 /* number of peers */);

	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		// If we are already tracking a transfer `trid`, it must be unassigned, and we can initiate payload-receives for all unassigned pilots right away.
		auto& ut = std::get<unassigned_transfer>(entry->second);
		for(auto& pilot : ut.pilots) {
			handle_gather_request_pilot(*gr, pilot);
		}
		entry->second = gather_transfer{gr};
	} else {
		// Otherwise, we insert the transfer as pending and wait for the first pilots to arrive.
		m_transfers.emplace(trid, gather_transfer{gr});
	}
	m_active_transfers.push_back(trid);

	return make_async_event<gather_receive_event>(gr);
}

void receive_arbiter::poll_communicator() {
	// Try completing all pending payload sends / receives by polling their communicator events
	std::erase_if(m_active_transfers, [&](const transfer_id& trid) {
		const auto entry = m_transfers.find(trid);
		assert(entry != m_transfers.end());
		const bool is_complete = std::visit([](auto& transfer) { return transfer.do_complete(); }, entry->second);
		if(is_complete) { m_transfers.erase(entry); }
		return is_complete;
	});

	for(const auto& pilot : m_comm->poll_inbound_pilots()) {
		if(const auto entry = m_transfers.find(pilot.message.transfer_id); entry != m_transfers.end()) {
			// If we already know a the transfer id, initiate pending payload-receives or add the pilot to the unassigned-list.
			matchbox::match(
			    entry->second,                 //
			    [&](unassigned_transfer& ut) { //
				    ut.pilots.push_back(pilot);
			    },
			    [&](multi_region_transfer& mrt) {
				    // find the unique region-request this pilot belongs to
				    const auto rr = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(), [&](const stable_region_request& rr) {
					    assert((region_intersection(rr->incomplete_region, pilot.message.box) != pilot.message.box)
					           == region_intersection(rr->incomplete_region, pilot.message.box).empty());
					    return region_intersection(rr->incomplete_region, pilot.message.box) == pilot.message.box;
				    });
				    if(rr != mrt.active_requests.end()) {
					    handle_region_request_pilot(**rr, pilot, mrt.elem_size);
				    } else {
					    mrt.unassigned_pilots.push_back(pilot);
				    }
			    },
			    [&](gather_transfer& gt) { //
				    handle_gather_request_pilot(*gt.request, pilot);
			    });
		} else {
			// If we haven't seen the transfer id before, create a new unassigned_transfer for it.
			m_transfers.emplace(pilot.message.transfer_id, unassigned_transfer{{pilot}});
		}
	}
}

void receive_arbiter::handle_region_request_pilot(region_request& rr, const inbound_pilot& pilot, const size_t elem_size) {
	assert(region_intersection(rr.incomplete_region, pilot.message.box) == pilot.message.box);
	assert(rr.allocated_box.covers(pilot.message.box));

	// Initiate a strided payload-receive directly into the allocation passed to receive() / begin_split_receive()
	const auto offset_in_allocation = pilot.message.box.get_offset() - rr.allocated_box.get_offset();
	const communicator::stride stride{
	    rr.allocated_box.get_range(),
	    subrange<3>{offset_in_allocation, pilot.message.box.get_range()},
	    elem_size,
	};
	auto event = m_comm->receive_payload(pilot.from, pilot.message.id, rr.allocation, stride);
	rr.incoming_fragments.push_back({pilot.message.box, std::move(event)});
}

void receive_arbiter::handle_gather_request_pilot(gather_request& gr, const inbound_pilot& pilot) {
	if(pilot.message.box.empty()) {
		// Peers will send a pilot with an empty box to signal that they don't contribute to a reduction
		assert(gr.num_incomplete_chunks > 0);
		gr.num_incomplete_chunks -= 1;
	} else {
		// Initiate a region-receive with a simple stride to address the chunk id in the allocation
		const communicator::stride stride{range_cast<3>(range(m_num_nodes)), subrange(id_cast<3>(id(pilot.from)), range_cast<3>(range(1))), gr.chunk_size};
		auto event = m_comm->receive_payload(pilot.from, pilot.message.id, gr.allocation, stride);
		gr.incoming_chunks.push_back(incoming_gather_chunk{std::move(event)});
	}
}

} // namespace celerity::detail
