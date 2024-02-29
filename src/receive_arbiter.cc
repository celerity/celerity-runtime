#include "receive_arbiter.h"
#include "grid.h"
#include "instruction_graph.h"

#include <exception>
#include <memory>

namespace celerity::detail {


receive_arbiter::receive_arbiter(communicator& comm) : m_comm(&comm), m_num_nodes(comm.get_num_nodes()) { assert(m_num_nodes > 0); }

receive_arbiter::~receive_arbiter() { assert(std::uncaught_exceptions() > 0 || m_transfers.empty()); }

bool receive_arbiter::receive_event::is_complete() const {
	return matchbox::match(
	    m_state,                                     //
	    [](const completed_state&) { return true; }, //
	    [](const region_transfer_state& rts) { return rts.request.expired(); },
	    [](const subregion_transfer_state& sts) {
		    const auto rr = sts.request.lock();
		    return rr == nullptr || region_intersection(rr->incomplete_region, sts.awaited_region).empty();
	    },
	    [](const gather_transfer_state& gts) { return gts.request.expired(); });
}

receive_arbiter::stable_region_request& receive_arbiter::begin_receive_region(
    const transfer_id& trid, const region<3>& request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	assert(allocated_box.covers(bounding_box(request)));

	multi_region_transfer* mrt = nullptr;
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		matchbox::match(
		    entry->second, //
		    [&](unassigned_transfer& ut) { mrt = &utils::replace(entry->second, multi_region_transfer(elem_size, std::move(ut.pilots))); },
		    [&](multi_region_transfer& existing_mrt) { mrt = &existing_mrt; },
		    [&](gather_transfer& gt) { utils::panic("calling receive_arbiter::begin_receive on an active gather transfer"); });
	} else {
		mrt = &utils::replace(m_transfers[trid], multi_region_transfer(elem_size));
	}

	assert(std::all_of(mrt->active_requests.begin(), mrt->active_requests.end(),
	    [&](const stable_region_request& rr) { return region_intersection(rr->incomplete_region, request).empty(); }));
	auto& rr = mrt->active_requests.emplace_back(std::make_shared<region_request>(request, allocation, allocated_box));

	const auto last_unassigned_pilot = std::remove_if(mrt->unassigned_pilots.begin(), mrt->unassigned_pilots.end(), [&](const inbound_pilot& pilot) {
		assert((region_intersection(rr->incomplete_region, pilot.message.box) != pilot.message.box)
		       == region_intersection(rr->incomplete_region, pilot.message.box).empty());
		if(region_intersection(rr->incomplete_region, pilot.message.box) == pilot.message.box) {
			handle_region_request_pilot(*rr, pilot, elem_size);
			return true;
		}
		return false;
	});
	mrt->unassigned_pilots.erase(last_unassigned_pilot, mrt->unassigned_pilots.end());

	return rr;
}

void receive_arbiter::begin_split_receive(
    const transfer_id& trid, const region<3>& request, void* const allocation, const box<3>& allocated_box, const size_t elem_size) {
	begin_receive_region(trid, request, allocation, allocated_box, elem_size);
}

async_event receive_arbiter::await_split_receive_subregion(const transfer_id& trid, const region<3>& subregion) {
	const auto transfer_it = m_transfers.find(trid);
	if(transfer_it == m_transfers.end()) { return make_complete_event(); }

	auto& mrt = std::get<multi_region_transfer>(transfer_it->second);

#ifndef NDEBUG
	const auto awaited_bounds = bounding_box(subregion);
	assert(std::all_of(mrt.active_requests.begin(), mrt.active_requests.end(), [&](const stable_region_request& rr) {
		// all boxes from the awaited region must be contained in a single allocation
		const auto overlap = box_intersection(rr->allocated_box, awaited_bounds);
		return overlap.empty() || overlap == awaited_bounds;
	}));
#endif

	const auto req_it = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(),
	    [&](const stable_region_request& rr) { return !region_intersection(rr->incomplete_region, subregion).empty(); });
	if(req_it == mrt.active_requests.end()) { return make_complete_event(); }

	return make_async_event<receive_event>(*req_it, subregion);
}

async_event receive_arbiter::receive(const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size) {
	return make_async_event<receive_event>(begin_receive_region(trid, request, allocation, allocated_box, elem_size));
}

async_event receive_arbiter::gather_receive(const transfer_id& trid, void* allocation, size_t node_chunk_size) {
	auto gr = std::make_shared<gather_request>(allocation, node_chunk_size, m_num_nodes - 1 /* number of peers */);
	auto event = make_async_event<receive_event>(gr);
	if(const auto entry = m_transfers.find(trid); entry != m_transfers.end()) {
		auto& ut = std::get<unassigned_transfer>(entry->second);
		for(auto& pilot : ut.pilots) {
			handle_gather_request_pilot(*gr, pilot);
		}
		entry->second = gather_transfer{std::move(gr)};
	} else {
		m_transfers.emplace(trid, gather_transfer{std::move(gr)});
	}
	return event;
}

bool receive_arbiter::region_request::do_complete() {
	const auto incomplete_fragments_end = std::remove_if(incoming_fragments.begin(), incoming_fragments.end(), [&](const incoming_region_fragment& fragment) {
		const bool is_complete = fragment.communication.is_complete();
		if(is_complete) { incomplete_region = region_difference(incomplete_region, fragment.box); }
		return is_complete;
	});
	incoming_fragments.erase(incomplete_fragments_end, incoming_fragments.end());
	assert(!incomplete_region.empty() || incoming_fragments.empty());
	return incomplete_region.empty();
}

bool receive_arbiter::multi_region_transfer::do_complete() {
	const auto incomplete_req_end =
	    std::remove_if(active_requests.begin(), active_requests.end(), [&](stable_region_request& rr) { return rr->do_complete(); });
	active_requests.erase(incomplete_req_end, active_requests.end());
	return active_requests.empty() && unassigned_pilots.empty();
}

bool receive_arbiter::gather_request::do_complete() {
	const auto incomplete_chunks_end = std::remove_if(incoming_chunks.begin(), incoming_chunks.end(), [&](const incoming_gather_chunk& chunk) {
		const bool is_complete = chunk.communication.is_complete();
		if(is_complete) {
			assert(num_incomplete_chunks > 0);
			num_incomplete_chunks -= 1;
		}
		return is_complete;
	});
	incoming_chunks.erase(incomplete_chunks_end, incoming_chunks.end());
	return num_incomplete_chunks == 0;
}

bool receive_arbiter::gather_transfer::do_complete() { return request->do_complete(); }

bool receive_arbiter::unassigned_transfer::do_complete() { // NOLINT(readability-make-member-function-const)
	// an unassigned_transfer inside receive_arbiter::m_transfers is never empty.
	assert(!pilots.empty());
	return false;
}

void receive_arbiter::poll_communicator() {
	for(auto entry = m_transfers.begin(); entry != m_transfers.end();) {
		if(std::visit([](auto& transfer) { return transfer.do_complete(); }, entry->second)) {
			entry = m_transfers.erase(entry);
		} else {
			++entry;
		}
	}

	for(const auto& pilot : m_comm->poll_inbound_pilots()) {
		if(const auto entry = m_transfers.find(pilot.message.transfer_id); entry != m_transfers.end()) {
			matchbox::match(
			    entry->second, //
			    [&](unassigned_transfer& ut) { ut.pilots.push_back(pilot); },
			    [&](multi_region_transfer& mrt) {
				    const auto rr = std::find_if(mrt.active_requests.begin(), mrt.active_requests.end(), [&](const stable_region_request& rr) {
					    assert((region_intersection(rr->incomplete_region, pilot.message.box) != pilot.message.box)
					           == region_intersection(rr->incomplete_region, pilot.message.box).empty());
					    return region_intersection(rr->incomplete_region, pilot.message.box) == pilot.message.box;
				    });
				    if(rr != mrt.active_requests.end()) {
					    handle_region_request_pilot(**rr, pilot, mrt.elem_size); // elem_size is set when transfer_region is inserted
				    } else {
					    mrt.unassigned_pilots.push_back(pilot);
				    }
			    },
			    [&](gather_transfer& gt) { handle_gather_request_pilot(*gt.request, pilot); });
		} else {
			m_transfers.emplace(pilot.message.transfer_id, unassigned_transfer{{pilot}});
		}
	}
}

void receive_arbiter::handle_region_request_pilot(region_request& rr, const inbound_pilot& pilot, const size_t elem_size) {
	assert(region_intersection(rr.incomplete_region, pilot.message.box) == pilot.message.box);
	assert(rr.allocated_box.covers(pilot.message.box));
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
		// peers will send a pilot with an empty box to signal that they don't contribute to a reduction.
		assert(gr.num_incomplete_chunks > 0);
		gr.num_incomplete_chunks -= 1;
	} else {
		const communicator::stride stride{range_cast<3>(range(m_num_nodes)), subrange(id_cast<3>(id(pilot.from)), range_cast<3>(range(1))), gr.chunk_size};
		auto event = m_comm->receive_payload(pilot.from, pilot.message.id, gr.allocation, stride);
		gr.incoming_chunks.push_back(incoming_gather_chunk{std::move(event)});
	}
}

} // namespace celerity::detail
