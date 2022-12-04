#include "buffer_transfer_manager.h"

#include <cassert>
#include <climits>

#include "buffer_manager.h"
#include "log.h"
#include "reduction_manager.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	inline constexpr size_t send_recv_unit_bytes = 64;

	mpi_support::data_type make_send_recv_unit() {
		MPI_Datatype unit;
		MPI_Type_contiguous(send_recv_unit_bytes, MPI_BYTE, &unit);
		MPI_Type_commit(&unit);
		return mpi_support::data_type(unit);
	}

	buffer_transfer_manager::buffer_transfer_manager(buffer_manager& bm, reduction_manager& rm)
	    : m_buffer_mngr(bm), m_reduction_mngr(rm), m_send_recv_unit(make_send_recv_unit()) {}

	std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::push(
	    const node_id target, const transfer_id trid, const buffer_id bid, const subrange<3>& sr, const reduction_id rid) {
		auto t_handle = std::make_shared<transfer_handle>();
		// We are blocking the caller until the buffer has been copied and submitted to MPI
		// TODO: Investigate doing this in worker thread
		// --> This probably needs some kind of heuristic, as for small (e.g. ghost cell) transfers the overhead of threading is way too big
		const auto element_size = m_buffer_mngr.get_buffer_info(bid).element_size;

		unique_frame_ptr<data_frame> frame(from_payload_count, sr.range.size() * element_size, /* packet_size_bytes */ send_recv_unit_bytes);
		frame->sr = sr;
		frame->bid = bid;
		frame->rid = rid;
		frame->trid = trid;
		m_buffer_mngr.get_buffer_data(bid, sr, frame->data);

		assert(frame.get_size_bytes() % send_recv_unit_bytes == 0);
		const size_t frame_units = frame.get_size_bytes() / send_recv_unit_bytes;
		CELERITY_TRACE("Ready to send {} of buffer {} ({} * {}B) to {}", sr, bid, frame_units, send_recv_unit_bytes, target);

		// Start transmitting data
		MPI_Request req;
		assert(frame_units <= static_cast<size_t>(std::numeric_limits<int>::max()));
		MPI_Isend(frame.get_pointer(), static_cast<int>(frame_units), m_send_recv_unit, static_cast<int>(target), mpi_support::TAG_DATA_TRANSFER,
		    MPI_COMM_WORLD, &req);

		auto transfer = std::make_unique<transfer_out>();
		transfer->handle = t_handle;
		transfer->request = req;
		transfer->frame = std::move(frame);
		m_outgoing_transfers.push_back(std::move(transfer));

		return t_handle;
	}

	std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::await_push(
	    const transfer_id trid, const buffer_id bid, const GridRegion<3>& expected_region, const reduction_id rid) {
		std::shared_ptr<transfer_handle> t_handle;
		// Check to see if we have (fully) received the data already
		const auto buffer_transfer = std::pair{bid, trid};
		if(auto rm_it = m_requests.find(buffer_transfer); rm_it != m_requests.end()) {
			request_manager& rm = rm_it->second;
			t_handle = rm.request_region(expected_region);
			if(rm.can_drain()) {
				rm.drain_transfers([&](std::unique_ptr<transfer_in> t) {
					assert(t->frame->bid == bid);
					assert(t->frame->rid == rid);
					commit_transfer(*t);
				});
				if(rm.fully_drained()) { m_requests.erase(buffer_transfer); }
			}
		} else {
			request_manager rm;
			t_handle = rm.request_region(expected_region);
			// Store new manager so we can match against it once the transfer received
			m_requests.try_emplace(buffer_transfer, std::move(rm));
		}

		return t_handle;
	}

	void buffer_transfer_manager::poll() {
		poll_incoming_transfers();
		update_incoming_transfers();
		update_outgoing_transfers();
	}

	void buffer_transfer_manager::poll_incoming_transfers() {
		MPI_Status status;
		int flag;
		MPI_Message msg;
		MPI_Improbe(MPI_ANY_SOURCE, mpi_support::TAG_DATA_TRANSFER, MPI_COMM_WORLD, &flag, &msg, &status);
		if(flag == 0) {
			// No incoming transfers at the moment
			return;
		}
		int frame_units;
		MPI_Get_count(&status, m_send_recv_unit, &frame_units);

		auto transfer = std::make_unique<transfer_in>();
		transfer->source_nid = static_cast<node_id>(status.MPI_SOURCE);
		transfer->frame = unique_frame_ptr<data_frame>(from_size_bytes, static_cast<size_t>(frame_units) * send_recv_unit_bytes);

		// Start receiving data
		MPI_Imrecv(transfer->frame.get_pointer(), frame_units, m_send_recv_unit, &msg, &transfer->request);
		m_incoming_transfers.push_back(std::move(transfer));

		CELERITY_TRACE("Receiving incoming data of size {} * {}B from {}", frame_units, send_recv_unit_bytes, status.MPI_SOURCE);
	}

	void buffer_transfer_manager::update_incoming_transfers() {
		for(auto it = m_incoming_transfers.begin(); it != m_incoming_transfers.end();) {
			auto& transfer = *it;
			int flag;
			MPI_Test(&transfer->request, &flag, MPI_STATUS_IGNORE);
			if(flag == 0) {
				++it;
				continue;
			}

			// Check whether we already have an await push request
			const auto buffer_transfer = std::pair{transfer->frame->bid, transfer->frame->trid};
			if(auto rm_it = m_requests.find(buffer_transfer); rm_it != m_requests.end()) {
				request_manager& rm = rm_it->second;
				rm.add_transfer(std::move(*it));
				if(rm.can_drain()) {
					rm.drain_transfers([&](std::unique_ptr<transfer_in> t) { commit_transfer(*t); });
					if(rm.fully_drained()) { m_requests.erase(buffer_transfer); }
				}
			} else {
				request_manager rm;
				rm.add_transfer(std::move(*it));
				// Store new manager so we can match against it once the corresponding await push is posted
				m_requests.try_emplace(buffer_transfer, std::move(rm));
			}

			it = m_incoming_transfers.erase(it);
		}
	}

	void buffer_transfer_manager::update_outgoing_transfers() {
		for(auto it = m_outgoing_transfers.begin(); it != m_outgoing_transfers.end();) {
			auto& t = *it;
			int flag;
			MPI_Test(&t->request, &flag, MPI_STATUS_IGNORE);
			if(flag == 0) {
				++it;
				continue;
			}
			t->handle->complete = true;
			it = m_outgoing_transfers.erase(it);
		}
	}

	void buffer_transfer_manager::commit_transfer(transfer_in& transfer) {
		const auto& frame = *transfer.frame;
		auto payload = std::move(transfer.frame).into_payload_ptr();

		if(frame.rid) {
			// In some rare situations the local runtime might not yet know about this reduction. Busy wait until it does.
			while(!m_reduction_mngr.has_reduction(frame.rid)) {}
			m_reduction_mngr.push_overlapping_reduction_data(frame.rid, transfer.source_nid, std::move(payload));
		} else {
			// In some rare situations the local runtime might not yet know about this buffer. Busy wait until it does.
			while(!m_buffer_mngr.has_buffer(frame.bid)) {}
			m_buffer_mngr.set_buffer_data(frame.bid, frame.sr, std::move(payload));
		}
	}

} // namespace detail
} // namespace celerity
