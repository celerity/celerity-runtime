#include "buffer_transfer_manager.h"

#include <cassert>

#include "buffer_manager.h"
#include "log.h"
#include "mpi_support.h"
#include "reduction_manager.h"
#include "runtime.h"

namespace celerity {
namespace detail {

	std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::push(const command_pkg& pkg) {
		assert(pkg.get_command_type() == command_type::PUSH);
		auto t_handle = std::make_shared<transfer_handle>();
		// We are blocking the caller until the buffer has been copied and submitted to MPI
		// TODO: Investigate doing this in worker thread
		// --> This probably needs some kind of heuristic, as for small (e.g. ghost cell) transfers the overhead of threading is way too big
		const push_data& data = std::get<push_data>(pkg.data);

		auto transfer = std::make_unique<transfer_out>();
		transfer->data =
		    runtime::get_instance().get_buffer_manager().get_buffer_data(data.bid, cl::sycl::range<3>(data.sr.offset[0], data.sr.offset[1], data.sr.offset[2]),
		        cl::sycl::range<3>(data.sr.range[0], data.sr.range[1], data.sr.range[2]));

		CELERITY_TRACE("Ready to send {} of buffer {} ({} B) to {}", data.sr, data.bid, transfer->data.get_size(), data.target);

		transfer->handle = t_handle;
		transfer->header.sr = data.sr;
		transfer->header.bid = data.bid;
		transfer->header.rid = data.rid;
		transfer->header.push_cid = pkg.cid;
		transfer->data_type =
		    mpi_support::build_single_use_composite_type({{sizeof(data_header), &transfer->header}, {transfer->data.get_size(), transfer->data.get_pointer()}});

		// Start transmitting data
		MPI_Isend(MPI_BOTTOM, 1, *transfer->data_type, static_cast<int>(data.target), mpi_support::TAG_DATA_TRANSFER, MPI_COMM_WORLD, &transfer->request);
		outgoing_transfers.push_back(std::move(transfer));

		return t_handle;
	}

	std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::await_push(const command_pkg& pkg) {
		assert(pkg.get_command_type() == command_type::AWAIT_PUSH);
		const auto& data = std::get<await_push_data>(pkg.data);

		std::shared_ptr<incoming_transfer_handle> t_handle;
		// Check to see if we have (fully) received the push already
		if(push_blackboard.count(data.source_cid) != 0) {
			t_handle = push_blackboard[data.source_cid];
			push_blackboard.erase(data.source_cid);
			assert(t_handle->transfer != nullptr);
			assert(t_handle->transfer->header.bid == data.bid);
			assert(t_handle->transfer->header.rid == data.rid);
			assert(t_handle->transfer->header.sr == data.sr);
			assert(t_handle->complete);
			commit_transfer(*t_handle->transfer);
		} else {
			t_handle = std::make_shared<incoming_transfer_handle>();
			// Store new handle so we can mark it as complete when the push is received
			push_blackboard[data.source_cid] = t_handle;
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
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count);
		const size_t data_size = count - sizeof(data_header);

		auto transfer = std::make_unique<transfer_in>();
		transfer->source_nid = static_cast<node_id>(status.MPI_SOURCE);
		// Since we don't know the dimensions of the data yet, we'll just allocate the buffer as if it were one very large element.
		// We'll reinterpret the data later, once the transfer is completed.
		transfer->data = raw_buffer_data{data_size, cl::sycl::range<3>(1, 1, 1)};
		transfer->data_type =
		    mpi_support::build_single_use_composite_type({{sizeof(data_header), &transfer->header}, {data_size, transfer->data.get_pointer()}});

		// Start receiving data
		MPI_Imrecv(MPI_BOTTOM, 1, *transfer->data_type, &msg, &transfer->request);
		incoming_transfers.push_back(std::move(transfer));

		CELERITY_TRACE("Receiving incoming data of size {} B from {}", data_size, status.MPI_SOURCE);
	}

	void buffer_transfer_manager::update_incoming_transfers() {
		for(auto it = incoming_transfers.begin(); it != incoming_transfers.end();) {
			auto& transfer = *it;
			int flag;
			MPI_Test(&transfer->request, &flag, MPI_STATUS_IGNORE);
			if(flag == 0) {
				++it;
				continue;
			}

			// Check whether we already have an await push request
			std::shared_ptr<incoming_transfer_handle> t_handle = nullptr;
			if(push_blackboard.count(transfer->header.push_cid) != 0) {
				t_handle = push_blackboard[transfer->header.push_cid];
				push_blackboard.erase(transfer->header.push_cid);
				assert(t_handle.use_count() > 1 && "Dangling await push request");
				t_handle->transfer = std::move(*it);
				commit_transfer(*t_handle->transfer);
				t_handle->complete = true;
			} else {
				t_handle = std::make_shared<incoming_transfer_handle>();
				push_blackboard[transfer->header.push_cid] = t_handle;
				t_handle->transfer = std::move(*it);
				t_handle->complete = true;
			}
			it = incoming_transfers.erase(it);
		}
	}

	void buffer_transfer_manager::update_outgoing_transfers() {
		for(auto it = outgoing_transfers.begin(); it != outgoing_transfers.end();) {
			auto& t = *it;
			int flag;
			MPI_Test(&t->request, &flag, MPI_STATUS_IGNORE);
			if(flag == 0) {
				++it;
				continue;
			}
			t->handle->complete = true;
			it = outgoing_transfers.erase(it);
		}
	}

	void buffer_transfer_manager::commit_transfer(transfer_in& transfer) {
		const auto& header = transfer.header;
		size_t elem_size = transfer.data.get_size() / (header.sr.range[0] * header.sr.range[1] * header.sr.range[2]);
		transfer.data.reinterpret(elem_size, cl::sycl::range<3>(header.sr.range[0], header.sr.range[1], header.sr.range[2]));
		if(header.rid) {
			auto& rm = runtime::get_instance().get_reduction_manager();
			// In some rare situations the local runtime might not yet know about this reduction. Busy wait until it does.
			while(!rm.has_reduction(header.rid)) {}
			rm.push_overlapping_reduction_data(header.rid, transfer.source_nid, std::move(transfer.data));
		} else {
			auto& bm = runtime::get_instance().get_buffer_manager();
			// In some rare situations the local runtime might not yet know about this buffer. Busy wait until it does.
			while(!bm.has_buffer(header.bid)) {}
			bm.set_buffer_data(header.bid, header.sr.offset, std::move(transfer.data));
		}
		transfer.data = {};
	}

} // namespace detail
} // namespace celerity
