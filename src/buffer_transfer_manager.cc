#include "buffer_transfer_manager.h"

#include <cassert>
#include <limits>

#include "runtime.h"

namespace celerity {

buffer_transfer_manager::~buffer_transfer_manager() {
	for(auto dt : mpi_byte_size_data_types) {
		MPI_Type_free(&dt.second);
	}
}

void buffer_transfer_manager::poll() {
	poll_transfers();
	update_incoming_transfers();
	update_outgoing_transfers();
}

void buffer_transfer_manager::poll_transfers() {
	MPI_Status status;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_DATA_TRANSFER, MPI_COMM_WORLD, &flag, &status);
	if(flag == 0) {
		// No incoming transfers at the moment
		return;
	}

	int count;
	MPI_Get_count(&status, MPI_CHAR, &count);
	const int data_size = count - sizeof(data_header);

	auto transfer = std::make_unique<transfer_in>();
	transfer->data.resize(data_size);

	// Build data type
	MPI_Datatype transfer_data_type;
	MPI_Datatype block_types[2] = {MPI_BYTE, MPI_BYTE};
	int block_lengths[2] = {sizeof(data_header), data_size};
	// We use absolute displacements here (= pointers), since it's easier that way to get the actual memory location within the vector
	MPI_Aint disps[2] = {(MPI_Aint)&transfer->header, (MPI_Aint)&transfer->data[0]};
	MPI_Type_create_struct(2, block_lengths, disps, block_types, &transfer_data_type);
	MPI_Type_commit(&transfer_data_type);
	// Store data type so it can be free'd after the transfer is complete
	transfer->data_type = transfer_data_type;

	// Start receiving data
	MPI_Irecv(MPI_BOTTOM, 1, transfer_data_type, status.MPI_SOURCE, CELERITY_MPI_TAG_DATA_TRANSFER, MPI_COMM_WORLD, &transfer->request);
	incoming_transfers.push_back(std::move(transfer));

	transfer_logger->info("Receiving incoming data of size {} from {}", data_size, status.MPI_SOURCE);
}

// TODO: Copy buffer subrange in case we want to overwrite it (handle here or on job-level?)
std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::push(const command_pkg& pkg) {
	assert(pkg.cmd == command::PUSH);
	const push_data& data = pkg.data.push;

	auto t_handle = std::make_shared<transfer_handle>();
	auto data_handle =
	    runtime::get_instance().get_buffer_data(data.bid, cl::sycl::range<3>(data.subrange.offset0, data.subrange.offset1, data.subrange.offset2),
	        cl::sycl::range<3>(data.subrange.range0, data.subrange.range1, data.subrange.range2));

	// Build subarray data type
	MPI_Datatype subarray_data_type;
	MPI_Type_create_subarray(data_handle.dimensions, data_handle.full_size.data(), data_handle.subsize.data(), data_handle.offsets.data(), MPI_ORDER_C,
	    get_byte_size_data_type(data_handle.element_size), &subarray_data_type);
	MPI_Type_commit(&subarray_data_type);

	auto transfer = std::make_unique<transfer_out>(data_handle);
	transfer->handle = t_handle;
	transfer->header.subrange = data.subrange;
	transfer->header.bid = data.bid;
	transfer->header.push_cid = pkg.cid;

	// Build full data type with header
	MPI_Datatype transfer_data_type;
	MPI_Datatype block_types[2] = {MPI_BYTE, subarray_data_type};
	int block_lengths[2] = {sizeof(data_header), 1};
	// We use absolute displacements here (= pointers), so we can obtain header and data from different locations
	MPI_Aint disps[2] = {(MPI_Aint)&transfer->header, (MPI_Aint)transfer->get_raw_ptr()};
	MPI_Type_create_struct(2, block_lengths, disps, block_types, &transfer_data_type);
	MPI_Type_commit(&transfer_data_type);
	// Store data type so it can be free'd after the transfer is complete
	transfer->data_type = transfer_data_type;

	// Start transmitting data
	MPI_Isend(MPI_BOTTOM, 1, transfer_data_type, static_cast<int>(data.target), CELERITY_MPI_TAG_DATA_TRANSFER, MPI_COMM_WORLD, &transfer->request);
	outgoing_transfers.push_back(std::move(transfer));

	return t_handle;
}

std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::await_push(const command_pkg& pkg) {
	assert(pkg.cmd == command::AWAIT_PUSH);
	const await_push_data& data = pkg.data.await_push;

	std::shared_ptr<incoming_transfer_handle> t_handle;
	// Check to see if we have (fully) received the push already
	if(push_blackboard.count(data.source_cid) != 0) {
		t_handle = push_blackboard[data.source_cid];
		push_blackboard.erase(data.source_cid);
		assert(t_handle->transfer != nullptr);
		assert(t_handle->transfer->header.bid == data.bid);
		assert(t_handle->transfer->header.subrange == data.subrange);
		write_data_to_buffer(std::move(t_handle->transfer));
	} else {
		t_handle = std::make_shared<incoming_transfer_handle>();
		// Store new handle so we can mark it as complete when the push is received
		push_blackboard[data.source_cid] = t_handle;
	}

	return t_handle;
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
			write_data_to_buffer(std::move(*it));
		} else {
			t_handle = std::make_shared<incoming_transfer_handle>();
			push_blackboard[transfer->header.push_cid] = t_handle;
			t_handle->transfer = std::move(*it);
		}

		t_handle->complete = true;
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

		// Transfer cannot be accessed after this!
		it = outgoing_transfers.erase(it);
	}
}

void buffer_transfer_manager::write_data_to_buffer(std::unique_ptr<transfer_in>&& transfer) {
	// FIXME: This check doesn't work for e.g. transfering just a single column: (0, 0, 0) - (256, 1, 1) of a 2D or 3D buffer!
	int dimensions = 3;
	const auto& header = transfer->header;
	if(transfer->header.subrange.range2 == 1) {
		dimensions = 2;
		if(transfer->header.subrange.range1 == 1) { dimensions = 1; }
	}
	// FIXME: It's not ideal that we set raw_data_range::full_size and element_size to all zeros here.
	detail::raw_data_range dr{&transfer->data[0], dimensions, {(int)header.subrange.range0, (int)header.subrange.range1, (int)header.subrange.range2},
	    {(int)header.subrange.offset0, (int)header.subrange.offset1, (int)header.subrange.offset2}, {0, 0, 0}, 0};
	runtime::get_instance().set_buffer_data(header.bid, dr);
}

MPI_Datatype buffer_transfer_manager::get_byte_size_data_type(size_t byte_size) {
	if(mpi_byte_size_data_types.count(byte_size) != 0) { return mpi_byte_size_data_types[byte_size]; }
	MPI_Datatype data_type;
	assert(byte_size < std::numeric_limits<int>::max());
	MPI_Type_contiguous(static_cast<int>(byte_size), MPI_BYTE, &data_type);
	MPI_Type_commit(&data_type);
	mpi_byte_size_data_types[byte_size] = data_type;
	return data_type;
}

} // namespace celerity
