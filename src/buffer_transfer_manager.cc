#include "buffer_transfer_manager.h"

#include <cassert>

#include "runtime.h"

namespace celerity {

buffer_transfer_manager::~buffer_transfer_manager() {
	for(auto dt : mpi_byte_size_data_types) {
		MPI_Type_free(&dt.second);
	}
}

void buffer_transfer_manager::poll() {
	poll_requests();
	poll_transfers();
	update_transfers();
}

void buffer_transfer_manager::poll_requests() {
	MPI_Status status;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_DATA_REQUEST, MPI_COMM_WORLD, &flag, &status);
	if(flag == 0) return;
	command_pkg pkg;
	MPI_Recv(&pkg, sizeof(command_pkg), MPI_BYTE, status.MPI_SOURCE, CELERITY_MPI_TAG_DATA_REQUEST, MPI_COMM_WORLD, &status);
	runtime::get_instance().schedule_buffer_send(status.MPI_SOURCE, pkg);
}

void buffer_transfer_manager::poll_transfers() {
	MPI_Status status;
	int flag;
	MPI_Iprobe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_DATA_TRANSFER, MPI_COMM_WORLD, &flag, &status);
	if(flag == 0) return;

	int count;
	MPI_Get_count(&status, MPI_CHAR, &count);
	int data_size = count - sizeof(data_header);

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
	incoming_transfers.insert(std::move(transfer));

	transfer_logger->info("Receiving incoming data of size {} from {}", data_size, status.MPI_SOURCE);
}

std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::send(node_id to, const command_pkg& pkg) {
	assert(pkg.cmd == command::PULL);
	const pull_data& data = pkg.data.pull;

	// Check whether we already have an await pull request, and if so, use that handle
	std::shared_ptr<transfer_handle> handle = nullptr;
	for(auto& p : active_handles[pkg.tid][data.bid]) {
		if(p.first.cmd != command::AWAIT_PULL) continue;
		if(p.first.data.await_pull.subrange == data.subrange) {
			handle = p.second;
			break;
		}
	}

	if(handle == nullptr) {
		handle = std::make_shared<transfer_handle>();
		// Store new handle so we can return it when the await pull finally comes in
		// TODO: It's confusing that we're storing PULL commands for both our own requests, as well as incoming requests
		// We may want to instead store an AWAIT PULL here
		active_handles[pkg.tid][data.bid].push_back(std::make_pair(pkg, handle));
	}

	auto data_handle =
	    runtime::get_instance().get_buffer_data(data.bid, cl::sycl::range<3>(data.subrange.offset0, data.subrange.offset1, data.subrange.offset2),
	        cl::sycl::range<3>(data.subrange.range0, data.subrange.range1, data.subrange.range2));

	// Build subarray data type
	MPI_Datatype subarray_data_type;
	MPI_Type_create_subarray(data_handle.dimensions, data_handle.full_size.data(), data_handle.subsize.data(), data_handle.offsets.data(), MPI_ORDER_C,
	    get_byte_size_data_type(data_handle.element_size), &subarray_data_type);
	MPI_Type_commit(&subarray_data_type);

	auto transfer = std::make_unique<transfer_out>(data_handle);
	transfer->handle = handle;
	transfer->header.subrange = pkg.data.pull.subrange;
	transfer->header.bid = pkg.data.pull.bid;
	transfer->header.tid = pkg.tid;

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
	MPI_Isend(MPI_BOTTOM, 1, transfer_data_type, to, CELERITY_MPI_TAG_DATA_TRANSFER, MPI_COMM_WORLD, &transfer->request);
	outgoing_transfers.insert(std::move(transfer));

	transfer_logger->info("Serving incoming PULL request for buffer {} from node {}", pkg.data.pull.bid, to);
	return handle;
}

// TODO: Copy buffer subrange in case we want to overwrite it (handle here or on job-level?)
std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::await_pull(const command_pkg& pkg) {
	assert(pkg.cmd == command::AWAIT_PULL);
	const await_pull_data& data = pkg.data.await_pull;

	// Check to see if we have a data transfer running already
	for(auto& p : active_handles[pkg.tid][data.bid]) {
		if(p.first.cmd != command::PULL) continue;
		if(p.first.data.pull.subrange == data.subrange) {
			// TODO: We may want to replace the stored pkg with the new one to signal that we are expecting that transfer
			// so we can remove it from the list of active handles upon completion
			return p.second;
		}
	}

	auto handle = std::make_shared<transfer_handle>();
	// Store new handle so we can use it when the pull comes in
	active_handles[pkg.tid][data.bid].push_back(std::make_pair(pkg, handle));
	return handle;
}

void buffer_transfer_manager::update_transfers() {
	for(auto it = incoming_transfers.begin(); it != incoming_transfers.end();) {
		auto& t = *it;
		int flag;
		MPI_Test(&t->request, &flag, MPI_STATUS_IGNORE);
		if(flag == 0) {
			++it;
			continue;
		}

		std::shared_ptr<transfer_handle> handle = nullptr;
		pull_data pdata;
		for(auto& p : active_handles[t->header.tid][t->header.bid]) {
			if(p.first.cmd != command::PULL) continue;
			// TODO: Remove matching handle
			if(p.first.data.pull.subrange == t->header.subrange) {
				// TODO: assert that the pull source is not this node (since this could also be an incoming pull request, see ::handle_request)
				handle = p.second;
				pdata = p.first.data.pull;
				break;
			}
		}
		assert(handle != nullptr && "Received unrequested data");
		handle->complete = true;

		// TODO: Assert t->data.size();
		int dimensions = 3;
		if(pdata.subrange.range2 == 0) {
			dimensions = 2;
			if(pdata.subrange.range1 == 0) { dimensions = 1; }
		}
		// FIXME: It's not ideal that we set raw_data_range::full_size and element_size to all zeros here.
		detail::raw_data_range dr{&t->data[0], dimensions, {(int)pdata.subrange.range0, (int)pdata.subrange.range1, (int)pdata.subrange.range2},
		    {(int)pdata.subrange.offset0, (int)pdata.subrange.offset1, (int)pdata.subrange.offset2}, {0, 0, 0}, 0};
		runtime::get_instance().set_buffer_data(pdata.bid, dr);

		// Transfer cannot be accessed after this!
		it = incoming_transfers.erase(it);
	}

	for(auto it = outgoing_transfers.begin(); it != outgoing_transfers.end();) {
		auto& t = *it;
		int flag;
		MPI_Test(&t->request, &flag, MPI_STATUS_IGNORE);
		if(flag == 0) {
			++it;
			continue;
		}
		t->handle->complete = true;
		// TODO: REMOVE HANDLE FROM active_handles IFF ::await_pull has been called already

		// Transfer cannot be accessed after this!
		it = outgoing_transfers.erase(it);
	}
}

MPI_Datatype buffer_transfer_manager::get_byte_size_data_type(size_t byte_size) {
	if(mpi_byte_size_data_types.count(byte_size) != 0) { return mpi_byte_size_data_types[byte_size]; }
	MPI_Datatype data_type;
	MPI_Type_contiguous(byte_size, MPI_BYTE, &data_type);
	MPI_Type_commit(&data_type);
	mpi_byte_size_data_types[byte_size] = data_type;
	return data_type;
}

std::shared_ptr<const buffer_transfer_manager::transfer_handle> buffer_transfer_manager::pull(const command_pkg& pkg) {
	assert(pkg.cmd == command::PULL);
	const pull_data& data = pkg.data.pull;

	// We basically just forward the command using a different tag, then we wait (asynchronously) for the response transfer
	MPI_Send(&pkg, sizeof(command_pkg), MPI_BYTE, data.source, CELERITY_MPI_TAG_DATA_REQUEST, MPI_COMM_WORLD);

	auto handle = std::make_shared<transfer_handle>();
	active_handles[pkg.tid][data.bid].push_back(std::make_pair(pkg, handle));
	return handle;
}

} // namespace celerity
