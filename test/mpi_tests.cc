#include "communicator.h"
#include "host_utils.h"
#include "mpi_communicator.h"
#include "test_utils.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>


using namespace celerity;
using namespace celerity::detail;


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator sends and receives pilot messages", "[mpi]") {
	mpi_communicator comm(MPI_COMM_WORLD);
	if(comm.get_num_nodes() <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const auto make_pilot_message = [&](const node_id sender, const node_id receiver) {
		const auto p2p_id = 1 + sender * comm.get_num_nodes() + receiver;
		const message_id msgid = p2p_id * 13;
		const buffer_id bid = p2p_id * 11;
		const task_id consumer_tid = p2p_id * 17;
		const reduction_id rid = p2p_id * 19;
		const transfer_id trid(consumer_tid, bid, rid);
		const box<3> box = {id{p2p_id, p2p_id * 2, p2p_id * 3}, id{p2p_id * 4, p2p_id * 5, p2p_id * 6}};
		return outbound_pilot{receiver, pilot_message{msgid, trid, box}};
	};

	for(node_id to = 0; to < comm.get_num_nodes(); ++to) {
		if(to == comm.get_local_node_id()) continue;
		comm.send_outbound_pilot(make_pilot_message(comm.get_local_node_id(), to));
	}

	size_t num_pilots_received = 0;
	while(num_pilots_received < comm.get_num_nodes() - 1) {
		for(const auto& pilot : comm.poll_inbound_pilots()) {
			CAPTURE(pilot.from, comm.get_local_node_id());
			const auto expect = make_pilot_message(pilot.from, comm.get_local_node_id());
			CHECK(pilot.message.id == expect.message.id);
			CHECK(pilot.message.transfer_id == expect.message.transfer_id);
			CHECK(pilot.message.box == expect.message.box);
			++num_pilots_received;
		}
	}
	CHECK(num_pilots_received == comm.get_num_nodes() - 1);
}


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator sends and receives payloads", "[mpi]") {
	mpi_communicator comm(MPI_COMM_WORLD);
	if(comm.get_num_nodes() <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const auto make_msgid = [&](const node_id sender, const node_id receiver) { //
		return message_id(1 + sender * comm.get_num_nodes() + receiver);
	};

	const communicator::stride stride{{12, 11, 11}, {{1, 0, 3}, {5, 4, 6}}, sizeof(int)};

	std::vector<std::vector<int>> send_buffers;
	std::vector<std::vector<int>> receive_buffers;
	std::vector<async_event> events;
	for(node_id other = 0; other < comm.get_num_nodes(); ++other) {
		if(other == comm.get_local_node_id()) continue;

		auto& send = send_buffers.emplace_back(stride.allocation.size());
		std::iota(send.begin(), send.end(), make_msgid(comm.get_local_node_id(), other));
		auto& receive = receive_buffers.emplace_back(stride.allocation.size());
		events.push_back(comm.send_payload(other, make_msgid(comm.get_local_node_id(), other), send.data(), stride));
		events.push_back(comm.receive_payload(other, make_msgid(other, comm.get_local_node_id()), receive.data(), stride));
	}

	while(!events.empty()) {
		const auto end_incomplete = std::remove_if(events.begin(), events.end(), std::mem_fn(&async_event::is_complete));
		events.erase(end_incomplete, events.end());
	}

	auto received = receive_buffers.begin();
	for(node_id other = 0; other < comm.get_num_nodes(); ++other) {
		if(other == comm.get_local_node_id()) continue;

		std::vector<int> other_send(stride.allocation.size());
		std::iota(other_send.begin(), other_send.end(), make_msgid(other, comm.get_local_node_id()));
		std::vector<int> expected(stride.allocation.size());
		experimental::for_each_item(stride.subrange.range, [&](const item<3>& item) {
			const auto id = stride.subrange.offset + item.get_id();
			const auto linear_index = get_linear_index(stride.allocation, id);
			expected[linear_index] = other_send[linear_index];
		});
		CHECK(*received == expected);
		++received;
	}
}


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator correctly transfers scalars between strides of different dimensionality", "[mpi]") {
	mpi_communicator comm(MPI_COMM_WORLD);
	if(comm.get_num_nodes() <= 1) { SKIP("test must be run on at least 2 ranks"); }
	if(comm.get_local_node_id() >= 2) return; // needs exactly 2 nodes

	const auto send_dims = GENERATE(values<size_t>({0, 1, 2, 3}));
	const auto recv_dims = GENERATE(values<size_t>({0, 1, 2, 3}));
	CAPTURE(send_dims, recv_dims);

	constexpr communicator::stride dim_strides[] = {
	    {{1, 1, 1}, {{0, 0, 0}, {1, 1, 1}}, 4}, // 0-dimensional
	    {{2, 1, 1}, {{1, 0, 0}, {1, 1, 1}}, 4}, // 1-dimensional
	    {{2, 3, 1}, {{1, 2, 0}, {1, 1, 1}}, 4}, // 2-dimensional
	    {{2, 3, 5}, {{1, 2, 3}, {1, 1, 1}}, 4}, // 3-dimensional
	};

	const auto& send_stride = dim_strides[send_dims];
	const auto& recv_stride = dim_strides[recv_dims];

	std::vector<int> buf(dim_strides[3].allocation.size());
	async_event evt;
	if(comm.get_local_node_id() == 1) {
		buf[get_linear_index(send_stride.allocation, send_stride.subrange.offset)] = 42;
		evt = comm.send_payload(0, 99, buf.data(), send_stride);
	} else {
		evt = comm.receive_payload(1, 99, buf.data(), recv_stride);
	}
	while(!evt.is_complete()) {}

	if(comm.get_local_node_id() == 0) {
		std::vector<int> expected(dim_strides[3].allocation.size());
		expected[get_linear_index(recv_stride.allocation, recv_stride.subrange.offset)] = 42;
		CHECK(buf == expected);
	}
}


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator correctly transfers boxes that map to different subranges on sender and receiver", "[mpi]") {
	mpi_communicator comm(MPI_COMM_WORLD);
	if(comm.get_num_nodes() <= 1) { SKIP("test must be run on at least 2 ranks"); }
	if(comm.get_local_node_id() >= 2) return; // needs exactly 2 nodes

	const auto dims = GENERATE(values<int>({1, 2, 3}));
	CAPTURE(dims);

	range box_range{3, 4, 5};
	range sender_allocation{10, 7, 11};
	id sender_offset{1, 2, 3};
	range receiver_allocation{8, 10, 13};
	id receiver_offset{2, 0, 4};
	// manually truncate to runtime value `dims`
	for(int d = dims; d < 3; ++d) {
		box_range[d] = 1;
		sender_allocation[d] = 1;
		sender_offset[d] = 0;
		receiver_allocation[d] = 1;
		receiver_offset[d] = 0;
	}

	std::vector<int> send_buf(sender_allocation.size());
	std::vector<int> recv_buf(receiver_allocation.size());

	std::iota(send_buf.begin(), send_buf.end(), 0);

	async_event evt;
	if(comm.get_local_node_id() == 1) {
		evt = comm.send_payload(0, 99, send_buf.data(), communicator::stride{sender_allocation, subrange{sender_offset, box_range}, sizeof(int)});
	} else {
		evt = comm.receive_payload(1, 99, recv_buf.data(), communicator::stride{receiver_allocation, subrange{receiver_offset, box_range}, sizeof(int)});
	}
	while(!evt.is_complete()) {}

	if(comm.get_local_node_id() == 0) {
		std::vector<int> expected(receiver_allocation.size());
		experimental::for_each_item(box_range, [&](const item<3>& item) {
			const auto sender_idx = get_linear_index(sender_allocation, sender_offset + item.get_id());
			const auto receiver_idx = get_linear_index(receiver_allocation, receiver_offset + item.get_id());
			expected[receiver_idx] = send_buf[sender_idx];
		});
		CHECK(recv_buf == expected);
	}
}


// TODO implement and test transfers > 2Gi elements
