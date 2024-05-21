#include "../test_utils.h"

#include "communicator.h"
#include "mpi_communicator.h"
#include "types.h"

#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>


using namespace celerity;
using namespace celerity::detail;
using namespace std::chrono_literals;


namespace celerity::detail {

struct mpi_communicator_testspy {
	static size_t get_num_active_outbound_pilots(const mpi_communicator& comm) { return comm.m_outbound_pilots.size(); }
	static size_t get_num_cached_array_types(const mpi_communicator& comm) { return comm.m_array_type_cache.size(); }
	static size_t get_num_cached_scalar_types(const mpi_communicator& comm) { return comm.m_scalar_type_cache.size(); }
};

} // namespace celerity::detail


TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator sends and receives pilot messages", "[mpi]") {
	mpi_communicator comm(collective_clone_from, MPI_COMM_WORLD);
	const auto num_nodes = comm.get_num_nodes();
	const auto self = comm.get_local_node_id();
	CAPTURE(num_nodes, self);

	if(num_nodes <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const auto make_pilot_message = [&](const node_id sender, const node_id receiver) {
		// Compute a unique id for the (sender, receiver) tuple and base all other members of the pilot message on this ID to test that we receive the correct
		// pilots on the correct nodes (and everything remains uncorrupted).
		const auto p2p_id = 1 + sender * num_nodes + receiver;
		const message_id msgid = p2p_id * 13;
		const buffer_id bid = p2p_id * 11;
		const task_id consumer_tid = p2p_id * 17;
		const reduction_id rid = p2p_id * 19;
		const transfer_id trid(consumer_tid, bid, rid);
		const box<3> box = {id{p2p_id, p2p_id * 2, p2p_id * 3}, id{p2p_id * 4, p2p_id * 5, p2p_id * 6}};
		return outbound_pilot{receiver, pilot_message{msgid, trid, box}};
	};

	// Send a pilot from each node to each other node
	for(node_id other = 0; other < num_nodes; ++other) {
		if(other == self) continue;
		comm.send_outbound_pilot(make_pilot_message(self, other));
	}

	size_t num_pilots_received = 0;
	while(num_pilots_received < num_nodes - 1) {
		// busy-wait for all expected pilots to arrive
		for(const auto& pilot : comm.poll_inbound_pilots()) {
			CAPTURE(pilot.from);
			const auto expect = make_pilot_message(pilot.from, self);
			CHECK(pilot.message.id == expect.message.id);
			CHECK(pilot.message.transfer_id == expect.message.transfer_id);
			CHECK(pilot.message.box == expect.message.box);
			++num_pilots_received;
		}
	}

	SUCCEED("it didn't deadlock 👏");
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator sends and receives payloads", "[mpi]") {
	mpi_communicator comm(collective_clone_from, MPI_COMM_WORLD);
	const auto num_nodes = comm.get_num_nodes();
	const auto self = comm.get_local_node_id();
	CAPTURE(num_nodes, self);

	if(num_nodes <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const auto make_msgid = [=](const node_id sender, const node_id receiver) { //
		return message_id(1 + sender * num_nodes + receiver);
	};

	constexpr static communicator::stride stride{{12, 11, 11}, {{1, 0, 3}, {5, 4, 6}}, sizeof(int)};

	std::vector<std::vector<int>> send_buffers;
	std::vector<std::vector<int>> receive_buffers;
	std::vector<async_event> events;
	for(node_id other = 0; other < num_nodes; ++other) {
		if(other == self) continue;

		// allocate a send buffer and fill with a (sender, receiver) specific pattern that can be tested after receiving
		auto& send = send_buffers.emplace_back(stride.allocation_range.size());
		std::iota(send.begin(), send.end(), make_msgid(self, other));

		// allocate and zero-fill a receive buffer (zero is never part of a payload)
		auto& receive = receive_buffers.emplace_back(stride.allocation_range.size());

		// start send and receive operations
		events.push_back(comm.send_payload(other, make_msgid(self, other), send.data(), stride));
		events.push_back(comm.receive_payload(other, make_msgid(other, self), receive.data(), stride));
	}

	// busy-wait for all send / receive events to complete
	while(!events.empty()) {
		const auto end_incomplete = std::remove_if(events.begin(), events.end(), std::mem_fn(&async_event::is_complete));
		events.erase(end_incomplete, events.end());
	}

	auto received = receive_buffers.begin();
	for(node_id other = 0; other < num_nodes; ++other) {
		if(other == self) continue;

		// reconstruct the expected receive buffer
		std::vector<int> other_send(stride.allocation_range.size());
		std::iota(other_send.begin(), other_send.end(), make_msgid(other, self));
		std::vector<int> expected(stride.allocation_range.size());
		test_utils::for_each_in_range(stride.transfer.range, stride.transfer.offset, [&](const id<3>& id) {
			const auto linear_index = get_linear_index(stride.allocation_range, id);
			expected[linear_index] = other_send[linear_index];
		});

		CHECK(*received == expected);
		++received; // not equivalent to receive_buffers[other] because we skip `other == self`
	}
}

// We require that it's well-defined to send a scalar from an n-dimensional stride and receive it in an m-dimensional stride, since stride dimensionality is
// determined from effective allocation dimensionality, which can vary between participating nodes depending on the size of their buffer host allocations.
TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator correctly transfers scalars between strides of different dimensionality", "[mpi]") {
	// All GENERATEs must happen before an early-return, otherwise different nodes will execute this test case different numbers of times
	const auto send_dims = GENERATE(values<size_t>({0, 1, 2, 3}));
	const auto recv_dims = GENERATE(values<size_t>({0, 1, 2, 3}));
	CAPTURE(send_dims, recv_dims);

	mpi_communicator comm(collective_clone_from, MPI_COMM_WORLD);
	const auto num_nodes = comm.get_num_nodes();
	const auto local_node_id = comm.get_local_node_id();
	CAPTURE(num_nodes, local_node_id);

	if(num_nodes <= 1) { SKIP("test must be run on at least 2 ranks"); }
	if(local_node_id >= 2) return; // needs exactly 2 nodes to participate

	constexpr communicator::stride dim_strides[] = {
	    {{1, 1, 1}, {{0, 0, 0}, {1, 1, 1}}, 4}, // 0-dimensional
	    {{2, 1, 1}, {{1, 0, 0}, {1, 1, 1}}, 4}, // 1-dimensional
	    {{2, 3, 1}, {{1, 2, 0}, {1, 1, 1}}, 4}, // 2-dimensional
	    {{2, 3, 5}, {{1, 2, 3}, {1, 1, 1}}, 4}, // 3-dimensional
	};

	const auto& send_stride = dim_strides[send_dims];
	const auto& recv_stride = dim_strides[recv_dims];

	std::vector<int> buf(dim_strides[3].allocation_range.size());
	async_event evt;
	if(local_node_id == 1) { // sender
		buf[get_linear_index(send_stride.allocation_range, send_stride.transfer.offset)] = 42;
		evt = comm.send_payload(0, 99, buf.data(), send_stride);
	} else { // receiver
		evt = comm.receive_payload(1, 99, buf.data(), recv_stride);
	}
	// busy-wait for event
	while(!evt.is_complete()) {}

	if(local_node_id == 0) { // receiver
		std::vector<int> expected(dim_strides[3].allocation_range.size());
		expected[get_linear_index(recv_stride.allocation_range, recv_stride.transfer.offset)] = 42;
		CHECK(buf == expected);
	}
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "mpi_communicator correctly transfers boxes that map to different subranges on sender and receiver", "[mpi]") {
	// All GENERATEs must happen before an early-return, otherwise different nodes will execute this test case a different number of times
	const auto dims = GENERATE(values<int>({1, 2, 3}));
	CAPTURE(dims);

	mpi_communicator comm(collective_clone_from, MPI_COMM_WORLD);
	const auto num_nodes = comm.get_num_nodes();
	const auto local_node_id = comm.get_local_node_id();
	CAPTURE(num_nodes, local_node_id);

	if(num_nodes <= 1) { SKIP("test must be run on at least 2 ranks"); }
	if(local_node_id >= 2) return; // needs exactly 2 nodes

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
	if(local_node_id == 1) { // sender
		evt = comm.send_payload(0, 99, send_buf.data(), communicator::stride{sender_allocation, subrange{sender_offset, box_range}, sizeof(int)});
	} else { // receiver
		evt = comm.receive_payload(1, 99, recv_buf.data(), communicator::stride{receiver_allocation, subrange{receiver_offset, box_range}, sizeof(int)});
	}
	while(!evt.is_complete()) {} // busy-wait for evt

	if(local_node_id == 0) {
		std::vector<int> expected(receiver_allocation.size());
		test_utils::for_each_in_range(box_range, [&](const id<3>& id) {
			const auto sender_idx = get_linear_index(sender_allocation, sender_offset + id);
			const auto receiver_idx = get_linear_index(receiver_allocation, receiver_offset + id);
			expected[receiver_idx] = send_buf[sender_idx];
		});
		CHECK(recv_buf == expected);
	}
}

TEST_CASE_METHOD(test_utils::mpi_fixture, "collectives are concurrent between distinct mpi_communicators", "[mpi][smoke-test]") {
	constexpr static size_t concurrency = 16;

	// create a bunch of communicators that we can then operate on from concurrent threads
	std::vector<std::unique_ptr<communicator>> roots;
	for(size_t i = 0; i < concurrency; ++i) {
		roots.push_back(std::make_unique<mpi_communicator>(collective_clone_from, MPI_COMM_WORLD));
	}

	// for each communicator, spawn a thread that creates more communicators
	std::vector<std::vector<std::unique_ptr<communicator>>> concurrent_clones(concurrency);
	std::vector<std::thread> concurrent_threads(concurrency);
	for(size_t i = 0; i < concurrency; ++i) {
		concurrent_threads[i] = std::thread([&, i] {
			for(size_t j = 0; j < concurrency; ++j) {
				concurrent_clones[i].push_back(roots[i]->collective_clone());
				std::this_thread::sleep_for(10ms); // ensure the OS doesn't serialize all threads by chance
			}
		});
	}
	for(size_t i = 0; i < concurrency; ++i) {
		concurrent_threads[i].join();
	}

	// flip the iteration order and issue a barrier from each new collective group
	for(size_t i = 0; i < concurrency; ++i) {
		concurrent_threads[i] = std::thread([&, i] {
			for(size_t j = 0; j < concurrency; ++j) {
				concurrent_clones[j][i]->collective_barrier();
				std::this_thread::sleep_for(10ms); // ensure the OS doesn't serialize all threads by chance
			}
		});
	}
	for(size_t i = 0; i < concurrency; ++i) {
		concurrent_threads[i].join();
	}

	// ~mpi_communicator is also a collective operation; and it shouldn't matter if we destroy parents before their children
	roots.clear();

	for(size_t i = 0; i < concurrency; ++i) {
		concurrent_threads[i] = std::thread([&, i] {
			concurrent_clones[i].clear(); // ~mpi_communicator is a collective operation
		});
	}
	for(size_t i = 0; i < concurrency; ++i) {
		concurrent_threads[i].join();
	}

	SUCCEED("it didn't deadlock or crash 🎉");
}

TEST_CASE("mpi_communicator normalizes strides to cache and re-uses MPI data types", "[mpi]") {
	static const std::vector<std::vector<communicator::stride>> sets_of_equivalent_strides{
	    // strides only differ in allocation size / offset in dim 0 and can be normalized by adjusting the base pointer
	    {
	        {{13, 12, 11}, {{1, 0, 3}, {5, 4, 6}}, sizeof(int)},
	        {{5, 12, 11}, {{0, 0, 3}, {5, 4, 6}}, sizeof(int)},
	        {{20, 12, 11}, {{4, 0, 3}, {5, 4, 6}}, sizeof(int)},
	    },
	    {
	        {{13, 1, 1}, {{1, 0, 0}, {5, 1, 1}}, sizeof(int)},
	        {{1, 13, 1}, {{0, 1, 0}, {1, 5, 1}}, sizeof(int)},
	        {{1, 1, 13}, {{0, 0, 1}, {1, 1, 5}}, sizeof(int)},
	    },
	};
	// All GENERATEs must happen before an early-return, otherwise different nodes will execute this test case different numbers of times
	const auto equivalent_strides = GENERATE(from_range(sets_of_equivalent_strides));

	mpi_communicator comm(collective_clone_from, MPI_COMM_WORLD);
	const auto num_nodes = comm.get_num_nodes();
	const auto self = comm.get_local_node_id();
	CAPTURE(num_nodes, self);

	if(num_nodes <= 1) { SKIP("test must be run on at least 2 ranks"); }
	if(self >= 2) return; // needs exactly 2 nodes
	const node_id peer = 1 - self;

	message_id msgid = 0;
	for(int repeat = 0; repeat < 2; ++repeat) {
		CAPTURE(repeat);
		for(const auto& stride : equivalent_strides) {
			CAPTURE(stride.allocation_range, stride.transfer, stride.element_size);
			std::vector<int> send_buf(stride.allocation_range.size());
			std::iota(send_buf.begin(), send_buf.end(), 1);
			const auto send_evt = comm.send_payload(peer, msgid, std::as_const(send_buf).data(), stride);

			std::vector<int> recv_buf(stride.allocation_range.size());
			const auto recv_event = comm.receive_payload(peer, msgid, recv_buf.data(), stride);

			while(!send_evt.is_complete() || !recv_event.is_complete()) {} // busy-wait for events to complete

			std::vector<int> expected(stride.allocation_range.size());
			test_utils::for_each_in_range(stride.transfer.range, stride.transfer.offset, [&](const id<3>& id) {
				const auto linear_id = get_linear_index(stride.allocation_range, id);
				expected[linear_id] = send_buf[linear_id];
			});

			CHECK(recv_buf == expected);
			++msgid;
		}
	}

	CHECK(mpi_communicator_testspy::get_num_cached_array_types(comm) == 1);  // all strides we sent/received were equivalent under normalization
	CHECK(mpi_communicator_testspy::get_num_cached_scalar_types(comm) == 1); // only scalar type used was int
}

TEST_CASE("successfully sent pilots are garbage-collected by communicator", "[mpi]") {
	mpi_communicator comm(collective_clone_from, MPI_COMM_WORLD);
	const auto num_nodes = comm.get_num_nodes();
	const auto self = comm.get_local_node_id();
	CAPTURE(num_nodes, self);

	if(num_nodes <= 1) { SKIP("test must be run on at least 2 ranks"); }

	const bool participate = self < 2; // needs exactly 2 participating nodes
	const node_id peer = 1 - self;

	if(participate) {
		for(int i = 0; i < 3; ++i) {
			comm.send_outbound_pilot(outbound_pilot{peer, pilot_message{0, transfer_id{0, 0, 0}, box<3>{id{0, 0, 0}, id{1, 1, 1}}}});
		}
		CHECK(mpi_communicator_testspy::get_num_active_outbound_pilots(comm) <= 3);

		size_t num_received_pilots = 0;
		while(num_received_pilots < 3) {
			num_received_pilots += comm.poll_inbound_pilots().size();
		}
	}

	comm.collective_barrier(); // hope that this also means all p2p transfers have complete...

	if(participate) {
		// send_outbound_pilot will garbage-collect all finished pilot-sends
		comm.send_outbound_pilot(outbound_pilot{peer, pilot_message{0, transfer_id{0, 0, 0}, box<3>{id{0, 0, 0}, id{1, 1, 1}}}});
		CHECK(mpi_communicator_testspy::get_num_active_outbound_pilots(comm) <= 1);
	}
}
