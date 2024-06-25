#include "buffer_storage.h" // for memcpy_strided_host
#include "receive_arbiter.h"
#include "test_utils.h"

#include <algorithm>
#include <map>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;

/// A mock communicator implementation that allows tests to manually push inbound pilots and incoming receive payloads that the receive_arbiter is waiting for.
class mock_recv_communicator : public communicator {
  public:
	/// `num_nodes` and `local_node_id` simply are the values reported by the respective getters.
	explicit mock_recv_communicator(const size_t num_nodes, const node_id local_node_id) : m_num_nodes(num_nodes), m_local_nid(local_node_id) {}
	mock_recv_communicator(const mock_recv_communicator&) = delete;
	mock_recv_communicator(mock_recv_communicator&&) = delete;
	mock_recv_communicator& operator=(const mock_recv_communicator&) = delete;
	mock_recv_communicator& operator=(mock_recv_communicator&&) = delete;
	~mock_recv_communicator() override { CHECK(m_pending_recvs.empty()); }

	size_t get_num_nodes() const override { return m_num_nodes; }
	node_id get_local_node_id() const override { return m_local_nid; }

	void send_outbound_pilot(const outbound_pilot& /* pilot */) override {
		utils::panic("unimplemented"); // receive_arbiter does not send stuff
	}

	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override { return std::move(m_inbound_pilots); }

	[[nodiscard]] async_event send_payload(
	    const node_id /* to */, const message_id /* outbound_pilot_tag */, const void* const /* base */, const stride& /* stride */) override {
		utils::panic("unimplemented"); // receive_arbiter does not send stuff
	}

	[[nodiscard]] async_event receive_payload(const node_id from, const message_id msgid, void* const base, const stride& stride) override {
		const auto key = std::pair(from, msgid);
		REQUIRE(m_pending_recvs.count(key) == 0);
		completion_flag flag = std::make_shared<bool>(false);
		m_pending_recvs.emplace(key, std::tuple(base, stride, flag));
		return make_async_event<mock_event>(flag);
	}

	void push_inbound_pilot(const inbound_pilot& pilot) { m_inbound_pilots.push_back(pilot); }

	void complete_receiving_payload(const node_id from, const message_id msgid, const void* const src, const range<3>& src_range) {
		const auto key = std::pair(from, msgid);
		const auto [dest, stride, flag] = m_pending_recvs.at(key);
		REQUIRE(src_range == stride.transfer.range);
		memcpy_strided_host(src, dest, stride.element_size, src_range, zeros, stride.allocation_range, stride.transfer.offset, stride.transfer.range);
		*flag = true;
		m_pending_recvs.erase(key);
	}

	std::unique_ptr<communicator> collective_clone() override { utils::panic("unimplemented"); }
	void collective_barrier() override { utils::panic("unimplemented"); }

  private:
	using completion_flag = std::shared_ptr<bool>;

	class mock_event final : public async_event_impl {
	  public:
		explicit mock_event(const completion_flag& flag) : m_flag(flag) {}
		bool is_complete() override { return *m_flag; }

	  private:
		completion_flag m_flag;
	};

	size_t m_num_nodes;
	node_id m_local_nid;
	std::vector<inbound_pilot> m_inbound_pilots;
	std::unordered_map<std::pair<node_id, message_id>, std::tuple<void*, stride, completion_flag>, utils::pair_hash> m_pending_recvs;
};

/// Instructs the test loop to perform a specific operation on receive_arbiter or mock_recv_communicator in order to execute tests in all possible event orders
/// with the help of `enumerate_all_event_orders`.
struct receive_event {
	enum {
		call_to_receive, ///< `receive_arbiter::receive` or `begin_split_receive` is called
		incoming_pilot,  ///< `communicator::poll_inbound_pilots` returns a pilot with matching transfer id
		incoming_data,   ///< The async_event from `communicator::receive_payload` completes (after call_to_receive and incoming_pilot)
	} transition;

	/// For call_to_receive, the index in requested_regions; for incoming_pilot/incoming_data, the index in incoming_fragments
	size_t which;

	friend bool operator==(const receive_event& lhs, const receive_event& rhs) { return lhs.transition == rhs.transition && lhs.which == rhs.which; }
	friend bool operator!=(const receive_event& lhs, const receive_event& rhs) { return !(lhs == rhs); }
};

template <>
struct Catch::StringMaker<receive_event> {
	static std::string convert(const receive_event& event) {
		switch(event.transition) {
		case receive_event::call_to_receive: return fmt::format("call_to_receive[{}]", event.which);
		case receive_event::incoming_pilot: return fmt::format("incoming_pilot[{}]", event.which);
		case receive_event::incoming_data: return fmt::format("incoming_data[{}]", event.which);
		default: abort();
		}
	}
};

/// Enumerates all O(N!) possible `receive_event` orders that would complete the `requested_regions` with the `incoming_fragments`.
std::vector<std::vector<receive_event>> enumerate_all_event_orders(
    const std::vector<region<3>>& requested_regions, const std::vector<box<3>>& incoming_fragments) //
{
	constexpr static auto permutation_order = [](const receive_event& lhs, const receive_event& rhs) {
		if(lhs.transition < rhs.transition) return true;
		if(lhs.transition > rhs.transition) return false;
		return lhs.which < rhs.which;
	};

	// construct the first permutation according to permutation_order
	std::vector<receive_event> current_permutation;
	for(size_t region_id = 0; region_id < requested_regions.size(); ++region_id) {
		current_permutation.push_back({receive_event::call_to_receive, region_id});
	}
	for(size_t fragment_id = 0; fragment_id < incoming_fragments.size(); ++fragment_id) {
		current_permutation.push_back({receive_event::incoming_pilot, fragment_id});
	}
	for(size_t fragment_id = 0; fragment_id < incoming_fragments.size(); ++fragment_id) {
		current_permutation.push_back({receive_event::incoming_data, fragment_id});
	}

	// helper: get the index within current_permutation
	const auto index_of = [&](const receive_event& event) {
		for(size_t i = 0; i < current_permutation.size(); ++i) {
			if(current_permutation[i].transition == event.transition && current_permutation[i].which == event.which) return i;
		}
		abort();
	};

	// collect all legal permutations (i.e. pilots are received before data, and calls to receive() also happen before receiving data)
	std::vector<std::vector<receive_event>> transition_orders;
	for(;;) {
		bool is_valid_order = true;
		for(size_t fragment_id = 0; fragment_id < incoming_fragments.size(); ++fragment_id) {
			is_valid_order &= index_of({receive_event::incoming_pilot, fragment_id}) < index_of({receive_event::incoming_data, fragment_id});
		}
		for(size_t region_id = 0; region_id < requested_regions.size(); ++region_id) {
			const auto receive_called_at = index_of({receive_event::call_to_receive, region_id});
			for(size_t i = 0; i < receive_called_at; ++i) {
				if(current_permutation[i].transition == receive_event::incoming_data) {
					is_valid_order &= region_intersection(incoming_fragments[current_permutation[i].which], requested_regions[region_id]).empty();
				}
			}
		}
		if(is_valid_order) { transition_orders.push_back(current_permutation); }

		if(!std::next_permutation(current_permutation.begin(), current_permutation.end(), permutation_order)) {
			// we wrapped around to the first permutation according to permutation_order
			return transition_orders;
		}
	}
}

TEST_CASE("receive_arbiter aggregates receives from multiple incoming fragments", "[receive_arbiter]") {
	static const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	static const box<3> alloc_box = {{2, 1, 0}, {39, 10, 10}};
	static const std::vector<box<3>> incoming_fragments{
	    box<3>({4, 2, 1}, {22, 9, 4}),
	    box<3>({4, 2, 4}, {22, 9, 8}),
	    box<3>({22, 2, 1}, {37, 9, 4}),
	};
	static const std::vector<region<3>> requested_regions{
	    region(box_vector<3>(incoming_fragments.begin(), incoming_fragments.end())),
	};
	static const size_t elem_size = sizeof(int);

	const auto& event_order = GENERATE(from_range(enumerate_all_event_orders(requested_regions, incoming_fragments)));
	CAPTURE(event_order);

	const auto receive_method = GENERATE(values<std::string>({"single", "split_await"}));
	CAPTURE(receive_method);

	mock_recv_communicator comm(4, 0);
	receive_arbiter ra(comm);

	std::vector<int> allocation(alloc_box.get_range().size());
	std::optional<async_event> receive;

	for(const auto& [transition, which] : event_order) {
		const node_id peer = 1 + which;
		const message_id msgid = 10 + which;
		CAPTURE(transition, which, peer, msgid);

		// only the last event (always an incoming_data transition) will complete the receive
		if(receive.has_value()) { CHECK(!receive->is_complete()); }

		switch(transition) {
		case receive_event::call_to_receive: {
			CHECK_FALSE(receive.has_value());
			if(receive_method == "single") {
				receive = ra.receive(trid, requested_regions[0], allocation.data(), alloc_box, elem_size);
			} else if(receive_method == "split_await") {
				ra.begin_split_receive(trid, requested_regions[0], allocation.data(), alloc_box, elem_size);
				receive = ra.await_split_receive_subregion(trid, requested_regions[0]);
			}
			break;
		}
		case receive_event::incoming_pilot: {
			comm.push_inbound_pilot(inbound_pilot{peer, pilot_message{msgid, trid, incoming_fragments[which]}});
			break;
		}
		case receive_event::incoming_data: {
			std::vector<int> fragment(incoming_fragments[which].get_range().size(), static_cast<int>(peer));
			comm.complete_receiving_payload(peer, msgid, fragment.data(), incoming_fragments[which].get_range());
			break;
		}
		}
		ra.poll_communicator();
	}

	REQUIRE(receive.has_value());
	CHECK(receive->is_complete());

	// it is legal to `await` a transfer that has already been completed and is not tracked by the receive_arbiter anymore
	CHECK(ra.await_split_receive_subregion(trid, requested_regions[0]).is_complete());
	CHECK(ra.await_split_receive_subregion(trid, incoming_fragments[0]).is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	for(size_t which = 0; which < incoming_fragments.size(); ++which) {
		const auto& box = incoming_fragments[which];
		const node_id peer = 1 + which;
		test_utils::for_each_in_range(box.get_range(), box.get_offset() - alloc_box.get_offset(), [&](const id<3>& id_in_allocation) {
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected_allocation[linear_index] = static_cast<int>(peer);
		});
	}
	CHECK(allocation == expected_allocation);
}

TEST_CASE("receive_arbiter can complete await-receives through differently-shaped overlapping fragments", "[receive_arbiter]") {
	static const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	static const box<3> alloc_box = {{2, 1, 0}, {19, 20, 1}};
	static const std::vector<region<3>> requested_regions{
	    box<3>{{4, 1, 0}, {19, 18, 1}},
	};
	static const std::vector<region<3>> awaited_regions{
	    region<3>{{{{4, 1, 0}, {14, 10, 1}}, {{14, 1, 0}, {19, 18, 1}}}},
	    box<3>{{4, 10, 0}, {14, 18, 1}},
	};
	static const std::vector<box<3>> incoming_fragments{
	    box<3>{{4, 1, 0}, {14, 18, 1}},
	    box<3>{{14, 1, 0}, {19, 18, 1}},
	};
	static const size_t elem_size = sizeof(int);

	const auto& event_order = GENERATE(from_range(enumerate_all_event_orders(requested_regions, incoming_fragments)));
	CAPTURE(event_order);

	mock_recv_communicator comm(2, 0);
	receive_arbiter ra(comm);

	const node_id peer = 1;

	std::vector<int> allocation(alloc_box.get_range().size());
	std::optional<async_event> awaits[2];
	region<3> region_received;

	for(const auto& [transition, which] : event_order) {
		const message_id msgid = 10 + which;
		CAPTURE(transition, which, peer, msgid);

		// check that fragments[0] completes awaits[1]
		for(size_t await_id = 0; await_id < 2; ++await_id) {
			if(!awaits[await_id].has_value()) continue;
			CHECK(awaits[await_id]->is_complete() == (region_intersection(region_received, awaited_regions[await_id]) == awaited_regions[await_id]));
		}

		switch(transition) {
		case receive_event::call_to_receive: {
			ra.begin_split_receive(trid, requested_regions[0], allocation.data(), alloc_box, elem_size);
			awaits[0] = ra.await_split_receive_subregion(trid, awaited_regions[0]);
			awaits[1] = ra.await_split_receive_subregion(trid, awaited_regions[1]);
			break;
		}
		case receive_event::incoming_pilot: {
			comm.push_inbound_pilot(inbound_pilot{peer, pilot_message{msgid, trid, incoming_fragments[which]}});
			break;
		}
		case receive_event::incoming_data: {
			std::vector<int> fragment(incoming_fragments[which].get_range().size(), static_cast<int>(1 + which));
			comm.complete_receiving_payload(peer, msgid, fragment.data(), incoming_fragments[which].get_range());
			region_received = region_union(region_received, incoming_fragments[which]);
			break;
		}
		}
		ra.poll_communicator();
	}

	REQUIRE(awaits[0].has_value());
	REQUIRE(awaits[0]->is_complete());
	REQUIRE(awaits[1].has_value());
	REQUIRE(awaits[1]->is_complete());

	// it is legal to `await` a transfer that has already been completed and is not tracked by the receive_arbiter anymore
	CHECK(ra.await_split_receive_subregion(trid, requested_regions[0]).is_complete());
	CHECK(ra.await_split_receive_subregion(trid, incoming_fragments[0]).is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	for(size_t which = 0; which < incoming_fragments.size(); ++which) {
		const auto& box = incoming_fragments[which];
		test_utils::for_each_in_range(box.get_range(), box.get_offset() - alloc_box.get_offset(), [&](const id<3>& id_in_allocation) {
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected_allocation[linear_index] = static_cast<int>(1 + which);
		});
	}
	CHECK(allocation == expected_allocation);
}

TEST_CASE("receive_arbiter immediately completes await-receives for which all corresponding fragments have already been received", "[receive_arbiter]") {
	static const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	static const box<3> alloc_box = {{2, 1, 0}, {19, 20, 1}};
	static const std::vector<region<3>> requested_regions{
	    box<3>{{4, 1, 0}, {19, 18, 1}},
	};
	static const std::vector<region<3>> awaited_regions{
	    region<3>{{{{4, 1, 0}, {14, 10, 1}}, {{14, 1, 0}, {19, 18, 1}}}},
	    box<3>{{4, 10, 0}, {14, 18, 1}},
	};
	static const std::vector<box<3>> incoming_fragments{
	    box<3>{{4, 1, 0}, {14, 18, 1}},
	    box<3>{{14, 1, 0}, {19, 18, 1}},
	};
	static const size_t elem_size = sizeof(int);

	mock_recv_communicator comm(2, 0);
	receive_arbiter ra(comm);

	const node_id peer = 1;

	std::vector<int> allocation(alloc_box.get_range().size());
	region<3> region_received;

	const auto receive_fragment = [&](const size_t which) {
		const message_id msgid = 10 + which;
		comm.push_inbound_pilot(inbound_pilot{peer, pilot_message{msgid, trid, incoming_fragments[which]}});
		ra.poll_communicator();
		std::vector<int> fragment(incoming_fragments[which].get_range().size(), static_cast<int>(1 + which));
		comm.complete_receiving_payload(peer, msgid, fragment.data(), incoming_fragments[which].get_range());
		ra.poll_communicator();
		region_received = region_union(region_received, incoming_fragments[which]);
	};

	ra.begin_split_receive(trid, requested_regions[0], allocation.data(), alloc_box, elem_size);
	receive_fragment(0);

	auto await0 = ra.await_split_receive_subregion(trid, awaited_regions[0]);
	CHECK_FALSE(await0.is_complete());
	auto await1 = ra.await_split_receive_subregion(trid, awaited_regions[1]);
	CHECK(await1.is_complete());

	receive_fragment(1);
	CHECK(await0.is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	for(size_t which = 0; which < incoming_fragments.size(); ++which) {
		const auto& box = incoming_fragments[which];
		test_utils::for_each_in_range(box.get_range(), box.get_offset() - alloc_box.get_offset(), [&](const id<3>& id_in_allocation) {
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected_allocation[linear_index] = static_cast<int>(1 + which);
		});
	}
	CHECK(allocation == expected_allocation);
}

TEST_CASE("receive_arbiter handles multiple receive instructions for the same transfer id", "[receive_arbiter]") {
	static const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	static const box<3> alloc_box = {{0, 0, 0}, {20, 20, 1}};
	static const std::vector incoming_fragments{
	    box<3>({2, 2, 0}, {8, 18, 1}),
	    box<3>({12, 2, 0}, {18, 18, 1}),
	};
	static const std::vector requested_regions{
	    region(incoming_fragments[0]),
	    region(incoming_fragments[1]),
	};
	static const size_t elem_size = sizeof(int);

	const auto& event_order = GENERATE(from_range(enumerate_all_event_orders(requested_regions, incoming_fragments)));
	CAPTURE(event_order);

	mock_recv_communicator comm(3, 0);
	receive_arbiter ra(comm);

	std::vector<int> allocation(alloc_box.get_range().size());
	std::map<node_id, async_event> events;

	for(const auto& [transition, which] : event_order) {
		const node_id peer = 1 + which;
		const message_id msgid = 10 + which;
		CAPTURE(transition, which, peer, msgid);

		switch(transition) {
		case receive_event::call_to_receive: {
			events.emplace(peer, ra.receive(trid, requested_regions[which], allocation.data(), alloc_box, elem_size));
			break;
		}
		case receive_event::incoming_pilot: {
			comm.push_inbound_pilot(inbound_pilot{peer, pilot_message{msgid, trid, incoming_fragments[which]}});
			break;
		}
		case receive_event::incoming_data: {
			std::vector<int> fragment(incoming_fragments[which].get_range().size(), static_cast<int>(peer));
			comm.complete_receiving_payload(peer, msgid, fragment.data(), incoming_fragments[which].get_range());
			break;
		}
		}
		ra.poll_communicator();

		if(events.count(peer) > 0) { CHECK(events.at(peer).is_complete() == (transition == receive_event::incoming_data)); }
	}

	for(auto& [from, event] : events) {
		CAPTURE(from);
		CHECK(event.is_complete());
	}

	std::vector<int> expected(alloc_box.get_range().size());
	for(size_t from = 0; from < incoming_fragments.size(); ++from) {
		const auto& box = incoming_fragments[from];
		test_utils::for_each_in_range(box.get_range(), box.get_offset() - alloc_box.get_offset(), [&, from = from](const id<3>& id_in_allocation) {
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected[linear_index] = static_cast<int>(1 + from);
		});
	}
	CHECK(allocation == expected);
}

TEST_CASE("receive_arbiter::gather_receive works", "[receive_arbiter]") {
	static const transfer_id trid(task_id(2), buffer_id(0), reduction_id(1));
	static const box<3> unit_box{{0, 0, 0}, {1, 1, 1}};
	static const std::vector requested_regions{region(unit_box)};
	static const std::vector incoming_fragments{unit_box, unit_box, unit_box}; // each fragment is the chunk from a peer
	static const size_t chunk_size = sizeof(int);

	const auto& event_order = GENERATE(from_range(enumerate_all_event_orders(requested_regions, incoming_fragments)));
	CAPTURE(event_order);

	mock_recv_communicator comm(4, 0);
	receive_arbiter ra(comm);

	std::vector<int> allocation(comm.get_num_nodes(), -1);
	std::optional<async_event> receive;

	for(const auto& [transition, which] : event_order) {
		CAPTURE(transition, which);

		// only the last event (always an incoming_data transition) will complete the receive
		if(receive.has_value()) { CHECK(!receive->is_complete()); }

		switch(transition) {
		case receive_event::call_to_receive: {
			receive = ra.gather_receive(trid, allocation.data(), chunk_size);
			break;
		}
		case receive_event::incoming_pilot: {
			const node_id peer = 1 + which;
			const message_id msgid = 10 + which;
			CAPTURE(peer, msgid);
			comm.push_inbound_pilot(inbound_pilot{peer, pilot_message{msgid, trid, incoming_fragments[which]}});
			break;
		}
		case receive_event::incoming_data: {
			const node_id peer = 1 + which;
			const message_id msgid = 10 + which;
			CAPTURE(peer, msgid);
			std::vector<int> fragment(incoming_fragments[which].get_range().size(), static_cast<int>(peer));
			comm.complete_receiving_payload(peer, msgid, fragment.data(), incoming_fragments[which].get_range());
			break;
		}
		}
		ra.poll_communicator();
	}

	REQUIRE(receive.has_value());
	CHECK(receive->is_complete());
	CHECK(allocation == std::vector{-1 /* unchanged */, 1, 2, 3});
}

// peers will send a pilot with an empty box to signal that they don't contribute to a reduction
TEST_CASE("receive_arbiter knows how to handle empty pilot boxes in gathers", "[receive_arbiter]") {
	const transfer_id trid(task_id(2), buffer_id(0), reduction_id(1));
	const box<3> empty_box;
	static const box<3> unit_box{{0, 0, 0}, {1, 1, 1}};
	static const std::vector requested_regions{region(unit_box)};
	static const std::vector incoming_fragments{unit_box, empty_box};
	static const size_t chunk_size = sizeof(int);

	const auto& event_order = GENERATE(from_range(enumerate_all_event_orders(requested_regions, incoming_fragments)));
	CAPTURE(event_order);

	mock_recv_communicator comm(3, 0);
	receive_arbiter ra(comm);

	std::vector<int> allocation(comm.get_num_nodes(), -1);
	std::optional<async_event> receive;

	for(const auto& [transition, which] : event_order) {
		CAPTURE(transition, which);

		switch(transition) {
		case receive_event::call_to_receive: {
			receive = ra.gather_receive(trid, allocation.data(), chunk_size);
			break;
		}
		case receive_event::incoming_pilot: {
			const node_id peer = 1 + which;
			const message_id msgid = 10 + which;
			CAPTURE(peer, msgid);
			comm.push_inbound_pilot(inbound_pilot{peer, pilot_message{msgid, trid, incoming_fragments[which]}});
			break;
		}
		case receive_event::incoming_data: {
			if(!incoming_fragments[which].empty()) {
				const node_id peer = 1 + which;
				const message_id msgid = 10 + which;
				CAPTURE(peer, msgid);
				std::vector<int> fragment(incoming_fragments[which].get_range().size(), static_cast<int>(peer));
				comm.complete_receiving_payload(peer, msgid, fragment.data(), incoming_fragments[which].get_range());
			}
			break;
		}
		}
		ra.poll_communicator();
	}

	REQUIRE(receive.has_value());
	CHECK(receive->is_complete());

	// only a single chunk, `1`, is actually written to `allocation`
	CHECK(allocation == std::vector{-1, 1, -1});
}
