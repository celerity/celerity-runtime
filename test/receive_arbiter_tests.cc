#include "host_utils.h"
#include "nd_memory.h"
#include "receive_arbiter.h"
#include "test_utils.h"

#include <algorithm>
#include <map>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;

class mock_recv_communicator : public communicator {
  public:
	explicit mock_recv_communicator(const size_t num_nodes, const node_id local_node_id) : m_num_nodes(num_nodes), m_local_nid(local_node_id) {}
	mock_recv_communicator(mock_recv_communicator&&) = default;
	mock_recv_communicator& operator=(mock_recv_communicator&&) = default;
	mock_recv_communicator(const mock_recv_communicator&) = delete;
	mock_recv_communicator& operator=(const mock_recv_communicator&) = delete;
	~mock_recv_communicator() override { CHECK(m_pending_recvs.empty()); }

	size_t get_num_nodes() const override { return m_num_nodes; }
	node_id get_local_node_id() const override { return m_local_nid; }
	void send_outbound_pilot(const outbound_pilot& /* pilot */) override { utils::panic("unimplemented"); }

	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override { return std::move(m_inbound_pilots); }

	[[nodiscard]] async_event send_payload(
	    const node_id /* to */, const message_id /* outbound_pilot_tag */, const void* const /* base */, const stride& /* stride */) override {
		utils::panic("unimplemented");
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
		REQUIRE(src_range == stride.subrange.range);
		nd_copy_host(src, dest, src_range, stride.allocation, zeros, stride.subrange.offset, stride.subrange.range, stride.element_size);
		*flag = true;
		m_pending_recvs.erase(key);
	}

	collective_group* get_collective_root() override { utils::panic("unimplemented"); }

  private:
	using completion_flag = std::shared_ptr<bool>;

	class mock_event final : public async_event_base {
	  public:
		explicit mock_event(const completion_flag& flag) : m_flag(flag) {}
		bool is_complete() const override { return *m_flag; }

	  private:
		completion_flag m_flag;
	};

	size_t m_num_nodes;
	node_id m_local_nid;
	std::vector<inbound_pilot> m_inbound_pilots;
	std::unordered_map<std::pair<node_id, message_id>, std::tuple<void*, stride, completion_flag>, utils::pair_hash> m_pending_recvs;
};

// from https://en.cppreference.com/w/cpp/algorithm/next_permutation - replace with std::next_permutation for C++20
template <class BidirIt>
bool next_permutation(BidirIt first, BidirIt last) {
	auto r_first = std::make_reverse_iterator(last);
	auto r_last = std::make_reverse_iterator(first);
	auto left = std::is_sorted_until(r_first, r_last);

	if(left != r_last) {
		auto right = std::upper_bound(r_first, left, *left);
		std::iter_swap(left, right);
	}

	std::reverse(left.base(), last);
	return left != r_last;
}

enum class receive_event { call_to_receive, incoming_pilot, incoming_data };

template <>
struct Catch::StringMaker<receive_event> {
	static std::string convert(const receive_event& event) {
		switch(event) {
		case receive_event::call_to_receive: return "call_to_receive";
		case receive_event::incoming_pilot: return "incoming_pilot";
		case receive_event::incoming_data: return "incoming_data";
		default: abort();
		}
	}
};

std::vector<std::vector<std::pair<receive_event, node_id>>> enumerate_receive_event_orders(const std::vector<node_id>& peers) {
	constexpr auto permutation_order = [](const std::pair<receive_event, node_id>& lhs, const std::pair<receive_event, node_id>& rhs) {
		if(lhs.first < rhs.first) return true;
		if(lhs.first > rhs.first) return false;
		return lhs.second < rhs.second;
	};

	// start with the first permutation according to permutation_order
	std::vector<std::pair<receive_event, node_id>> current_permutation;
	for(const auto event : {receive_event::call_to_receive, receive_event::incoming_pilot, receive_event::incoming_data}) {
		for(const auto nid : peers) {
			current_permutation.emplace_back(event, nid);
		}
	}
	const auto index_of = [&](const receive_event event, const node_id nid) {
		return std::find(current_permutation.begin(), current_permutation.end(), std::pair{event, nid}) - current_permutation.begin();
	};

	std::vector<std::vector<std::pair<receive_event, node_id>>> event_orders;
	for(;;) {
		bool is_valid_order = true;
		for(const auto nid : peers) {
			is_valid_order &= index_of(receive_event::call_to_receive, nid) < index_of(receive_event::incoming_data, nid);
			is_valid_order &= index_of(receive_event::incoming_pilot, nid) < index_of(receive_event::incoming_data, nid);
		}
		if(is_valid_order) { event_orders.push_back(current_permutation); }
		if(!next_permutation(current_permutation.begin(), current_permutation.end(), permutation_order)) return event_orders;
	}
}

// TODO use this -^ in more tests instead of the `pilots_before` hack


TEST_CASE("receive_arbiter aggregates receives of subsets", "[receive_arbiter]") {
	const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	const box<3> alloc_box = {{2, 1, 0}, {39, 10, 10}};
	const box<3> recv_box = {{4, 2, 1}, {37, 9, 8}};
	const size_t elem_size = sizeof(int);

	const std::tuple<node_id, int, box<3>> fragments_meta[] = {
	    {node_id(1), 15, box<3>({4, 2, 1}, {22, 9, 4})},
	    {node_id(2), 14, box<3>({4, 2, 4}, {22, 9, 8})},
	    {node_id(3), 12, box<3>({22, 2, 1}, {37, 9, 4})},
	    {node_id(4), 13, box<3>({22, 2, 4}, {37, 9, 8})},
	};
	constexpr size_t num_fragments = std::size(fragments_meta);
	CAPTURE(num_fragments);

	std::vector<std::vector<int>> fragments;
	for(const auto& [from, msgid, box] : fragments_meta) {
		fragments.emplace_back(box.get_range().size(), static_cast<int>(from));
	}
	std::vector<inbound_pilot> pilots;
	for(const auto& [from, msgid, box] : fragments_meta) {
		pilots.push_back(inbound_pilot{from, pilot_message{msgid, trid, box}});
	}

	mock_recv_communicator comm(8, 0);
	receive_arbiter ra(comm);

	const size_t num_pilots_pushed_before_begin = GENERATE(values<size_t>({0, num_fragments / 2, num_fragments}));
	CAPTURE(num_pilots_pushed_before_begin);
	for(size_t i = 0; i < num_pilots_pushed_before_begin; ++i) {
		comm.push_inbound_pilot(pilots[i]);
	}
	ra.poll_communicator();

	std::vector<int> allocation(alloc_box.get_range().size());
	ra.begin_split_receive(trid, recv_box, allocation.data(), alloc_box, elem_size);

	const size_t num_pilots_completed_before_await = std::min(num_pilots_pushed_before_begin, GENERATE(values<size_t>({0, num_fragments / 2, num_fragments})));
	CAPTURE(num_pilots_completed_before_await);
	for(size_t i = 0; i < num_pilots_completed_before_await; ++i) {
		const auto& [from, msgid, box] = fragments_meta[i];
		comm.complete_receiving_payload(from, msgid, fragments[i].data(), box.get_range());
	}
	ra.poll_communicator();

	const auto event = ra.await_split_receive_subregion(trid, recv_box);

	for(size_t i = num_pilots_completed_before_await; i < num_pilots_pushed_before_begin; ++i) {
		const auto& [from, msgid, box] = fragments_meta[i];
		comm.complete_receiving_payload(from, msgid, fragments[i].data(), box.get_range());
	}
	ra.poll_communicator();

	for(size_t i = num_pilots_pushed_before_begin; i < num_fragments; ++i) {
		comm.push_inbound_pilot(pilots[i]);
	}
	ra.poll_communicator();

	for(size_t i = num_pilots_pushed_before_begin; i < num_fragments; ++i) {
		const auto& [from, msgid, box] = fragments_meta[i];
		comm.complete_receiving_payload(from, msgid, fragments[i].data(), box.get_range());
	}
	ra.poll_communicator();

	CHECK(event.is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	for(const auto& [from, msgid, box] : fragments_meta) {
		experimental::for_each_item(box.get_range(), [&, from = from, &box = box](const item<3>& it) {
			const auto id_in_allocation = box.get_offset() - alloc_box.get_offset() + it.get_id();
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected_allocation[linear_index] = static_cast<int>(from);
		});
	}
	CHECK(allocation == expected_allocation);
}

TEST_CASE("receive_arbiter can await supersets of incoming fragments", "[receive_arbiter]") {
	const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	const box<3> alloc_box = {{2, 1, 0}, {19, 20, 1}};
	const region<3> recv_regions[] = {
	    region<3>{{{{4, 1, 0}, {14, 10, 1}}, {{14, 1, 0}, {19, 18, 1}}}},
	    box<3>{{4, 10, 0}, {14, 18, 1}},
	};
	const box<3> fragment_box = {{4, 1, 0}, {19, 18, 1}}; // union of recv_regions
	const region<3> full_recv_region = fragment_box;
	const size_t elem_size = sizeof(int);
	const node_id from = 1;
	const message_id msgid = 15;

	mock_recv_communicator comm(2, 0);
	receive_arbiter ra(comm);

	comm.push_inbound_pilot(inbound_pilot{from, pilot_message{msgid, trid, fragment_box}});
	ra.poll_communicator();

	std::vector<int> allocation(alloc_box.get_range().size());
	ra.begin_split_receive(trid, full_recv_region, allocation.data(), alloc_box, elem_size);
	const auto event_0 = ra.await_split_receive_subregion(trid, recv_regions[0]);
	const auto event_1 = ra.await_split_receive_subregion(trid, recv_regions[1]);
	ra.poll_communicator();

	CHECK(!event_0.is_complete());
	CHECK(!event_1.is_complete());

	std::vector<int> fragment(fragment_box.get_range().size(), static_cast<int>(from));
	comm.complete_receiving_payload(from, msgid, fragment.data(), fragment_box.get_range());
	ra.poll_communicator();

	CHECK(event_0.is_complete());
	CHECK(event_1.is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	experimental::for_each_item(fragment_box.get_range(), [&](const item<3>& it) {
		const auto id_in_allocation = fragment_box.get_offset() - alloc_box.get_offset() + it.get_id();
		const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
		expected_allocation[linear_index] = static_cast<int>(from);
	});
	CHECK(allocation == expected_allocation);
}

TEST_CASE("receive_arbiter single-instruction receive works", "[receive_arbiter]") {
	const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	const box<3> alloc_box = {{2, 1, 0}, {19, 20, 1}};
	const region<3> recv_region{{{{4, 1, 0}, {14, 10, 1}}, {{14, 1, 0}, {19, 18, 1}}}};
	const size_t num_fragments = recv_region.get_boxes().size();
	const size_t elem_size = sizeof(int);
	const node_id from = 1;

	mock_recv_communicator comm(2, 0);
	receive_arbiter ra(comm);

	const auto num_pilots_before_receive = GENERATE(values<size_t>({0, 1, 2}));
	CAPTURE(num_pilots_before_receive);
	for(size_t i = 0; i < num_pilots_before_receive; ++i) {
		comm.push_inbound_pilot(inbound_pilot{from, pilot_message{static_cast<int>(i), trid, recv_region.get_boxes()[i]}});
	}
	ra.poll_communicator();

	std::vector<int> allocation(alloc_box.get_range().size());
	auto event = ra.receive(trid, recv_region, allocation.data(), alloc_box, elem_size);
	CHECK(!event.is_complete());

	for(size_t i = num_pilots_before_receive; i < num_fragments; ++i) {
		comm.push_inbound_pilot(inbound_pilot{from, pilot_message{static_cast<int>(i), trid, recv_region.get_boxes()[i]}});
	}
	ra.poll_communicator();

	for(size_t i = 0; i < num_fragments; ++i) {
		CHECK(!event.is_complete());
		const auto& fragment_box = recv_region.get_boxes()[i];
		std::vector<int> fragment(fragment_box.get_range().size(), static_cast<int>(from));
		comm.complete_receiving_payload(from, static_cast<int>(i), fragment.data(), fragment_box.get_range());
		ra.poll_communicator();
	}
	CHECK(event.is_complete());

	std::vector<int> expected_allocation(alloc_box.get_range().size());
	for(const auto& fragment_box : recv_region.get_boxes()) {
		experimental::for_each_item(fragment_box.get_range(), [&](const item<3>& it) {
			const auto id_in_allocation = fragment_box.get_offset() - alloc_box.get_offset() + it.get_id();
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected_allocation[linear_index] = static_cast<int>(from);
		});
	}
	CHECK(allocation == expected_allocation);
}

TEST_CASE("receive_arbiter::gather_receive works", "[receive_arbiter]") {
	const transfer_id trid(task_id(2), buffer_id(0), reduction_id(1));
	const size_t chunk_size = 4;
	const box<3> unit_box{{0, 0, 0}, {1, 1, 1}};

	mock_recv_communicator comm(4, 0);
	receive_arbiter ra(comm);

	const auto max_pilot_before_gather = GENERATE(values<size_t>({1, 2, 4}));
	CAPTURE(max_pilot_before_gather);
	for(size_t i = 1; i < max_pilot_before_gather; ++i) {
		comm.push_inbound_pilot(inbound_pilot{node_id(i), pilot_message{static_cast<int>(i), trid, unit_box}});
	}

	std::vector<uint8_t> gather_allocation(chunk_size * comm.get_num_nodes());
	// my (local) chunk is default-initialized to the correct value 0

	auto event = ra.gather_receive(trid, gather_allocation.data(), chunk_size);
	CHECK(!event.is_complete());

	for(size_t i = max_pilot_before_gather; i < comm.get_num_nodes(); ++i) {
		comm.push_inbound_pilot(inbound_pilot{node_id(i), pilot_message{static_cast<int>(i), trid, unit_box}});
	}
	ra.poll_communicator();
	CHECK(!event.is_complete());

	for(size_t i = 1; i < comm.get_num_nodes(); ++i) {
		std::vector<uint8_t> fragment(chunk_size, static_cast<uint8_t>(i));
		comm.complete_receiving_payload(node_id(i), static_cast<int>(i), fragment.data(), unit_box.get_range());
	}
	ra.poll_communicator();
	CHECK(event.is_complete());

	std::vector<uint8_t> expected_allocation(chunk_size * comm.get_num_nodes());
	for(size_t i = 0; i < comm.get_num_nodes(); ++i) {
		for(size_t j = 0; j < chunk_size; ++j) {
			expected_allocation[i * chunk_size + j] = static_cast<uint8_t>(i);
		}
	}
	CHECK(gather_allocation == expected_allocation);
}

TEST_CASE("receive_arbiter handles multiple receive instructions for the same transfer id", "[receive_arbiter]") {
	const transfer_id trid(task_id(1), buffer_id(420), no_reduction_id);
	const box<3> alloc_box = {{0, 0, 0}, {20, 20, 1}};
	const std::map<node_id, box<3>> receives{
	    {node_id(1), box<3>({2, 2, 0}, {8, 18, 1})},
	    {node_id(2), box<3>({12, 2, 0}, {18, 18, 1})},
	};
	const size_t elem_size = sizeof(int);

	const auto& event_order = GENERATE(from_range(enumerate_receive_event_orders({node_id(1), node_id(2)})));
	CAPTURE(event_order);

	mock_recv_communicator comm(3, 0);
	receive_arbiter ra(comm);

	std::vector<int> allocation(alloc_box.get_range().size());
	std::map<node_id, async_event> events;

	for(const auto& [event, from] : event_order) {
		CAPTURE(event, from);

		switch(event) {
		case receive_event::call_to_receive: {
			events.emplace(from, ra.receive(trid, receives.at(from), allocation.data(), alloc_box, elem_size));
			break;
		}
		case receive_event::incoming_pilot: {
			comm.push_inbound_pilot(inbound_pilot{from, pilot_message{static_cast<int>(from), trid, receives.at(from)}});
			break;
		}
		case receive_event::incoming_data: {
			std::vector<int> fragment(receives.at(from).get_range().size(), static_cast<int>(from));
			comm.complete_receiving_payload(from, static_cast<int>(from), fragment.data(), receives.at(from).get_range());
			break;
		}
		}
		ra.poll_communicator();

		if(events.count(from) > 0) { CHECK(events.at(from).is_complete() == (event == receive_event::incoming_data)); }
	}

	for(auto& [from, event] : events) {
		CAPTURE(from);
		CHECK(event.is_complete());
	}

	std::vector<int> expected(alloc_box.get_range().size());
	for(const auto& [from, box] : receives) {
		experimental::for_each_item(box.get_range(), [&, from = from, box = box](const item<3>& it) {
			const auto id_in_allocation = box.get_offset() - alloc_box.get_offset() + it.get_id();
			const auto linear_index = get_linear_index(alloc_box.get_range(), id_in_allocation);
			expected[linear_index] = static_cast<int>(from);
		});
	}
	CHECK(allocation == expected);
}

// peers will send a pilot with an empty box to signal that they don't contribute to a reduction
TEST_CASE("receive_arbiter knows how to handle empty pilot boxes in gathers", "[receive_arbiter]") {
	const transfer_id trid(task_id(2), buffer_id(0), reduction_id(1));
	const size_t chunk_size = 4;
	const box<3> empty_box;
	const box<3> unit_box{{0, 0, 0}, {1, 1, 1}};

	mock_recv_communicator comm(3, 0);
	receive_arbiter ra(comm);

	// Catch2 does not support values<bool>() because of the std::vector<bool> specialization
	const auto empty_pilot_first = static_cast<bool>(GENERATE(values<int>({false, true})));
	const auto num_pilots_before_gather = GENERATE(values<size_t>({0, 1, 2}));
	CAPTURE(empty_pilot_first, num_pilots_before_gather);

	std::vector<inbound_pilot> pilots{
	    inbound_pilot{node_id(1), pilot_message{1, trid, empty_box}},
	    inbound_pilot{node_id(2), pilot_message{2, trid, unit_box}},
	};
	if(!empty_pilot_first) { std::swap(pilots[0], pilots[1]); }

	for(size_t i = 0; i < num_pilots_before_gather; ++i) {
		comm.push_inbound_pilot(pilots[i]);
	}

	std::vector<uint8_t> gather_allocation(chunk_size * comm.get_num_nodes());
	auto event = ra.gather_receive(trid, gather_allocation.data(), chunk_size);
	ra.poll_communicator();
	CHECK(!event.is_complete());

	for(size_t i = num_pilots_before_gather; i < 2; ++i) {
		comm.push_inbound_pilot(pilots[i]);
	}
	ra.poll_communicator();
	CHECK(!event.is_complete());

	int fragment = 42;
	comm.complete_receiving_payload(node_id(2), static_cast<int>(2), &fragment, unit_box.get_range());
	ra.poll_communicator();

	CHECK(event.is_complete());
}
