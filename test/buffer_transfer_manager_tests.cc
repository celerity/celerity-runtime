#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "buffer_transfer_manager.h"

#include <mpi.h>

using namespace celerity;
using namespace celerity::detail;

namespace celerity::detail {
struct buffer_transfer_manager_testspy {
	static size_t get_request_count(const buffer_transfer_manager& btm) { return btm.m_requests.size(); }
};
} // namespace celerity::detail

namespace {
template <typename Predicate>
void poll_until(buffer_transfer_manager& btm, Predicate p) {
	while(!p()) {
		// Poll several times to avoid relying on implementation details (e.g. whether a newly incoming transfer can complete within a single poll)
		for(int i = 0; i < 100; ++i) {
			btm.poll();
		}
	}
}

class buffer_transfer_manager_fixture : public test_utils::mpi_fixture {
  public:
	buffer_transfer_manager_fixture() : m_config(nullptr, nullptr) {
		MPI_Comm_size(MPI_COMM_WORLD, &m_num_ranks);
		MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
		m_local_devices.init(m_config);
		m_buffer_mngr = std::make_unique<buffer_manager>(m_local_devices, [](auto, auto) {});
		m_buffer_transfer_mngr = std::make_unique<buffer_transfer_manager>(*m_buffer_mngr, m_reduction_mngr);
	}

	void require_num_ranks(const size_t num_ranks) const { REQUIRE(m_num_ranks == static_cast<int>(num_ranks)); }

	int get_rank() const { return m_rank; }

	buffer_id create_buffer(const size_t range_1d) {
		const range<3> buf_range{range_1d, 1, 1};
		const auto bid = m_buffer_mngr->register_buffer<char, 1>(range<3>{range_1d, 1, 1});
		auto info = m_buffer_mngr->access_host_buffer<char, 1>(bid, access_mode::discard_write, buf_range, {});
		std::memset(info.ptr, m_rank + 1, buf_range.size());
		return bid;
	}

	void verify_buffer_ranges(const buffer_id bid, const std::vector<std::pair<subrange<1>, node_id>>& owned_by) {
		for(auto& [sr, nid] : owned_by) {
			auto info = m_buffer_mngr->access_host_buffer<char, 1>(bid, access_mode::read, range_cast<3>(sr.range), id_cast<3>(sr.offset));
			for(size_t i = 0; i < sr.range[0]; ++i) {
				const size_t idx = sr.offset[0] + i - info.backing_buffer_offset[0];
				REQUIRE_LOOP(static_cast<char*>(info.ptr)[idx] == static_cast<char>(nid) + 1);
			}
		}
	}

	buffer_transfer_manager& get_buffer_transfer_manager() { return *m_buffer_transfer_mngr; }

  private:
	int m_num_ranks = 0;
	int m_rank = -1;
	local_devices m_local_devices;
	config m_config;
	std::unique_ptr<buffer_manager> m_buffer_mngr;
	reduction_manager m_reduction_mngr;
	std::unique_ptr<buffer_transfer_manager> m_buffer_transfer_mngr;
};


} // namespace

// TODO: We should also test this with more than one rank pushing
TEST_CASE_METHOD(buffer_transfer_manager_fixture, "multiple pushes can satisfy a single await push") {
	require_num_ranks(2);
	buffer_transfer_manager& btm = get_buffer_transfer_manager();

	const auto bid = create_buffer(128);
	const transfer_id trid = 123;
	const reduction_id rid = 0;

	// Awaited region can be disjoint!
	const auto sr_1 = subrange<3>({0, 0, 0}, {32, 1, 1});
	const auto sr_2 = subrange<3>({96, 0, 0}, {32, 1, 1});

	if(get_rank() == 0) {
		const auto handle_1 = btm.push(1, trid, bid, sr_1, rid);
		const auto handle_2 = btm.push(1, trid, bid, sr_2, rid);
		poll_until(btm, [&]() { return handle_1->complete && handle_2->complete; });
		SUCCEED();
	}

	if(get_rank() == 1) {
		const GridRegion<3> expected = GridRegion<3>::merge(subrange_to_grid_box(sr_1), subrange_to_grid_box(sr_2));
		const auto handle = btm.await_push(trid, bid, expected, rid);
		poll_until(btm, [&]() { return handle->complete; });
		verify_buffer_ranges(bid, {{subrange_cast<1>(sr_1), 0}, {{32, 64}, 1}, {subrange_cast<1>(sr_2), 0}});
	}
}

// This is required for multi-gpu support: Local chunks might generate separate await push commands
// while other nodes only see a single distributed chunk requiring the union of those await pushes.
TEST_CASE_METHOD(buffer_transfer_manager_fixture, "a single push can satisfy multiple await pushes") {
	require_num_ranks(2);
	buffer_transfer_manager& btm = get_buffer_transfer_manager();

	const auto bid = create_buffer(128);
	const transfer_id trid = 123;
	const reduction_id rid = 0;

	const auto sr = subrange<3>({32, 0, 0}, {64, 1, 1});

	if(get_rank() == 0) {
		const auto handle = btm.push(1, trid, bid, sr, rid);
		poll_until(btm, [&]() { return handle->complete; });
		SUCCEED();
	}

	if(get_rank() == 1) {
		// NB: We have to post both awaits before polling for completion as the incoming transfer doesn't match either exactly so they can't be fast-tracked.
		const auto handle_1 = btm.await_push(trid, bid, subrange_to_grid_box(subrange<3>({32, 0, 0}, {32, 1, 1})), rid);
		const auto handle_2 = btm.await_push(trid, bid, subrange_to_grid_box(subrange<3>({64, 0, 0}, {32, 1, 1})), rid);
		poll_until(btm, [&]() { return handle_2->complete; });
		verify_buffer_ranges(bid, {{{0, 32}, 1}, {subrange_cast<1>(sr), 0}, {{96, 32}, 1}});
	}
}

TEST_CASE_METHOD(buffer_transfer_manager_fixture, "multiple pushes can satisfy multiple await pushes for a single transfer") {
	require_num_ranks(2);
	buffer_transfer_manager& btm = get_buffer_transfer_manager();

	const auto bid = create_buffer(128);
	const transfer_id trid = 123;
	const reduction_id rid = 0;

	// Awaited region can be disjoint!
	const auto sr_1 = subrange<3>({0, 0, 0}, {32, 1, 1});
	const auto sr_2 = subrange<3>({96, 0, 0}, {32, 1, 1});

	if(get_rank() == 0) {
		const auto handle_1 = btm.push(1, trid, bid, sr_1, rid);
		const auto handle_2 = btm.push(1, trid, bid, sr_2, rid);
		poll_until(btm, [&]() { return handle_1->complete && handle_2->complete; });
		SUCCEED();
	}

	if(get_rank() == 1) {
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> handle_1;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> handle_2;

		handle_1 = btm.await_push(trid, bid, subrange_to_grid_box(sr_1), rid);
		handle_2 = btm.await_push(trid, bid, subrange_to_grid_box(sr_2), rid);

		poll_until(btm, [&]() { return handle_1->complete && handle_2->complete; });
		verify_buffer_ranges(bid, {{subrange_cast<1>(sr_1), 0}, {{32, 64}, 1}, {subrange_cast<1>(sr_2), 0}});
	}
}

// TEST_CASE_METHOD(buffer_transfer_manager_fixture, "multiple pushes can satisfy multiple await pushes for a single transfer") {
// 	require_num_ranks(2);
// 	buffer_transfer_manager& btm = get_buffer_transfer_manager();

// 	const auto bid = create_buffer(128);
// 	const transfer_id trid = 123;
// 	const reduction_id rid = 0;

// 	// Awaited region can be disjoint!
// 	const auto sr_1 = subrange<3>({0, 0, 0}, {32, 1, 1});
// 	const auto sr_2 = subrange<3>({96, 0, 0}, {32, 1, 1});

// 	// If a partial await push matches what has been received so far exactly, we can fast-track the request and complete it right away.
// 	// Additional pushes and await pushes for the same transfer will then be handled independently afterwards.
// 	const bool enable_fast_track = GENERATE(true, false);

// 	if(get_rank() == 0) {
// 		const auto handle_1 = btm.push(1, trid, bid, sr_1, rid);
// 		if(enable_fast_track) {
// 			// Wait until 1 has received the first transfer
// 			MPI_Barrier(MPI_COMM_WORLD);
// 		}
// 		const auto handle_2 = btm.push(1, trid, bid, sr_2, rid);
// 		poll_until(btm, [&]() { return handle_1->complete && handle_2->complete; });
// 		SUCCEED();
// 	}

// 	if(get_rank() == 1) {
// 		// Wait until we have received some data so we can reliably check for handle_1's completion below
// 		poll_until(btm, [&]() { return buffer_transfer_manager_testspy::get_request_count(btm) > 0; });

// 		std::shared_ptr<const buffer_transfer_manager::transfer_handle> handle_1;
// 		std::shared_ptr<const buffer_transfer_manager::transfer_handle> handle_2;

// 		if(enable_fast_track) {
// 			handle_1 = btm.await_push(trid, bid, subrange_to_grid_box(sr_1), rid);
// 			CHECK(handle_1->complete);
// 			MPI_Barrier(MPI_COMM_WORLD);
// 			handle_2 = btm.await_push(trid, bid, subrange_to_grid_box(sr_2), rid);
// 		} else {
// 			handle_1 = btm.await_push(trid, bid, subrange_to_grid_box(sr_2), rid);
// 			// Since we posted the await for the second subrange before the first the BTM now waits for the other before it completes both,
// 			// even though the data to satisfy this request should already be available.
// 			// TODO: This is a quality-of-implementation issue and could be further optimized, although solving it in general might prove to be tricky
// 			CHECK(!handle_1->complete);
// 			handle_2 = btm.await_push(trid, bid, subrange_to_grid_box(sr_1), rid);
// 		}

// 		poll_until(btm, [&]() { return handle_1->complete && handle_2->complete; });
// 		verify_buffer_ranges(bid, {{subrange_cast<1>(sr_1), 0}, {{32, 64}, 1}, {subrange_cast<1>(sr_2), 0}});
// 	}
// }
