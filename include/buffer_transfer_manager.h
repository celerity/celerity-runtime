#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#include <mpi.h>
#pragma clang diagnostic pop

#include <gch/small_vector.hpp>

#include "buffer_storage.h"
#include "frame.h"
#include "grid.h"
#include "mpi_support.h"
#include "types.h"
#include "utils.h"

namespace celerity {
namespace detail {

	class buffer_manager;
	class reduction_manager;

	class buffer_transfer_manager {
		friend struct buffer_transfer_manager_testspy;

	  public:
		struct transfer_handle {
			bool complete = false;
		};

		struct data_frame {
			using payload_type = std::byte;

			// variable-sized structure
			data_frame() = default;
			data_frame(const data_frame&) = delete;
			data_frame& operator=(const data_frame&) = delete;

			buffer_id bid;
			reduction_id rid; // zero if this does not belong to a reduction
			subrange<3> sr;
			transfer_id trid;
			alignas(std::max_align_t) payload_type data[]; // max_align to allow reinterpret_casting a pointer to this member to any buffer element pointer
		};

		// unique_frame_ptr assumes that the flexible payload member begins at exactly sizeof(Frame) bytes
		static_assert(offsetof(data_frame, data) == sizeof(data_frame));

		buffer_transfer_manager(buffer_manager& bm, reduction_manager& rm);

		~buffer_transfer_manager() {
			assert(m_incoming_transfers.empty());
			assert(m_outgoing_transfers.empty());
			assert(m_requests.empty());
		}

		// TODO: Receiving data_frame is not great for encapsulation... Instead provide member function to get a frame + transfer event?
		std::shared_ptr<const transfer_handle> push(const node_id target, unique_frame_ptr<data_frame> frame);
		std::shared_ptr<const transfer_handle> await_push(
		    const transfer_id trid, const buffer_id bid, const GridRegion<3>& expected_region, const reduction_id rid);

		/**
		 * @brief Polls for incoming transfers and updates the status of existing ones.
		 */
		void poll();

	  private:
		struct transfer_in {
			node_id source_nid;
			MPI_Request request;
			unique_frame_ptr<data_frame> frame;
		};

		class request_manager {
		  public:
			request_manager() = default;
			request_manager(const request_manager&) = delete;
			request_manager(request_manager&&) = default;

#if defined(CELERITY_DEBUG)
			~request_manager() { assert(m_received_region == m_expected_region); }
#endif

			std::shared_ptr<transfer_handle> request_region(GridRegion<3> region) {
#if defined(CELERITY_DEBUG)
				if(!m_expected_region.has_value()) {
					m_expected_region = region;
				} else {
					// Multiple await push requests can be associated with this transfer.
					// However, they all must require disjunct parts of the incoming data.
					assert(GridRegion<3>::intersect(*m_expected_region, region).empty());
					m_expected_region = GridRegion<3>::merge(*m_expected_region, region);
				}
#endif
				m_requests.emplace_back(request_state{std::make_shared<transfer_handle>(), std::move(region), {}});
				const size_t idx = m_requests.size() - 1;
				for(size_t i = 0; i < m_transfers.size(); ++i) {
					match(i, idx);
				}
				return m_requests.back().handle;
			}

			void add_transfer(std::unique_ptr<transfer_in>&& t) {
#if defined(CELERITY_DEBUG)
				assert(m_received_region != m_expected_region);
				const auto box = subrange_to_grid_box(t->frame->sr);
				assert(GridRegion<3>::intersect(m_received_region, box).empty());
				m_received_region = GridRegion<3>::merge(m_received_region, box);
#endif
				GridRegion<3> remainder = subrange_to_grid_box(t->frame->sr);
				m_transfers.emplace_back(transfer_state{std::move(t), std::move(remainder), gch::small_vector<size_t>{}});
				const size_t idx = m_transfers.size() - 1;
				for(size_t i = 0; i < m_requests.size(); ++i) {
					match(idx, i);
				}
			}

			bool can_drain() const { return !m_drainable_transfers.empty(); }

			bool fully_drained() const { return m_drain_count == m_transfers.size() && m_received_region == m_expected_region; }

			template <typename Callback>
			void drain_transfers(Callback&& cb) {
				for(size_t i : m_drainable_transfers) {
					assert(m_transfers[i].transfer != nullptr);
					assert(m_transfers[i].remainder.empty());
					cb(std::move(m_transfers[i].transfer));
					assert(!m_transfers[i].requests.empty());
					for(size_t j : m_transfers[i].requests) {
						assert(m_requests[j].unsatisfied.empty());
						assert(m_requests[j].handle.use_count() > 1 && "Dangling await push request");
						m_requests[j].handle->complete = true;
					}
				}
				m_drain_count += m_drainable_transfers.size();
				m_drainable_transfers.clear();
			}

		  private:
			struct transfer_state {
				std::unique_ptr<transfer_in> transfer;
				// Part of the transfer not yet requested
				GridRegion<3> remainder;
				// Array indices of requests that depend on this transfer
				gch::small_vector<size_t> requests;
			};

			struct request_state {
				// Handle for user to query state of request
				std::shared_ptr<transfer_handle> handle;
				// Part of the request without transfer
				GridRegion<3> unsatisfied;
				// Array indices of transfers satisfy this request
				gch::small_vector<size_t> transfers;
			};

			std::vector<transfer_state> m_transfers;
			std::vector<request_state> m_requests;
			std::vector<size_t> m_drainable_transfers;
			size_t m_drain_count = 0;

#if defined(CELERITY_DEBUG)
			std::optional<GridRegion<3>> m_expected_region; // This will only be set once the (first) await push job has started
			GridRegion<3> m_received_region;
#endif

			void match(const size_t transfer_idx, const size_t request_idx) {
				auto& ts = m_transfers[transfer_idx];
				auto& rs = m_requests[request_idx];
				if(auto is = GridRegion<3>::intersect(rs.unsatisfied, ts.remainder); !is.empty()) {
					assert(ts.transfer != nullptr); // Hasn't been drained
					rs.unsatisfied = GridRegion<3>::difference(rs.unsatisfied, is);
					rs.transfers.push_back(transfer_idx);
					ts.remainder = GridRegion<3>::difference(ts.remainder, is);
					ts.requests.push_back(request_idx);
					update_drainability(transfer_idx);
				}
			}

			// TODO: Oof - the must be a simpler way of doing this...
			void update_drainability(const size_t transfer_idx) {
				if(!m_transfers[transfer_idx].remainder.empty()) return; // Early exit

				std::unordered_set<size_t> visited_transfers;
				std::unordered_set<size_t> visited_requests;
				std::queue<size_t> to_visit;
				to_visit.push(transfer_idx);

				while(!to_visit.empty()) {
					const size_t t_idx = to_visit.front();
					// If two transfers share more than one request, we may end up enqueuing them several times.
					if(visited_transfers.count(t_idx)) {
						to_visit.pop();
						continue;
					}
					visited_transfers.insert(t_idx);
					auto& ts = m_transfers[t_idx];
					to_visit.pop();

					if(!ts.remainder.empty()) {
						// At least one transfer in the connected graph hasn't been fully requested, cannot drain.
						return;
					}

					for(size_t i : ts.requests) {
						if(visited_requests.count(i)) continue;
						visited_requests.insert(i);
						auto& rs = m_requests[i];
						if(!rs.unsatisfied.empty()) {
							// At least one unsatisfied request in the connected graph, cannot drain.
							return;
						}
						for(size_t j : rs.transfers) {
							if(j == t_idx) continue;
							to_visit.push(j);
							// NOCOMMIT TODO: Think about this - can it happen? Possibly with 3 transfers that share requests?
							assert(visited_transfers.count(j) == 0);
						}
					}
				}

				for(auto i : visited_transfers) {
					m_drainable_transfers.push_back(i);
				}
			}
		};

		struct transfer_out {
			std::shared_ptr<transfer_handle> handle;
			MPI_Request request;
			unique_frame_ptr<data_frame> frame;
		};

		buffer_manager& m_buffer_mngr;
		reduction_manager& m_reduction_mngr;

		std::list<std::unique_ptr<transfer_in>> m_incoming_transfers;
		std::list<std::unique_ptr<transfer_out>> m_outgoing_transfers;

		// We store a dedicated request manager for each buffer_id/transfer_id combination currently in flight.
		// A manager may be created upon encountering
		//  - incoming transfers that have not yet been requested through ::await_push, as well as
		//  - still outstanding transfers that have been requested through ::await_push
		std::unordered_map<std::pair<buffer_id, transfer_id>, request_manager, utils::pair_hash> m_requests;

		mpi_support::data_type m_send_recv_unit;

		void poll_incoming_transfers();
		void update_incoming_transfers();
		void update_outgoing_transfers();

		void commit_transfer(transfer_in& transfer);
	};

} // namespace detail
} // namespace celerity
