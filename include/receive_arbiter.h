#pragma once

#include "communicator.h"
#include "pilot.h"

#include <unordered_map>

namespace celerity::detail {

class receive_arbiter {
  private:
	struct region_request;
	struct gather_request;
	// shared_ptr for pointer stability (referenced by receive_arbiter::event)
	using stable_region_request = std::shared_ptr<region_request>;
	using stable_gather_request = std::shared_ptr<gather_request>;

  public:
	explicit receive_arbiter(communicator& comm);
	receive_arbiter(const receive_arbiter&) = delete;
	receive_arbiter(receive_arbiter&&) = default;
	receive_arbiter& operator=(const receive_arbiter&) = delete;
	receive_arbiter& operator=(receive_arbiter&&) = default;
	~receive_arbiter();

	[[nodiscard]] async_event receive(const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size);

	void begin_split_receive(const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size);
	[[nodiscard]] async_event await_split_receive_subregion(const transfer_id& trid, const region<3>& subregion);

	// "gather receives" are a temporary solution until we implement inter-node reductions through MPI collectives.
	[[nodiscard]] async_event gather_receive(const transfer_id& trid, void* allocation, size_t node_chunk_size);

	void poll_communicator();

  private:
	class receive_event final : public async_event_base {
	  public:
		explicit receive_event(const stable_region_request& rr) : m_state(region_transfer_state{rr}) {}
		explicit receive_event(const stable_region_request& rr, const region<3>& awaited_subregion)
		    : m_state(subregion_transfer_state{rr, awaited_subregion}) {}
		explicit receive_event(const stable_gather_request& gr) : m_state(gather_transfer_state{gr}) {}

		bool is_complete() const override;

	  private:
		friend class receive_arbiter;

		struct completed_state {};
		struct region_transfer_state {
			std::weak_ptr<const region_request> request;
		};
		struct subregion_transfer_state {
			std::weak_ptr<const region_request> request;
			region<3> awaited_region;
		};
		struct gather_transfer_state {
			std::weak_ptr<const gather_request> request;
		};
		using state = std::variant<region_transfer_state, subregion_transfer_state, gather_transfer_state>;

		state m_state;
	};

	struct incoming_region_fragment {
		detail::box<3> box;
		async_event communication;
	};

	struct region_request {
		void* allocation;
		box<3> allocated_box;
		region<3> incomplete_region;
		std::vector<incoming_region_fragment> incoming_fragments;

		region_request(region<3> requested_region, void* const allocation, const box<3>& allocated_bounding_box)
		    : allocation(allocation), allocated_box(allocated_bounding_box), incomplete_region(std::move(requested_region)) {}
		bool do_complete();
	};

	struct multi_region_transfer {
		size_t elem_size;
		std::vector<stable_region_request> active_requests;
		std::vector<inbound_pilot> unassigned_pilots;
		explicit multi_region_transfer(const size_t elem_size) : elem_size(elem_size) {}
		explicit multi_region_transfer(const size_t elem_size, std::vector<inbound_pilot>&& unassigned_pilots)
		    : elem_size(elem_size), unassigned_pilots(std::move(unassigned_pilots)) {}
		bool do_complete();
	};

	struct incoming_gather_chunk {
		async_event communication;
	};

	struct gather_request {
		void* allocation;
		size_t chunk_size;
		size_t num_incomplete_chunks;
		std::vector<incoming_gather_chunk> incoming_chunks;

		gather_request(void* const allocation, const size_t chunk_size, const size_t num_total_chunks)
		    : allocation(allocation), chunk_size(chunk_size), num_incomplete_chunks(num_total_chunks) {}
		bool do_complete();
	};

	struct gather_transfer {
		stable_gather_request request;
		bool do_complete();
	};

	struct unassigned_transfer {
		std::vector<inbound_pilot> pilots;
		bool do_complete();
	};

	using transfer = std::variant<unassigned_transfer, multi_region_transfer, gather_transfer>;

	communicator* m_comm;
	size_t m_num_nodes;
	std::unordered_map<transfer_id, transfer> m_transfers;

	stable_region_request& begin_receive_region(
	    const transfer_id& trid, const region<3>& request, void* allocation, const box<3>& allocated_box, size_t elem_size);

	void handle_region_request_pilot(region_request& rr, const inbound_pilot& pilot, size_t elem_size);
	void handle_gather_request_pilot(gather_request& gr, const inbound_pilot& pilot);
};

} // namespace celerity::detail
