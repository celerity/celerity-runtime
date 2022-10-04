#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#include <mpi.h>
#pragma clang diagnostic pop

#include "buffer_storage.h"
#include "command.h"
#include "frame.h"
#include "types.h"

namespace celerity {
namespace detail {

	class buffer_transfer_manager {
	  public:
		struct transfer_handle {
			bool complete = false;
		};

		buffer_transfer_manager();

		std::shared_ptr<const transfer_handle> push(const command_pkg& pkg);
		std::shared_ptr<const transfer_handle> await_push(const command_pkg& pkg);

		/**
		 * @brief Polls for incoming transfers and updates the status of existing ones.
		 */
		void poll();

	  private:
		struct data_frame {
			using payload_type = std::byte;

			// variable-sized structure
			data_frame() = default;
			data_frame(const data_frame&) = delete;
			data_frame& operator=(const data_frame&) = delete;

			buffer_id bid;
			reduction_id rid; // zero if this does not belong to a reduction
			subrange<3> sr;
			transaction_id trid;
			alignas(std::max_align_t) payload_type data[]; // max_align to allow reinterpret_casting a pointer to this member to any buffer element pointer
		};

		// unique_frame_ptr assumes that the flexible payload member begins at exactly sizeof(Frame) bytes
		static_assert(offsetof(data_frame, data) == sizeof(data_frame));

		struct transfer_in {
			node_id source_nid;
			MPI_Request request;
			unique_frame_ptr<data_frame> frame;
		};

		struct incoming_transfer_handle : transfer_handle {
			std::unique_ptr<transfer_in> transfer;
		};

		struct transfer_out {
			std::shared_ptr<transfer_handle> handle;
			MPI_Request request;
			unique_frame_ptr<data_frame> frame;
		};

		std::list<std::unique_ptr<transfer_in>> m_incoming_transfers;
		std::list<std::unique_ptr<transfer_out>> m_outgoing_transfers;

		// Here we store two types of handles:
		//  - Incoming pushes that have not yet been requested through ::await_push
		//  - Still outstanding pushes that have been requested through ::await_push
		std::unordered_map<std::pair<node_id, transaction_id>, std::shared_ptr<incoming_transfer_handle>, utils::pair_hash> m_push_blackboard;

		mpi_support::data_type m_send_recv_unit;

		void poll_incoming_transfers();
		void update_incoming_transfers();
		void update_outgoing_transfers();

		static void commit_transfer(transfer_in& transfer);
	};

} // namespace detail
} // namespace celerity
