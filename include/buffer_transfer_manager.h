#pragma once

#include <list>
#include <memory>
#include <unordered_map>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#include <mpi.h>
#pragma clang diagnostic pop

#include "buffer_storage.h"
#include "command.h"
#include "mpi_support.h"
#include "types.h"

namespace celerity {
namespace detail {

	class buffer_transfer_manager {
	  public:
		struct transfer_handle {
			bool complete = false;
		};

		std::shared_ptr<const transfer_handle> push(const command_pkg& pkg);
		std::shared_ptr<const transfer_handle> await_push(const command_pkg& pkg);

		/**
		 * @brief Polls for incoming transfers and updates the status of existing ones.
		 */
		void poll();

	  private:
		struct data_header {
			buffer_id bid;
			reduction_id rid; // zero if this does not belong to a reduction
			subrange<3> sr;
			command_id push_cid;
		};

		struct transfer_in {
			node_id source_nid;
			MPI_Request request;
			data_header header;
			raw_buffer_data data;
			mpi_support::single_use_data_type data_type;
		};

		struct incoming_transfer_handle : transfer_handle {
			std::unique_ptr<transfer_in> transfer;
		};

		struct transfer_out {
			std::shared_ptr<transfer_handle> handle;
			MPI_Request request;
			data_header header;
			raw_buffer_data data;
			mpi_support::single_use_data_type data_type;
		};

		std::list<std::unique_ptr<transfer_in>> incoming_transfers;
		std::list<std::unique_ptr<transfer_out>> outgoing_transfers;

		// Here we store two types of handles:
		//  - Incoming pushes that have not yet been requested through ::await_push
		//  - Still outstanding pushes that have been requested through ::await_push
		std::unordered_map<command_id, std::shared_ptr<incoming_transfer_handle>> push_blackboard;

		void poll_incoming_transfers();
		void update_incoming_transfers();
		void update_outgoing_transfers();

		static void commit_transfer(transfer_in& transfer);
	};

} // namespace detail
} // namespace celerity
