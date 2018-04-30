#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <mpi.h>

#include "buffer_storage.h"
#include "command.h"
#include "logger.h"
#include "types.h"

namespace celerity {

constexpr int CELERITY_MPI_TAG_CMD = 0;
constexpr int CELERITY_MPI_TAG_DATA_REQUEST = 1;
constexpr int CELERITY_MPI_TAG_DATA_TRANSFER = 2;

class buffer_transfer_manager {
  public:
	struct transfer_handle {
		bool complete = false;
	};

	buffer_transfer_manager(std::shared_ptr<logger> transfer_logger) : transfer_logger(transfer_logger) {}

	/**
	 * Checks for (and handles) incoming data requests and transfers.
	 */
	void poll();

	std::shared_ptr<const transfer_handle> await_pull(const command_pkg& pkg);

	std::shared_ptr<const transfer_handle> pull(const command_pkg& pkg);

	std::shared_ptr<const transfer_handle> send(node_id to, const command_pkg& pkg);

  private:
	struct data_header {
		buffer_id bid;
		task_id tid;
		command_subrange subrange;
	};

	struct transfer_in {
		MPI_Request request;
		data_header header;
		std::vector<char> data;

		// This data type is constructed for every individual transfer
		MPI_Datatype data_type = 0;
		~transfer_in() {
			if(data_type != 0) { MPI_Type_free(&data_type); }
		}
	};

	struct transfer_out {
		std::shared_ptr<transfer_handle> handle;
		MPI_Request request;
		data_header header;

		transfer_out(detail::raw_data_read_handle data_handle) : data_handle(data_handle) {}
		void* get_raw_ptr() const { return data_handle.base_ptr; }

		// This data type is constructed for every individual transfer
		MPI_Datatype data_type = 0;
		~transfer_out() {
			if(data_type != 0) { MPI_Type_free(&data_type); }
		}

	  private:
		detail::raw_data_read_handle data_handle;
	};

	std::unordered_map<task_id, std::unordered_map<buffer_id, std::vector<std::pair<command_pkg, std::shared_ptr<transfer_handle>>>>> active_handles;

	std::unordered_set<std::unique_ptr<transfer_in>> incoming_transfers;
	std::unordered_set<std::unique_ptr<transfer_out>> outgoing_transfers;

	std::shared_ptr<logger> transfer_logger;

	void poll_requests();
	void poll_transfers();
	void update_transfers();
};


} // namespace celerity
