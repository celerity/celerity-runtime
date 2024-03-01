#pragma once

#include "communicator.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <mpi.h>

namespace celerity::detail {

struct collective_clone_from_tag {
} inline constexpr collective_clone_from{};

class mpi_communicator final : public communicator {
  public:
	explicit mpi_communicator(collective_clone_from_tag tag, MPI_Comm mpi_comm);

	mpi_communicator(const mpi_communicator&) = delete;
	mpi_communicator(mpi_communicator&&) = delete;
	mpi_communicator& operator=(const mpi_communicator&) = delete;
	mpi_communicator& operator=(mpi_communicator&&) = delete;
	~mpi_communicator() override;

	size_t get_num_nodes() const override;
	node_id get_local_node_id() const override;

	void send_outbound_pilot(const outbound_pilot& pilot) override;
	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override;

	[[nodiscard]] async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) override;
	[[nodiscard]] async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) override;

	[[nodiscard]] std::unique_ptr<communicator> collective_clone() override;
	void collective_barrier() override;

	MPI_Comm get_native() const { return m_mpi_comm; }

  private:
	struct datatype_deleter {
		void operator()(MPI_Datatype dtype) const;
	};
	using unique_datatype = std::unique_ptr<std::remove_pointer_t<MPI_Datatype>, datatype_deleter>;

	struct in_flight_pilot {
		std::unique_ptr<pilot_message> message; // std::unique_ptr: pointer must be stable after being passed to MPI_Isend/Irecv
		MPI_Request request = MPI_REQUEST_NULL;
	};

	MPI_Comm m_mpi_comm = MPI_COMM_NULL;

	in_flight_pilot m_inbound_pilot;
	std::vector<in_flight_pilot> m_outbound_pilots;

	std::unordered_map<size_t, unique_datatype> m_scalar_type_cache;
	std::unordered_map<stride, unique_datatype> m_array_type_cache;

	void begin_receiving_pilot();

	MPI_Datatype get_scalar_type(size_t bytes);
	MPI_Datatype get_array_type(const stride& stride);
};

} // namespace celerity::detail
