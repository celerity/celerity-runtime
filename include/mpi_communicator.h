#pragma once

#include "async_event.h"
#include "communicator.h"
#include "pilot.h"
#include "types.h"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <mpi.h>


namespace celerity::detail {

/// Constructor tag for mpi_communicator
struct collective_clone_from_tag {
} inline constexpr collective_clone_from{};

/// MPI implementation of the `communicator` interface.
///
/// Wraps an `MPI_Comm`, manages strided MPI datatypes for sends / receives and optionally maintains an inbound / outbound queue of pilot messages.
class mpi_communicator final : public communicator {
  public:
	/// Creates a new `mpi_communicator` by cloning the given `MPI_Comm`, which must not be `MPI_COMM_NULL`.
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

	/// Returns the underlying `MPI_Comm`. The result is never `MPI_COMM_NULL`.
	MPI_Comm get_native() const { return m_mpi_comm; }

  private:
	friend struct mpi_communicator_testspy;

	struct datatype_deleter {
		void operator()(MPI_Datatype dtype) const;
	};
	using unique_datatype = std::unique_ptr<std::remove_pointer_t<MPI_Datatype>, datatype_deleter>;

	/// Keeps a stable pointer to a `pilot_message` alive during an asynchronous pilot send / receive operation.
	struct in_flight_pilot {
		std::unique_ptr<pilot_message> message;
		MPI_Request request = MPI_REQUEST_NULL;
	};

	MPI_Comm m_mpi_comm = MPI_COMM_NULL;

	in_flight_pilot m_inbound_pilot; ///< continually Irecv'd into after the first call to poll_inbound_pilots()
	std::vector<in_flight_pilot> m_outbound_pilots;

	std::unordered_map<size_t, unique_datatype> m_scalar_type_cache;
	std::unordered_map<stride, unique_datatype> m_array_type_cache;

	MPI_Datatype get_scalar_type(size_t bytes);
	MPI_Datatype get_array_type(const stride& stride);
};

} // namespace celerity::detail
