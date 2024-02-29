#pragma once

#include "communicator.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include <mpi.h>

namespace celerity::detail {

class mpi_communicator final : public communicator {
  public:
	// TODO reconsider the collective_group inner class - maybe the communicator itself should be cloned instead, that way we can rely on MPI for thread safety.
	// Divergence block chain also needs a communicator in a thread foreign to the executor, and sharing the instance (which we need to do at least for the
	// MPI_comm_dup call?) would require us to add locks to make the class thread safe.
	class collective_group final : public communicator::collective_group {
	  public:
		collective_group(const collective_group&) = delete;
		collective_group(collective_group&&) = delete;
		collective_group& operator=(const collective_group&) = delete;
		collective_group& operator=(collective_group&&) = delete;

		collective_group* clone() override;
		void barrier() override;

		MPI_Comm get_mpi_comm() const { return m_comm; }

	  private:
		friend class mpi_communicator;

		mpi_communicator* m_owner;
		MPI_Comm m_comm;

		collective_group(mpi_communicator* owner, const MPI_Comm comm) : m_owner(owner), m_comm(comm) {}
		~collective_group() = default;
	};

	explicit mpi_communicator(MPI_Comm comm);

	mpi_communicator(mpi_communicator&&) = default;
	mpi_communicator& operator=(mpi_communicator&&) = default;
	mpi_communicator(const mpi_communicator&) = delete;
	mpi_communicator& operator=(const mpi_communicator&) = delete;
	~mpi_communicator() override;

	size_t get_num_nodes() const override;
	node_id get_local_node_id() const override;
	void send_outbound_pilot(const outbound_pilot& pilot) override;
	[[nodiscard]] std::vector<inbound_pilot> poll_inbound_pilots() override;
	[[nodiscard]] async_event send_payload(node_id to, message_id msgid, const void* base, const stride& stride) override;
	[[nodiscard]] async_event receive_payload(node_id from, message_id msgid, void* base, const stride& stride) override;

	collective_group* get_collective_root() override;

  private:
	struct datatype_deleter {
		void operator()(MPI_Datatype dtype) const;
	};
	using unique_datatype = std::unique_ptr<std::remove_pointer_t<MPI_Datatype>, datatype_deleter>;

	struct in_flight_pilot {
		// std::unique_ptr: pointer must be stable
		std::unique_ptr<pilot_message> message = std::make_unique<pilot_message>();
		MPI_Request request = MPI_REQUEST_NULL;
	};

	inline constexpr static int pilot_exchange_tag = 0;
	inline constexpr static int first_message_tag = 10; // TODO mpi_support.h defines its own tag type for graph printing, unify these

	MPI_Comm m_root_comm;

	std::vector<collective_group*> m_collective_groups;

	in_flight_pilot m_inbound_pilot; // TODO do we want to have multiple of these buffers around to increase throughput?
	std::vector<in_flight_pilot> m_outbound_pilots;

	std::unordered_map<size_t, unique_datatype> m_scalar_type_cache;
	std::unordered_map<stride, unique_datatype> m_array_type_cache;

	void begin_receive_pilot();
	MPI_Datatype get_scalar_type(size_t bytes);
	MPI_Datatype get_array_type(const stride& stride);
};

} // namespace celerity::detail
