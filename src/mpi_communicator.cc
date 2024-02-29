#include "mpi_communicator.h"
#include "instruction_graph.h"
#include "ranges.h"

#include <cstddef>

#include <mpi.h>

namespace celerity::detail {

class mpi_event final : public async_event_base {
  public:
	mpi_event(MPI_Request req) : m_req(req) {}
	mpi_event(const async_event&) = delete;
	mpi_event(async_event&&) = delete;
	mpi_event& operator=(const async_event&) = delete;
	mpi_event& operator=(async_event&&) = delete;
	~mpi_event() override {
		// MPI_Request_free is always incorrect for our use case: events originate from an Isend or Irecv, which must ensure that the user-provided buffer
		// remains until the operation has completed.
		MPI_Wait(&m_req, MPI_STATUS_IGNORE);
	}

	bool is_complete() const override {
		int flag = -1;
		MPI_Test(&m_req, &flag, MPI_STATUS_IGNORE);
		return flag != 0;
	}

  private:
	mutable MPI_Request m_req;
};

mpi_communicator::collective_group* mpi_communicator::collective_group::clone() {
	MPI_Comm new_comm = MPI_COMM_NULL;
	MPI_Comm_dup(m_comm, &new_comm);
	const auto group = new collective_group(m_owner, new_comm); // NOLINT(cppcoreguidelines-owning-memory)
	m_owner->m_collective_groups.emplace_back(group);
	return group;
}

void mpi_communicator::collective_group::barrier() { MPI_Barrier(m_comm); }

mpi_communicator::mpi_communicator(const MPI_Comm comm) : m_root_comm(comm), m_collective_groups{new collective_group(this, comm)} { begin_receive_pilot(); }

mpi_communicator::~mpi_communicator() {
	for(auto& outbound : m_outbound_pilots) {
		MPI_Wait(&outbound.request, MPI_STATUS_IGNORE);
	}
	MPI_Cancel(&m_inbound_pilot.request);
	MPI_Wait(&m_inbound_pilot.request, MPI_STATUS_IGNORE);

	for(size_t i = 1 /* all except groups[0] (root) */; i < m_collective_groups.size(); ++i) {
		MPI_Comm_free(&m_collective_groups[m_collective_groups.size() - i]->m_comm);
	}
	for(const auto group : m_collective_groups) {
		delete group; // NOLINT(cppcoreguidelines-owning-memory)
	}
}

size_t mpi_communicator::get_num_nodes() const {
	int size = -1;
	MPI_Comm_size(m_root_comm, &size);
	return static_cast<size_t>(size);
}

node_id mpi_communicator::get_local_node_id() const {
	int rank = -1;
	MPI_Comm_rank(m_root_comm, &rank);
	return static_cast<node_id>(rank);
}

void mpi_communicator::send_outbound_pilot(const outbound_pilot& pilot) {
	CELERITY_DEBUG("[mpi] pilot -> N{} (MSG{}, {}, {})", pilot.to, pilot.message.id, pilot.message.transfer_id, pilot.message.box);

	assert(pilot.to < get_num_nodes());
	assert(pilot.to != get_local_node_id());

	// initiate Isend as early as possible
	in_flight_pilot newly_in_flight;
	*newly_in_flight.message = pilot.message;
	MPI_Isend(newly_in_flight.message.get(), sizeof *newly_in_flight.message, MPI_BYTE, static_cast<int>(pilot.to), pilot_exchange_tag, m_root_comm,
	    &newly_in_flight.request);

	// collect finished sends (TODO rate-limit this to avoid quadratic behavior)
	for(auto& already_in_flight : m_outbound_pilots) {
		int flag = -1;
		MPI_Test(&already_in_flight.request, &flag, MPI_STATUS_IGNORE);
	}
	const auto last_incomplete_outbound_pilot = std::remove_if(m_outbound_pilots.begin(), m_outbound_pilots.end(),
	    [](const in_flight_pilot& already_in_flight) { return already_in_flight.request == MPI_REQUEST_NULL; });
	m_outbound_pilots.erase(last_incomplete_outbound_pilot, m_outbound_pilots.end());

	// keep allocation until Isend has completed
	m_outbound_pilots.push_back(std::move(newly_in_flight));
}

std::vector<inbound_pilot> mpi_communicator::poll_inbound_pilots() {
	std::vector<inbound_pilot> received_pilots; // vector: MPI might have received and buffered multiple inbound pilots, collect them
	for(;;) {
		int flag = -1;
		MPI_Status status;
		MPI_Test(&m_inbound_pilot.request, &flag, &status);
		if(flag == 0) return received_pilots;

		const inbound_pilot pilot{static_cast<node_id>(status.MPI_SOURCE), *m_inbound_pilot.message};
		begin_receive_pilot(); // initiate next receive asap

		CELERITY_DEBUG("[mpi] pilot <- N{} (MSG{}, {} {})", pilot.from, pilot.message.id, pilot.message.transfer_id, pilot.message.box);
		received_pilots.push_back(pilot);
	}
}

async_event mpi_communicator::send_payload(const node_id to, const message_id msgid, const void* const base, const stride& stride) {
	CELERITY_DEBUG("[mpi] payload -> N{} (MSG{}) from {} ({}) {}x{}", to, msgid, base, stride.allocation, stride.subrange, stride.element_size);

	assert(to < get_num_nodes());
	assert(to != get_local_node_id());

	MPI_Request req = MPI_REQUEST_NULL;
	// TODO normalize stride and adjust base in order to re-use more datatypes
	MPI_Isend(base, 1, get_array_type(stride), static_cast<int>(to), first_message_tag + static_cast<int>(msgid), m_root_comm, &req);
	return make_async_event<mpi_event>(req);
}

async_event mpi_communicator::receive_payload(const node_id from, const message_id msgid, void* const base, const stride& stride) {
	CELERITY_DEBUG("[mpi] payload <- N{} (MSG{}) into {} ({}) {}x{}", from, msgid, base, stride.allocation, stride.subrange, stride.element_size);

	assert(from < get_num_nodes());
	assert(from != get_local_node_id());

	MPI_Request req = MPI_REQUEST_NULL;
	// TODO normalize stride and adjust base in order to re-use more datatypes
	MPI_Irecv(base, 1, get_array_type(stride), static_cast<int>(from), first_message_tag + static_cast<int>(msgid), m_root_comm, &req);
	return make_async_event<mpi_event>(req);
}

mpi_communicator::collective_group* mpi_communicator::get_collective_root() { return m_collective_groups.front(); }

void mpi_communicator::begin_receive_pilot() {
	assert(m_inbound_pilot.request == MPI_REQUEST_NULL);
	MPI_Irecv(
	    m_inbound_pilot.message.get(), sizeof *m_inbound_pilot.message, MPI_BYTE, MPI_ANY_SOURCE, pilot_exchange_tag, m_root_comm, &m_inbound_pilot.request);
}

MPI_Datatype mpi_communicator::get_scalar_type(const size_t bytes) {
	if(const auto it = m_scalar_type_cache.find(bytes); it != m_scalar_type_cache.end()) { return it->second.get(); }

	assert(bytes <= INT_MAX);
	MPI_Datatype type = MPI_DATATYPE_NULL;
	MPI_Type_contiguous(static_cast<int>(bytes), MPI_BYTE, &type);
	MPI_Type_commit(&type);
	m_scalar_type_cache.emplace(bytes, unique_datatype(type));
	return type;
}

MPI_Datatype mpi_communicator::get_array_type(const stride& stride) {
	if(const auto it = m_array_type_cache.find(stride); it != m_array_type_cache.end()) { return it->second.get(); }

	const int dims = detail::get_effective_dims(stride.allocation);
	assert(detail::get_effective_dims(stride.subrange) <= dims);

	if(dims == 0) { return get_scalar_type(stride.element_size); }

	// TODO - for 1D, use pointer adjustment and MPI_Type_contiguous to allow pointing inside allocations with > INT_MAX extents
	// TODO - for any dimensionality, do pointer adjustment to re-use MPI data types
	// TODO - can we get runaway behavior by constructing too many MPI data types?
	// TODO - have an explicit call for creating cached MPI types? These can be invoked whenever we send or receive a pilot

	int size_array[3];
	int subsize_array[3];
	int start_array[3];
	for(int d = 0; d < 3; ++d) {
		// TODO support transfers > 2Gi elements, at least in the 1d case - either through typing magic here, or by splitting sends / recvs in the iggen
		assert(stride.allocation[d] <= INT_MAX);
		size_array[d] = static_cast<int>(stride.allocation[d]);
		assert(stride.subrange.range[d] <= INT_MAX);
		subsize_array[d] = static_cast<int>(stride.subrange.range[d]);
		assert(stride.subrange.offset[d] <= INT_MAX);
		start_array[d] = static_cast<int>(stride.subrange.offset[d]);
	}

	MPI_Datatype type = MPI_DATATYPE_NULL;
	MPI_Type_create_subarray(dims, size_array, subsize_array, start_array, MPI_ORDER_C, get_scalar_type(stride.element_size), &type);
	MPI_Type_commit(&type);
	m_array_type_cache.emplace(stride, unique_datatype(type));
	return type;
}

void mpi_communicator::datatype_deleter::operator()(MPI_Datatype dtype) const { //
	MPI_Type_free(&dtype);
}

} // namespace celerity::detail
