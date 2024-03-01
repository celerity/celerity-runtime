#include "mpi_communicator.h"
#include "log.h"
#include "ranges.h"

#include <cstddef>

#include <mpi.h>

namespace celerity::detail::mpi_detail {

class event final : public async_event_base {
  public:
	explicit event(MPI_Request req) : m_req(req) {}

	event(const event&) = delete;
	event(event&&) = delete;
	event& operator=(const event&) = delete;
	event& operator=(event&&) = delete;

	~event() override {
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

constexpr int pilot_exchange_tag = mpi_support::TAG_COMMUNICATOR;
constexpr int first_message_tag = pilot_exchange_tag + 1;

constexpr int message_id_to_tag(const message_id msgid) {
	assert(msgid <= static_cast<message_id>(INT_MAX - first_message_tag));
	return first_message_tag + static_cast<int>(msgid);
}

constexpr int node_id_to_rank(const node_id nid) {
	assert(nid <= static_cast<node_id>(INT_MAX));
	return static_cast<int>(nid);
}

constexpr node_id rank_to_node_id(const int rank) {
	assert(rank >= 0);
	return static_cast<node_id>(rank);
}

} // namespace celerity::detail::mpi_detail

namespace celerity::detail {

mpi_communicator::mpi_communicator(collective_clone_from_tag /* tag */, MPI_Comm mpi_comm) {
	assert(mpi_comm != MPI_COMM_NULL);
#if MPI_VERSION < 3
	MPI_Comm_dup(mpi_comm, &m_mpi_comm);
#else
	MPI_Info info;
	MPI_Info_create(&info);
	MPI_Info_set(info, "mpi_assert_no_any_tag", "true");
	MPI_Info_set(info, "mpi_assert_exact_length", "true");
	MPI_Info_set(info, "mpi_assert_allow_overtaking", "true");
	MPI_Comm_dup_with_info(mpi_comm, info, &m_mpi_comm);
	MPI_Info_free(&info);
#endif
}

mpi_communicator::~mpi_communicator() {
	for(auto& outbound : m_outbound_pilots) {
		MPI_Wait(&outbound.request, MPI_STATUS_IGNORE);
	}
	if(m_inbound_pilot.request != MPI_REQUEST_NULL) {
		MPI_Cancel(&m_inbound_pilot.request);
		MPI_Wait(&m_inbound_pilot.request, MPI_STATUS_IGNORE);
	}

	// MPI_Comm_free is itself a collective, but since this call happens from a destructor we implicitly guarantee that it cant' be re-ordered against any
	// other collective operation on this communicator.
	MPI_Comm_free(&m_mpi_comm);
}

size_t mpi_communicator::get_num_nodes() const {
	int size = -1;
	MPI_Comm_size(m_mpi_comm, &size);
	assert(size > 0);
	return static_cast<size_t>(size);
}

node_id mpi_communicator::get_local_node_id() const {
	int rank = -1;
	MPI_Comm_rank(m_mpi_comm, &rank);
	return mpi_detail::rank_to_node_id(rank);
}

void mpi_communicator::send_outbound_pilot(const outbound_pilot& pilot) {
	CELERITY_DEBUG("[mpi] pilot -> N{} (MSG{}, {}, {})", pilot.to, pilot.message.id, pilot.message.transfer_id, pilot.message.box);

	assert(pilot.to < get_num_nodes());
	assert(pilot.to != get_local_node_id());

	// initiate Isend as early as possible
	in_flight_pilot newly_in_flight;
	newly_in_flight.message = std::make_unique<pilot_message>(pilot.message);
	MPI_Isend(newly_in_flight.message.get(), sizeof *newly_in_flight.message, MPI_BYTE, mpi_detail::node_id_to_rank(pilot.to), mpi_detail::pilot_exchange_tag,
	    m_mpi_comm, &newly_in_flight.request);

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
	if(m_inbound_pilot.request == MPI_REQUEST_NULL) {
		// this is the first call to poll_inbound_pilots
		begin_receiving_pilot();
	}

	std::vector<inbound_pilot> received_pilots; // vector: MPI might have received and buffered multiple inbound pilots, collect them
	for(;;) {
		int flag = -1;
		MPI_Status status;
		MPI_Test(&m_inbound_pilot.request, &flag, &status);
		if(flag == 0) return received_pilots;

		const inbound_pilot pilot{mpi_detail::rank_to_node_id(status.MPI_SOURCE), *m_inbound_pilot.message};
		begin_receiving_pilot(); // immediately initiate the next receive

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
	MPI_Isend(base, 1, get_array_type(stride), mpi_detail::node_id_to_rank(to), mpi_detail::message_id_to_tag(msgid), m_mpi_comm, &req);
	return make_async_event<mpi_detail::event>(req);
}

async_event mpi_communicator::receive_payload(const node_id from, const message_id msgid, void* const base, const stride& stride) {
	CELERITY_DEBUG("[mpi] payload <- N{} (MSG{}) into {} ({}) {}x{}", from, msgid, base, stride.allocation, stride.subrange, stride.element_size);

	assert(from < get_num_nodes());
	assert(from != get_local_node_id());

	MPI_Request req = MPI_REQUEST_NULL;
	// TODO normalize stride and adjust base in order to re-use more datatypes
	MPI_Irecv(base, 1, get_array_type(stride), mpi_detail::node_id_to_rank(from), mpi_detail::message_id_to_tag(msgid), m_mpi_comm, &req);
	return make_async_event<mpi_detail::event>(req);
}

std::unique_ptr<communicator> mpi_communicator::collective_clone() { return std::make_unique<mpi_communicator>(collective_clone_from, m_mpi_comm); }

void mpi_communicator::collective_barrier() { MPI_Barrier(m_mpi_comm); }

void mpi_communicator::begin_receiving_pilot() {
	assert(m_inbound_pilot.request == MPI_REQUEST_NULL);
	if(m_inbound_pilot.message == nullptr) { m_inbound_pilot.message = std::make_unique<pilot_message>(); }
	MPI_Irecv(m_inbound_pilot.message.get(), sizeof *m_inbound_pilot.message, MPI_BYTE, MPI_ANY_SOURCE, mpi_detail::pilot_exchange_tag, m_mpi_comm,
	    &m_inbound_pilot.request);
}

MPI_Datatype mpi_communicator::get_scalar_type(const size_t bytes) {
	if(const auto it = m_scalar_type_cache.find(bytes); it != m_scalar_type_cache.end()) { return it->second.get(); }

	assert(bytes > 0);
	assert(bytes <= static_cast<size_t>(INT_MAX));
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

	// TODO - for any dimensionality, do pointer adjustment to re-use MPI data types
	// TODO - can we get runaway behavior by constructing too many MPI data types?
	// TODO - have an explicit call for creating cached MPI types? These can be invoked whenever we send or receive a pilot

	int size_array[3];
	int subsize_array[3];
	int start_array[3];
	for(int d = 0; d < 3; ++d) {
		// The instruction graph generator should only ever emit transfers which can be described with an "integer" stride
		assert(stride.allocation[d] <= static_cast<size_t>(INT_MAX));
		assert(stride.subrange.range[d] <= static_cast<size_t>(INT_MAX));
		assert(stride.subrange.offset[d] <= static_cast<size_t>(INT_MAX));
		size_array[d] = static_cast<int>(stride.allocation[d]);
		subsize_array[d] = static_cast<int>(stride.subrange.range[d]);
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
