#include "mpi_communicator.h"
#include "log.h"
#include "mpi_support.h"
#include "ranges.h"

#include <climits>
#include <cstddef>

#include <mpi.h>

namespace celerity::detail::mpi_detail {

/// async_event wrapper around an MPI_Request.
class mpi_event final : public async_event_impl {
  public:
	explicit mpi_event(MPI_Request req) : m_req(req) {}

	mpi_event(const mpi_event&) = delete;
	mpi_event(mpi_event&&) = delete;
	mpi_event& operator=(const mpi_event&) = delete;
	mpi_event& operator=(mpi_event&&) = delete;

	~mpi_event() override {
		// MPI_Request_free is always incorrect for our use case: events originate from an Isend or Irecv, which must ensure that the user-provided buffer
		// remains live until the operation has completed.
		MPI_Wait(&m_req, MPI_STATUS_IGNORE);
	}

	bool is_complete() override {
		int flag = -1;
		MPI_Test(&m_req, &flag, MPI_STATUS_IGNORE);
		return flag != 0;
	}

  private:
	MPI_Request m_req;
};

constexpr int pilot_exchange_tag = mpi_support::TAG_COMMUNICATOR;
constexpr int first_message_tag = pilot_exchange_tag + 1;

constexpr int message_id_to_mpi_tag(message_id msgid) {
	// If the resulting tag would overflow INT_MAX in a long-running program with many nodes, we wrap around to `first_message_tag` instead, assuming that
	// there will never be a way to cause temporal ambiguity between transfers that are 2^31 message ids apart.
	msgid %= static_cast<message_id>(INT_MAX - first_message_tag);
	return first_message_tag + static_cast<int>(msgid);
}

constexpr int node_id_to_mpi_rank(const node_id nid) {
	assert(nid <= static_cast<node_id>(INT_MAX));
	return static_cast<int>(nid);
}

constexpr node_id mpi_rank_to_node_id(const int rank) {
	assert(rank >= 0);
	return static_cast<node_id>(rank);
}

/// Strides that only differ e.g. in their dim0 allocation size are equivalent when adjusting the base pointer. This not only improves mpi_communicator type
/// cache efficiency, but is in fact necessary to make sure all boxes that instruction_graph_generator emits for send instructions and inbound pilots
/// are representable in the 32-bit integer world of MPI.
/// @tparam Void Either `void` or `const void`.
template <typename Void>
constexpr std::tuple<Void*, communicator::stride> normalize_strided_pointer(Void* ptr, communicator::stride stride) {
	using byte_pointer_t = std::conditional_t<std::is_const_v<Void>, const std::byte*, std::byte*>;

	// drop leading buffer dimensions with extent 1, which allows us to do pointer adjustment in d1 / d2
	while(stride.allocation_range[0] == 1 && stride.allocation_range[1] * stride.allocation_range[2] > 1) {
		stride.allocation_range[0] = stride.allocation_range[1], stride.allocation_range[1] = stride.allocation_range[2], stride.allocation_range[2] = 1;
		stride.transfer.range[0] = stride.transfer.range[1], stride.transfer.range[1] = stride.transfer.range[2], stride.transfer.range[2] = 1;
		stride.transfer.offset[0] = stride.transfer.offset[1], stride.transfer.offset[1] = stride.transfer.offset[2], stride.transfer.offset[2] = 0;
	}

	// adjust base pointer to remove the offset
	const auto offset_elements = stride.transfer.offset[0] * stride.allocation_range[1] * stride.allocation_range[2];
	ptr = static_cast<byte_pointer_t>(ptr) + offset_elements * stride.element_size;
	stride.transfer.offset[0] = 0;

	// clamp allocation size to subrange (MPI will not access memory beyond subrange.range anyway)
	stride.allocation_range[0] = stride.transfer.range[0];

	// TODO we can normalize further if we accept arbitrarily large scalar types (via MPI contiguous / struct types):
	// 	 - collapse fast dimensions if contiguous via `stride.element_size *= stride.subrange.range[d]`
	//   - factorize stride coordinates: `element_size *= gcd(allocation[0], offset[0], range[0], allocation[1], ...)`
	// Doing all this will complicate instruction_graph_generator_detail::split_into_communicator_compatible_boxes though.
	return {ptr, stride};
}

} // namespace celerity::detail::mpi_detail

namespace celerity::detail {

mpi_communicator::mpi_communicator(const collective_clone_from_tag /* tag */, const MPI_Comm mpi_comm) : m_mpi_comm(MPI_COMM_NULL) {
	assert(mpi_comm != MPI_COMM_NULL);
#if MPI_VERSION < 3
	// MPI 2 only has Comm_dup - we assume that the user has not done any obscure things to MPI_COMM_WORLD
	MPI_Comm_dup(mpi_comm, &m_mpi_comm);
#else
	// MPI >= 3.0 provides MPI_Comm_dup_with_info, which allows us to reset all implementation hints on the communicator to our liking
	MPI_Info info;
	MPI_Info_create(&info);
	// See the OpenMPI manpage for MPI_Comm_set_info for keys and values
	MPI_Info_set(info, "mpi_assert_no_any_tag", "true");       // promise never to use MPI_ANY_TAG (we _do_ use MPI_ANY_SOURCE for pilots)
	MPI_Info_set(info, "mpi_assert_exact_length", "true");     // promise to exactly match sizes between corresponding MPI_Send and MPI_Recv calls
	MPI_Info_set(info, "mpi_assert_allow_overtaking", "true"); // we do not care about message ordering since we disambiguate by tag
	MPI_Comm_dup_with_info(mpi_comm, info, &m_mpi_comm);
	MPI_Info_free(&info);
#endif
}

mpi_communicator::~mpi_communicator() {
	// All asynchronous sends / receives must have completed at this point - unfortunately we have no easy way of checking this here.

	// Await the completion of all outbound pilot sends. The blocking-wait should usually be unnecessary because completion of payload-sends should imply
	// completion of the outbound-pilot sends, although there is no real guarantee of this given MPI's freedom to buffer transfers however it likes.
	// MPI_Wait will also free the async request, so we use this function unconditionally.
	for(auto& outbound : m_outbound_pilots) {
		MPI_Wait(&outbound.request, MPI_STATUS_IGNORE);
	}

	// We always re-start the pilot Irecv immediately, so we need to MPI_Cancel the last such request (and then free it using MPI_Wait).
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
	return mpi_detail::mpi_rank_to_node_id(rank);
}

void mpi_communicator::send_outbound_pilot(const outbound_pilot& pilot) {
	CELERITY_DEBUG("[mpi] pilot -> N{} (MSG{}, {}, {})", pilot.to, pilot.message.id, pilot.message.transfer_id, pilot.message.box);

	assert(pilot.to < get_num_nodes());
	assert(pilot.to != get_local_node_id());

	// Initiate Isend as early as possible to hide latency.
	in_flight_pilot newly_in_flight;
	newly_in_flight.message = std::make_unique<pilot_message>(pilot.message);
	MPI_Isend(newly_in_flight.message.get(), sizeof *newly_in_flight.message, MPI_BYTE, mpi_detail::node_id_to_mpi_rank(pilot.to),
	    mpi_detail::pilot_exchange_tag, m_mpi_comm, &newly_in_flight.request);

	// Collect finished sends (TODO consider rate-limiting this to avoid quadratic behavior)
	constexpr auto pilot_send_finished = [](in_flight_pilot& already_in_flight) {
		int flag = -1;
		MPI_Test(&already_in_flight.request, &flag, MPI_STATUS_IGNORE);
		return already_in_flight.request == MPI_REQUEST_NULL;
	};
	m_outbound_pilots.erase(std::remove_if(m_outbound_pilots.begin(), m_outbound_pilots.end(), pilot_send_finished), m_outbound_pilots.end());

	// Keep allocation until Isend has completed
	m_outbound_pilots.push_back(std::move(newly_in_flight));
}

std::vector<inbound_pilot> mpi_communicator::poll_inbound_pilots() {
	// Irecv needs to be called initially, and after receiving each pilot to enqueue the next operation.
	const auto begin_receiving_next_pilot = [this] {
		assert(m_inbound_pilot.message != nullptr);
		assert(m_inbound_pilot.request == MPI_REQUEST_NULL);
		MPI_Irecv(m_inbound_pilot.message.get(), sizeof *m_inbound_pilot.message, MPI_BYTE, MPI_ANY_SOURCE, mpi_detail::pilot_exchange_tag, m_mpi_comm,
		    &m_inbound_pilot.request);
	};

	if(m_inbound_pilot.request == MPI_REQUEST_NULL) {
		// This is the first call to poll_inbound_pilots, spin up the pilot-receiving machinery - we don't do this unconditionally in the constructor
		// because communicators for collective groups do not deal with pilots
		m_inbound_pilot.message = std::make_unique<pilot_message>();
		begin_receiving_next_pilot();
	}

	// MPI might have received and buffered multiple inbound pilots, collect all of them in a loop
	std::vector<inbound_pilot> received_pilots;
	for(;;) {
		int flag = -1;
		MPI_Status status;
		MPI_Test(&m_inbound_pilot.request, &flag, &status);
		if(flag == 0 /* incomplete */) {
			return received_pilots; // no more pilots in queue, we're done collecting
		}

		const inbound_pilot pilot{mpi_detail::mpi_rank_to_node_id(status.MPI_SOURCE), *m_inbound_pilot.message};
		begin_receiving_next_pilot(); // initiate the next receive asap

		CELERITY_DEBUG("[mpi] pilot <- N{} (MSG{}, {} {})", pilot.from, pilot.message.id, pilot.message.transfer_id, pilot.message.box);
		received_pilots.push_back(pilot);
	}
}

async_event mpi_communicator::send_payload(const node_id to, const message_id msgid, const void* const base, const stride& stride) {
	CELERITY_DEBUG("[mpi] payload -> N{} (MSG{}) from {} ({}) {}x{}", to, msgid, base, stride.allocation_range, stride.transfer, stride.element_size);

	assert(to < get_num_nodes());
	assert(to != get_local_node_id());

	MPI_Request req = MPI_REQUEST_NULL;
	const auto [adjusted_base, normalized_stride] = mpi_detail::normalize_strided_pointer(base, stride);
	MPI_Isend(
	    adjusted_base, 1, get_array_type(normalized_stride), mpi_detail::node_id_to_mpi_rank(to), mpi_detail::message_id_to_mpi_tag(msgid), m_mpi_comm, &req);
	return make_async_event<mpi_detail::mpi_event>(req);
}

async_event mpi_communicator::receive_payload(const node_id from, const message_id msgid, void* const base, const stride& stride) {
	CELERITY_DEBUG("[mpi] payload <- N{} (MSG{}) into {} ({}) {}x{}", from, msgid, base, stride.allocation_range, stride.transfer, stride.element_size);

	assert(from < get_num_nodes());
	assert(from != get_local_node_id());

	MPI_Request req = MPI_REQUEST_NULL;
	const auto [adjusted_base, normalized_stride] = mpi_detail::normalize_strided_pointer(base, stride);
	MPI_Irecv(
	    adjusted_base, 1, get_array_type(normalized_stride), mpi_detail::node_id_to_mpi_rank(from), mpi_detail::message_id_to_mpi_tag(msgid), m_mpi_comm, &req);
	return make_async_event<mpi_detail::mpi_event>(req);
}

std::unique_ptr<communicator> mpi_communicator::collective_clone() { return std::make_unique<mpi_communicator>(collective_clone_from, m_mpi_comm); }

void mpi_communicator::collective_barrier() { MPI_Barrier(m_mpi_comm); }

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

	const int dims = detail::get_effective_dims(stride.allocation_range);
	assert(detail::get_effective_dims(stride.transfer) <= dims);

	// MPI (understandably) does not recognize a 0-dimensional subarray as a scalar
	if(dims == 0) { return get_scalar_type(stride.element_size); }

	// TODO - can we get runaway behavior from constructing too many MPI data types, especially with Spectrum MPI?
	// TODO - eagerly create MPI types ahead-of-time whenever we send or receive a pilot to reduce latency?

	int size_array[3];
	int subsize_array[3];
	int start_array[3];
	for(int d = 0; d < 3; ++d) {
		// The instruction graph generator must only ever emit transfers which can be described with a signed-int stride
		assert(stride.allocation_range[d] <= static_cast<size_t>(INT_MAX));
		assert(stride.transfer.range[d] <= static_cast<size_t>(INT_MAX));
		assert(stride.transfer.offset[d] <= static_cast<size_t>(INT_MAX));
		size_array[d] = static_cast<int>(stride.allocation_range[d]);
		subsize_array[d] = static_cast<int>(stride.transfer.range[d]);
		start_array[d] = static_cast<int>(stride.transfer.offset[d]);
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
