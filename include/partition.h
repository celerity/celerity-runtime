#pragma once

#include "mpi_communicator.h" // TODO only used for type cast - move that function to .cc file
#include "ranges.h"
#include "utils.h"

#include <mpi.h>

namespace celerity {
template <int Dims>
class partition;
}

namespace celerity::experimental {
class collective_partition;
}

namespace celerity::detail {

template <int Dims>
partition<Dims> make_partition(const range<Dims>& global_size, const subrange<Dims>& range) {
	return partition<Dims>(global_size, range);
}

experimental::collective_partition make_collective_partition(const range<1>& global_size, const subrange<1>& range, const communicator& collective_comm);

} // namespace celerity::detail

namespace celerity {

/**
 * Represents the sub-range of the iteration space handled by each host in a host_task.
 */
template <int Dims>
class partition {
  public:
	/** The subrange handled by this host. */
	const subrange<Dims>& get_subrange() const { return m_range; }

	/** The size of the entire iteration space */
	const range<Dims>& get_global_size() const { return m_global_size; }

  private:
	range<Dims> m_global_size;
	subrange<Dims> m_range;

  protected:
	friend partition<Dims> detail::make_partition<Dims>(const range<Dims>& global_size, const subrange<Dims>& range);

	explicit partition(const range<Dims>& global_size, const subrange<Dims>& range) : m_global_size(global_size), m_range(range) {}
};

} // namespace celerity

namespace celerity::experimental {

/**
 * A one-dimensional partition, additionally carrying the MPI communicator of the collective group.
 */
class collective_partition : public partition<1> {
  public:
	MPI_Comm get_collective_mpi_comm() const { return detail::utils::as<detail::mpi_communicator>(m_collective_comm)->get_native(); }

	size_t get_node_index() const { return get_subrange().offset[0]; }

	size_t get_num_nodes() const { return get_global_size()[0]; }

  protected:
	friend collective_partition detail::make_collective_partition(
	    const range<1>& global_size, const subrange<1>& range, const detail::communicator& collective_comm);

	const detail::communicator* m_collective_comm;

	collective_partition(const range<1>& global_size, const subrange<1>& range, const detail::communicator& collective_comm)
	    : partition<1>(global_size, range), m_collective_comm(&collective_comm) {}
};

} // namespace celerity::experimental

namespace celerity::detail {

inline experimental::collective_partition make_collective_partition(
    const range<1>& global_size, const subrange<1>& range, const detail::communicator& collective_comm) {
	return experimental::collective_partition(global_size, range, collective_comm);
}

} // namespace celerity::detail
