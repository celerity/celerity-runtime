#pragma once

#include "ranges.h"
#include "version.h"

#include <cstddef>

#if CELERITY_ENABLE_MPI
#include <mpi.h>
#endif


namespace celerity {
template <int Dims>
class partition;
}

namespace celerity::experimental {
class collective_partition;
}

namespace celerity::detail {

class communicator;

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
#if CELERITY_ENABLE_MPI
	MPI_Comm get_collective_mpi_comm() const; // defined in mpi_communicator.cc
#else
	template <typename E = void>
	void get_collective_mpi_comm() const {
		static_assert(detail::constexpr_false<E> && "MPI support is not enabled (CELERITY_ENABLE_MPI=OFF)");
	}
#endif

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
