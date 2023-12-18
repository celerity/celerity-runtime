#pragma once

#include <memory>

#include <mpi.h>

#include "communicator.h"

namespace celerity::detail {
class mpi_communicator : public communicator {
  public:
	mpi_communicator(MPI_Comm comm) : m_comm(comm) {}

  private:
	MPI_Comm m_comm;

	void allgather_inplace_impl(std::byte* sendrecvbuf, const int sendrecvcount) override {
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sendrecvbuf, sendrecvcount, MPI_BYTE, m_comm);
	};

	void allgather_impl(const std::byte* sendbuf, const int sendcount, std::byte* recvbuf, const int recvcount) override {
		MPI_Allgather(sendbuf, sendcount, MPI_BYTE, recvbuf, recvcount, MPI_BYTE, m_comm);
	};

	void barrier_impl() override { MPI_Barrier(m_comm); }

	size_t num_nodes_impl() override {
		int size = -1;
		MPI_Comm_size(m_comm, &size);
		return static_cast<size_t>(size);
	}

	node_id local_nid_impl() override {
		int rank = -1;
		MPI_Comm_rank(m_comm, &rank);
		return static_cast<node_id>(rank);
	}
};
} // namespace celerity::detail
