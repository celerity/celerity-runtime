#pragma once

#include "types.h"

namespace celerity::detail {

/*
 * @brief Defines an interface for a communicator that can be used to communicate between nodes.
 *
 * This interface is used to abstract away the communication between nodes. This allows us to use different communication backends during testing and
 * runtime. For example, we can use MPI for the runtime and a custom implementation for testing.
 */
class communicator {
  public:
	communicator() = default;
	communicator(const communicator&) = delete;
	communicator(communicator&&) noexcept = default;

	communicator& operator=(const communicator&) = delete;
	communicator& operator=(communicator&&) noexcept = default;

	virtual ~communicator() = default;

	template <typename S>
	void allgather_inplace(S* sendrecvbuf, const int sendrecvcount) {
		allgather_inplace_impl(reinterpret_cast<std::byte*>(sendrecvbuf), sendrecvcount * sizeof(S));
	}

	template <typename S, typename R>
	void allgather(const S* sendbuf, const int sendcount, R* recvbuf, const int recvcount) {
		allgather_impl(reinterpret_cast<const std::byte*>(sendbuf), sendcount * sizeof(S), reinterpret_cast<std::byte*>(recvbuf), recvcount * sizeof(R));
	}

	void barrier() { barrier_impl(); }

	size_t get_num_nodes() { return num_nodes_impl(); }

	node_id get_local_nid() { return local_nid_impl(); }

  protected:
	virtual void allgather_inplace_impl(std::byte* sendrecvbuf, const int sendrecvcount) = 0;
	virtual void allgather_impl(const std::byte* sendbuf, const int sendcount, std::byte* recvbuf, const int recvcount) = 0;
	virtual void barrier_impl() = 0;
	virtual size_t num_nodes_impl() = 0;
	virtual node_id local_nid_impl() = 0;
};
} // namespace celerity::detail