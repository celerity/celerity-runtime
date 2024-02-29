#pragma once

#include <chrono>
#include <unordered_map>

#include <CL/sycl.hpp>

#include <ctpl_stl.h>
#include <mpi.h>

#include "async_event.h"
#include "log.h"
#include "named_threads.h"
#include "types.h"

namespace celerity {

template <int Dims>
class partition;

namespace experimental {
	class collective_partition;
}

namespace detail {

	template <int Dims>
	partition<Dims> make_partition(const range<Dims>& global_size, const subrange<Dims>& range) {
		return partition<Dims>(global_size, range);
	}

	experimental::collective_partition make_collective_partition(const range<1>& global_size, const subrange<1>& range, MPI_Comm comm);

} // namespace detail

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

/**
 * A one-dimensional partition, additionally carrying the MPI communicator of the collective group.
 */
class experimental::collective_partition : public partition<1> {
  public:
	MPI_Comm get_collective_mpi_comm() const { return m_comm; }

	size_t get_node_index() const { return get_subrange().offset[0]; }

	size_t get_num_nodes() const { return get_global_size()[0]; }

  protected:
	friend collective_partition detail::make_collective_partition(const range<1>& global_size, const subrange<1>& range, MPI_Comm comm);

	MPI_Comm m_comm;

	collective_partition(const range<1>& global_size, const subrange<1>& range, MPI_Comm comm) : partition<1>(global_size, range), m_comm(comm) {}
};


namespace detail {

	inline experimental::collective_partition make_collective_partition(const range<1>& global_size, const subrange<1>& range, MPI_Comm comm) {
		return experimental::collective_partition(global_size, range, comm);
	}

	/**
	 * The @p host_queue provides a thread pool to submit host tasks.
	 */
	// TODO (IDAG) rework this entire class:
	//	- got rid of execution_info (we just remove CELERITY_PROFILE_KERNEL)
	//	- create a thread_pool class that can also be used by for memory alloctation etc
	class host_queue {
	  public:
		class future_event final : public async_event_base {
		  public:
			future_event(std::future<void> future) : m_future(std::move(future)) {}
			bool is_complete() const override { return m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

		  private:
			std::future<void> m_future;
		};

		host_queue() {
			// TODO what is a good thread count for the non-collective thread pool?
			m_pools.emplace(std::piecewise_construct, std::tuple{non_collective_group_id}, std::tuple{4, m_next_pool_id++});
		}

		// Concurrent, blocking collectives must run in separate threads to avoid deadlocks (the executor might schedule collectives in different orders on
		// different nodes)
		void require_collective_group(const collective_group_id cgid) {
			if(m_pools.count(cgid) == 0) { m_pools.emplace(std::piecewise_construct, std::tuple{cgid}, std::tuple{1 /* n_threads */, m_next_pool_id++}); }
		}

		template <typename Fn>
		async_event submit(Fn&& fn) {
			return submit(collective_group_id{0}, std::forward<Fn>(fn));
		}

		template <typename Fn>
		async_event submit(collective_group_id cgid, Fn&& fn) {
			auto future = m_pools.at(cgid).pool.push([fn = std::forward<Fn>(fn)](int) {
				try {
					fn();
				} catch(std::exception& e) { CELERITY_ERROR("exception in thread pool: {}", e.what()); } catch(...) {
					// TODO (IDAG) this should probably abort?
					CELERITY_ERROR("unknown exception in thread pool");
				}
			});
			return make_async_event<future_event>(std::move(future));
		}

	  private:
		struct thread_pool {
			ctpl::thread_pool pool;

			thread_pool(size_t n_threads, size_t id) : pool(n_threads) {
				for(size_t i = 0; i < n_threads; ++i) {
					auto& worker = pool.get_thread(i);
					set_thread_name(worker.native_handle(), fmt::format("cy-worker-{}.{}", id, i));
				}
			}
		};

		std::unordered_map<collective_group_id, thread_pool> m_pools;
		size_t m_next_pool_id = 0;
	};

} // namespace detail
} // namespace celerity
