#pragma once

#include <chrono>
#include <memory>
#include <unordered_map>

#include <CL/sycl.hpp>

#include <ctpl_stl.h>
#include <mpi.h>

#include "config.h"
#include "types.h"

namespace celerity {

template <int Dims>
class partition;

namespace experimental {
	class collective_partition;
}

namespace detail {

	template <int Dims>
	class sized_partition_base {
	  public:
		explicit sized_partition_base(const celerity::range<Dims>& global_size, const subrange<Dims>& range)
		    : global_size(range_cast<Dims>(global_size)), range(range) {}

		/** The subrange handled by this host. */
		const subrange<Dims>& get_subrange() const { return range; }

		/** The size of the entire iteration space */
		const celerity::range<Dims>& get_global_size() const { return global_size; }

	  private:
		celerity::range<Dims> global_size;
		subrange<Dims> range;
	};

	template <int Dims>
	partition<Dims> make_partition(const celerity::range<Dims>& global_size, const subrange<Dims>& range) {
		return partition<Dims>(global_size, range);
	}

	partition<0> make_0d_partition();

	experimental::collective_partition make_collective_partition(const celerity::range<1>& global_size, const subrange<1>& range, MPI_Comm comm);

} // namespace detail

/**
 * Represents the sub-range of the iteration space handled by each host in a host_task.
 */
template <int Dims>
class partition : public detail::sized_partition_base<Dims> {
  protected:
	friend partition<Dims> detail::make_partition<Dims>(const celerity::range<Dims>& global_size, const subrange<Dims>& range);

	partition(const celerity::range<Dims>& global_size, const subrange<Dims>& range) : detail::sized_partition_base<Dims>(global_size, range) {}
};

/**
 * A one-dimensional partition, additionally carrying the MPI communicator of the collective group.
 */
class experimental::collective_partition : public partition<1> {
  public:
	MPI_Comm get_collective_mpi_comm() const { return comm; }

	size_t get_node_index() const { return get_subrange().offset[0]; }

	size_t get_num_nodes() const { return get_global_size()[0]; }

  protected:
	friend collective_partition detail::make_collective_partition(const celerity::range<1>& global_size, const subrange<1>& range, MPI_Comm comm);

	MPI_Comm comm;

	collective_partition(const celerity::range<1>& global_size, const subrange<1>& range, MPI_Comm comm) : partition<1>(global_size, range), comm(comm) {}
};

template <>
class partition<0> {
  private:
	partition() noexcept = default;

	friend partition<0> detail::make_0d_partition();
};


namespace detail {

	inline experimental::collective_partition make_collective_partition(const celerity::range<1>& global_size, const subrange<1>& range, MPI_Comm comm) {
		return experimental::collective_partition(global_size, range, comm);
	}

	inline partition<0> make_0d_partition() { return {}; }

	/**
	 * The @p host_queue provides a thread pool to submit host tasks.
	 */
	class host_queue {
	  public:
		struct execution_info {
			using time_point = std::chrono::steady_clock::time_point;

			time_point submit_time{};
			time_point start_time{};
			time_point end_time{};
		};

		host_queue() {
			// TODO what is a good thread count for the non-collective thread pool?
			threads.emplace(std::piecewise_construct, std::tuple{0}, std::tuple{MPI_COMM_NULL, 4});
		}

		void require_collective_group(collective_group_id cgid) {
			if(threads.find(cgid) != threads.end()) return;
			assert(cgid != 0);
			MPI_Comm comm;
			MPI_Comm_dup(MPI_COMM_WORLD, &comm);
			threads.emplace(std::piecewise_construct, std::tuple{cgid}, std::tuple{comm, 1});
		}

		template <typename Fn>
		std::future<execution_info> submit(Fn&& fn) {
			return submit(collective_group_id{0}, std::forward<Fn>(fn));
		}

		template <typename Fn>
		std::future<execution_info> submit(collective_group_id cgid, Fn&& fn) {
			auto it = threads.find(cgid);
			assert(it != threads.end());
			return it->second.thread.push([fn = std::forward<Fn>(fn), submit_time = std::chrono::steady_clock::now(), comm = it->second.comm](int) {
				auto start_time = std::chrono::steady_clock::now();
				try {
					fn(comm);
				} catch(std::exception& e) { CELERITY_ERROR("exception in thread pool: {}", e.what()); } catch(...) {
					CELERITY_ERROR("unknown exception in thread pool");
				}
				auto end_time = std::chrono::steady_clock::now();
				return execution_info{submit_time, start_time, end_time};
			});
		}

		/**
		 * @brief Waits until all currently submitted operations have completed.
		 */
		void wait() {
			for(auto& ct : threads) {
				ct.second.thread.stop(true /* isWait */);
			}
		}

	  private:
		struct comm_thread {
			MPI_Comm comm;
			ctpl::thread_pool thread;

			comm_thread(MPI_Comm comm, size_t n_threads) : comm(comm), thread(n_threads) {}
		};

		std::unordered_map<collective_group_id, comm_thread> threads;
	};

} // namespace detail
} // namespace celerity
