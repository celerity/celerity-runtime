#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <ctpl.h>
#include <mpi.h>

#include "buffer_storage.h"
#include "config.h"
#include "device_queue.h"
#include "logger.h"
#include "mpi_support.h"
#include "types.h"

namespace celerity {
namespace detail {

	class buffer_storage_base;
	class graph_generator;
	class scheduler;
	class executor;
	class task_manager;

	class runtime {
	  public:
		static void init(int* argc, char** argv[]);
		static bool is_initialized() { return instance != nullptr; }
		static runtime& get_instance();

		~runtime();

		/**
		 * @brief Starts the runtime and all its internal components and worker threads.
		 *
		 * @param user_device This optional device can be provided by the user, overriding any other device selection strategy.
		 */
		void startup(cl::sycl::device* user_device);
		void shutdown();

		/**
		 * @brief Whether this node is the master node.
		 *
		 * This function should generally be used with care, as branching based on node type can lead
		 * to diverging Celerity user code (i.e. task definitions, buffers etc), which causes undefined behavior.
		 */
		bool is_master_node() const { return is_master; }

		task_manager& get_task_manager() const;

		device_queue& get_device_queue() const { return *queue; }

		buffer_id register_buffer(cl::sycl::range<3> range, std::shared_ptr<buffer_storage_base> buf_storage, bool host_initialized);

		/**
		 * This is currently a no-op. We don't know whether it is safe to free a buffer.
		 * TODO: We could mark when a buffer is no longer needed in the task graph, and free the memory accordingly.
		 */
		void unregister_buffer(buffer_id bid) {}

		/**
		 * @brief Checks whether the buffer with id \p bid has already been registered with the runtime.
		 *
		 * This is useful in rare situations where worker nodes might receive data for buffers they haven't registered yet.
		 */
		bool has_buffer(buffer_id bid) const {
			std::lock_guard<std::mutex> lock(buffer_mutex);
			return buffer_ptrs.count(bid) == 1;
		}

		std::shared_ptr<raw_data_read_handle> get_buffer_data(buffer_id bid, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) const {
			std::lock_guard<std::mutex> lock(buffer_mutex);
			assert(buffer_ptrs.count(bid) == 1);
			return buffer_ptrs.at(bid)->get_data(queue->get_sycl_queue(), offset, range);
		}

		void set_buffer_data(buffer_id bid, const raw_data_handle& dh) {
			std::lock_guard<std::mutex> lock(buffer_mutex);
			assert(buffer_ptrs.count(bid) == 1);
			buffer_ptrs[bid]->set_data(queue->get_sycl_queue(), dh);
		}

		std::shared_ptr<logger> get_logger() const { return default_logger; }

		/**
		 * @brief Executes the given functor \p fun in a thread pool.
		 * @internal
		 */
		template <typename Functor>
		auto execute_async_pooled(Functor&& fun) -> std::future<decltype(fun())> {
			// Wrap inside lambda to get rid of mandatory thread id argument provided by CTPL
			return thread_pool.push([fun](int id) { return fun(); });
		}

	  private:
		static std::unique_ptr<runtime> instance;
		std::shared_ptr<logger> default_logger;
		std::shared_ptr<logger> graph_logger;

		// Whether the runtime is active, i.e. between startup() and shutdown().
		bool is_active = false;

		std::unique_ptr<config> cfg;
		std::unique_ptr<device_queue> queue;
		size_t num_nodes;
		bool is_master;

		size_t buffer_count = 0;
		std::unordered_map<buffer_id, std::shared_ptr<buffer_storage_base>> buffer_ptrs;
		mutable std::mutex buffer_mutex;

		// The graph generator and scheduler are only constructed on the master node.
		std::shared_ptr<graph_generator> ggen;
		std::unique_ptr<scheduler> schdlr;

		std::shared_ptr<task_manager> task_mngr;
		std::unique_ptr<executor> exec;

		// TODO: What is a good size for this?
		ctpl::thread_pool thread_pool{3};

		struct flush_handle {
			command_pkg pkg;
			std::vector<command_id> dependencies;
			MPI_Request req;
			mpi_support::single_use_data_type data_type;
		};
		std::deque<flush_handle> active_flushes;

		runtime(int* argc, char** argv[]);
		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;

		void flush_command(node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies);

#ifdef CELERITY_TEST
		// ------------------------------------------ TESTING UTILS ------------------------------------------
		// We have to jump through some hoops to be able to re-initialize the runtime for unit testing.
		// MPI does not like being initialized more than once per process, so we have to skip that part for
		// re-initialization.
		// ---------------------------------------------------------------------------------------------------

	  public:
		/**
		 * Initializes the runtime singleton without running the MPI lifecycle more than once per process.
		 */
		static void init_for_testing() {
			if(instance == nullptr) {
				init(nullptr, nullptr);
				return;
			}
			test_skip_mpi_lifecycle = true;
			instance.reset();
			init(nullptr, nullptr);
			test_skip_mpi_lifecycle = false;
		}
#endif
	  private:
		static bool test_skip_mpi_lifecycle;
	};

} // namespace detail
} // namespace celerity
