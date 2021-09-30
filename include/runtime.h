#pragma once

#include <deque>
#include <limits>
#include <memory>

#include <mpi.h>

#include "command.h"
#include "config.h"
#include "device_queue.h"
#include "host_queue.h"
#include "logger.h"
#include "mpi_support.h"
#include "types.h"

namespace celerity {
namespace detail {

	class buffer_manager;
	class reduction_manager;
	class graph_generator;
	class graph_serializer;
	class command_graph;
	class scheduler;
	class executor;
	class task_manager;

	class runtime_already_started_error : public std::runtime_error {
	  public:
		runtime_already_started_error() : std::runtime_error("The Celerity runtime has already been started") {}
	};

	class runtime {
	  public:
		/**
		 * @param user_device This optional device can be provided by the user, overriding any other device selection strategy.
		 */
		static void init(int* argc, char** argv[], cl::sycl::device* user_device = nullptr);
		static bool is_initialized() { return instance != nullptr; }
		static runtime& get_instance();

		~runtime();

		/**
		 * @brief Starts the runtime and all its internal components and worker threads.
		 */
		void startup();

		void shutdown() noexcept;

		void sync() noexcept;

		bool is_master_node() const { return local_nid == 0; }

		size_t get_num_nodes() const { return num_nodes; }

		task_manager& get_task_manager() const;

		host_queue& get_host_queue() const { return *h_queue; }

		device_queue& get_device_queue() const { return *d_queue; }

		buffer_manager& get_buffer_manager() const;

		reduction_manager& get_reduction_manager() const;

		std::shared_ptr<logger> get_logger() const { return default_logger; }

		/**
		 * @brief Broadcasts the specified control command to all workers.
		 * @internal
		 */
		void broadcast_control_command(command_type cmd, const command_data& data);

	  private:
		static std::unique_ptr<runtime> instance;
		std::shared_ptr<logger> default_logger;
		std::shared_ptr<logger> graph_logger;

		// Whether the runtime is active, i.e. between startup() and shutdown().
		bool is_active = false;

		bool is_shutting_down = false;

		std::unique_ptr<config> cfg;
		std::unique_ptr<host_queue> h_queue;
		std::unique_ptr<device_queue> d_queue;
		size_t num_nodes;
		node_id local_nid;

		uint64_t sync_id = 0;

		// We reserve the upper half of command IDs for control commands.
		command_id next_control_command_id = command_id(1) << (std::numeric_limits<command_id::underlying_t>::digits - 1);

		// These management classes are only constructed on the master node.
		std::unique_ptr<command_graph> cdag;
		std::shared_ptr<graph_generator> ggen;
		std::shared_ptr<graph_serializer> gsrlzr;
		std::unique_ptr<scheduler> schdlr;

		std::unique_ptr<buffer_manager> buffer_mngr;
		std::unique_ptr<reduction_manager> reduction_mngr;
		std::unique_ptr<task_manager> task_mngr;
		std::unique_ptr<executor> exec;

		struct flush_handle {
			command_pkg pkg;
			std::vector<command_id> dependencies;
			MPI_Request req;
			mpi_support::single_use_data_type data_type;
		};
		std::deque<flush_handle> active_flushes;

		runtime(int* argc, char** argv[], cl::sycl::device* user_device = nullptr);
		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;

		void handle_buffer_registered(buffer_id bid);
		void handle_buffer_unregistered(buffer_id bid);

		/**
		 * @brief Destroys the runtime if it is no longer active and all buffers have been unregistered.
		 */
		void maybe_destroy_runtime() const;

		void flush_command(node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies);

		// ------------------------------------------ TESTING UTILS ------------------------------------------
		// We have to jump through some hoops to be able to re-initialize the runtime for unit testing.
		// MPI does not like being initialized more than once per process, so we have to skip that part for
		// re-initialization.
		// ---------------------------------------------------------------------------------------------------

	  public:
		/**
		 * @brief Enables test mode, which ensures the MPI lifecycle methods are only called once per process.
		 */
		static void enable_test_mode() {
			assert(!is_initialized() && !test_mode);
			// Initialize normally one time to setup MPI
			init(nullptr, nullptr, nullptr);
			test_mode = true;
			teardown();
		}

		static void teardown() {
			assert(test_mode);
			instance.reset();
		}

		static void finish_test_mode() {
			assert(test_mode && !instance);
			MPI_Finalize();
		}

	  private:
		inline static bool test_mode = false;
	};

} // namespace detail
} // namespace celerity
