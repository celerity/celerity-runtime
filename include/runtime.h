#pragma once

#include <deque>
#include <limits>
#include <memory>

#include <mpi.h>

#include "command.h"
#include "config.h"
#include "device_queue.h"
#include "host_queue.h"
#include "mpi_support.h"
#include "types.h"

namespace celerity {

namespace experimental::bench::detail {
	class user_benchmarker;
} // namespace experimental::bench::detail

namespace detail {

	class buffer_manager;
	class reduction_manager;
	class graph_generator;
	class graph_serializer;
	class command_graph;
	class scheduler;
	class executor;
	class task_manager;
	class host_object_manager;

	class runtime_already_started_error : public std::runtime_error {
	  public:
		runtime_already_started_error() : std::runtime_error("The Celerity runtime has already been started") {}
	};

	class runtime {
		friend struct runtime_testspy;

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

		experimental::bench::detail::user_benchmarker& get_user_benchmarker() const { return *user_bench; }

		host_queue& get_host_queue() const { return *h_queue; }

		device_queue& get_device_queue() const { return *d_queue; }

		buffer_manager& get_buffer_manager() const;

		reduction_manager& get_reduction_manager() const;

		host_object_manager& get_host_object_manager() const;

	  private:
		inline static bool mpi_initialized = false;
		inline static bool mpi_finalized = false;

		static void mpi_initialize_once(int* argc, char*** argv);
		static void mpi_finalize_once();

		static std::unique_ptr<runtime> instance;

		// Whether the runtime is active, i.e. between startup() and shutdown().
		bool is_active = false;

		bool is_shutting_down = false;

		std::unique_ptr<config> cfg;
		std::unique_ptr<experimental::bench::detail::user_benchmarker> user_bench;
		std::unique_ptr<host_queue> h_queue;
		std::unique_ptr<device_queue> d_queue;
		size_t num_nodes;
		node_id local_nid;

		// These management classes are only constructed on the master node.
		std::unique_ptr<command_graph> cdag;
		std::shared_ptr<graph_generator> ggen;
		std::shared_ptr<graph_serializer> gsrlzr;
		std::unique_ptr<scheduler> schdlr;

		std::unique_ptr<buffer_manager> buffer_mngr;
		std::unique_ptr<reduction_manager> reduction_mngr;
		std::unique_ptr<host_object_manager> host_object_mngr;
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
		// Switches to test mode, where MPI will be initialized through test_case_enter() instead of runtime::runtime(). Called on Catch2 startup.
		static void test_mode_enter() {
			assert(!mpi_initialized);
			test_mode = true;
		}

		// Finalizes MPI if it was ever initialized in test mode. Called on Catch2 shutdown.
		static void test_mode_exit() {
			assert(test_mode && !test_active && !mpi_finalized);
			if(mpi_initialized) mpi_finalize_once();
		}

		// Initializes MPI for tests, if it was not initialized before
		static void test_require_mpi() {
			assert(test_mode && !test_active);
			if(!mpi_initialized) mpi_initialize_once(nullptr, nullptr);
		}

		// Allows the runtime to be transitively instantiated in tests. Called from runtime_fixture.
		static void test_case_enter() {
			assert(test_mode && !test_active && mpi_initialized);
			test_active = true;
		}

		static bool test_runtime_was_instantiated() {
			assert(test_mode && test_active);
			return instance != nullptr;
		}

		// Deletes the runtime instance, which happens only in tests. Called from runtime_fixture.
		static void test_case_exit() {
			assert(test_mode && test_active);
			instance.reset();
			test_active = false;
		}

	  private:
		inline static bool test_mode = false;
		inline static bool test_active = false;
	};

} // namespace detail
} // namespace celerity
