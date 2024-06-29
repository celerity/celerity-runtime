#pragma once

#include <memory>

#include "buffer_manager.h"
#include "command.h"
#include "config.h"
#include "device_queue.h"
#include "host_queue.h"
#include "recorders.h"
#include "types.h"

namespace celerity {

namespace experimental::bench::detail {
	class user_benchmarker;
} // namespace experimental::bench::detail

namespace detail {

	class buffer_manager;
	class reduction_manager;
	class graph_serializer;
	class command_graph;
	class scheduler;
	class legacy_executor;
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
		 * @param user_device_or_selector This optional device (overriding any other device selection strategy) or device selector can be provided by the user.
		 */
		static void init(int* argc, char** argv[], device_or_selector user_device_or_selector = auto_select_device{});

		static bool is_initialized() { return instance != nullptr; }
		static runtime& get_instance();

		~runtime();

		/**
		 * @brief Starts the runtime and all its internal components and worker threads.
		 */
		void startup();

		void shutdown();

		void sync();

		node_id get_local_nid() const { return m_local_nid; }

		size_t get_num_nodes() const { return m_num_nodes; }

		task_manager& get_task_manager() const;

		experimental::bench::detail::user_benchmarker& get_user_benchmarker() const { return *m_user_bench; }

		host_queue& get_host_queue() const { return *m_h_queue; }

		device_queue& get_device_queue() const { return *m_d_queue; }

		buffer_manager& get_buffer_manager() const;

		reduction_manager& get_reduction_manager() const;

		template <typename DataT, int Dims>
		buffer_id create_buffer(const range<3>& range, const DataT* host_init_ptr) {
			const auto bid = m_buffer_mngr->register_buffer<DataT, Dims>(range, host_init_ptr);
			this->register_buffer(bid, range, host_init_ptr != nullptr);
			return bid;
		}

		void set_buffer_debug_name(buffer_id bid, const std::string& debug_name);

		void destroy_buffer(buffer_id bid);

		host_object_id create_host_object();

		void destroy_host_object(host_object_id hoid);

		// returns the combined command graph of all nodes on node 0, an empty string on other nodes
		std::string gather_command_graph() const;

		bool is_dry_run() const { return m_cfg->is_dry_run(); }

	  private:
		inline static bool m_mpi_initialized = false;
		inline static bool m_mpi_finalized = false;

		static void mpi_initialize_once(int* argc, char*** argv);
		static void mpi_finalize_once();

		static std::unique_ptr<runtime> instance;

		// Whether the runtime is active, i.e. between startup() and shutdown().
		bool m_is_active = false;

		bool m_is_shutting_down = false;

		std::unique_ptr<config> m_cfg;
		std::unique_ptr<experimental::bench::detail::user_benchmarker> m_user_bench;
		std::unique_ptr<host_queue> m_h_queue;
		std::unique_ptr<device_queue> m_d_queue;
		size_t m_num_nodes;
		node_id m_local_nid;

		// These management classes are only constructed on the master node.
		std::unique_ptr<command_graph> m_cdag;
		std::unique_ptr<scheduler> m_schdlr;

		std::unique_ptr<buffer_manager> m_buffer_mngr;
		std::unique_ptr<reduction_manager> m_reduction_mngr;
		std::unique_ptr<host_object_manager> m_host_object_mngr;
		std::unique_ptr<task_manager> m_task_mngr;
		std::unique_ptr<legacy_executor> m_exec;

		std::unique_ptr<detail::task_recorder> m_task_recorder;
		std::unique_ptr<detail::command_recorder> m_command_recorder;

		runtime(int* argc, char** argv[], device_or_selector user_device_or_selector);
		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;

		// The outlined non-templated part of create_buffer (so we don't need definitions of task_manager or distributed_graph_generator in this header)
		void register_buffer(buffer_id bid, const range<3>& range, bool host_initialized);

		/**
		 * @brief Destroys the runtime if it is no longer active and all buffers have been unregistered.
		 */
		void maybe_destroy_runtime() const;

		// ------------------------------------------ TESTING UTILS ------------------------------------------
		// We have to jump through some hoops to be able to re-initialize the runtime for unit testing.
		// MPI does not like being initialized more than once per process, so we have to skip that part for
		// re-initialization.
		// ---------------------------------------------------------------------------------------------------

	  public:
		// Switches to test mode, where MPI will be initialized through test_case_enter() instead of runtime::runtime(). Called on Catch2 startup.
		static void test_mode_enter() {
			assert(!m_mpi_initialized);
			m_test_mode = true;
		}

		// Finalizes MPI if it was ever initialized in test mode. Called on Catch2 shutdown.
		static void test_mode_exit() {
			assert(m_test_mode && !m_test_active && !m_mpi_finalized);
			if(m_mpi_initialized) mpi_finalize_once();
		}

		// Initializes MPI for tests, if it was not initialized before
		static void test_require_mpi() {
			assert(m_test_mode && !m_test_active);
			if(!m_mpi_initialized) mpi_initialize_once(nullptr, nullptr);
		}

		// Allows the runtime to be transitively instantiated in tests. Called from runtime_fixture.
		static void test_case_enter() {
			assert(m_test_mode && !m_test_active && m_mpi_initialized);
			m_test_active = true;
		}

		static bool test_runtime_was_instantiated() {
			assert(m_test_mode && m_test_active);
			return instance != nullptr;
		}

		// Deletes the runtime instance, which happens only in tests. Called from runtime_fixture.
		static void test_case_exit();

	  private:
		inline static bool m_test_mode = false;
		inline static bool m_test_active = false;
	};

} // namespace detail
} // namespace celerity
