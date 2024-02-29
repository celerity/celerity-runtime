#pragma once

#include <memory>
#include <unordered_set>

#include "config.h"
#include "device_selection.h"
#include "executor.h"
#include "recorders.h"
#include "scheduler.h"
#include "types.h"

namespace celerity {

namespace experimental::bench::detail {
	class user_benchmarker;
} // namespace experimental::bench::detail

namespace detail {

	class host_queue;
	class runtime_reduction;
	class task_manager;
	struct host_object_instance;

	class runtime final : private abstract_scheduler::delegate, private executor::delegate {
		friend struct runtime_testspy;

	  public:
		/**
		 * @param user_device_or_selector This optional device (overriding any other device selection strategy) or device selector can be provided by the user.
		 */
		static void init(int* argc, char** argv[], const devices_or_selector& user_device_or_selector = auto_select_devices{});

		static bool has_instance() { return s_instance != nullptr; }

		static runtime& get_instance();

		~runtime();

		void sync(detail::epoch_action action);

		node_id get_local_nid() const { return m_local_nid; }

		size_t get_num_nodes() const { return m_num_nodes; }

		size_t get_num_local_devices() const { return m_num_local_devices; }

		task_manager& get_task_manager() const;

		experimental::bench::detail::user_benchmarker& get_user_benchmarker() const { return *m_user_bench; }

		host_queue& get_host_queue() const { return *m_h_queue; }

		void create_queue();

		void destroy_queue();

		allocation_id create_user_allocation(void* ptr);

		buffer_id create_buffer(int dims, const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid);

		void set_buffer_debug_name(buffer_id bid, const std::string& debug_name);

		void destroy_buffer(buffer_id bid);

		host_object_id create_host_object(std::unique_ptr<host_object_instance> instance = nullptr);

		void destroy_host_object(host_object_id hoid);

		reduction_id create_reduction(std::unique_ptr<runtime_reduction> reduction);

		// Returns the combined command graph of all nodes on node 0, an empty string on other nodes
		// Call *must* be preceded by a scheduler shutdown or a call to slow_full_sync() / scheduler_testspy::wait_idle()
		std::string gather_command_graph() const;

		bool is_dry_run() const { return m_cfg->is_dry_run(); }

	  private:
		inline static bool s_mpi_initialized = false;
		inline static bool s_mpi_finalized = false;

		static void mpi_initialize_once(int* argc, char*** argv);
		static void mpi_finalize_once();

		static std::unique_ptr<runtime> s_instance;

		// `runtime` is not thread safe except for its delegate implementations, so we store the id of the thread where it was instantiated (the application
		// thread) in order to throw if the user attempts to issue a runtime operation from any other thread. One case where this may happen unintentionally
		// is capturing a buffer into a host-task by value, where this capture is the last reference to the buffer: The runtime would attempt to destroy itself
		// from a thread that it also needs to await, which would at least cause a deadlock. This variable is immutable, so reading it from a different thread
		// for the purpose of the check is safe.
		std::thread::id m_application_thread;

		std::unique_ptr<config> m_cfg;
		std::unique_ptr<experimental::bench::detail::user_benchmarker> m_user_bench;
		std::unique_ptr<host_queue> m_h_queue;
		size_t m_num_nodes;
		node_id m_local_nid;
		size_t m_num_local_devices;

		// track all instances of celerity::distr_queue, celerity::buffer and celerity::host_object to know when to destroy s_instance
		bool m_has_live_queue = false;
		std::unordered_set<buffer_id> m_live_buffers;
		std::unordered_set<host_object_id> m_live_host_objects;

		buffer_id m_next_buffer_id = 0;
		raw_allocation_id m_next_user_allocation_id = 1;
		host_object_id m_next_host_object_id = 0;
		reduction_id m_next_reduction_id = 1;

		std::unique_ptr<scheduler> m_schdlr;

		std::unique_ptr<task_manager> m_task_mngr;
		std::unique_ptr<executor> m_exec;

		std::optional<task_id> m_latest_horizon_reached; // only accessed by executor thread

		std::unique_ptr<detail::task_recorder> m_task_recorder;
		std::unique_ptr<detail::command_recorder> m_command_recorder;
		std::unique_ptr<detail::instruction_recorder> m_instruction_recorder;

		runtime(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector);
		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;

		// Throw if not called from m_application_thread (see that variable for more info on the matter). Since there are thread-safe and non thread-safe member
		// functions, we call this check at the beginning of all the non-safe ones.
		void require_call_from_application_thread() const;

		// abstract_scheduler::delegate
		void flush_instructions(std::vector<const instruction*> instr) override;
		void flush_outbound_pilots(std::vector<outbound_pilot> pilot) override;

		// instruction_executor::delegate
		void horizon_reached(task_id horizon_tid) override;
		void epoch_reached(task_id epoch_tid) override;

		bool is_unreferenced() const;

		/**
		 * @brief Destroys the runtime if it is no longer active and all buffers and host objects have been unregistered.
		 */
		static void destroy_instance_if_unreferenced();

		// ------------------------------------------ TESTING UTILS ------------------------------------------
		// We have to jump through some hoops to be able to re-initialize the runtime for unit testing.
		// MPI does not like being initialized more than once per process, so we have to skip that part for
		// re-initialization.
		// ---------------------------------------------------------------------------------------------------

	  public:
		// Switches to test mode, where MPI will be initialized through test_case_enter() instead of runtime::runtime(). Called on Catch2 startup.
		static void test_mode_enter() {
			assert(!s_mpi_initialized);
			s_test_mode = true;
		}

		// Finalizes MPI if it was ever initialized in test mode. Called on Catch2 shutdown.
		static void test_mode_exit() {
			assert(s_test_mode && !s_test_active && !s_mpi_finalized);
			if(s_mpi_initialized) mpi_finalize_once();
		}

		// Initializes MPI for tests, if it was not initialized before
		static void test_require_mpi() {
			assert(s_test_mode && !s_test_active);
			if(!s_mpi_initialized) mpi_initialize_once(nullptr, nullptr);
		}

		// Allows the runtime to be transitively instantiated in tests. Called from runtime_fixture.
		static void test_case_enter() {
			assert(s_test_mode && !s_test_active && s_mpi_initialized && s_instance == nullptr);
			s_test_active = true;
			s_test_runtime_was_instantiated = false;
		}

		static bool test_runtime_was_instantiated() {
			assert(s_test_mode);
			return s_test_runtime_was_instantiated;
		}

		// Deletes the runtime instance, which happens only in tests. Called from runtime_fixture.
		static void test_case_exit() {
			assert(s_test_mode && s_test_active);
			s_instance.reset(); // for when the test case explicitly initialized the runtime but did not successfully construct a queue / buffer / ...
			s_test_active = false;
		}

	  private:
		inline static bool s_test_mode = false;
		inline static bool s_test_active = false;
		inline static bool s_test_runtime_was_instantiated = false;
	};

} // namespace detail
} // namespace celerity
