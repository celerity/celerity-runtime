#pragma once

#include "cgf.h"
#include "device_selection.h"
#include "ranges.h"
#include "types.h"

#include <memory>


namespace celerity {
namespace detail {

	class host_queue;
	class reducer;
	struct host_object_instance;

	class runtime {
		friend struct runtime_testspy;

	  public:
		/**
		 * @param user_device_or_selector This optional device (overriding any other device selection strategy) or device selector can be provided by the user.
		 */
		static void init(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector = auto_select_devices{});

		static bool has_instance() { return s_instance != nullptr; }

		static void shutdown();

		static runtime& get_instance();

		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;
		runtime& operator=(const runtime&) = delete;
		runtime& operator=(runtime&&) = delete;

		virtual ~runtime() = default;

		virtual task_id submit(raw_command_group&& cg) = 0;

		virtual task_id fence(buffer_access access, std::unique_ptr<task_promise> fence_promise) = 0;

		virtual task_id fence(host_object_effect effect, std::unique_ptr<task_promise> fence_promise) = 0;

		virtual task_id sync(detail::epoch_action action) = 0;

		virtual void create_queue() = 0;

		virtual void destroy_queue() = 0;

		virtual allocation_id create_user_allocation(void* ptr) = 0;

		virtual buffer_id create_buffer(const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid) = 0;

		virtual void set_buffer_debug_name(buffer_id bid, const std::string& debug_name) = 0;

		virtual void destroy_buffer(buffer_id bid) = 0;

		virtual host_object_id create_host_object(std::unique_ptr<host_object_instance> instance /* optional */) = 0;

		virtual void destroy_host_object(host_object_id hoid) = 0;

		virtual reduction_id create_reduction(std::unique_ptr<reducer> reducer) = 0;

		virtual bool is_dry_run() const = 0;

		virtual void set_scheduler_lookahead(experimental::lookahead lookahead) = 0;

		virtual void flush_scheduler() = 0;

	  protected:
		inline static bool s_mpi_initialized = false;
		inline static bool s_mpi_finalized = false;

		static void mpi_initialize_once(int* argc, char*** argv);
		static void mpi_finalize_once();

		static std::unique_ptr<runtime> s_instance;

		runtime() = default;

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

	  protected:
		inline static bool s_test_mode = false;
		inline static bool s_test_active = false;
		inline static bool s_test_runtime_was_instantiated = false;
	};

	/// Returns the combined command graph of all nodes on node 0, an empty string on other nodes
	std::string gather_command_graph(const std::string& graph_str, const size_t num_nodes, const node_id local_nid);

} // namespace detail
} // namespace celerity
