#pragma once

#include "cgf.h"
#include "device_selector.h"
#include "ranges.h"
#include "types.h"

#include <memory>


namespace celerity {
namespace detail {

	class backend;
	class host_queue;
	class reducer;
	struct host_object_instance;

	class runtime {
		friend struct runtime_testspy;

	  public:
		/**
		 * @param user_device_or_selector This optional device (overriding any other device selection strategy) or device selector can be provided by the user.
		 */
		static void init(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector = auto_select_devices{}, const bool init_mpi = true);

		static bool has_instance() { return s_instance.m_impl != nullptr; }

		static void shutdown();

		static runtime& get_instance();

		runtime(const runtime&) = delete;
		runtime(runtime&&) = delete;
		runtime& operator=(const runtime&) = delete;
		runtime& operator=(runtime&&) = delete;
		~runtime() = default;

		task_id submit(raw_command_group&& cg);

		task_id fence(buffer_access access, std::unique_ptr<task_promise> fence_promise);

		task_id fence(host_object_effect effect, std::unique_ptr<task_promise> fence_promise);

		task_id sync(detail::epoch_action action);

		void create_queue();

		void destroy_queue();

		allocation_id create_user_allocation(void* ptr);

		buffer_id create_buffer(const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid);

		void set_buffer_debug_name(buffer_id bid, const std::string& debug_name);

		void destroy_buffer(buffer_id bid);

		host_object_id create_host_object(std::unique_ptr<host_object_instance> instance /* optional */);

		void destroy_host_object(host_object_id hoid);

		reduction_id create_reduction(std::unique_ptr<reducer> reducer);

		bool is_dry_run() const;

		void set_scheduler_lookahead(experimental::lookahead lookahead);

		void flush_scheduler();

		void initialize_new_loop_template();

		void begin_loop_iteration();

		void complete_loop_iteration();

		void finalize_loop_template();

		// Hacks for RnD
		backend* NOCOMMIT_get_backend_ptr() const;
		node_id NOCOMMIT_get_local_nid() const;
		size_t NOCOMMIT_get_num_nodes() const;
		size_t NOCOMMIT_get_num_local_devices() const;

		const std::vector<sycl::device>& NOCOMMIT_get_sycl_devices() const;

		void leak_memory();

	  private:
		class impl;

		static bool s_mpi_initialized;
		static bool s_mpi_finalized;

		static bool s_test_mode;
		static bool s_test_active;
		static bool s_test_runtime_was_instantiated;

		static void mpi_initialize_once(int* argc, char*** argv);
		static void mpi_finalize_once();

		static runtime s_instance;

		std::unique_ptr<impl> m_impl;

		runtime() = default;
	};

	/// Returns the combined command graph of all nodes on node 0, an empty string on other nodes
	std::string gather_command_graph(const std::string& graph_str, const size_t num_nodes, const node_id local_nid);

} // namespace detail
} // namespace celerity
