#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "handler.h"
#include "host_queue.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

namespace celerity {
namespace detail {

	class reduction_manager;
	class logger;
	using task_callback = std::function<void(task_id)>;

	class task_manager {
		friend struct task_manager_testspy;
		using buffer_writers_map = std::unordered_map<buffer_id, region_map<std::optional<task_id>>>;

	  public:
		task_manager(size_t num_collective_nodes, host_queue* queue, reduction_manager* reduction_mgr);

		virtual ~task_manager() = default;

		template <typename CGF, typename... Hints>
		task_id create_task(CGF cgf, Hints... hints) {
			task_id tid;
			{
				std::lock_guard lock(task_mutex);
				tid = get_new_tid();

				prepass_handler cgh(tid, std::make_unique<command_group_storage<CGF>>(cgf), num_collective_nodes);
				cgf(cgh);

				task& task_ref = register_task_internal(std::move(cgh).into_task());

				compute_dependencies(tid);
				if(queue) queue->require_collective_group(task_ref.get_collective_group_id());
				if(need_new_horizon()) { generate_task_horizon(); }
			}
			invoke_callbacks(tid);
			return tid;
		}

		/**
		 * @brief Registers a new callback that will be called whenever a new task is created.
		 */
		void register_task_callback(task_callback cb) { task_callbacks.push_back(cb); }

		/**
		 * @brief Adds a new buffer for dependency tracking
		 * @arg host_initialized Whether this buffer has been initialized using a host pointer (i.e., it contains useful data before any write-task)
		 */
		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized);

		/**
		 * @brief Checks whether a task has already been registered with the queue.
		 *
		 * This is useful for scenarios where the master node sends out commands concerning tasks
		 * that have not yet been registered through the local execution of the user program.
		 */
		bool has_task(task_id tid) const;

		const task* get_task(task_id tid) const;

		/**
		 * @brief Returns the id of the INIT task which acts as a surrogate for the host-initialization of buffers.
		 *
		 * While this is always 0, having this method makes code dealing with the INIT task more explicit.
		 */
		task_id get_init_task_id() const {
			assert(init_task_id == 0);
			return init_task_id;
		}

		void print_graph(logger& graph_logger) const;

		/**
		 * @brief Shuts down the task_manager, freeing all stored tasks.
		 */
		void shutdown() { task_map.clear(); }

		unsigned get_max_pseudo_critical_path_length() const { return max_pseudo_critical_path_length; }

		const std::unordered_set<task*>& get_execution_front() { return execution_front; }

		void set_horizon_step(const int step) { task_horizon_step_size = step; }

	  private:
		const size_t num_collective_nodes;
		host_queue* queue;

		reduction_manager* reduction_mngr;

		task_id next_task_id = 0;
		const task_id init_task_id;
		std::unordered_map<task_id, std::unique_ptr<task>> task_map;

		// We store a map of which task last wrote to a certain region of a buffer.
		// NOTE: This represents the state after the latest performed pre-pass.
		buffer_writers_map buffers_last_writers;

		std::unordered_map<collective_group_id, task_id> last_collective_tasks;

		// For simplicity we use a single mutex to control access to all task-related (i.e. the task graph, task_map, ...) data structures.
		mutable std::mutex task_mutex;

		std::vector<task_callback> task_callbacks;

		// maximum critical path length in the task graph before inserting a horizon
		int task_horizon_step_size = 4;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		int max_pseudo_critical_path_length = 0;
		int previous_horizon_critical_path_length = 0;

		task* previous_horizon_task = nullptr;

		// Set of tasks with no dependents
		std::unordered_set<task*> execution_front;

		task_id get_new_tid();

		task& register_task_internal(std::unique_ptr<task> task);

		void invoke_callbacks(task_id tid);

		void add_dependency(task* depender, task* dependee, dependency_kind kind = dependency_kind::TRUE_DEP);

		bool need_new_horizon() const;

		void generate_task_horizon();

		void compute_dependencies(task_id tid);
	};

} // namespace detail
} // namespace celerity
