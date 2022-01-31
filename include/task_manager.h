#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "handler.h"
#include "host_queue.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

namespace celerity {
namespace detail {

	class reduction_manager;
	using task_callback = std::function<void(task_id, task_type)>;

	class task_manager {
		friend struct task_manager_testspy;
		using buffer_writers_map = std::unordered_map<buffer_id, region_map<std::optional<task_id>>>;

	  public:
		constexpr inline static task_id initial_epoch_task = 0;

		task_manager(size_t num_collective_nodes, host_queue* queue, reduction_manager* reduction_mgr);

		virtual ~task_manager() = default;

		template <typename CGF, typename... Hints>
		task_id create_task(CGF cgf, Hints... hints) {
			task_id tid;
			task_type type;
			{
				std::lock_guard lock(task_mutex);
				tid = get_new_tid();

				prepass_handler cgh(tid, std::make_unique<command_group_storage<CGF>>(cgf), num_collective_nodes);
				cgf(cgh);

				task& task_ref = register_task_internal(std::move(cgh).into_task());
				type = task_ref.get_type();

				compute_dependencies(tid);
				if(queue) queue->require_collective_group(task_ref.get_collective_group_id());
				clean_up_pre_horizon_tasks();
			}
			invoke_callbacks(tid, type);
			if(need_new_horizon()) { generate_task_horizon(); }
			return tid;
		}

		task_id end_epoch();

		/**
		 * @brief Registers a new callback that will be called whenever a new task is created.
		 */
		void register_task_callback(task_callback cb) { task_callbacks.push_back(cb); }

		/**
		 * @brief Adds a new buffer for dependency tracking
		 * @arg host_initialized Whether this buffer has been initialized using a host pointer (i.e., it contains useful data before any write-task)
		 */
		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized);

		const task* find_task(task_id tid) const;

		/**
		 * @brief Checks whether a task has already been registered with the queue.
		 *
		 * This is useful for scenarios where the master node sends out commands concerning tasks
		 * that have not yet been registered through the local execution of the user program.
		 */
		bool has_task(task_id tid) const;

		const task* get_task(task_id tid) const;

		std::optional<std::string> print_graph(size_t max_nodes) const;

		/**
		 * @brief Shuts down the task_manager, freeing all stored tasks.
		 */
		void shutdown() { task_map.clear(); }

		void set_horizon_step(const int step) {
			assert(step >= 0);
			task_horizon_step_size = step;
		}

		/**
		 * @brief Notifies the task manager that the given horizon has been executed (used for task deletion)
		 */
		void notify_horizon_executed(task_id tid);

		void notify_epoch_ended(task_id epoch_tid);

		/**
		 * Returns the number of tasks created during the lifetime of the task_manager,
		 * including tasks that have already been deleted.
		 */
		task_id get_total_task_count() const { return next_task_id; }

		/**
		 * Returns the number of tasks currently being managed by the task_manager.
		 */
		task_id get_current_task_count() const { return task_map.size(); }

	  private:
		const size_t num_collective_nodes;
		host_queue* queue;

		reduction_manager* reduction_mngr;

		task_id next_task_id = 1;
		// The current epoch is used as the last writer for host-initialized buffers.
		// To ensure correct ordering, all tasks that have no other true-dependencies depend on this task.
		// This is useful so we can correctly generate anti-dependencies onto tasks that read host-initialized buffers.
		task_id current_epoch;
		std::unordered_map<task_id, std::unique_ptr<task>> task_map;

		// We store a map of which task last wrote to a certain region of a buffer.
		// NOTE: This represents the state after the latest performed pre-pass.
		buffer_writers_map buffers_last_writers;

		std::unordered_map<collective_group_id, task_id> last_collective_tasks;

		// Stores which host object was last affected by which task.
		std::unordered_map<host_object_id, task_id> host_object_last_effects;

		// For simplicity we use a single mutex to control access to all task-related (i.e. the task graph, task_map, ...) data structures.
		mutable std::mutex task_mutex;

		std::vector<task_callback> task_callbacks;

		// maximum critical path length in the task graph before inserting a horizon
		int task_horizon_step_size = 4;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		int max_pseudo_critical_path_length = 0;
		int current_horizon_critical_path_length = 0;

		// The latest horizon task created. Will be applied as last writer once the next horizon is created.
		task* current_horizon_task = nullptr;

		// Queue of horizon tasks for which the associated commands were executed.
		// Only accessed in task_manager::notify_horizon_executed, which is always called from the executor thread - no locking needed.
		std::queue<task_id> horizon_deletion_queue;
		// marker task id for "nothing to delete" - we can safely use 0 here
		static constexpr task_id nothing_to_delete = 0;
		// task_id ready for deletion, 0 if nothing to delete (set on notify, used on new task creation)
		std::atomic<task_id> delete_before_epoch = nothing_to_delete;
		// How many horizons to delay before deleting tasks
		static constexpr int horizon_deletion_lag = 3;

		// Set of tasks with no dependents
		std::unordered_set<task*> execution_front;

		inline task_id get_new_tid() { return next_task_id++; }

		task& register_task_internal(std::unique_ptr<task> task);

		void invoke_callbacks(task_id tid, task_type type);

		void add_dependency(task* depender, task* dependee, dependency_kind kind, dependency_origin origin);

		inline bool need_new_horizon() const { return max_pseudo_critical_path_length - current_horizon_critical_path_length >= task_horizon_step_size; }

		int get_max_pseudo_critical_path_length() const { return max_pseudo_critical_path_length; }

		task& reduce_execution_front(std::unique_ptr<task> reducer);

		void apply_epoch(task* epoch);

		const std::unordered_set<task*>& get_execution_front() { return execution_front; }

		void generate_task_horizon();

		// Needs to be called while task map accesses are safe (ie. mutex is locked)
		void clean_up_pre_horizon_tasks();

		void compute_dependencies(task_id tid);
	};

} // namespace detail
} // namespace celerity
